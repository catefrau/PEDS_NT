"""Data loading: train/val/test splitting, caching, phi features, baselines."""
import os
import glob
import pickle

import numpy as np
import jax.numpy as jnp

from NTcode_config_data.config_def import GeometryConfig
from NTcode_config_data.config_run import GEO_CYL as GEO
from solvers.NTdiffusion.diffusion_solver import predict_xs
from PEDS_subdivision import context
from PEDS_subdivision.context import update_geo
from PEDS_subdivision.physics_solver import _run_NT_solver


def data_loader(*arrays, batch_size: int):
    n = arrays[0].shape[0]
    for start in range(0, n, batch_size):
        yield tuple(arr[start:start + batch_size] for arr in arrays)


def compute_batch_baselines(params_raw: np.ndarray, geo: GeometryConfig) -> jnp.ndarray:
    """Polynomial-regression XS for every sample. Returns [N, 3, 12]."""
    raw = np.stack([predict_xs(update_geo(geo, params_raw[i])) for i in range(len(params_raw))])
    floored = np.maximum(raw, 1e-6)
    masked = np.where(context.XS_MASK[None, :, :], floored, 0.0)
    return jnp.array(masked, dtype=jnp.float32)


def compute_phi_features(rawparams: np.ndarray) -> tuple:
    """
    Returns:
        phi_features: [N, G*3]  — volume-weighted mean flux per group per region
                                  order: [phi_g0_CR, phi_g0_Core, phi_g0_Mod,
                                          phi_g1_CR, phi_g1_Core, phi_g1_Mod]
    """
    phi_features = []

    for i, p in enumerate(rawparams):
        geo_i   = update_geo(GEO, p)
        xs_i    = np.array(predict_xs(geo_i), dtype=np.float32)
        k, phi_fwd_padded, _, _ = _run_NT_solver(xs_i, p, np.array([i], dtype=np.int32))

        # ── unpack geometry ───────────────────────────────────────────────────
        R       = geo_i.boundaries[-1].radius
        I       = int(R / geo_i.mesh_size)
        Delta_r = geo_i.mesh_size
        G       = geo_i.G

        # Region outer radii (CR_outer, core_outer, mod_outer)
        region_radii = [b.radius for b in geo_i.boundaries]

        # ── extract unpadded flux ─────────────────────────────────────────────
        # phi_fwd_padded layout: group g occupies indices [g*(I+1) : g*(I+1)+I]
        # (the +1 slot is the boundary point, left as zero)

        feats = []
        for g in range(G):
            phi_g = phi_fwd_padded[g * (I + 1) : g * (I + 1) + I]   # shape (I,)

            r_prev = 0.0
            for r_reg in region_radii:
                # Cell i has centre at (i + 0.5) * Delta_r
                centres = np.array([(ic + 0.5) * Delta_r for ic in range(I)])
                mask    = (centres >= r_prev) & (centres < r_reg)

                if mask.any():
                    weighted_mean = np.mean(phi_g[mask])
                    feats.append(float(weighted_mean))
                else:
                    # This region has no cells (e.g. CR radius < mesh_size)
                    feats.append(0.0)

                r_prev = r_reg

        phi_features.append(feats)  # length = G * 3

        if i % 20 == 0:
            print(f"  phi_reg precompute {i}/{len(rawparams)}", flush=True)

    return np.array(phi_features, dtype=np.float32)
    

def derive_split_indices(filepath, train_size, val_size, test_size,
                          train_seed=42, holdout_seed=0):
    """Pure index selection — no phi/XS computation, so it's cheap to call standalone."""
    data  = np.load(filepath, allow_pickle=True)
    keffs = np.array(data['keffs'], dtype=np.float32)
    sorted_idx = np.argsort(keffs)
    n_bins = 10

    holdout_rng  = np.random.default_rng(holdout_seed)
    val_per_bin  = max(1, val_size  // n_bins)
    test_per_bin = max(1, test_size // n_bins)

    val_idx, test_idx, pool_idx = [], [], []
    for bin_indices in np.array_split(sorted_idx, n_bins):
        bin_indices = bin_indices.copy()
        holdout_rng.shuffle(bin_indices)
        n_v = min(val_per_bin,  len(bin_indices) - 1)
        n_t = min(test_per_bin, len(bin_indices) - 1 - n_v)
        val_idx.extend(bin_indices[:n_v].tolist())
        test_idx.extend(bin_indices[n_v:n_v + n_t].tolist())
        pool_idx.extend(bin_indices[n_v + n_t:].tolist())

    val_idx  = np.array(val_idx[:val_size])
    test_idx = np.array(test_idx[:test_size])
    pool_idx = np.array(pool_idx)

    train_rng = np.random.default_rng(train_seed)
    train_rng.shuffle(pool_idx)
    train_idx = pool_idx[:train_size]
    if len(train_idx) < train_size or len(val_idx) < val_size or len(test_idx) < test_size:
        raise ValueError(
            f"Not enough samples! Dataset has ~{len(pool_idx) + len(val_idx) + len(test_idx)} total, "
            f"but requested train={train_size}, val={val_size}, test={test_size}. "
            f"Got: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )
    return train_idx, val_idx, test_idx


def _allocate_counts(total: int, weights: np.ndarray,
                     capacities: np.ndarray | None = None) -> np.ndarray:
    """
    Allocate an integer `total` across bins using largest-remainder rounding.
    Optional `capacities` cap each bin; any overflow is redistributed.
    """
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if total == 0:
        return np.zeros_like(weights, dtype=int)

    w = np.array(weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if np.all(w == 0):
        w = np.ones_like(w, dtype=np.float64)
    w = w / w.sum()

    raw = total * w
    alloc = np.floor(raw).astype(int)
    remainder = int(total - alloc.sum())
    if remainder > 0:
        order = np.argsort(-(raw - alloc))
        alloc[order[:remainder]] += 1

    if capacities is None:
        return alloc

    caps = np.array(capacities, dtype=int)
    if np.any(caps < 0):
        raise ValueError("capacities must be non-negative")

    alloc = np.minimum(alloc, caps)
    deficit = int(total - alloc.sum())
    if deficit <= 0:
        return alloc

    while deficit > 0:
        spare = caps - alloc
        candidates = np.where(spare > 0)[0]
        if len(candidates) == 0:
            break
        order = candidates[np.argsort(-spare[candidates])]
        for b in order:
            if deficit == 0:
                break
            alloc[b] += 1
            deficit -= 1

    if alloc.sum() != total:
        raise ValueError(
            f"Could not allocate requested {total} samples under capacities "
            f"(max possible {caps.sum()})."
        )
    return alloc


def _assign_param_strata(rawparams: np.ndarray,
                         n_bins_per_param: int = 2,
                         bin_edges_per_param=None):
    """
    Digitize each parameter into bins and (optionally) pack a joint id.

    Default: equal-count (quantile) bins per parameter.
    Optional `bin_edges_per_param`: list of length n_params, each an array of
    edges (length n_bins+1) for fixed-width / custom ranges.

    Returns
    -------
    stratum_ids : np.ndarray[int64], shape [N]  (joint id; diagnostic only)
    edges_list  : list of edge arrays actually used (one per parameter)
    bin_matrix  : np.ndarray[int], shape [N, n_params] per-parameter bin index
    """
    rawparams = np.asarray(rawparams, dtype=np.float64)
    if rawparams.ndim != 2:
        raise ValueError(f"rawparams must be 2D [N, P], got shape {rawparams.shape}")
    n, n_params = rawparams.shape
    if n_bins_per_param < 1 and bin_edges_per_param is None:
        raise ValueError("n_bins_per_param must be >= 1")

    edges_list = []
    bin_cols = []
    for p in range(n_params):
        if bin_edges_per_param is not None:
            edges = np.asarray(bin_edges_per_param[p], dtype=np.float64)
            if edges.ndim != 1 or len(edges) < 2:
                raise ValueError(
                    f"bin_edges_per_param[{p}] must be 1D with >= 2 edges, got {edges.shape}"
                )
        else:
            qs = np.linspace(0.0, 1.0, int(n_bins_per_param) + 1)
            edges = np.quantile(rawparams[:, p], qs).astype(np.float64)
            # Break ties so digitize bins stay well-defined.
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = np.nextafter(edges[i - 1], np.inf)

        edges_list.append(edges)
        b = np.digitize(rawparams[:, p], edges[1:-1], right=False)
        b = np.clip(b, 0, len(edges) - 2)
        bin_cols.append(b.astype(np.int64))

    bin_matrix = np.stack(bin_cols, axis=1)  # [N, P]
    n_bins = np.array([len(e) - 1 for e in edges_list], dtype=np.int64)
    # Pack multi-index into a single integer stratum id (diagnostics / legacy).
    stratum_ids = bin_matrix[:, 0].copy()
    for p in range(1, n_params):
        stratum_ids = stratum_ids * int(n_bins[p]) + bin_matrix[:, p]
    return stratum_ids.astype(np.int64), edges_list, bin_matrix


def _param_label_matrix(bin_matrix: np.ndarray, n_bins_per_param: int):
    """
    One-hot multi-label matrix over per-parameter bins.

    label index = param_index * n_bins + bin_index
    Shape: [N, n_params * n_bins]
    """
    bin_matrix = np.asarray(bin_matrix, dtype=np.int64)
    n, n_params = bin_matrix.shape
    n_bins = int(n_bins_per_param)
    n_labels = n_params * n_bins
    labels = np.zeros((n, n_labels), dtype=bool)
    for p in range(n_params):
        b = np.clip(bin_matrix[:, p], 0, n_bins - 1)
        labels[np.arange(n), p * n_bins + b] = True
    return labels


def _even_spaced_indices(n: int, k: int) -> np.ndarray:
    """Pick k distinct positions spread across range(n)."""
    if k <= 0:
        return np.array([], dtype=np.int64)
    if k > n:
        raise ValueError(f"Cannot pick k={k} distinct indices from n={n}")
    if k == n:
        return np.arange(n, dtype=np.int64)
    # linspace then uniquify / repair collisions by shifting
    raw = np.round(np.linspace(0, n - 1, k)).astype(np.int64)
    raw = np.clip(raw, 0, n - 1)
    chosen = []
    used = set()
    for p in raw:
        q = int(p)
        while q in used and q + 1 < n:
            q += 1
        while q in used and q - 1 >= 0:
            q -= 1
        if q in used:
            # fallback scan
            for r in range(n):
                if r not in used:
                    q = r
                    break
        used.add(q)
        chosen.append(q)
    return np.array(chosen, dtype=np.int64)


def derive_split_indices_by_params(filepath, train_size, val_size, test_size,
                                   train_seed=42, holdout_seed=0,
                                   n_bins_per_param=2, bin_edges_per_param=None,
                                   balanced=False,
                                   fuel_r_idx=2, fuel_r_weight=None):
    """
    Stratified train/val/test selection over input-parameter marginals.

    Priority:
      1. Hard equal coverage of every ``fuel_r`` quantile bin in train/val/test.
      2. Within each fuel_r bin, sort by the other parameters' bins and take
         evenly spaced holdouts so secondary ranges stay covered too.

    Holdout semantics: ``holdout_seed`` fixes val/test; ``train_seed`` selects
    train from the remaining pool inside each fuel_r bin.
    """
    del balanced, fuel_r_weight

    data = np.load(filepath, allow_pickle=True)
    rawparams = np.array(data["params_raw"], dtype=np.float32)
    n_samples = len(rawparams)
    requested_total = int(train_size + val_size + test_size)
    if requested_total > n_samples:
        raise ValueError(
            f"Requested train+val+test={requested_total} exceeds available "
            f"samples={n_samples}"
        )

    _, edges_list, bin_matrix = _assign_param_strata(
        rawparams,
        n_bins_per_param=n_bins_per_param,
        bin_edges_per_param=bin_edges_per_param,
    )
    n_params = bin_matrix.shape[1]
    n_bins = int(n_bins_per_param)
    fuel_r_idx = int(fuel_r_idx)
    if not (0 <= fuel_r_idx < n_params):
        raise ValueError(f"fuel_r_idx={fuel_r_idx} out of range for n_params={n_params}")

    secondary_params = [p for p in range(n_params) if p != fuel_r_idx]
    fuel_bins = bin_matrix[:, fuel_r_idx]
    bin_members = [np.where(fuel_bins == b)[0].astype(np.int64) for b in range(n_bins)]
    capacities = np.array([len(m) for m in bin_members], dtype=int)

    uniform = np.ones(n_bins, dtype=np.float64)
    holdout_total = int(val_size + test_size)
    holdout_counts = _allocate_counts(holdout_total, uniform, capacities=capacities)
    val_counts = _allocate_counts(
        int(val_size), holdout_counts.astype(np.float64), capacities=holdout_counts
    )
    test_counts = holdout_counts - val_counts
    remaining_cap = capacities - holdout_counts
    train_counts = _allocate_counts(
        int(train_size), uniform, capacities=remaining_cap
    )

    if (int(train_counts.sum()) != int(train_size)
            or int(val_counts.sum()) != int(val_size)
            or int(test_counts.sum()) != int(test_size)):
        raise ValueError(
            "fuel_r primary allocation mismatch: "
            f"train={train_counts.sum()} (wanted {train_size}), "
            f"val={val_counts.sum()} (wanted {val_size}), "
            f"test={test_counts.sum()} (wanted {test_size})"
        )

    holdout_rng = np.random.default_rng(holdout_seed)
    train_rng = np.random.default_rng(train_seed)

    train_parts, val_parts, test_parts = [], [], []
    for b in range(n_bins):
        members = bin_members[b].copy()
        n_val = int(val_counts[b])
        n_test = int(test_counts[b])
        n_train = int(train_counts[b])
        need_here = n_val + n_test + n_train
        if need_here == 0:
            continue
        if len(members) < need_here:
            raise ValueError(
                f"fuel_r bin {b}: have {len(members)}, need {need_here}"
            )

        # Sort by secondary param bins so even spacing covers their ranges.
        # Tiny holdout-seeded jitter breaks ties without destroying order.
        h_rng = np.random.default_rng(holdout_rng.integers(0, 2**31 - 1))
        jitter = h_rng.random(len(members))
        sort_keys = [jitter]
        for p in reversed(secondary_params):
            sort_keys.append(bin_matrix[members, p].astype(np.float64))
        order = np.lexsort(sort_keys)  # last key is primary
        sorted_members = members[order]

        holdout_n = n_val + n_test
        if holdout_n > 0:
            holdout_pos = _even_spaced_indices(len(sorted_members), holdout_n)
            holdout_pos_sorted = np.sort(holdout_pos)
            holdout_samples = sorted_members[holdout_pos_sorted]
            # Round-robin into val/test along the spaced sequence so both
            # track the same secondary sweep.
            val_b, test_b = [], []
            nv, nt = n_val, n_test
            for s in holdout_samples:
                if nv > 0 and nv >= nt:
                    val_b.append(int(s)); nv -= 1
                elif nt > 0:
                    test_b.append(int(s)); nt -= 1
                elif nv > 0:
                    val_b.append(int(s)); nv -= 1
            val_b = np.array(val_b, dtype=np.int64)
            test_b = np.array(test_b, dtype=np.int64)
        else:
            val_b = np.array([], dtype=np.int64)
            test_b = np.array([], dtype=np.int64)

        held = set(val_b.tolist()) | set(test_b.tolist())
        pool = np.array([int(i) for i in members.tolist() if int(i) not in held],
                        dtype=np.int64)
        t_rng = np.random.default_rng(train_rng.integers(0, 2**31 - 1))
        t_rng.shuffle(pool)
        if len(pool) < n_train:
            raise ValueError(
                f"fuel_r bin {b}: train pool {len(pool)} < need {n_train}"
            )
        # Prefer evenly spaced train picks from pool sorted by secondary bins
        # so train also covers secondary ranges uniformly inside the bin.
        if n_train > 0 and len(pool) > n_train:
            pool_jitter = t_rng.random(len(pool))
            pool_keys = [pool_jitter]
            for p in reversed(secondary_params):
                pool_keys.append(bin_matrix[pool, p].astype(np.float64))
            pool_order = np.lexsort(pool_keys)
            pool_sorted = pool[pool_order]
            train_pos = _even_spaced_indices(len(pool_sorted), n_train)
            train_b = pool_sorted[train_pos]
        else:
            train_b = pool[:n_train].astype(np.int64)

        train_parts.append(train_b)
        val_parts.append(val_b)
        test_parts.append(test_b)

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64)

    train_rng.shuffle(train_idx)
    holdout_rng.shuffle(val_idx)
    holdout_rng.shuffle(test_idx)

    full_label_mat = _param_label_matrix(bin_matrix, n_bins)
    meta = {
        "n_strata": int(n_bins),
        "n_labels": int(n_params * n_bins),
        "n_bins_per_param": n_bins,
        "balanced": True,
        "method": "fuel_r_primary_even_spaced",
        "fuel_r_idx": fuel_r_idx,
        "edges_list": edges_list,
        "fuel_r_train_counts": train_counts,
        "fuel_r_val_counts": val_counts,
        "fuel_r_test_counts": test_counts,
        "train_label_counts": full_label_mat[train_idx].sum(axis=0).astype(int),
        "val_label_counts": full_label_mat[val_idx].sum(axis=0).astype(int),
        "test_label_counts": full_label_mat[test_idx].sum(axis=0).astype(int),
    }
    return train_idx, val_idx, test_idx, meta


def _save_split_cache(cache_path, train_idx, val_idx, test_idx, metadata):
    """Save splits to disk for reuse across runs."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    payload = {
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "metadata": metadata,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)


def _as_1d_int_indices(indices, name):
    """Normalize cached index containers to a flat integer NumPy array."""
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _validate_split_indices(train_idx, val_idx, test_idx,
                            dataset_size, expected_train=None, expected_val=None,
                            expected_test=None, context="split"):
    """
    Ensure cached/derived splits are safe to use:
      - correct lengths (when expected_* provided)
      - all indices in bounds
      - no duplicates within each set
      - train/val/test are mutually disjoint
    Returns normalized int64 arrays.
    """
    if dataset_size is None or int(dataset_size) <= 0:
        raise ValueError(f"[{context}] invalid dataset_size={dataset_size}")
    dataset_size = int(dataset_size)

    train_idx = _as_1d_int_indices(train_idx, "train_idx")
    val_idx   = _as_1d_int_indices(val_idx, "val_idx")
    test_idx  = _as_1d_int_indices(test_idx, "test_idx")

    if expected_train is not None and len(train_idx) != int(expected_train):
        raise ValueError(
            f"[{context}] train length mismatch: expected {int(expected_train)}, got {len(train_idx)}"
        )
    if expected_val is not None and len(val_idx) != int(expected_val):
        raise ValueError(
            f"[{context}] val length mismatch: expected {int(expected_val)}, got {len(val_idx)}"
        )
    if expected_test is not None and len(test_idx) != int(expected_test):
        raise ValueError(
            f"[{context}] test length mismatch: expected {int(expected_test)}, got {len(test_idx)}"
        )

    for name, arr in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if arr.size and (arr.min() < 0 or arr.max() >= dataset_size):
            raise ValueError(
                f"[{context}] {name} indices out of bounds for dataset_size={dataset_size}"
            )
        if np.unique(arr).size != arr.size:
            raise ValueError(f"[{context}] duplicate indices inside {name} split")

    if np.intersect1d(train_idx, val_idx).size:
        raise ValueError(f"[{context}] train/val overlap detected")
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError(f"[{context}] train/test overlap detected")
    if np.intersect1d(val_idx, test_idx).size:
        raise ValueError(f"[{context}] val/test overlap detected")

    return train_idx, val_idx, test_idx


def _dataset_cache_tag(filepath: str) -> str:
    """Stable dataset id used in cache filenames and metadata checks."""
    return os.path.splitext(os.path.basename(filepath))[0]


def _find_largest_smaller_cache(cache_dir, dataset_tag, train_seed, holdout_seed, val_size, test_size, train_size):
    """Find the largest cached train_size < current request for this dataset only."""
    pattern = os.path.join(
        cache_dir,
        f"split_ds{dataset_tag}_t*_ts{train_seed}_hs{holdout_seed}_v{val_size}_t{test_size}.pkl",
    )
    cached_files = glob.glob(pattern)

    candidates = []
    for fpath in cached_files:
        fname = os.path.basename(fpath)
        try:
            parts = fname.split("_")
            t_part = next(p for p in parts if p.startswith("t") and p[1:].isdigit())
            t_val = int(t_part[1:])
            if t_val < train_size:
                candidates.append((t_val, fpath))
        except Exception:
            continue

    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    return candidates[0][0], candidates[0][1]


def load_or_create_split_cache(filepath, train_size, val_size, test_size,
                                train_seed=42, holdout_seed=0, cache_path=None,
                                use_cache=True):
    """
    Load train/val/test splits from cache if available and compatible.
    If dataset grew: val/test stay the same, train extends with new samples.
    
    Fallback: if exact train_size not found, looks for largest smaller train_size
    and extends from there.
    """
    data = np.load(filepath, allow_pickle=True)
    current_dataset_size = len(data['params'])
    dataset_tag = _dataset_cache_tag(filepath)
    print(f"  Split request: train={train_size}, val={val_size}, test={test_size}, "
          f"train_seed={train_seed}, holdout_seed={holdout_seed}, "
          f"dataset_size={current_dataset_size}")
    print(f"  Dataset cache tag: {dataset_tag}")

    if not use_cache:
        print("  Split cache: DISABLED (USE_SPLIT_CACHE=False)")
        print(f"  [cache bypass] computing fresh splits (train_seed={train_seed})…")
        train_idx, val_idx, test_idx = derive_split_indices(
            filepath, train_size, val_size, test_size, train_seed, holdout_seed
        )
        train_idx, val_idx, test_idx = _validate_split_indices(
            train_idx, val_idx, test_idx,
            dataset_size=current_dataset_size,
            expected_train=train_size,
            expected_val=val_size,
            expected_test=test_size,
            context="fresh split / cache disabled"
        )
        return train_idx, val_idx, test_idx

    if cache_path is None:
        cache_dir = os.path.join(os.path.dirname(filepath) or ".", ".split_cache")
    else:
        cache_dir = cache_path

    os.makedirs(cache_dir, exist_ok=True)

    cache_fname = (
        f"split_ds{dataset_tag}_t{train_size}_ts{train_seed}_hs{holdout_seed}_v{val_size}_t{test_size}.pkl"
    )
    cache_path = os.path.join(cache_dir, cache_fname)
    print(f"  Split cache: {cache_path}")
    
    # ── Try to load EXACT cache ──────────────────────────────────────────────
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            meta = cache["metadata"]
            old_train_idx = cache["train_idx"]
            old_val_idx   = cache["val_idx"]
            old_test_idx  = cache["test_idx"]
            old_dataset_size = meta.get("dataset_size")
            
            # Seeds + dataset identity match: cache belongs to this request
            if (meta.get("dataset_tag") == dataset_tag and
                meta.get("dataset_path") == os.path.abspath(filepath) and
                meta.get("train_seed") == train_seed and
                meta.get("holdout_seed") == holdout_seed and
                meta.get("val_size") == val_size and
                meta.get("test_size") == test_size):
                
                # Dataset unchanged: reuse exactly
                if current_dataset_size == old_dataset_size:
                    if meta.get("train_size") == train_size:
                        old_train_idx, old_val_idx, old_test_idx = _validate_split_indices(
                            old_train_idx, old_val_idx, old_test_idx,
                            dataset_size=current_dataset_size,
                            expected_train=train_size,
                            expected_val=val_size,
                            expected_test=test_size,
                            context="exact cache"
                        )
                        print(f"  [cache hit] reusing {len(old_train_idx)} train, "
                              f"{len(old_val_idx)} val, {len(old_test_idx)} test samples")
                        return old_train_idx, old_val_idx, old_test_idx
                
                # Dataset grew: extend training set
                if current_dataset_size > old_dataset_size:
                    old_train_idx, old_val_idx, old_test_idx = _validate_split_indices(
                        old_train_idx, old_val_idx, old_test_idx,
                        dataset_size=old_dataset_size,
                        expected_train=meta.get("train_size"),
                        expected_val=val_size,
                        expected_test=test_size,
                        context="exact cache / dataset grew"
                    )
                    print("FOUND AN EXACT CACHE, BUT DATASET GREW")
                    if len(old_train_idx) > train_size:
                        raise ValueError(
                            f"Cached train split ({len(old_train_idx)}) is larger than "
                            f"requested train_size={train_size}"
                        )
                    print(f"  [cache hit, dataset grew] {old_dataset_size} → {current_dataset_size} samples")
                    print(f"    extending train from {len(old_train_idx)} to {train_size}…")
                    
                    old_all = np.union1d(old_train_idx, np.union1d(old_val_idx, old_test_idx))
                    new_pool = np.setdiff1d(np.arange(current_dataset_size), old_all)
                    n_needed = train_size - len(old_train_idx)
                    
                    if len(new_pool) < n_needed:
                        raise ValueError(
                            f"Not enough new samples ({len(new_pool)}) to reach "
                            f"train_size={train_size} (need {n_needed} more)"
                        )
                    
                    extend_rng = np.random.default_rng(train_seed)
                    extend_rng.shuffle(new_pool)
                    new_indices = new_pool[:n_needed]
                    
                    train_idx = np.concatenate([old_train_idx, new_indices])
                    val_idx   = old_val_idx
                    test_idx  = old_test_idx
                    train_idx, val_idx, test_idx = _validate_split_indices(
                        train_idx, val_idx, test_idx,
                        dataset_size=current_dataset_size,
                        expected_train=train_size,
                        expected_val=val_size,
                        expected_test=test_size,
                        context="extended exact cache"
                    )
                    
                    meta["dataset_size"] = current_dataset_size
                    meta["train_size"] = train_size
                    _save_split_cache(cache_path, train_idx, val_idx, test_idx, meta)
                    return train_idx, val_idx, test_idx
        
        except Exception as e:
            print(f"  [cache load failed] {e}, trying fallback…")
    
    # ── Fallback: look for largest smaller cached train_size ─────────────────
    prev_train_size, prev_cache_path = _find_largest_smaller_cache(
        cache_dir, dataset_tag, train_seed, holdout_seed, val_size, test_size, train_size
    )
    
    if prev_cache_path is not None:
        print(f"  [fallback] found previous cache for train_size={prev_train_size}")
        print(f"    extending from {prev_cache_path} to {train_size}…")
        
        try:
            with open(prev_cache_path, "rb") as f:
                cache = pickle.load(f)
            meta = cache["metadata"]
            old_train_idx = cache["train_idx"]
            old_val_idx   = cache["val_idx"]
            old_test_idx  = cache["test_idx"]
            old_dataset_size = meta.get("dataset_size")
            if (meta.get("dataset_tag") != dataset_tag or
                meta.get("dataset_path") != os.path.abspath(filepath)):
                raise ValueError("fallback cache belongs to a different dataset")
            old_train_idx, old_val_idx, old_test_idx = _validate_split_indices(
                old_train_idx, old_val_idx, old_test_idx,
                dataset_size=old_dataset_size,
                expected_train=prev_train_size,
                expected_val=val_size,
                expected_test=test_size,
                context="fallback cache"
            )
            if len(old_train_idx) > train_size:
                raise ValueError(
                    f"Fallback cached train split ({len(old_train_idx)}) is larger than "
                    f"requested train_size={train_size}"
                )

            old_all = np.union1d(old_train_idx, np.union1d(old_val_idx, old_test_idx))
            new_pool = np.setdiff1d(np.arange(current_dataset_size), old_all)
            n_needed = train_size - len(old_train_idx)

            if len(new_pool) < n_needed:
                raise ValueError(
                    f"Not enough new samples ({len(new_pool)}) to extend train "
                    f"from {len(old_train_idx)} to {train_size} (need {n_needed})"
                )

            extend_rng = np.random.default_rng(train_seed)
            extend_rng.shuffle(new_pool)
            new_indices = new_pool[:n_needed]
            
            train_idx = np.concatenate([old_train_idx, new_indices])
            val_idx   = old_val_idx
            test_idx  = old_test_idx
            train_idx, val_idx, test_idx = _validate_split_indices(
                train_idx, val_idx, test_idx,
                dataset_size=current_dataset_size,
                expected_train=train_size,
                expected_val=val_size,
                expected_test=test_size,
                context="extended fallback cache"
            )

            meta["dataset_size"] = current_dataset_size
            meta["train_size"]   = train_size
            _save_split_cache(cache_path, train_idx, val_idx, test_idx, meta)
            print(f"    ✓ extended: {len(old_train_idx)} → {len(train_idx)} samples")
            return train_idx, val_idx, test_idx
        
        except Exception as e:
            print(f"  [fallback failed] {e}, computing from scratch…")
    
    # ── Cache miss: compute from scratch ─────────────────────────────────────
    print(f"  [cache miss] computing new splits (train_seed={train_seed})…")
    train_idx, val_idx, test_idx = derive_split_indices(
        filepath, train_size, val_size, test_size, train_seed, holdout_seed
    )
    train_idx, val_idx, test_idx = _validate_split_indices(
        train_idx, val_idx, test_idx,
        dataset_size=current_dataset_size,
        expected_train=train_size,
        expected_val=val_size,
        expected_test=test_size,
        context="fresh split"
    )
    
    meta = {
        "train_size": train_size, "val_size": val_size, "test_size": test_size,
        "train_seed": train_seed, "holdout_seed": holdout_seed,
        "dataset_size": current_dataset_size,
        "dataset_tag": dataset_tag,
        "dataset_path": os.path.abspath(filepath),
    }
    _save_split_cache(cache_path, train_idx, val_idx, test_idx, meta)
    return train_idx, val_idx, test_idx

def load_data(filepath, train_size, val_size, test_size,
              train_seed=42, holdout_seed=0, split_cache_path=None, cache_dir=None,
              use_split_cache=True):

    if split_cache_path is not None:
        cache_dir = os.path.dirname(split_cache_path)
    
    data      = np.load(filepath, allow_pickle=True)
    geoms     = np.array(data['params'],     dtype=np.float32)
    keffs     = np.array(data['keffs'],      dtype=np.float32)
    rawparams = np.array(data['params_raw'], dtype=np.float32)
    print(f"  Dataset loaded: {filepath}")
    print(f"  Requested split sizes: train={train_size}, val={val_size}, test={test_size}")
    print(f"  Split seeds: train_seed={train_seed}, holdout_seed={holdout_seed}")
    if not use_split_cache:
        print("  Split cache dir: <disabled>")
    elif cache_dir is None:
        print("  Split cache dir: <default next to dataset>")
    else:
        print(f"  Split cache dir: {cache_dir}")
    print(f"  Dataset samples available: {len(geoms)}")

    train_idx, val_idx, test_idx = load_or_create_split_cache(
        filepath, train_size, val_size, test_size, train_seed, holdout_seed,
        cache_path=cache_dir, use_cache=use_split_cache)

    print(f"  Train k range: {keffs[train_idx].min():.3f} – {keffs[train_idx].max():.3f}")
    print(f"  Val   k range: {keffs[val_idx].min():.3f}  – {keffs[val_idx].max():.3f}")
    print(f"  Test  k range: {keffs[test_idx].min():.3f}  – {keffs[test_idx].max():.3f}")

    print("Precomputing φ_reg features (done once)…")
    train_phi_features = compute_phi_features(rawparams[train_idx])
    val_phi_features    = compute_phi_features(rawparams[val_idx])
    test_phi_features   = compute_phi_features(rawparams[test_idx])

    return (
        (train_idx, geoms[train_idx], keffs[train_idx], rawparams[train_idx], train_phi_features),
        (val_idx, geoms[val_idx],   keffs[val_idx],   rawparams[val_idx],   val_phi_features),
        (test_idx, geoms[test_idx],  keffs[test_idx],  rawparams[test_idx],  test_phi_features),
    )
    

def load_data_balanced_ranges(filepath, train_size, val_size, test_size,
                               train_seed=42, holdout_seed=0,
                               split_cache_path=None, cache_dir=None,
                               bin_edges=None):
    """
    Fixed-range balanced split by k_eff.
    Uses the same number of samples from every k_eff range for train, val, AND test.

    Returns the same outputs as your current loaddata().
    """

    data = np.load(filepath, allow_pickle=True)
    geoms = np.array(data["params"], dtype=np.float32)
    keffs = np.array(data["keffs"], dtype=np.float32)
    rawparams = np.array(data["params_raw"], dtype=np.float32)

    if bin_edges is None:
        bin_edges = np.array([0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975,
                              1.00, 1.025, 1.05, 1.075, 1.10, 1.125, 1.15, 
                              1.175, 1.20, 1.225, 1.25, 1.275, 1.30, 1.35], dtype=np.float32)

    if split_cache_path is not None:
        cache_dir = os.path.dirname(split_cache_path)

    train_rng = np.random.default_rng(train_seed)
    nbins = len(bin_edges) - 1

    trainidx, validx, testidx = [], [], []
    bin_indices = []
    bin_labels = []

    print("\n=== Balanced fixed-range split (train/val/test) ===")
    print(f"Split seeds: train_seed={train_seed}, holdout_seed={holdout_seed}")

    holdout_rng = np.random.default_rng(holdout_seed)

    for b in range(nbins):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        if b < nbins - 1:
            idx = np.where((keffs >= lo) & (keffs < hi))[0]
            label = f"[{lo:.2f}, {hi:.2f})"
        else:
            idx = np.where((keffs >= lo) & (keffs <= hi))[0]
            label = f"[{lo:.2f}, {hi:.2f}]"
        holdout_rng.shuffle(idx)
        bin_indices.append(idx)
        bin_labels.append(label)

    capacities = np.array([len(idx) for idx in bin_indices], dtype=int)
    requested_total = int(train_size + val_size + test_size)
    total_available = int(capacities.sum())
    if requested_total > total_available:
        raise ValueError(
            f"Requested train+val+test={requested_total} exceeds available in bins={total_available}"
        )

    # Step 1: allocate train+val across bins evenly (capped by per-bin capacity)
    uniform_weights = np.ones(nbins, dtype=np.float64)
    requested_tv = int(train_size + val_size)
    tv_counts = _allocate_counts(requested_tv, uniform_weights, capacities=capacities)

    # Step 2: split each bin's tv allocation into val/train proportionally
    val_counts = _allocate_counts(val_size, tv_counts.astype(np.float64), capacities=tv_counts)
    train_counts = tv_counts - val_counts

    if int(train_counts.sum()) != int(train_size) or int(val_counts.sum()) != int(val_size):
        raise ValueError(
            "Internal split allocation mismatch: "
            f"train={train_counts.sum()} (wanted {train_size}), "
            f"val={val_counts.sum()} (wanted {val_size})"
        )

    # Step 3: allocate test evenly across bins, capped by whatever capacity is LEFT
    # after train+val are removed from each bin.
    remaining_capacity = capacities - tv_counts
    test_counts = _allocate_counts(test_size, uniform_weights, capacities=remaining_capacity)

    if int(test_counts.sum()) != int(test_size):
        raise ValueError(
            "Internal split allocation mismatch: "
            f"test={test_counts.sum()} (wanted {test_size})"
        )

    print(
        f"Requested sizes: train={train_size}, val={val_size}, "
        f"test={test_size}, nbins={nbins}"
    )

    for b in range(nbins):
        idx = bin_indices[b]
        n_val = int(val_counts[b])
        n_train = int(train_counts[b])
        n_test = int(test_counts[b])

        # Keep val/test deterministic across train seeds by deriving holdouts first.
        val_bin = idx[:n_val]
        test_bin = idx[n_val:n_val + n_test]
        train_pool = idx[n_val + n_test:]
        train_pool = train_pool.copy()
        train_rng.shuffle(train_pool)
        train_bin = train_pool[:n_train]

        validx.extend(val_bin.tolist())
        trainidx.extend(train_bin.tolist())
        testidx.extend(test_bin.tolist())

        print(
            f"{bin_labels[b]}: available={len(idx):3d}, "
            f"train={n_train:3d}, val={n_val:3d}, test={n_test:3d}"
        )

    trainidx = np.array(trainidx, dtype=int)
    validx = np.array(validx, dtype=int)
    testidx = np.array(testidx, dtype=int)

    train_rng.shuffle(trainidx)
    holdout_rng.shuffle(validx)
    holdout_rng.shuffle(testidx)

    print(f"\nFinal train size = {len(trainidx)}")
    print(f"Final val size   = {len(validx)}")
    print(f"Final test size  = {len(testidx)}")
    print(f"Train k range {keffs[trainidx].min():.3f} to {keffs[trainidx].max():.3f}")
    print(f"Val   k range {keffs[validx].min():.3f} to {keffs[validx].max():.3f}")
    print(f"Test  k range {keffs[testidx].min():.3f} to {keffs[testidx].max():.3f}")
    print(f"Train keff mean {keffs[trainidx].mean():.3f} std {keffs[trainidx].std():.3f}")
    print(f"Val   keff mean {keffs[validx].mean():.3f} std {keffs[validx].std():.3f}")
    print(f"Test  keff mean {keffs[testidx].mean():.3f} std {keffs[testidx].std():.3f}")

    print("Precomputing phi_reg features (done once)...")
    trainphifeatures = compute_phi_features(rawparams[trainidx])
    valphifeatures = compute_phi_features(rawparams[validx])
    testphifeatures = compute_phi_features(rawparams[testidx])

    return (
        (trainidx, geoms[trainidx], keffs[trainidx], rawparams[trainidx], trainphifeatures),
        (validx, geoms[validx], keffs[validx], rawparams[validx], valphifeatures),
        (testidx, geoms[testidx], keffs[testidx], rawparams[testidx], testphifeatures),
    )


def load_data_param_stratified(filepath, train_size, val_size, test_size,
                               train_seed=42, holdout_seed=0,
                               n_bins_per_param=None, bin_edges_per_param=None,
                               balanced=False):
    """
    Load dataset and split train/val/test by stratifying over input parameters
    (params_raw), not keff.

    Primary: equal coverage of every fuel_r quantile bin in train/val/test.
    Secondary: within each fuel_r bin, evenly spaced samples across the other
    parameter bins so no geometry sub-range is left only in train.

    Returns the same nested tuple layout as load_data().
    """
    if n_bins_per_param is None:
        n_bins_per_param = context.PARAM_STRAT_BINS

    data = np.load(filepath, allow_pickle=True)
    geoms = np.array(data["params"], dtype=np.float32)
    keffs = np.array(data["keffs"], dtype=np.float32)
    rawparams = np.array(data["params_raw"], dtype=np.float32)

    print("\n=== Parameter-stratified split (fuel_r primary + even-spaced) ===")
    print(f"  Dataset loaded: {filepath}")
    print(f"  Requested sizes: train={train_size}, val={val_size}, test={test_size}")
    print(f"  Split seeds: train_seed={train_seed}, holdout_seed={holdout_seed}")
    print(f"  Bins per parameter: {n_bins_per_param}")
    print(f"  Primary param: fuel_r (hard coverage per bin)")
    print(f"  Parameters: {context.PARAM_NAMES}")
    print(f"  Dataset samples available: {len(geoms)}")

    train_idx, val_idx, test_idx, meta = derive_split_indices_by_params(
        filepath, train_size, val_size, test_size,
        train_seed=train_seed, holdout_seed=holdout_seed,
        n_bins_per_param=n_bins_per_param,
        bin_edges_per_param=bin_edges_per_param,
        balanced=balanced,
        fuel_r_idx=context.PARAM_NAMES.index("fuel_r") if "fuel_r" in context.PARAM_NAMES else 2,
    )
    train_idx, val_idx, test_idx = _validate_split_indices(
        train_idx, val_idx, test_idx,
        dataset_size=len(geoms),
        expected_train=train_size,
        expected_val=val_size,
        expected_test=test_size,
        context="param-stratified split",
    )

    print(f"  Method: {meta.get('method')} | primary fuel_r bins={meta.get('n_strata')}")
    if "fuel_r_train_counts" in meta:
        print(f"  fuel_r bin quotas train/val/test: "
              f"{meta['fuel_r_train_counts'].tolist()} / "
              f"{meta['fuel_r_val_counts'].tolist()} / "
              f"{meta['fuel_r_test_counts'].tolist()}")
    print(f"  Final sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"  Train k range: {keffs[train_idx].min():.3f} – {keffs[train_idx].max():.3f}")
    print(f"  Val   k range: {keffs[val_idx].min():.3f} – {keffs[val_idx].max():.3f}")
    print(f"  Test  k range: {keffs[test_idx].min():.3f} – {keffs[test_idx].max():.3f}")

    print_param_bin_counts("TRAIN", rawparams[train_idx], edges_list=meta["edges_list"])
    print_param_bin_counts("VAL",   rawparams[val_idx],   edges_list=meta["edges_list"])
    print_param_bin_counts("TEST",  rawparams[test_idx],  edges_list=meta["edges_list"])

    print("Precomputing φ_reg features (done once)…")
    train_phi_features = compute_phi_features(rawparams[train_idx])
    val_phi_features = compute_phi_features(rawparams[val_idx])
    test_phi_features = compute_phi_features(rawparams[test_idx])

    return (
        (train_idx, geoms[train_idx], keffs[train_idx], rawparams[train_idx], train_phi_features),
        (val_idx, geoms[val_idx], keffs[val_idx], rawparams[val_idx], val_phi_features),
        (test_idx, geoms[test_idx], keffs[test_idx], rawparams[test_idx], test_phi_features),
    )


def print_keff_bin_counts(name, keffs):
    edges = np.array([0.75, 0.80, 0.85, 0.90, 0.95,
                      1.00, 1.05, 1.10, 1.15, 1.20, 1.25], dtype=np.float32)
    print(f"\n{name} bin counts:")
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        if b < len(edges) - 2:
            n = np.sum((keffs >= lo) & (keffs < hi))
            label = f"[{lo:.2f}, {hi:.2f})"
        else:
            n = np.sum((keffs >= lo) & (keffs <= hi))
            label = f"[{lo:.2f}, {hi:.2f}]"
        print(f"{label}: {n}")


def print_param_bin_counts(name, rawparams, edges_list=None, n_bins=None):
    """
    Print per-parameter bin occupancy for a split (marginal coverage check).
    Uses the same edges as the stratified split when `edges_list` is provided.
    """
    rawparams = np.asarray(rawparams, dtype=np.float64)
    if rawparams.size == 0:
        print(f"\n{name} param bin counts: (empty)")
        return
    n_params = rawparams.shape[1]
    if edges_list is None:
        if n_bins is None:
            n_bins = context.PARAM_STRAT_BINS
        _, edges_list, _ = _assign_param_strata(rawparams, n_bins_per_param=n_bins)

    print(f"\n{name} param bin counts (marginal):")
    for p in range(min(n_params, len(context.PARAM_NAMES))):
        edges = np.asarray(edges_list[p], dtype=np.float64)
        nb = len(edges) - 1
        pname = context.PARAM_NAMES[p] if p < len(context.PARAM_NAMES) else f"param_{p}"
        parts = []
        for b in range(nb):
            lo, hi = edges[b], edges[b + 1]
            if b < nb - 1:
                n = int(np.sum((rawparams[:, p] >= lo) & (rawparams[:, p] < hi)))
                label = f"[{lo:.3g},{hi:.3g})"
            else:
                n = int(np.sum((rawparams[:, p] >= lo) & (rawparams[:, p] <= hi)))
                label = f"[{lo:.3g},{hi:.3g}]"
            parts.append(f"{label}:{n}")
        print(f"  {pname}: " + "  ".join(parts))


