"""
xs_heatmap.py  —  XS Heatmap: Baseline | Epoch corrections | %Change | Final
==============================================================================
Layout per row:  Baseline | Epoch1 | Epoch2 | ... | %Change | Final XS

Coloring:
  Baseline & Final : per-XS-type log colorscale (Blues/Oranges/Purples/Greens/Reds)
                     shared range across ALL regions+groups for that type
  Epoch columns    : diverging RdBu_r, symmetric ±max_correction, shared across epochs
  %Change column   : diverging RdBu_r, symmetric ±max_%change, shows (final-base)/base*100

==================   plot_xs_subplots   =====================================
One subplot per XS type. Each subplot: (n_regions x 2G) cells.
  Left G cols  = Baseline (polynomial regression)
  Right G cols = Final XS (baseline + NN correction)
Shared log colorscale within each subplot.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.ticker import LogFormatter
import os

REGION_NAMES = ["B4C rod", "Fuel annulus", "Water"]

XS_TYPE_CMAPS = {
    "D":         ("Blues",   "D"),
    "Sigma_a":   ("Oranges", "Σ_a"),
    "nuSigma_f": ("Purples", "νΣ_f"),
    "Sigma_s":   ("Greens",  "Σ_s"),
    "chi":       ("Reds",    "χ"),
}

XS_TYPES = [
    ("D",               "D (diffusion coeff.)",    "Blues"),
    ("Sigma_a",         "Sigma_a (absorption)",     "Oranges"),
    ("nuSigma_f",       "nuSigma_f (fission)",      "Purples"),
    ("Sigma_s_diag",    "Sigma_s g->g (self)",      "Greens"),
    ("Sigma_s_offdiag", "Sigma_s g->g' (cross)",   "YlGn"),
    ("chi",             "chi (fission spectrum)",   "Reds"),
]


def _build_row_meta(G):
    rows, col = [], 0
    for g in range(G):
        rows.append({"label": f"D g{g+1}",           "xs_type": "D",         "col": col}); col += 1
    for g in range(G):
        rows.append({"label": f"Σ_a g{g+1}",         "xs_type": "Sigma_a",   "col": col}); col += 1
    for g in range(G):
        rows.append({"label": f"νΣ_f g{g+1}",        "xs_type": "nuSigma_f", "col": col}); col += 1
    for g1 in range(G):
        for g2 in range(G):
            rows.append({"label": f"Σ_s {g1+1}→{g2+1}", "xs_type": "Sigma_s", "col": col}); col += 1
    for g in range(G):
        rows.append({"label": f"χ g{g+1}",            "xs_type": "chi",       "col": col}); col += 1
    return rows


def _type_ranges(baseline, final_xs, row_meta, n_regions):
    buckets = {t: [] for t in XS_TYPE_CMAPS}
    for r in range(n_regions):
        for rm in row_meta:
            for arr in [baseline, final_xs]:
                v = float(arr[r, rm["col"]])
                if v > 0:
                    buckets[rm["xs_type"]].append(v)
    result = {}
    for t, vals in buckets.items():
        result[t] = (min(vals)*0.85, max(vals)*1.15) if vals else (1e-12, 1.0)
    return result


def _pct_change_abs_max(baseline, final_xs, row_meta, n_regions):
    vals = []
    for r in range(n_regions):
        for rm in row_meta:
            b = float(baseline[r, rm["col"]])
            f = float(final_xs[r, rm["col"]])
            if b != 0:
                vals.append(abs((f - b) / b * 100))
    return max(vals) if vals else 1.0


def _corr_abs_max(snapshots):
    vals = [v for snap in snapshots for v in snap.flatten() if v != 0]
    return max(abs(v) for v in vals) if vals else 1e-4


def _lum(rgba):
    return 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]

def _tc(rgba):
    return "white" if _lum(rgba) < 0.45 else "black"


def plot_xs_heatmap(
    baseline,
    snapshots,
    epoch_labels,
    final_xs,
    G=2,
    save_path="xs_heatmap.png",
    plot_interval_note="",
):
    row_meta   = _build_row_meta(G)
    n_rows     = len(row_meta)
    n_regions  = baseline.shape[0]
    n_snap     = len(snapshots)
    total_rows = n_rows * n_regions

    # columns: 0=Baseline, 1..n_snap=epochs, n_snap+1=%change, n_snap+2=Final
    n_data_cols = 1 + n_snap + 1 + 1

    # ── norms ────────────────────────────────────────────────────────────────
    t_ranges   = _type_ranges(baseline, final_xs, row_meta, n_regions)
    type_norms = {}
    for t, (vmin, vmax) in t_ranges.items():
        cmap_name = XS_TYPE_CMAPS[t][0]
        cmap = plt.get_cmap(cmap_name)
        norm = LogNorm(vmin=vmin, vmax=vmax) if vmin > 0 else Normalize(vmin=vmin, vmax=vmax)
        type_norms[t] = (norm, cmap)

    corr_max  = _corr_abs_max(snapshots) if snapshots else 1e-4
    corr_norm = TwoSlopeNorm(vmin=-corr_max, vcenter=0.0, vmax=corr_max)
    corr_cmap = plt.get_cmap("RdBu_r")

    pct_max  = _pct_change_abs_max(baseline, final_xs, row_meta, n_regions)
    pct_norm = TwoSlopeNorm(vmin=-pct_max, vcenter=0.0, vmax=pct_max)
    pct_cmap = plt.get_cmap("PiYG")   # green=increase, pink=decrease

    # ── figure sizing ────────────────────────────────────────────────────────
    col_w    = 1.6
    row_h    = 0.27
    left_pad = 2.0     # row labels
    right_pad = 0.3    # small right margin — colorbars placed via fig.add_axes
    top_pad  = 1.5
    bot_pad  = 0.5
    cb_area  = 3.2     # extra width on the right for colorbars

    fig_w = left_pad + n_data_cols * col_w + right_pad + cb_area
    fig_h = top_pad  + total_rows * row_h  + bot_pad

    fig = plt.figure(figsize=(fig_w, fig_h))
    # main axes occupies left portion, colorbars go in right portion
    ax_left   = left_pad / fig_w
    ax_width  = (n_data_cols * col_w) / fig_w
    ax_bottom = bot_pad / fig_h
    ax_height = (total_rows * row_h) / fig_h

    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
    ax.set_xlim(0, n_data_cols)
    ax.set_ylim(0, total_rows)
    ax.axis("off")

    title = "XS Heatmap: Baseline vs NN Corrections vs Final"
    if plot_interval_note:
        title += f"  [{plot_interval_note}]"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    def cx(dc):  return dc + 0.5
    def draw_cell(dc, grow, value, fc, fmt=".4g", fs=10.0):
        y_bot = total_rows - grow - 1
        rect  = plt.Rectangle(
            (dc + 0.04, y_bot + 0.07), 0.92, 0.86,
            facecolor=fc, edgecolor="white", linewidth=0.3,
            transform=ax.transData, clip_on=False
        )
        ax.add_patch(rect)
        txt = f"{value:{fmt}}" if value != 0 else "0"
        ax.text(cx(dc), y_bot + 0.5, txt,
                ha="center", va="center", fontsize=fs,
                color=_tc(fc), transform=ax.transData)

    def abs_fc(xs_type, v):
        norm, cmap = type_norms[xs_type]
        if v <= 0: return cmap(0.05)
        try:    return cmap(norm(v))
        except: return cmap(0.05)

    def corr_fc(v):
        return corr_cmap(corr_norm(np.clip(v, -corr_max, corr_max)))

    def pct_fc(v):
        return pct_cmap(pct_norm(np.clip(v, -pct_max, pct_max)))

    # ── column headers ───────────────────────────────────────────────────────
    hy  = total_rows + 0.25
    hkw = dict(ha="center", va="bottom", fontsize=8,
               transform=ax.transData, clip_on=False)
    ax.text(cx(0), hy + 0.5, "Baseline(poly. reg.)",
            color="#1a5fa8", fontweight="bold", **hkw)
    for si, lbl in enumerate(epoch_labels):
        ax.text(cx(1 + si), hy + 0.5, lbl, **hkw)
    ax.text(cx(1 + n_snap), hy + 0.5, "% Change(fin vs base)",
            color="#5a005a", fontweight="bold", **hkw)
    ax.text(cx(n_data_cols - 1), hy + 0.5, "Final XS(base+NN)",
            color="#8b1a00", fontweight="bold", **hkw)

    # thin vertical separator before %change and Final columns
    for sep_x in [1 + n_snap, n_data_cols - 1]:
        ax.axvline(sep_x, color="#aaaaaa", lw=0.8,
                   ymin=0, ymax=1)

    # ── draw all cells ───────────────────────────────────────────────────────
    for ri in range(n_regions):
        for mi, rm in enumerate(row_meta):
            grow = ri * n_rows + mi
            c    = rm["col"]
            xt   = rm["xs_type"]

            b_val = float(baseline[ri, c])
            f_val = float(final_xs[ri, c])
            pct   = (f_val - b_val) / b_val * 100 if b_val != 0 else 0.0

            draw_cell(0,              grow, b_val, abs_fc(xt, b_val))
            for si, snap in enumerate(snapshots):
                draw_cell(1 + si,     grow, float(snap[ri, c]), corr_fc(float(snap[ri, c])))
            draw_cell(1 + n_snap,     grow, pct,   pct_fc(pct),  fmt=".2f")
            draw_cell(n_data_cols-1,  grow, f_val, abs_fc(xt, f_val))

    # ── row labels ───────────────────────────────────────────────────────────
    for ri in range(n_regions):
        for mi, rm in enumerate(row_meta):
            grow = ri * n_rows + mi
            ax.text(-0.08, total_rows - grow - 0.5, rm["label"],
                    ha="right", va="center", fontsize=7.5,
                    transform=ax.transData)

    # ── region labels + dividers ─────────────────────────────────────────────
    for ri, rname in enumerate(REGION_NAMES[:n_regions]):
        y_mid = total_rows - (ri + 0.5) * n_rows
        ax.text(-1.0, y_mid, rname,
                ha="center", va="center", fontsize=9, fontweight="bold",
                rotation=90, transform=ax.transData)
        if ri > 0:
            y_frac = (total_rows - ri * n_rows) / total_rows
            ax.axhline(y_frac, color="gray", lw=1.2, ls="--")

    # ── colorbars — placed in the right margin via fig.add_axes ──────────────
    # absolute range: ax_left + ax_width = right edge of main plot in fig coords
    plot_right = ax_left + ax_width
    cb_x_start = plot_right + 0.02        # small gap after plot
    cb_w       = 0.018
    cb_gap_x   = 0.06
    cb_top     = ax_bottom + ax_height
    cb_bot     = ax_bottom
    usable_h   = ax_height

    n_types    = len(XS_TYPE_CMAPS)
    each_h     = usable_h / n_types - 0.012

    for ti, (tname, (cmap_name, short)) in enumerate(XS_TYPE_CMAPS.items()):
        norm, cmap = type_norms[tname]
        y0  = cb_bot + ti * (each_h + 0.012)
        sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([cb_x_start, y0, cb_w, each_h])
        fmt = LogFormatter(labelOnlyBase=False) if isinstance(norm, LogNorm) else "%.3g"
        cb  = fig.colorbar(sm, cax=cax, format=fmt)
        cb.set_label(short, fontsize=7, labelpad=3)
        cb.ax.tick_params(labelsize=5.5)

    # correction colorbar
    sm_c = plt.cm.ScalarMappable(cmap=corr_cmap, norm=corr_norm)
    sm_c.set_array([])
    cax_c = fig.add_axes([cb_x_start + cb_w + cb_gap_x,
                          cb_bot, cb_w, usable_h * 0.48 - 0.01])
    cb_c  = fig.colorbar(sm_c, cax=cax_c)
    cb_c.set_label("NN corr.", fontsize=7, labelpad=3)
    cb_c.ax.tick_params(labelsize=5.5)

    # % change colorbar
    sm_p = plt.cm.ScalarMappable(cmap=pct_cmap, norm=pct_norm)
    sm_p.set_array([])
    cax_p = fig.add_axes([cb_x_start + cb_w + cb_gap_x,
                          cb_bot + usable_h * 0.52, cb_w, usable_h * 0.48 - 0.01])
    cb_p  = fig.colorbar(sm_p, cax=cax_p)
    cb_p.set_label("% change", fontsize=7, labelpad=3)
    cb_p.ax.tick_params(labelsize=5.5)

    # ── save ─────────────────────────────────────────────────────────────────
    dirn = os.path.dirname(save_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    print(f"[xs_heatmap] saved to: {save_path}")
    plt.show()
    plt.close()




def _build_col_map(G):
    col, result = 0, {}
    result["D"] = list(range(col, col + G)); col += G
    result["Sigma_a"] = list(range(col, col + G)); col += G
    result["nuSigma_f"] = list(range(col, col + G)); col += G
    diag, offdiag = [], []
    for g1 in range(G):
        for g2 in range(G):
            (diag if g1 == g2 else offdiag).append(col); col += 1
    result["Sigma_s_diag"] = diag
    result["Sigma_s_offdiag"] = offdiag
    result["chi"] = list(range(col, col + G))
    return result


def _make_grid(xs, cols, n_regions):
    g = np.zeros((n_regions, len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        g[:, j] = xs[:, c]
    return g


def _safe_norm(vmin, vmax):
    if vmin > 0 and vmax > 0:
        return LogNorm(vmin=vmin * 0.9, vmax=vmax * 1.1)
    return Normalize(vmin=vmin - abs(vmin)*0.1,
                     vmax=vmax + abs(vmax)*0.1 + 1e-12)


def _col_labels(xs_key, G):
    if xs_key == "Sigma_s_diag":
        return [f"g{g+1}->g{g+1}" for g in range(G)]
    if xs_key == "Sigma_s_offdiag":
        return [f"g{g1+1}->g{g2+1}"
                for g1 in range(G) for g2 in range(G) if g1 != g2]
    return [f"g{g+1}" for g in range(G)]


def _fmt_val(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 10:   return f"{v:.2f}"
    if a >= 1:    return f"{v:.3f}"
    if a >= 0.01: return f"{v:.4f}"
    return f"{v:.2e}"


def plot_xs_subplots(
    baseline,
    final_xs,
    G=2,
    save_path="xs_subplots.png",
    suptitle="XS: Baseline vs Final (NN corrected)",
    epoch_label=None,
    sample_idx=None,
    geo_params=None,
    param_names=None,
    keff_ref=None,
    keff_pred=None,
    weight=None,
):
    """
    Portrait layout: 2 subplots per row, stacked vertically.
    Subplot order (rows of 2):
      D           | Sigma_a
      nuSigma_f   | chi
      Sigma_s g->g (self-scatter)  | Sigma_s g->g' (cross-scatter)

    Each subplot shows:
      Left  G cols : Baseline values
      Right G cols : Final (NN-corrected) values
                     with % change vs baseline annotated below each value

    Optional metadata displayed below the suptitle:
      epoch_label  : e.g. "Epoch 30"
      sample_idx   : integer, shown in title and used to label saved file
      geo_params   : array of 6 raw geometry values
      param_names  : list of 6 strings naming each parameter
      keff_ref     : reference k-eff (OpenMC)
      keff_pred    : predicted k-eff (PEDS)
    """
    n_reg   = baseline.shape[0]
    col_map = _build_col_map(G)

    # ── Build active subplot list in the desired portrait order ─────────────
    # Explicit ordering: D, Sigma_a, nuSigma_f, chi, Sigma_s_diag, Sigma_s_offdiag
    PORTRAIT_ORDER = [
        ("D",               "D (diffusion coeff.)",   "Blues"),
        ("Sigma_a",         "Σ_a (absorption)",        "Oranges"),
        ("nuSigma_f",       "νΣ_f (fission)",          "Purples"),
        ("chi",             "χ (fission spectrum)",    "Reds"),
        ("Sigma_s_diag",    "Σ_s g→g (self-scatter)",  "Greens"),
        ("Sigma_s_offdiag", "Σ_s g→g' (cross-scatter)","YlGn"),
    ]

    active = []
    for xs_key, label, cmap_name in PORTRAIT_ORDER:
        bg = _make_grid(baseline, col_map[xs_key], n_reg)
        fg = _make_grid(final_xs, col_map[xs_key], n_reg)
        if np.any(bg != 0) or np.any(fg != 0):
            active.append((xs_key, label, cmap_name, bg, fg))

    n_t   = len(active)
    NCOLS = 2                           # always 2 subplots per row
    n_row = (n_t + NCOLS - 1) // NCOLS  # ceil division

    # ── sizing constants ─────────────────────────────────────────────────────
    cell_w   = 1.40   # inches per data cell
    cell_h   = 0.78   # inches per region row
    cb_w_in  = 0.22   # colorbar width
    cb_pad   = 0.12   # gap between axes and colorbar
    ylabel_w = 1.10   # left margin for region y-labels
    xlabel_h = 0.65   # bottom margin for x-tick labels
    title_h  = 0.80   # header zone (XS type title + half/half subtitle)
    sp_hgap  = 0.70   # vertical gap between subplot rows
    sp_wgap  = 1.00   # horizontal gap between subplot columns

    sp_data_w = G * cell_w * 2          # Baseline cols + Final cols (no nan gap)
    sp_data_h = n_reg * cell_h

    sp_w = ylabel_w + sp_data_w + cb_pad + cb_w_in
    sp_h = title_h  + sp_data_h + xlabel_h

    fig_w = NCOLS * sp_w + (NCOLS - 1) * sp_wgap + 0.4
    fig_h = n_row  * sp_h + (n_row  - 1) * sp_hgap + 0.90  # extra at top for suptitle

    # ── extra vertical space at top when metadata info box is shown ─────────────
    has_info = any(x is not None for x in [geo_params, keff_ref, keff_pred])
    info_h   = 0.55 if has_info else 0.0   # extra inches reserved for info line(s)
    fig_h   += info_h

    fig = plt.figure(figsize=(fig_w, fig_h))

    # ── suptitle: main title + epoch + sample annotation ─────────────────────
    epoch_note  = f"  —  {epoch_label}"          if epoch_label  is not None else ""
    sample_note = f"  —  Sample {sample_idx}"    if sample_idx   is not None else ""
    fig.suptitle(
        suptitle + epoch_note + sample_note,
        fontsize=14, fontweight="bold",
        y=1.0 - 0.08 / fig_h,
    )

    # ── optional metadata info box ────────────────────────────────────────────
    if has_info:
        parts = []
        if keff_ref is not None and keff_pred is not None:
            delta_rho = abs(keff_pred - keff_ref) / (keff_pred * keff_ref) * 1e5
            parts.append(
                f"k_ref={keff_ref:.5f}   k_pred={keff_pred:.5f}"
                f"   Δρ={delta_rho:.0f} pcm"
            )
            if weight is not None:
                parts.append(f"weight={weight:.2f}")
        elif keff_ref is not None:
            parts.append(f"k_ref={keff_ref:.5f}")

        if geo_params is not None:
            names  = param_names if param_names is not None \
                     else [f"p{j}" for j in range(len(geo_params))]
            pairs  = "   ".join(f"{n}={float(v):.4f}" for n, v in zip(names, geo_params))
            parts.append(pairs)

        info_text = "\n".join(parts)
        fig.text(
            0.5, 1.0 - 0.38 / fig_h,
            info_text,
            ha="center", va="top",
            fontsize=9.5,
            color="#333333",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f5f5",
                      edgecolor="#bbbbbb", linewidth=0.8),
        )

    for i, (xs_key, label, cmap_name, bg, fg) in enumerate(active):
        row_i = i // NCOLS
        col_i = i %  NCOLS

        # ── axes position in figure fractions ────────────────────────────────
        # y=0 is figure bottom; subplots fill from top downward
        sp_x0_fig = (col_i * (sp_w + sp_wgap) + ylabel_w) / fig_w
        sp_y0_fig = ((n_row - 1 - row_i) * (sp_h + sp_hgap) + xlabel_h + 0.35) / fig_h
        sp_w_fig  = sp_data_w / fig_w
        sp_h_fig  = sp_data_h / fig_h

        ax = fig.add_axes([sp_x0_fig, sp_y0_fig, sp_w_fig, sp_h_fig])

        nx          = bg.shape[1]   # number of group-columns per half
        combo       = np.concatenate([bg, fg], axis=1)   # [n_reg, 2*nx]
        total_xcols = combo.shape[1]

        # shared colorscale from all positive values in both halves
        vals = np.concatenate([bg.flatten(), fg.flatten()])
        pos  = vals[vals > 0]
        pos  = pos if len(pos) > 0 else np.array([1e-10, 1.0])
        norm = _safe_norm(pos.min(), pos.max())

        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#e4e4e4")

        im = ax.imshow(combo, cmap=cmap, norm=norm,
                       aspect="auto", interpolation="nearest")

        # ── cell text ─────────────────────────────────────────────────────────
        # Baseline cells: value centred
        # Final cells:    value on upper line, % diff on lower line
        for r in range(n_reg):
            for c in range(total_xcols):
                v = combo[r, c]
                if np.isnan(v):
                    continue
                rgba = cmap(norm(v)) if v > 0 else cmap(0.0)
                lum  = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                tc   = "white" if lum < 0.45 else "black"

                is_final_col = c >= nx
                if is_final_col:
                    # compute % change vs the corresponding baseline cell
                    b_v = bg[r, c - nx]
                    pct = (v - b_v) / b_v * 100 if b_v != 0 else 0.0
                    pct_sign = "+" if pct >= 0 else ""
                    pct_str  = f"{pct_sign}{pct:.1f}%"

                    # value slightly above centre, % diff slightly below
                    ax.text(c, r - 0.18, _fmt_val(v),
                            ha="center", va="center",
                            fontsize=11, fontweight="bold", color=tc)
                    ax.text(c, r + 0.28, pct_str,
                            ha="center", va="center",
                            fontsize=10, color=tc,
                            style="italic")
                else:
                    ax.text(c, r, _fmt_val(v),
                            ha="center", va="center",
                            fontsize=11, fontweight="bold", color=tc)

        # ── x-axis: group labels ──────────────────────────────────────────────
        gl     = _col_labels(xs_key, G)
        xticks = list(range(nx)) + list(range(nx, 2 * nx))
        xlbls  = [f"B {l}" for l in gl] + [f"F {l}" for l in gl]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlbls, fontsize=11, rotation=35, ha="right")

        # ── y-axis: region names ──────────────────────────────────────────────
        ax.set_yticks(range(n_reg))
        ax.set_yticklabels(REGION_NAMES[:n_reg], fontsize=11)
        ax.tick_params(axis="y", length=0, pad=4)

        # ── header labels (placed in figure coords above the axes) ────────────
        title_y_fig   = sp_y0_fig + sp_h_fig   # top edge of data axes in fig coords

        mid_b_data    = (nx - 1) / 2
        mid_b_fig     = sp_x0_fig + (mid_b_data + 0.5) / total_xcols * sp_w_fig
        mid_f_data    = nx + (nx - 1) / 2
        mid_f_fig     = sp_x0_fig + (mid_f_data + 0.5) / total_xcols * sp_w_fig

        subtitle_y    = title_y_fig + 0.012 / fig_h
        xs_title_y    = title_y_fig + (title_h * 0.62) / fig_h

        fig.text(mid_b_fig, subtitle_y, "Baseline",
                 ha="center", va="bottom", fontsize=12,
                 color="#1a5fa8", fontweight="bold")
        fig.text(mid_f_fig, subtitle_y, "Final (NN)  [val  Δ%]",
                 ha="center", va="bottom", fontsize=12,
                 color="#8b1a00", fontweight="bold")

        sp_cx_fig = sp_x0_fig + sp_w_fig / 2
        fig.text(sp_cx_fig, xs_title_y, label,
                 ha="center", va="bottom", fontsize=13, fontweight="bold")

        # thin vertical separator between Baseline and Final halves
        ax.axvline(x=nx - 0.5, color="white", linewidth=4, zorder=3)

        # ── colorbar ──────────────────────────────────────────────────────────
        cb_x0_fig = sp_x0_fig + sp_w_fig + cb_pad / fig_w
        cax = fig.add_axes([cb_x0_fig, sp_y0_fig,
                             cb_w_in / fig_w, sp_h_fig])
        fmt = LogFormatter(labelOnlyBase=False) if isinstance(norm, LogNorm) else "%.3g"
        cb  = fig.colorbar(im, cax=cax, format=fmt)
        cb.ax.tick_params(labelsize=9)

    dirn = os.path.dirname(save_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    print(f"[xs_subplots] saved to: {save_path}")
    plt.show()
    plt.close()


