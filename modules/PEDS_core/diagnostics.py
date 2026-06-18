import jax
import jax.numpy as jnp
import numpy as np

from matrix_JAX_optimized import Aphi_Fphi_scan
from solvers.NTdiffusion.core.MG1D_eigenvalue_nregions import diffusion_setup
from solvers.NTdiffusion.diffusion_solver import (
    build_xs_callables, bc_to_coeffs, GEOMETRY_CODE, run_diffusion_solver
)
from NTcode_config_data.config_def import GeometryConfig


def solver_floor_check(train_rawparams, train_keffs,
                       GEO, NTdiff_solver, update_geo, predict_xs,
                       n_samples=20):
    """
    Bypasses the NN entirely. Directly optimizes XS to match OpenMC k.
    Tells you: what is the best the diffusion solver can ever achieve?
    """
    print("\n" + "="*60)
    print("=== SOLVER FLOOR CHECK ===")
    print("="*60)

    SENTINEL_ID = 9999
    sid = jnp.array([SENTINEL_ID], dtype=jnp.int32)
    results = []

    for i in range(min(n_samples, len(train_rawparams))):
        geo_i    = update_geo(GEO, train_rawparams[i])
        k_mc     = float(train_keffs[i])
        params_i = jnp.array(train_rawparams[i], dtype=jnp.float32)

        xs = jnp.array(predict_xs(geo_i), dtype=jnp.float32)
        print(f"before optimization, sample {i} has starting k = "
              f"{float(NTdiff_solver(xs, params_i, sid)):.4f} vs k_mc = {k_mc:.4f}"
              f"\n with a set of baseline xs = {predict_xs(geo_i)}")
        _ = NTdiff_solver(xs, params_i, sid)

        lr = 1e-4
        for step in range(30):
            _, vjp_fn = jax.vjp(NTdiff_solver, xs, params_i, sid)
            k_diff    = float(NTdiff_solver(xs, params_i, sid))

            rho_residual = (k_diff - k_mc) / (k_diff * k_mc)
            dL_dk        = 2.0 * rho_residual / (k_diff * k_mc)
            grad_xs      = vjp_fn(jnp.array(dL_dk))[0]
            xs           = xs - lr * grad_xs

            if step in (1, 10, 20, 29):
                print(f"xs after gradient opti: {xs}  "
                      f"rho_residual: {rho_residual*1e5:.1f} pcm  step: {step}")

            xs = jnp.clip(xs, 1e-6, None)

        k_final    = float(NTdiff_solver(xs, params_i, sid))
        delta_rho  = abs(k_final - k_mc) / (k_final * k_mc) * 1e5
        k_baseline = float(NTdiff_solver(
            jnp.array(predict_xs(geo_i), dtype=jnp.float32), params_i, sid))
        delta_rho_baseline = abs(k_baseline - k_mc) / (k_baseline * k_mc) * 1e5

        log_ratios = jnp.log(xs / jnp.array(predict_xs(geo_i)))
        print(f"  log_ratio range: [{float(log_ratios.min()):.2f}, {float(log_ratios.max()):.2f}]")
        print(f"  XS changed by factors: [{float(jnp.exp(log_ratios.min())):.2f}, "
              f"{float(jnp.exp(log_ratios.max())):.2f}]")

        flag = " ← SOLVER FLOOR (diffusion limit)" if delta_rho > 650 else " ✓ REACHABLE\n\n"
        print(f"  S{i:3d}  k_mc={k_mc:.4f}  "
              f"baseline_err={delta_rho_baseline:.0f} pcm  "
              f"optimized_err={delta_rho:.0f} pcm{flag}")
        results.append(delta_rho)

    results = np.array(results)
    print(f"\n  Mean optimized error : {results.mean():.0f} pcm")
    print(f"  Samples above 650 pcm: {(results > 650).sum()} / {len(results)}  "
          f"(these are diffusion-limited)")
    print("="*60 + "\n")
    return results


def physics_sanity_check(xs_baseline, params_raw_single, sample_id,
                         NTdiff_solver, xs_layout, GEO):
    xs = jnp.array(xs_baseline, dtype=jnp.float32)
    _, vjp_fn = jax.vjp(NTdiff_solver, xs, params_raw_single, sample_id)
    grad = np.array(vjp_fn(jnp.ones(()))[0])

    lay = xs_layout(2)
    print("=== Physics Sanity Check ===")
    for reg_idx, reg_name in enumerate(['b4c_rod', 'fuel_annulus', 'water']):
        print(f"\nRegion: {reg_name}")
        print(f"  ∂keff/∂D         = {grad[reg_idx, lay['D']]}")
        print(f"  ∂keff/∂Σ_a       = {grad[reg_idx, lay['Sigma_a']]}")
        print(f"  ∂keff/∂νΣ_f      = {grad[reg_idx, lay['nuSigma_f']]}")


def check_eigenvalue_residual_vs_solver(xs_tensor, geo_i, phi_fwd_flat, k,
                                        _GEO_DATA_CACHE, SLAY):
    R = geo_i.boundaries[-1].radius
    I = int(R / geo_i.mesh_size)
    G = geo_i.G
    BC_coeffs = bc_to_coeffs(geo_i.bc)
    geometry_code = GEOMETRY_CODE[geo_i.geometry]
    r_divisions = [b.radius for b in geo_i.boundaries[:-1]]

    D_fn, Sigma_a_fn, nuSigma_f_fn, Sigma_s_fn, chi_fn = \
        build_xs_callables(np.array(xs_tensor), geo_i)
    _, A_real, B_real = diffusion_setup(
        R, I, G, r_divisions,
        D_fn, Sigma_a_fn, nuSigma_f_fn, Sigma_s_fn, chi_fn,
        BC_coeffs, geometry_code
    )

    phi = np.array(phi_fwd_flat)
    Aphi_real = A_real @ phi
    Fphi_real = B_real @ phi

    geo_data = _GEO_DATA_CACHE[0]
    Aphi_scan, Fphi_scan = Aphi_Fphi_scan(xs_tensor, geo_data, SLAY, jnp.array(phi))

    diff_A = np.abs(np.array(Aphi_scan) - Aphi_real)
    diff_F = np.abs(np.array(Fphi_scan) - Fphi_real)

    print("=== Scan vs Real Solver Matrix Check ===")
    print(f"  ||A_scan·φ - A_real·φ||_inf  = {diff_A.max():.3e}  at flat idx {diff_A.argmax()}")
    print(f"  ||F_scan·φ - F_real·φ||_inf  = {diff_F.max():.3e}  at flat idx {diff_F.argmax()}")

    diff_A_2d = diff_A.reshape(G, I+1)
    diff_F_2d = diff_F.reshape(G, I+1)
    print("\n  Per-group A mismatch:")
    for g in range(G):
        i_w = int(diff_A_2d[g].argmax())
        print(f"    Group {g+1}: max={diff_A_2d[g,i_w]:.3e} at cell i={i_w}  "
              f"(i=0:{i_w==0}, i=I:{i_w==I})")
    print("\n  Per-group F mismatch:")
    for g in range(G):
        i_w = int(diff_F_2d[g].argmax())
        print(f"    Group {g+1}: max={diff_F_2d[g,i_w]:.3e} at cell i={i_w}  "
              f"(i=0:{i_w==0}, i=I:{i_w==I})")

    residual_real = Aphi_real - (1.0/k) * Fphi_real
    print(f"\n  ||A_real·φ - (1/k)·F_real·φ||_inf = {np.abs(residual_real).max():.3e}")
    print(f"  (should be ≈ solver tolerance ~1e-8)")


def diagnose_scan_vs_real_cellwise(xs_tensor, geo_i, phi_fwd_flat, k,
                                   _GEO_DATA_CACHE, SLAY):
    from solvers.NTdiffusion.core.MG1D_eigenvalue_nregions import diffusion_setup, create_grid

    R = geo_i.boundaries[-1].radius
    I = int(R / geo_i.mesh_size)
    G = geo_i.G
    Delta_r = geo_i.mesh_size
    BC_coeffs = bc_to_coeffs(geo_i.bc)
    geometry_code = GEOMETRY_CODE[geo_i.geometry]
    r_divisions = [b.radius for b in geo_i.boundaries[:-1]]

    D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn = build_xs_callables(np.array(xs_tensor), geo_i)
    _, A_real, _ = diffusion_setup(R, I, G, r_divisions,
                                   D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn,
                                   BC_coeffs, geometry_code)
    phi = np.array(phi_fwd_flat)
    phi2d_check = phi.reshape(G, I+1)

    print(f"\n=== Flux Sanity Check ===")
    print(f"  phi_fwd_flat norm       = {np.linalg.norm(phi):.6e}")
    print(f"  phi2d[g=0, i=0:5]      = {phi2d_check[0, :5]}")
    print(f"  phi2d[g=1, i=0:5]      = {phi2d_check[1, :5]}")
    print(f"  phi2d[g=0, ghost i=I]  = {phi2d_check[0, I]:.6e}  (expect 0.0)")
    print(f"  phi2d[g=1, ghost i=I]  = {phi2d_check[1, I]:.6e}  (expect 0.0)")
    print(f"  max(phi)               = {phi.max():.6e}")
    print(f"  min(phi)               = {phi.min():.6e}")

    Aphi_real = A_real @ phi
    geo_data = _GEO_DATA_CACHE[0]
    Aphi_scan = np.array(Aphi_Fphi_scan(xs_tensor, geo_data, SLAY, jnp.array(phi))[0])

    diff_2d      = (Aphi_scan - Aphi_real).reshape(G, I+1)
    Aphi_real_2d = Aphi_real.reshape(G, I+1)
    phi2d        = phi.reshape(G, I+1)

    _, centers, edges = create_grid(R, I)
    S = 2.0 * np.pi * edges
    V = np.pi * (edges[1:]**2 - edges[:-1]**2)
    region_of_cell = geo_data['region_of_cell']

    print(f"\n=== Cell-by-Cell A-matrix Decomposition ===")
    print(f"  Showing cells where |error| > 1e-3\n")

    for g in range(G):
        print(f"\n  --- Group {g+1} ---")
        for i in range(I):
            if abs(diff_2d[g, i]) < 1e-3:
                continue
            reg   = int(region_of_cell[i])
            reg_n = int(region_of_cell[min(i+1, I-1)])
            xs    = np.array(xs_tensor)

            D_g      = float(xs[reg,   SLAY['D']][g])
            D_g_next = float(xs[reg_n, SLAY['D']][g])
            Dplus    = 2*D_g*D_g_next / (D_g + D_g_next + 1e-30)
            Dminus   = 0.0
            if i > 0:
                reg_p    = int(region_of_cell[i-1])
                D_g_prev = float(xs[reg_p, SLAY['D']][g])
                Dminus   = 2*D_g_prev*D_g / (D_g_prev + D_g + 1e-30)

            Sig_a     = float(xs[reg, SLAY['Sigma_a']][g])
            Sig_s_mat = xs[reg, SLAY['Sigma_s']].reshape(G, G)
            Sig_s_out = sum(Sig_s_mat[g, gp] for gp in range(G) if gp != g)

            diag_total  = Dplus * S[i+1] / (Delta_r * V[i]) + Sig_a + Sig_s_out
            term_diag   = diag_total * phi2d[g, i]
            term_right  = -Dplus * S[i+1] / (Delta_r * V[i]) * phi2d[g, i+1]
            term_left   = Dminus * S[i] / (Delta_r * V[i]) * (phi2d[g,i] - phi2d[g, i-1]) if i > 0 else 0.0
            term_scatin = -sum(Sig_s_mat[gp, g] * phi2d[gp, i] for gp in range(G) if gp != g)
            manual_sum  = term_diag + term_right + term_left + term_scatin

            print(f"  cell i={i:3d}  reg={reg}  scan={Aphi_scan[g*(I+1)+i]:.6e}"
                  f"  real={Aphi_real_2d[g,i]:.6e}  err={diff_2d[g,i]:.3e}")
            print(f"    manual reconstruction = {manual_sum:.6e}")
            print(f"    terms: diag={term_diag:.4e}  right={term_right:.4e}"
                  f"  left={term_left:.4e}  scatin={term_scatin:.4e}")
            print(f"    Dplus={Dplus:.4f}  Dminus={Dminus:.4f}"
                  f"  S[i]={S[i]:.4f}  S[i+1]={S[i+1]:.4f}  V[i]={V[i]:.4f}")


def finite_difference_gradient_check(xs_tensor, params_raw_single, sample_id,
                                     NTdiff_solver, _NTdiff_fwd,
                                     _GEO_DATA_CACHE, SLAY, GEO,
                                     rel_eps=1e-3, abs_eps_floor=1e-6):
    from solvers.NTdiffusion.diffusion_solver import xs_layout

    xs = jnp.array(xs_tensor, dtype=jnp.float32)
    flat_xs = xs.flatten()
    n = flat_xs.shape[0]
    _ = NTdiff_solver(xs, params_raw_single, sample_id)
    geo_data = _GEO_DATA_CACHE[int(sample_id[0])]

    _, (_, k_frozen, phi_fwd_frozen, phi_adj_frozen, _, _) = \
        _NTdiff_fwd(xs, params_raw_single, sample_id)

    def scalar_fn(xs_):
        Aphi, Fphi = Aphi_Fphi_scan(xs_, geo_data, SLAY, phi_fwd_frozen)
        return jnp.dot(phi_adj_frozen, Aphi) - (1.0 / k_frozen) * jnp.dot(phi_adj_frozen, Fphi)

    _, vjp_fn = jax.vjp(NTdiff_solver, xs, params_raw_single, sample_id)
    grad_analytic = np.array(vjp_fn(jnp.ones(()))[0]).flatten()

    grad_fd  = np.zeros(n)
    skipped  = []
    for i in range(n):
        xi = float(flat_xs[i])
        print(f"FD grad check: component {i}/{n}, value={xi:.4e}")
        if abs(xi) < 1e-10:
            skipped.append(i)
            continue
        eps_i    = max(rel_eps * abs(xi), abs_eps_floor)
        xs_plus  = flat_xs.at[i].add(+eps_i).reshape(xs.shape)
        xs_minus = flat_xs.at[i].add(-eps_i).reshape(xs.shape)
        grad_fd[i] = (float(scalar_fn(xs_plus)) - float(scalar_fn(xs_minus))) / (2.0 * eps_i)

    active  = np.array([i for i in range(n) if i not in skipped])
    ga      = grad_analytic[active]
    gf      = grad_fd[active]
    abs_err = np.abs(ga - gf)
    rel_err = abs_err / (np.abs(gf) + 1e-30)

    def make_xs_label(i, xs_shape, lay, G):
        n_per_reg = xs_shape[1]
        reg    = i // n_per_reg
        offset = i % n_per_reg
        for name, sl in lay.items():
            indices = list(range(*sl.indices(n_per_reg)))
            if offset in indices:
                return f"reg{reg}_{name}_g{indices.index(offset)+1}"
        return f"reg{reg}_idx{offset}"

    lay = xs_layout(GEO.G)
    print(f"\n{'='*65}")
    print(f"  Gradient Check — rel_eps={rel_eps:.0e}, floor={abs_eps_floor:.0e}")
    print(f"{'='*65}")
    print(f"  {'Component':<25} {'analytic':>12} {'FD':>12} {'|err|':>10} {'rel_err':>10}")
    print(f"  {'-'*65}")
    for j, i in enumerate(active):
        a, f = ga[j], gf[j]
        ref_scale = max(abs(a), abs(f))
        rel = rel_err[j]
        if ref_scale < 1e-8:        flag = ""
        elif abs(f) < 1e-8:         flag = " FD_UNRESOLVED"
        elif abs(a) < 1e-8:         flag = " AD_ZERO !"
        elif rel > 0.05:            flag = " !"
        else:                       flag = ""
        label = make_xs_label(i, xs.shape, lay, GEO.G)
        print(f"  {label:<25} {a:>12.4e} {f:>12.4e} {abs_err[j]:>10.2e} {rel:>10.2e}{flag}")

    print(f"\n  Skipped (zero XS): {skipped}")
    print(f"  Max  |rel err| : {rel_err.max():.3e}")
    print(f"  Mean |rel err| : {rel_err.mean():.3e}")
    print(f"  Components > 5% error: {(rel_err > 0.05).sum()} / {len(active)}")
    return grad_analytic.reshape(xs.shape), grad_fd.reshape(xs.shape), rel_err


def diagnose_scan_autodiff(xs_tensor, geo_data, phi_fwd, phi_adj, k, SLAY):
    print("\n" + "="*65)
    print("=== Scan Autodiff Diagnostic (FD vs vjp on scan directly) ===")
    print("="*65)

    rel_eps          = 1e-2
    xs_dtype         = jnp.array(xs_tensor).dtype
    eps_machine      = float(jnp.finfo(xs_dtype).eps)
    grad_noise_floor = eps_machine * 1e2
    abs_floor        = 1e2 * eps_machine

    def scalar_fn(xs):
        Aphi, Fphi = Aphi_Fphi_scan(xs, geo_data, SLAY, phi_fwd)
        return jnp.dot(phi_adj, Aphi) - (1.0 / k) * jnp.dot(phi_adj, Fphi)

    grad_analytic = jax.grad(scalar_fn)(xs_tensor)
    xs_np  = np.array(xs_tensor)
    grad_fd = np.zeros_like(xs_np)
    eps_used = np.zeros_like(xs_np)

    for r in range(xs_np.shape[0]):
        for m in range(xs_np.shape[1]):
            val = xs_np[r, m]
            if abs(val) < abs_floor:
                continue
            eps_i = max(rel_eps * abs(val), abs_floor)
            eps_used[r, m] = eps_i
            xs_p = xs_np.copy(); xs_p[r, m] += eps_i
            xs_m = xs_np.copy(); xs_m[r, m] -= eps_i
            grad_fd[r, m] = (float(scalar_fn(jnp.array(xs_p))) -
                             float(scalar_fn(jnp.array(xs_m)))) / (2 * eps_i)

    grad_fd   = jnp.array(grad_fd)
    G         = geo_data['G']
    reg_names = [r['name'] for r in geo_data['regions']] if 'regions' in geo_data \
                else [f'reg{r}' for r in range(xs_np.shape[0])]
    xs_type_names = []
    for name, sl in SLAY.items():
        size = (sl.stop - sl.start) if isinstance(sl, slice) else 1
        for g in range(size):
            xs_type_names.append(f"{name}_g{g+1}")

    print(f"  dtype={xs_dtype}, eps_machine={eps_machine:.2e}, abs_floor={abs_floor:.2e}")
    print(f"  rel_eps={rel_eps:.0e}")
    print(f"\n  {'Component':<20} {'xs_val':>10s} {'eps_used':>10s} {'analytic':>12} {'FD':>12} {'rel_err':>10}")
    print(f"  {'-'*68}")

    all_rel = []
    for r, rname in enumerate(reg_names):
        for m, xname in enumerate(xs_type_names):
            a   = float(grad_analytic[r, m])
            f   = float(grad_fd[r, m])
            rel = abs(a - f) / (abs(f) + 1e-30)
            ref_scale = max(abs(a), abs(f))
            if ref_scale <= grad_noise_floor:               flag = "noise"
            elif abs(f) < grad_noise_floor or abs(f) == 0: flag = " FD_UNRESOLVED"
            elif abs(a) < grad_noise_floor or abs(a) == 0: flag = " AD_ZERO !"
            elif rel > 0.05:                                flag = " !"
            else:
                flag = ""
                all_rel.append(rel)
            print(f"  {rname+xname:20s}  {xs_np[r,m]:10.3e} {eps_used[r,m]:10.2e} "
                  f"{a:12.4e} {f:12.4e} {rel:10.2e} {flag}")

    if all_rel:
        mean_re = sum(all_rel)/len(all_rel)
        print(f"\n  Max rel error  : {max(all_rel):.3e}")
        print(f"  Mean rel error : {mean_re:.3e} (target < 1e-2)")
        if mean_re < 1e-2:
            print("  ✅ Scan autodiff is CORRECT")
        else:
            print("  ❌ Scan autodiff is WRONG — bug inside Aphi_Fphi_scan")
    print("="*65 + "\n")


def diagnose_full_vjp(xs_tensor, params_raw_single, sample_id,
                      NTdiff_solver, _NTdiff_fwd,
                      _GEO_DATA_CACHE, SLAY,
                      rel_eps=1e-2, abs_floor=1e-5):
    print("=" * 65)
    print("Full Custom VJP Diagnostic — FD vs. NTdiff_bwd")
    print("=" * 65)

    xs    = jnp.array(xs_tensor, dtype=jnp.float32)
    xs_np = np.array(xs)

    NTdiff_solver(xs, params_raw_single, sample_id)
    _, (_, k_frozen, phi_fwd_frozen, phi_adj_frozen, Fphi_frozen, _) = \
        _NTdiff_fwd(xs, params_raw_single, sample_id)
    geodata = _GEO_DATA_CACHE[int(sample_id[0])]
    print(f"  frozen k = {k_frozen:.6f}")

    _, vjp_fn     = jax.vjp(NTdiff_solver, xs, params_raw_single, sample_id)
    grad_analytic = np.array(vjp_fn(jnp.ones(()))[0])

    _, Fphi_frozen  = Aphi_Fphi_scan(xs, geodata, SLAY, phi_fwd_frozen)
    denom = (1.0 / k_frozen**2) * jnp.dot(phi_adj_frozen, Fphi_frozen)

    def scalar_fn(xs_in):
        Aphi, Fphi = Aphi_Fphi_scan(xs_in, geodata, SLAY, phi_fwd_frozen)
        num = jnp.dot(phi_adj_frozen, Aphi - (1.0 / k_frozen) * Fphi)
        return float(-num / denom)

    grad_fd  = np.zeros_like(xs_np)
    eps_used = np.zeros_like(xs_np)
    for r in range(xs_np.shape[0]):
        for m in range(xs_np.shape[1]):
            val = xs_np[r, m]
            if abs(val) < abs_floor:
                continue
            epsi = max(rel_eps * abs(val), abs_floor)
            eps_used[r, m] = epsi
            xsp = xs_np.copy(); xsp[r, m] += epsi
            xsm = xs_np.copy(); xsm[r, m] -= epsi
            grad_fd[r, m] = (scalar_fn(jnp.array(xsp)) - scalar_fn(jnp.array(xsm))) / (2 * epsi)

    xs_type_names = []
    for name, sl in SLAY.items():
        size = sl.stop - sl.start if isinstance(sl, slice) else 1
        for g in range(size):
            xs_type_names.append(f"{name}_g{g+1}")
    reg_names        = [f"reg{r}" for r in range(xs_np.shape[0])]
    grad_noise_floor = float(jnp.finfo(jnp.float32).eps) * 1e2

    print(f"\n  {'Component':<22} {'xs val':>10} {'eps':>10} {'analytic':>12} {'FD':>12} {'rel err':>10}")
    print(f"  {'-'*70}")
    all_rel = []
    for r, r_name in enumerate(reg_names):
        for m, x_name in enumerate(xs_type_names):
            a   = float(grad_analytic[r, m])
            f   = float(grad_fd[r, m])
            rel = abs(a - f) / (abs(f) + 1e-30)
            ref_scale = max(abs(a), abs(f))
            if ref_scale < grad_noise_floor:                flag = "  noise!"
            elif abs(f) < grad_noise_floor or abs(f) == 0: flag = " FD_UNRESOLVED"
            elif abs(a) < grad_noise_floor or abs(a) == 0: flag = " AD_ZERO !"
            elif rel > 0.05:                                flag = "  !"
            else:
                flag = ""
                all_rel.append(rel)
            print(f"  {r_name}/{x_name:<18} {xs_np[r,m]:>10.3e} {eps_used[r,m]:>10.2e} "
                  f"{a:>12.3e} {f:>12.3e} {rel:>10.2e}{flag}")

    print(f"  {'-'*70}")
    if all_rel:
        mean = sum(all_rel)/len(all_rel)
        print(f"  Max rel error  : {max(all_rel):.3e}")
        print(f"  Mean rel error : {mean:.3e} (target < 5e-2 for float32)")
        if mean < 5e-2:
            print("  ✓ Custom VJP is CORRECT")
        else:
            print("  ✗ Custom VJP is WRONG — bug inside NTdiff_bwd")
    print("=" * 65)


def full_check(train_rawparams, XS_BASELINE, GEO,
               NTdiff_solver, _NTdiff_fwd, _run_NT_solver,
               _GEO_DATA_CACHE, SLAY, update_geo):
    """
    Single entry point — call this once from main to run all diagnostic checks.
    """
    from solvers.NTdiffusion.diffusion_solver import xs_layout

    xs_check     = jnp.array(XS_BASELINE, dtype=jnp.float32)
    params_check = jnp.array(train_rawparams[0], dtype=jnp.float32)
    id_check     = jnp.array([0], dtype=jnp.int32)

    print("Warming up solver cache for sample 0...")
    _ = NTdiff_solver(xs_check, params_check, id_check)

    geo_i_check = update_geo(GEO, np.array(train_rawparams[0]))
    I_check     = int(geo_i_check.boundaries[-1].radius / geo_i_check.mesh_size)

    # ── Step 1: raw solver output ────────────────────────────────────────
    print("\n--- Step 1: Eigenvalue Residual Check ---")
    k_check, phi_fwd_raw, phi_adj_raw = run_diffusion_solver(
        np.array(xs_check), geo_i_check
    )
    print(f"\n=== Raw Solver Output ===")
    print(f"  k                        = {k_check:.6f}")
    print(f"  ||phi_fwd_raw||          = {np.linalg.norm(phi_fwd_raw):.6e}")
    print(f"  phi_fwd_raw[g=0, i=0:5] = {phi_fwd_raw[0, :5]}")
    print(f"  phi_fwd_raw[g=1, i=0:5] = {phi_fwd_raw[1, :5]}")

    N_flat = geo_i_check.G * (I_check + 1)
    phi_fwd_check_flat = np.zeros(N_flat, dtype=np.float32)
    for g in range(geo_i_check.G):
        phi_fwd_check_flat[g*(I_check+1) : g*(I_check+1)+I_check] = phi_fwd_raw[g, :]

    k_nt, phi_nt_flat, _ = _run_NT_solver(np.array(xs_check), params_check, id_check)
    print(f"\n=== _run_NT_solver vs manual flat ===")
    print(f"  ||phi_nt_flat||  = {np.linalg.norm(phi_nt_flat):.6e}")
    print(f"  Match?           = {np.allclose(phi_fwd_check_flat, phi_nt_flat, atol=1e-5)}")

    check_eigenvalue_residual_vs_solver(
        xs_check, geo_i_check, phi_fwd_check_flat, k_check,
        _GEO_DATA_CACHE, SLAY
    )

    # ── Step 2: physics sanity ───────────────────────────────────────────
    print("\n--- Step 2: Physics Sanity Check ---")
    physics_sanity_check(xs_check, params_check, id_check,
                         NTdiff_solver, xs_layout, GEO)

    # ── Step 3: cell assignments ─────────────────────────────────────────
    print("\n--- Step 3.1: Cell Assignment Check ---")
    diagnose_scan_vs_real_cellwise(
        xs_check, geo_i_check, phi_fwd_check_flat, k_check,
        _GEO_DATA_CACHE, SLAY
    )

    # ── Step 3.2: scan autodiff ──────────────────────────────────────────
    print("\n--- Step 3.2: Scan Autodiff Check ---")
    k_nt, phifwd, phiadj = _run_NT_solver(np.array(xs_check), params_check, id_check)
    diagnose_scan_autodiff(
        xs_check, _GEO_DATA_CACHE[0],
        jnp.array(phifwd), jnp.array(phiadj), k_nt, SLAY
    )

    # ── Step 4: full VJP ─────────────────────────────────────────────────
    print("\n--- Step 4: Full Custom VJP Check ---")
    diagnose_full_vjp(xs_check, params_check, id_check,
                      NTdiff_solver, _NTdiff_fwd,
                      _GEO_DATA_CACHE, SLAY)

    # ── Step 5: N_flat consistency ───────────────────────────────────────
    print("\nChecking N_flat distribution across training samples...")
    n_flats = {geo_i_check.G * (int(update_geo(GEO, train_rawparams[i]).boundaries[-1].radius
               / update_geo(GEO, train_rawparams[i]).mesh_size) + 1)
               for i in range(len(train_rawparams))}
    print(f"  Unique N_flat values: {sorted(n_flats)}")
    print(f"  Count: {len(n_flats)}")