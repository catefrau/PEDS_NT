import warnings
import numpy as np
import sys, os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from modules.PEDS_v7 import update_geo, GEO
from solvers.NTdiffusion.diffusion_solver import predict_xs, run_diffusion_solver, _XS_COLS, _CHI_COLS

data = np.load("../data/highfidelity/1000_clean.npz", allow_pickle=True)
rawparams, keffs_mc = data["params_raw"], data["keffs"]

neg = warn = bad_k = pcm = []
for i, p in enumerate(rawparams):
    geo = update_geo(GEO, np.array(p, dtype=np.float32))
    xs = predict_xs(geo)
    if (xs < 0).any() or not np.isfinite(xs).all():
        neg.append(i)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        k, phi, adj = run_diffusion_solver(xs, geo)
        if any("inverse_power" in str(x.filename) for x in w):
            warn.append(i)
    if not np.isfinite(k):
        bad_k.append(i)
    else:
        pcm.append(abs(k - keffs_mc[i]) / (k * keffs_mc[i]) * 1e5)

print(f"negative/non-finite XS: {len(neg)}/{len(rawparams)}")
print(f"inverse_power warnings: {len(warn)}/{len(rawparams)}")
print(f"non-finite k:           {len(bad_k)}/{len(rawparams)}")
if pcm:
    pcm = np.array(pcm)
    print(f"baseline |Δρ| pcm: mean={pcm.mean():.0f} median={np.median(pcm):.0f} max={pcm.max():.0f}")