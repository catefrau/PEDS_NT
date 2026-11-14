import numpy as np, glob, os
from scipy import stats
out=[]
P=out.append
files=sorted(glob.glob('data/highfidelity/*.npz'))
D={os.path.basename(f):np.load(f) for f in files}
names=list(D.values())[0]['param_names']
P("# High-fidelity dataset comparison\n")
P("Parameters: "+", ".join(map(str,names))+"\n")
P("## 1. Samples\n\n| file | N | NaN keff | duplicate rows | unique rows |\n|---|---|---|---|---|")
for n,d in D.items():
    x=d['params_raw'];u=len(np.unique(x,axis=0))
    P(f"| {n} | {len(x)} | {int(np.isnan(d['keffs']).sum())} | {len(x)-u} | {u} |")
P("\n## 2. Bounds\n\nStored bounds (bounds_lo / bounds_hi) vs. observed min/max of `params_raw`.\n")
for n,d in D.items():
    x=d['params_raw'];lo=d['bounds_lo'];hi=d['bounds_hi'];k=d['keffs']
    P(f"### {n}\n\n| param | stored lo | stored hi | obs min | obs max | span used |\n|---|---|---|---|---|---|")
    for i,nm in enumerate(names):
        P(f"| {nm} | {lo[i]:.4g} | {hi[i]:.4g} | {x[:,i].min():.4g} | {x[:,i].max():.4g} | {(x[:,i].max()-x[:,i].min())/(hi[i]-lo[i]):.3f} |")
    P(f"\nkeff: min {k.min():.4f}, max {k.max():.4f}, mean {k.mean():.4f}, std {k.std():.4f}, p1/p99 {np.percentile(k,1):.4f}/{np.percentile(k,99):.4f}\n")
    P(f"`params` (normalised) range: [{d['params'].min():.3f}, {d['params'].max():.3f}]\n")
P("## 3. LHS diagnostics\n")
P("Using `params_raw` mapped to [0,1] with the stored bounds (u). For a true LHS of N points, each of N equal bins per dimension has exactly one point (occupancy=1.0, max/bin=1). Pure random gives occupancy ≈0.632. Also: KS vs uniform, max |Spearman| between params, and occupancy at coarse stratification (N/10 bins), which merged/filtered LHS sets may retain.\n")
P("| file | N | occ(N bins) mean over dims | occ(N/10 bins) | max count/bin (N/10) | expected random occ(N/10) | KS p (min over dims) | mean KS D | max abs Spearman | chi2 p (N/10 bins, min) |\n|---|---|---|---|---|---|---|---|---|---|")
for n,d in D.items():
    x=d['params_raw'].astype(float);lo=d['bounds_lo'];hi=d['bounds_hi'];N=len(x)
    u=np.clip((x-lo)/(hi-lo),0,1-1e-12)
    occ=np.mean([len(np.unique((u[:,i]*N).astype(int)))/N for i in range(6)])
    m=max(N//10,2)
    c=[np.bincount((u[:,i]*m).astype(int),minlength=m) for i in range(6)]
    occm=np.mean([(ci>0).mean() for ci in c])
    mx=max(ci.max() for ci in c)
    ks=[stats.kstest(u[:,i],'uniform') for i in range(6)]
    chi=[stats.chisquare(ci).pvalue for ci in c]
    rho=stats.spearmanr(u).statistic;np.fill_diagonal(rho,0)
    P(f"| {n} | {N} | {occ:.3f} | {occm:.3f} | {mx} (mean {N/m:.0f}) | 1.000 | {min(k.pvalue for k in ks):.2g} | {np.mean([k.statistic for k in ks]):.4f} | {abs(rho).max():.3f} | {min(chi):.2g} |")
P("\nPer-dimension chi2 / bin-count spread (N/10 bins; std of counts vs sqrt(mean) expected for random, 0 for LHS-like):\n")
P("| file | "+" | ".join(map(str,names))+" |\n|---|"+"---|"*6)
for n,d in D.items():
    x=d['params_raw'].astype(float);lo=d['bounds_lo'];hi=d['bounds_hi'];N=len(x)
    u=np.clip((x-lo)/(hi-lo),0,1-1e-12);m=max(N//10,2)
    P(f"| {n} | "+" | ".join(f"{np.bincount((u[:,i]*m).astype(int),minlength=m).std():.2f} (rand {np.sqrt(N/m):.2f})" for i in range(6))+" |")
P("\n## 4. Overlap between files (identical param rows)\n")
ks_=list(D);rows={k:set(map(tuple,D[k]['params_raw'].round(6))) for k in ks_}
P("| | "+" | ".join(ks_)+" |\n|---|"+"---|"*len(ks_))
for a in ks_:
    P(f"| {a} | "+" | ".join(str(len(rows[a]&rows[b])) for b in ks_)+" |")
open('reports/highfidelity_stats.md','w').write("\n".join(out))
print("\n".join(out))
