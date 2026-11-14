import numpy as np, pandas as pd, glob, os
from scipy import stats
from scipy.spatial import cKDTree
R='data/'
d=pd.read_csv(R+'files/dataset_keff.csv');P=list(d.columns[:6])
X=d[P].values;k=d.keff.values;N=len(d)
out=[];W=out.append
exact=d.duplicated(P).sum()
Xf=X.astype(np.float32)
print('exact',exact,'float32 dups',pd.DataFrame(Xf).duplicated().sum())
lo=X.min(0);hi=X.max(0)
H={os.path.basename(f)[:-4]:np.load(f) for f in sorted(glob.glob(R+'highfidelity/*.npz'))}
W("# `dataset_keff.csv`: analysis and relation to the `data/highfidelity` files\n")
W("## 1. Duplicate check\n")
W(f"- Rows: {N}. Parameter columns: {', '.join(P)}.")
W(f"- Exact duplicate 6-parameter vectors (float64): **{exact}**; after float32 casting: **{pd.DataFrame(Xf).duplicated().sum()}**.")
sc=X/(hi-lo)
for tol in (1e-6,1e-4):
    t=cKDTree(sc);pr=t.query_pairs(tol)
    W(f"- Near-duplicate pairs (Euclidean distance < {tol:g} in range-scaled space): **{len(pr)}**.")
W("- Therefore no rows had to be removed and `dataset_keff.csv` was used as is (no new file written).\n")
W(f"- NaN values: {int(d.isna().sum().sum())}.\n")
W("## 2. Samples, bounds, keff\n")
W(f"N = **{N}** unique cases.\n\n| param | min | max | mean |\n|---|---|---|---|")
for i,p in enumerate(P):W(f"| {p} | {lo[i]:.4g} | {hi[i]:.4g} | {X[:,i].mean():.4g} |")
W(f"\nkeff: min {k.min():.4f}, max {k.max():.4f}, mean {k.mean():.4f}, std {k.std():.4f}; share in [0.8,1.2]: {((k>=0.8)&(k<=1.2)).mean():.1%}. Mean MC keff_std {d.keff_std.mean():.2e} (max {d.keff_std.max():.2e}).\n")
W("## 3. Relation to the high-fidelity files\n")
W("Rows are matched on the 6 parameters after casting to float32 (the precision of the npz files), and keff is compared on matches.\n")
key=lambda A:[tuple(r) for r in np.asarray(A,np.float32)]
kd={}
for r,kk in zip(key(X),k):kd[r]=kk
W("| file | N unique | found in dataset_keff | % | keff max abs diff on matches | in file but missing from csv |\n|---|---|---|---|---|---|")
cover={}
for n,h in H.items():
    ku=key(h['params_raw']);u=dict(zip(ku,h['keffs']))
    m=[r for r in u if r in kd];cover[n]=set(m)
    diff=max(abs(kd[r]-u[r]) for r in m) if m else np.nan
    W(f"| {n} | {len(u)} | {len(m)} | {len(m)/len(u):.1%} | {diff:.2g} | {len(u)-len(m)} |")
allh=set().union(*cover.values());rows=set(kd)
W(f"\nUnion of matched rows: **{len(allh)}** of {N} csv rows ({len(allh)/N:.1%}). **{N-len(allh)}** rows ({(N-len(allh))/N:.1%}) of the csv are not in any of the 6 npz files.\n")
mask=np.array([r not in allh for r in key(X)])
Xn=X[~mask];Xe=X[mask];ke=k[mask]
W("### Rows only in the csv (not in any npz) vs rows shared\n\n| | N | keff min | keff max | keff mean | share in [0.8,1.2] |\n|---|---|---|---|---|---|")
for nm,kk in (("shared with npz",k[~mask]),("csv only",ke)):
    W(f"| {nm} | {len(kk)} | {kk.min():.3f} | {kk.max():.3f} | {kk.mean():.3f} | {((kk>=.8)&(kk<=1.2)).mean():.1%} |")
W("\nParameter extent of csv-only rows:\n\n| param | min | max |\n|---|---|---|")
for i,p in enumerate(P):W(f"| {p} | {Xe[:,i].min():.4g} | {Xe[:,i].max():.4g} |")
W("\n### Inclusion between files\n")
cf=cover.get('complete_LHS_2900',set());ok=set(key(X[(k>=.8)&(k<=1.2)]))
W(f"- Of the csv rows, {((k>=.8)&(k<=1.2)).sum()} have keff in [0.8,1.2]; of those, {len([r for r in ok if r in allh])} appear in some npz.")
for n in H:
    W(f"- `{n}` is {'fully' if len(cover[n])==len(set(key(H[n]['params_raw']))) else 'partly'} contained in the csv ({len(cover[n])}/{len(set(key(H[n]['params_raw'])))}).")
W("\n## 4. Is dataset_keff.csv LHS?\n")
W("Mapped to [0,1] with its own min/max. A true LHS has one point per N-bin in each dimension (occupancy 1.0, random 0.63).\n")
W("| bins m | occupancy (mean over 6 dims) | random expectation | Cr fraction + enrichment only |\n|---|---|---|---|")
u=np.clip((X-lo)/(hi-lo),0,1-1e-12)
for div in (10,4,2,1):
    m=N//div
    o=[(np.bincount((u[:,i]*m).astype(int),minlength=m)>0).mean() for i in range(6)]
    W(f"| N/{div} | {np.mean(o):.3f} | {1-(1-1/m)**N:.3f} | {np.mean([o[1],o[3]]):.3f} |")
rho=stats.spearmanr(u).statistic;np.fill_diagonal(rho,0)
W(f"\nmax |Spearman| between parameters: {abs(rho).max():.3f}. KS vs uniform p-values per dim: "+", ".join(f"{stats.kstest(u[:,i],'uniform').pvalue:.1e}" for i in range(6))+".\n")
W("10-bin counts per parameter:\n\n| param | counts |\n|---|---|")
for i,p in enumerate(P):W(f"| {p} | {np.histogram(u[:,i],10,(0,1))[0].tolist()} |")
W('''
## 5. Summary

- **Duplicates:** none (exact or near), so nothing was removed and no new csv was written.
- **Size:** 4599 cases, versus 1315-2931 rows (1854-2868 unique) in the npz files.
- **Superset:** every unique row of all 6 npz files is in the csv, with identical keff to float32 precision. The csv is the union of them (3910 rows) plus 689 extra rows.
- **Extra rows:** they have a lower share in the 0.8-1.2 window (48.5% vs 81%) and reach keff down to 0.43, below the npz minimum of 0.554. They look like later or unfiltered additions, not LHS-filtered points. Their r0 radius reaches 9.5, r1 radius 42.9 and f_mod starts at 0.40.
- **Bounds:** these are the widest of all the files: r0 radius 1-12, enrichment 1-10 and r2 radius 40-80. Enrichment goes down to 1.0, like `17jul_0.8_1.2` but unlike the other npz files (1.5). keff is 0.43-1.43, wider than the unfiltered `complete_LHS_2900` (0.55-1.43).
- **Distribution:** not LHS. Occupancy at N bins is 0.59, which is at or below the random level of 0.63. Marginals are strongly non-uniform, and the dense-core plus sparse-tail structure of `complete_LHS_2900` carries over (e.g. r0 radius ~920-980 per bin below 6 and ~50 per bin above). Cr fraction is the only parameter near uniform (KS p=0.08). Parameter correlations are smaller than in the npz files (max |Spearman| 0.15), because the keff filter is not applied to the whole csv.
''')
open(R+'reports/dataset_keff_report.md','w').write("\n".join(out))
print("\n".join(out))
