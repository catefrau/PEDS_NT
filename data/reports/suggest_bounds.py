import numpy as np, pandas as pd
d=pd.read_csv('data/files/dataset_keff.csv');P=list(d.columns[:6])
X=d[P].values;k=d.keff.values;N=len(d)
rng=np.random.default_rng(0)
def ks(v,lo,hi):
    v=np.sort(v);n=len(v);u=(v-lo)/(hi-lo);i=np.arange(1,n+1)
    return max((i/n-u).max(),(u-(i-1)/n).max())
def score(m,lo,hi):
    D=[ks(X[m,j],lo[j],hi[j]) for j in range(6)]
    return np.mean(D)+0.5*np.max(D),D
def search(kw,nmin,iters=40):
    kmask=(k>=kw[0])&(k<=kw[1])
    grid=[np.unique(np.quantile(X[kmask,j],np.linspace(0,1,61))) for j in range(6)]
    best=None
    for r in range(iters):
        lo=np.array([g[0] for g in grid]);hi=np.array([g[-1] for g in grid])
        def ev(lo,hi):
            m=kmask&np.all((X>=lo)&(X<=hi),1)
            if m.sum()<nmin:return 1e9,None,m
            s,D=score(m,lo,hi);return s,D,m
        cur=ev(lo,hi)[0]
        for it in range(8):
            ch=False
            for j in rng.permutation(6):
                for side in (0,1):
                    bs=(cur,None)
                    for v in grid[j]:
                        l2,h2=lo.copy(),hi.copy()
                        if side==0:l2[j]=v
                        else:h2[j]=v
                        if l2[j]>=h2[j]:continue
                        s=ev(l2,h2)[0]
                        if s<bs[0]-1e-9:bs=(s,v)
                    if bs[1] is not None:
                        if side==0:lo[j]=bs[1]
                        else:hi[j]=bs[1]
                        cur=bs[0];ch=True
            if not ch:break
        # random perturbation start for next iter
        s,D,m=ev(lo,hi)
        if best is None or s<best[0]:best=(s,lo.copy(),hi.copy(),D,m)
        grid=[np.unique(np.quantile(X[kmask,j],np.sort(rng.uniform(0,1,61)))) if r%2 else grid[j] for j in range(6)]
        for j in range(6):grid[j]=np.unique(np.r_[grid[j],X[kmask,j].min(),X[kmask,j].max()])
    return best
def occ(m,lo,hi):
    n=m.sum();return [np.histogram(X[m,j],10,(lo[j],hi[j]))[0] for j in range(6)]
cases=[("A1",(0.6,1.4),2800),("A2",(0.7,1.3),2800),("A3",(0.65,1.35),2800),("B1",(0.7,1.3),1800),("B2",(0.6,1.4),1800)]
res=[]
base=np.all((X>=X.min(0))&(X<=X.max(0)),1)
s0,D0=score(base,X.min(0),X.max(0))
res.append("# Suggested parameter bounds for a more uniform `dataset_keff.csv`\n")
res.append(f"Baseline (full csv): N={N}, mean KS D={np.mean(D0):.3f}, max={np.max(D0):.3f}. KS D is the max distance to the uniform CDF over the bounds (0=perfectly uniform). All rows are taken from the existing csv: a box plus a keff window.\n")
for nm,kw,nm_min in cases:
    s,lo,hi,D,m=search(kw,nm_min)
    res.append(f"## {nm}: keff in [{kw[0]},{kw[1]}], N >= {nm_min}\n\nN = **{m.sum()}**, mean KS D = **{np.mean(D):.3f}**, max = {np.max(D):.3f}, keff range in subset {k[m].min():.3f}-{k[m].max():.3f}\n\n| param | lo | hi | KS D | 10-bin counts |\n|---|---|---|---|---|")
    h=occ(m,lo,hi)
    for j,p in enumerate(P):res.append(f"| {p} | {lo[j]:.4g} | {hi[j]:.4g} | {D[j]:.3f} | {h[j].tolist()} |")
    res.append("")
    pd.DataFrame({'lo':lo,'hi':hi},index=P).to_csv(f'data/reports/bounds_{nm}.csv')
    print(nm,m.sum(),np.mean(D),flush=True)
open('data/reports/suggested_bounds_report.md','w').write("\n".join(res))
print("\n".join(res))
