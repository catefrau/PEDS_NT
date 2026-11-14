import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
d=pd.read_csv('data/files/dataset_keff.csv');P=list(d.columns[:6])
lo=np.array([1,0.0,10,1,0.40,40.]);hi=np.array([6,0.98,39,10,0.75,59.])
KW=(0.65,1.35)
X=d[P].values;k=d.keff.values
m=(k>=KW[0])&(k<=KW[1])&np.all((X>=lo)&(X<=hi),1)
Xs=X[m];ks=k[m];print('N',m.sum(),'keff',ks.min(),ks.max())
ref=np.load('data/highfidelity/17jul_0.8_1.2.npz')
out=dict(params=((Xs-lo)/(hi-lo)).astype(np.float32),params_raw=Xs.astype(np.float32),keffs=ks.astype(np.float32),
 param_names=ref['param_names'],bounds_lo=lo.astype(np.float32),bounds_hi=hi.astype(np.float32))
for kk in ref.files:assert out[kk].dtype==ref[kk].dtype,kk
np.savez('data/files/dataset_A3_rounded.npz',**out)
dd=d[m].copy();dd.to_csv('data/files/dataset_A3_rounded.csv',index=False)
fig,ax=plt.subplots(3,3,figsize=(15,11))
a=ax[0,0];a.hist(ks,40,color='C0',edgecolor='k');a.axvline(KW[0],c='r',ls='--');a.axvline(KW[1],c='r',ls='--')
a.set_title(f'keff distribution (N={len(ks)})');a.set_xlabel('keff')
for j,p in enumerate(P):
    a=ax.flat[j+1];c,_,_=a.hist(Xs[:,j],10,(lo[j],hi[j]),color='C1',edgecolor='k')
    a.axhline(len(ks)/10,c='r',ls='--',label='uniform');a.set_title(f'{p}\n[{lo[j]:g}, {hi[j]:g}]',fontsize=9);a.set_ylabel('count / bin')
    a.set_ylim(0,max(c.max(),len(ks)/10)*1.15)
ax.flat[1].legend()
ax[2,1].axis('off');ax[2,2].axis('off')
plt.tight_layout();plt.savefig('data/reports/A3_rounded_distributions.png',dpi=130)
for j,p in enumerate(P):print(p,np.histogram(Xs[:,j],10,(lo[j],hi[j]))[0].tolist())
