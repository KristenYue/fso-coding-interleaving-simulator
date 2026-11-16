import numpy as np, json, sys, fso_lib as F
from qfunc import Q
def thr_interp(x,y,target=1e-3):
    x=np.asarray(x,float); y=np.asarray(y,float)
    for i in range(1,len(y)):
        if y[i]<target<=y[i-1]:
            l0,l1=np.log10(y[i-1]),np.log10(y[i])
            return x[i-1]+(np.log10(target)-l0)*(x[i]-x[i-1])/(l1-l0)
    return float('nan')
name=sys.argv[1]; sR2=F.TURBULENCE[name]
a,b=F.gg_params(sR2)
rng=np.random.default_rng(31415)
lo,hi={'Weak':(25,29),'Moderate':(32,36),'Strong':(41,45)}[name]
grid=np.arange(lo,hi,0.1)
blocks=[]
for k in range(3):
    h=F.draw_h(1_000_000,a,b,F.POINTING['Severe'],rng)
    ber=np.array([Q(h*np.sqrt(10**(g/10))).mean() for g in grid])
    blocks.append(ber)
allber=np.mean(blocks,axis=0)
t=thr_interp(grid,allber)
ts=[thr_interp(grid,bk) for bk in blocks]
sd=float(np.std(ts,ddof=1)/np.sqrt(3))
print(json.dumps({"regime":name,"sigma_R2":sR2,"alpha":round(a,4),"beta":round(b,4),
                  "SI":round(F.scintillation_index(a,b),4),
                  "threshold_db":round(float(t),3),"sem_db":round(sd,3),
                  "n_samples":3_000_000}))
