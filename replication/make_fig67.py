"""Figure 6-11: replication-campaign RS curves with group-aware CI,
cross-validation scatter, and the d_v sensitivity result."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def thr(x,y,t=1e-3):
    o=np.argsort(x); x=np.array(x)[o]; y=np.array(y)[o]
    for i in range(1,len(y)):
        if y[i]<t<=y[i-1]:
            return x[i-1]+(np.log10(t)-np.log10(y[i-1]))*(x[i]-x[i-1])/(np.log10(y[i])-np.log10(y[i-1]))
    return float('nan')

rs=[r for r in csv.DictReader(open('rs_groupaware.csv'))]
dv3=[r for r in csv.DictReader(open('s1_ldpc_dv3.csv'))]
t10=[r for r in csv.DictReader(open('10_turbulence_sensitivity.csv'))]
COL={1:"#1f77b4",8:"#2ca02c",32:"#ff7f0e",128:"#d62728"}
fig,ax=plt.subplots(2,2,figsize=(13.2,9.6))

# (a),(b) RS with group-aware CI
for k,Lc in enumerate((64,256)):
    a=ax[0,k]
    for D in (1,8,32,128):
        s=[r for r in rs if int(r['Lc_bits'])==Lc and int(r['D'])==D]
        s.sort(key=lambda r: float(r['ebn0_db']))
        x=[float(r['ebn0_db']) for r in s]; y=np.array([max(float(r['ber']),1e-7) for r in s])
        e=np.array([float(r['ber_ci95']) for r in s])
        a.semilogy(x,y,"o-",ms=3.5,lw=1.3,color=COL[D],label=f"D = {D}")
        a.fill_between(x,np.maximum(y-e,1e-8),y+e,color=COL[D],alpha=0.20,lw=0)
    a.axhline(1e-3,color="k",ls="--",lw=0.9); a.axvline(33.796,color="grey",ls=":",lw=1.2)
    a.set_title(f"({'ab'[k]}) RS(255,223), $L_c$ = {Lc} bits — group-aware 95 % CI",fontsize=10)
    a.set_xlabel("Information-bit $E_b/N_0$ (dB)"); a.set_ylabel("Decoded information BER")
    a.set_ylim(1e-6,1); a.grid(True,which="both",alpha=0.3); a.legend(fontsize=8)

# (c) cross-validation scatter
a=ax[1,0]
pri={('RS',64):[27.19,24.05,24.30,24.18],('RS',256):[33.21,25.82,24.30,24.21],
     ('LDPC',64):[31.63,26.15,23.09,21.62]}
rep={('RS',64):[27.42,24.42,24.41,24.29],('RS',256):[32.56,25.87,24.22,24.33],
     ('LDPC',64):[32.22,26.99,23.35,21.58]}
mk={'RS':'o','LDPC':'s'}; cl={64:"#2F6FB5",256:"#D07C1E"}
for (c,Lc) in pri:
    a.scatter(pri[(c,Lc)],rep[(c,Lc)],marker=mk[c],s=55,color=cl[Lc],
              edgecolor='k',lw=0.6,label=f"{c}, $L_c$={Lc}")
lim=[20,35]; a.plot(lim,lim,"k--",lw=1,label="perfect agreement")
a.fill_between(lim,[l-1 for l in lim],[l+1 for l in lim],color="grey",alpha=0.15,label="±1 dB")
a.set_xlim(lim); a.set_ylim(lim); a.set_aspect('equal')
a.set_xlabel("Primary campaign threshold (dB)"); a.set_ylabel("Replication threshold (dB)")
a.set_title("(c) Cross-validation of the two implementations",fontsize=10)
a.grid(alpha=0.3); a.legend(fontsize=8,loc="upper left")

# (d) dv sensitivity
a=ax[1,1]
for D in (1,8,32,128):
    s=[r for r in dv3 if int(r['D'])==D]; s.sort(key=lambda r: float(r['ebn0_db']))
    a.semilogy([float(r['ebn0_db']) for r in s],
               [max(float(r['ber']),1e-7) for r in s],"s-",ms=3.5,lw=1.4,color=COL[D],
               label=f"$d_v$=3, D={D}")
for D in (1,8,32,128):
    s=[r for r in t10 if r['turbulence']=='Moderate' and r['codec']=='LDPC' and int(r['D'])==D]
    s.sort(key=lambda r: float(r['ebn0_db']))
    a.semilogy([float(r['ebn0_db']) for r in s],
               [max(float(r['ber']),1e-7) for r in s],"--",lw=1.0,color=COL[D],alpha=0.55)
a.axhline(1e-3,color="k",ls="--",lw=0.9)
a.set_title("(d) LDPC column weight: $d_v$=3 (solid) vs $d_v$=2 (dashed)",fontsize=10)
a.set_xlabel("Information-bit $E_b/N_0$ (dB)"); a.set_ylabel("Decoded information BER")
a.set_ylim(1e-6,1); a.set_xlim(15,36); a.grid(True,which="both",alpha=0.3); a.legend(fontsize=7,ncol=2)

fig.suptitle("Replication campaign: uncertainty quantification, cross-validation and code-design sensitivity",fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.96])
for e in ("pdf","png","svg"): fig.savefig(f"fig6_11_replication.{e}",dpi=190,bbox_inches="tight",facecolor="white")
print("saved fig6_11_replication.*")
