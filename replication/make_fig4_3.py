"""Figure 4-3: naive bit-level Wilson interval vs group-aware interval."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows=[r for r in csv.DictReader(open('rs_groupaware.csv'))]
cfg=[]
for Lc in (64,256):
    for D in (1,8,32,128):
        c=[r for r in rows if int(r['Lc_bits'])==Lc and int(r['D'])==D]
        r=min(c,key=lambda r:abs(float(r['ber'])-1e-3))
        p=float(r['ber']); N=int(r['info_bits']); ga=float(r['ber_ci95']); z=1.96
        hw=z*np.sqrt(p*(1-p)/N + z*z/(4*N*N))/(1+z*z/N)
        cfg.append((Lc,D,int(r['groups']),p,hw,ga,ga/hw))
fig,ax=plt.subplots(1,2,figsize=(12.4,4.9))
lab=[f"$L_c$={c[0]}\n$D$={c[1]}" for c in cfg]; x=np.arange(len(cfg))
a=ax[0]
a.bar(x-0.2,[c[4]/c[3]*100 for c in cfg],0.4,label="naive bit-level Wilson",color="#C0392B")
a.bar(x+0.2,[c[5]/c[3]*100 for c in cfg],0.4,label="group-aware (Eq. 4.2)",color="#1E8A63")
a.set_xticks(x); a.set_xticklabels(lab,fontsize=8)
a.set_ylabel("95 % CI half-width, % of BER"); a.legend(fontsize=9)
a.set_title("(a) Reported uncertainty at BER $\\approx$ 10$^{-3}$",fontsize=10)
a.grid(axis="y",alpha=0.3)
b=ax[1]
b.bar(x,[c[6] for c in cfg],0.55,color="#2F6FB5")
b.plot(x,[np.sqrt(c[1]) for c in cfg],"k--o",ms=4,lw=1.3,label="$\\sqrt{D}$ prediction")
for i,c in enumerate(cfg): b.text(i,c[6]+0.4,f"{c[6]:.1f}×",ha="center",fontsize=8)
b.set_xticks(x); b.set_xticklabels(lab,fontsize=8)
b.set_ylabel("understatement factor"); b.legend(fontsize=9)
b.set_title("(b) How far the naive interval is too narrow",fontsize=10)
b.grid(axis="y",alpha=0.3); b.set_ylim(0,20)
fig.suptitle("Naive versus group-aware confidence intervals, RS(255,223), Moderate turbulence",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94])
for e in ("pdf","png","svg"): fig.savefig(f"fig4_3_group_aware_ci.{e}",dpi=200,bbox_inches="tight",facecolor="white")
print("saved fig4_3_group_aware_ci.*")
for c in cfg: print(f"  Lc={c[0]:<4} D={c[1]:<4} G={c[2]:<5} BER={c[3]:.2e}  naive±{c[4]:.2e}  group±{c[5]:.2e}  ratio {c[6]:.1f}x")
