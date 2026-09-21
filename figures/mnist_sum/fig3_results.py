import pathlib
import json,sys,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP=sys.argv[1]; R=str(pathlib.Path(__file__).resolve().parents[2] / "results")
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],
                     "font.size":8,"axes.linewidth":0.6,"pdf.fonttype":42})
INK="#14181a"; MUT="#5d6b6e"; C28="#2a78d6"; C14="#eb6834"; BASE="#9aa5a3"; CEIL="#0d6b5f"
A=json.load(open(R+"/mnist_sum_p28_500k/results.json"))
B=json.load(open(R+"/mnist_sum_p14_500k/results.json"))
fig,axes=plt.subplots(1,3,figsize=(7.0,2.5),gridspec_kw=dict(width_ratios=[1.5,0.78,1.12],wspace=0.42))

def band(D,grp,split,k):
    d=D[grp][split][k] if split else D[grp][k]
    return 100*d["value"],100*(d["value"]-d["ci_lo"]),100*(d["ci_hi"]-d["value"])

# (a) i2t
ax=axes[0]; keys=["perception","arithmetic","end_to_end"]; x=np.arange(3); w=0.19
for j,(D,c,nm) in enumerate(((A,C28,"P=28"),(B,C14,"P=14"))):
    for s,(split,al) in enumerate((("val_gs2",1.0),("val_holdout_gs2",.55))):
        vv=[band(D,"i2t",split,k) for k in keys]
        ax.bar(x+(2*j+s-1.5)*w,[v[0] for v in vv],w*0.92,
               yerr=[[v[1] for v in vv],[v[2] for v in vv]],color=c,alpha=al,
               edgecolor="white",linewidth=0.4,
               error_kw=dict(elinewidth=0.7,capsize=1.5,ecolor=INK),
               label=f"{nm} {'val' if s==0 else 'held-out'}")
ax.set_xticks(x); ax.set_xticklabels(["perception","arithmetic","end-to-end"],fontsize=6.9)
ax.set_ylim(95.6,100.5); ax.set_ylabel("% exactly correct",fontsize=7.2); ax.tick_params(labelsize=6.8)
ax.set_title("(a)  image $\\rightarrow$ text",loc="left",fontsize=8,pad=4,color=INK)
ax.text(0.015,0.935,"modal-sum baseline 6.5% (off scale)",transform=ax.transAxes,fontsize=5.9,color=MUT)
ax.legend(fontsize=5.7,frameon=False,ncol=2,loc="lower right",
          handlelength=1.1,columnspacing=0.9,handletextpad=0.4)
for sp in ("top","right"): ax.spines[sp].set_visible(False)

# (b) t2i
ax=axes[1]
v28=100*A["t2i"]["val_gs2"]["all_four"]["value"]; v14=100*B["t2i"]["val_gs2"]["all_four"]["value"]
ax.bar([0],[v28],0.42,color=C28); ax.bar([1],[v14],0.42,color=C14); ax.set_xlim(-0.55,2.25)
ax.axhline(99.78,color=CEIL,lw=1.0)
ax.text(1.58,99.78,"same-pool\nceiling 99.78",fontsize=5.7,color=CEIL,va="center",ha="left")
ax.axhline(96.09,color=BASE,ls=":",lw=1.0)
ax.text(1.58,96.09,"test-pool\nceiling 96.09",fontsize=5.7,color=MUT,va="center",ha="left")
for i,v in enumerate((v28,v14)): ax.text(i,v+0.10,f"{v:.2f}",ha="center",fontsize=6.3,color=INK)
ax.set_xticks([0,1]); ax.set_xticklabels(["P=28","P=14"],fontsize=6.9)
ax.set_ylim(95.6,100.5); ax.set_ylabel("% all four correct",fontsize=7.2); ax.tick_params(labelsize=6.8)
ax.set_title("(b)  text $\\rightarrow$ image",loc="left",fontsize=8,pad=4,color=INK)
for sp in ("top","right"): ax.spines[sp].set_visible(False)

# (c) joint
ax=axes[2]
kk=["text_wellformed","addends_match_image","arithmetic_valid","consistent"]
ll=["well-formed","addends match","arithmetic valid","consistent"]
y=np.arange(4)[::-1]; h=0.34
for j,(D,c,nm) in enumerate(((A,C28,"P=28"),(B,C14,"P=14"))):
    vv=[band(D,"joint","uncond",k) for k in kk]
    ax.barh(y+(0.5-j)*h,[v[0] for v in vv],h*0.9,
            xerr=[[v[1] for v in vv],[v[2] for v in vv]],color=c,edgecolor="white",linewidth=0.4,
            error_kw=dict(elinewidth=0.7,capsize=1.5,ecolor=INK),label=nm)
ax.set_yticks(y); ax.set_yticklabels([])
for yy,t in zip(y,ll): ax.text(95.72,yy+0.47,t,fontsize=6.3,va="center",ha="left",color=INK)
ax.set_ylim(-0.85,3.95)
ax.set_xlim(95.6,100.5); ax.set_xlabel("% of samples",fontsize=7.2); ax.tick_params(labelsize=6.8)
ax.set_title("(c)  joint co-generation",loc="left",fontsize=8,pad=4,color=INK)
ax.legend(fontsize=6.2,frameon=False,loc="lower right",handlelength=1.1,handletextpad=0.4)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.savefig(SP+"/fig_mnist_sum_results.pdf",bbox_inches="tight")
fig.savefig(SP+"/fig_mnist_sum_results.png",dpi=260,bbox_inches="tight")
print("wrote fig_mnist_sum_results")
