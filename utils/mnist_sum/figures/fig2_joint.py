import json,sys,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP=sys.argv[1]
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],
                     "font.size":8,"axes.linewidth":0.6,"pdf.fonttype":42})
INK="#14181a"; MUT="#5d6b6e"; OK="#0d6b5f"; BAD="#b3541e"
A=json.load(open(SP+"/assets.json")); D=np.load(SP+"/assets.npz")
rows=A["joint_scored"]; N=12
fig,axes=plt.subplots(2,6,figsize=(7.0,2.92))
for k,ax in enumerate(axes.ravel()):
    r=rows[k]; ax.imshow(D["joint_img"][k],cmap="gray_r",interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    c=OK if r["consistent"] else BAD
    for s in ax.spines.values(): s.set_color(c); s.set_linewidth(1.1 if not r["consistent"] else 0.7)
    t=r["text"].replace(" + ","+").replace(" = ","=")
    a,b=t.split("=")
    ax.set_xlabel(a+"\n= "+b,fontsize=5.5,color=INK,labelpad=2.5,linespacing=1.35)
    if not r["consistent"]:
        ax.text(0.5,1.06,"arithmetic slip",transform=ax.transAxes,ha="center",
                fontsize=5.6,color=BAD)
fig.suptitle("Joint co-generation: image and equation produced together from noise, unconditioned",
             fontsize=8.4,color=INK,y=1.045)
fig.text(0.5,-0.105,"First 12 samples in generation order, not selected. A teal border means the equation is exactly correct for the image beside it; "
         "the one\nfailure (2 of the 32 sampled) names all four digits correctly and miscounts the sum, 9+7+8+3 given as twenty-eight — the dominant error mode.",
         ha="center",fontsize=6.2,color=MUT,linespacing=1.6)
fig.tight_layout(h_pad=2.2,w_pad=0.7)
fig.savefig(SP+"/fig_mnist_sum_joint.pdf",bbox_inches="tight")
fig.savefig(SP+"/fig_mnist_sum_joint.png",dpi=260,bbox_inches="tight")
print("wrote fig_mnist_sum_joint")
