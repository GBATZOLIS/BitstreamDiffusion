import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
SP = sys.argv[1]
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],
                     "font.size":8,"axes.linewidth":0.6,"pdf.fonttype":42})
INK="#14181a"; MUT="#5d6b6e"; ACC="#0d6b5f"; IMGC="#2a78d6"; TXTC="#eb6834"; MRK="#9aa5a3"
A=json.load(open(SP+"/assets.json")); D=np.load(SP+"/assets.npz")

fig=plt.figure(figsize=(7.0,3.55))
gs=fig.add_gridspec(2,3,height_ratios=[1.32,1.0],width_ratios=[1.05,1.0,1.0],
                    hspace=0.46,wspace=0.30,left=0.045,right=0.985,top=0.90,bottom=0.075)

# ---------------- (a) the datum ----------------
ax=fig.add_subplot(gs[0,0]); ax.set_title("(a)  one example",loc="left",fontsize=8.5,pad=6,color=INK)
img=D["real_img"][0]; ax.imshow(img,cmap="gray_r",interpolation="nearest")
ax.axhline(27.5,color=ACC,lw=0.8,alpha=.85); ax.axvline(27.5,color=ACC,lw=0.8,alpha=.85)
for (yy,xx,t) in ((6,6,"TL"),(6,34,"TR"),(34,6,"BL"),(34,34,"BR")):
    ax.text(xx,yy,t,fontsize=6.2,color=ACC,ha="center",va="center",
            bbox=dict(fc="white",ec="none",alpha=.75,pad=0.8))
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color(MUT)
ax.set_xlabel("56$\\times$56, binary pixels\nnative 28$\\times$28 digits, never resized",
              fontsize=6.6,color=MUT,labelpad=3)

# ---------------- (b) the shared bitstream ----------------
ax=fig.add_subplot(gs[0,1:]); ax.set_title("(b)  written into one bitstream — no tokenizer on either side",
                                           loc="left",fontsize=8.5,pad=6,color=INK)
ax.set_xlim(0,121); ax.set_ylim(-1.75,2.05); ax.axis("off")
segs=[(0,1,MRK,""),(1,1,MRK,""),(2,3,TXTC,""),(5,1,MRK,""),
      (6,1,MRK,""),(7,112,IMGC,"image   112 positions"),(119,1,MRK,""),(120,1,MRK,"")]
for x,w,c,lab in segs:
    ax.add_patch(Rectangle((x,0.12),w,0.62,fc=c,ec="white",lw=0.35,alpha=.92))
    if lab: ax.text(x+w/2,0.43,lab,ha="center",va="center",fontsize=6.6,color="white")
ax.text(60.5,-0.30,"121 positions $\\times$ 28 bits  =  3,388 bits",
        ha="center",fontsize=7.4,color=INK)
ax.text(3.5,0.98,"text  3 pos\n84 bits (2.5%)",ha="center",va="bottom",fontsize=6.2,color=TXTC)
ax.text(63,0.98,"3,136 bits (92.6%)",ha="center",va="bottom",fontsize=6.2,color=IMGC)
ax.text(121,0.98,"6 markers\n168 bits",ha="right",va="bottom",fontsize=6.2,color=MUT)
ax.annotate("",xy=(2,0.10),xytext=(2,-0.72),arrowprops=dict(arrowstyle="-",color=TXTC,lw=0.6))
ax.annotate("",xy=(5,0.10),xytext=(22,-0.72),arrowprops=dict(arrowstyle="-",color=TXTC,lw=0.6))
ax.add_patch(Rectangle((2,-1.52),20,0.80,fc="none",ec=TXTC,lw=0.7))
for i in range(4):
    ax.add_patch(Rectangle((2.6+i*4.8,-1.40),4.2,0.56,fc=TXTC,ec="none",alpha=.22))
    ax.text(2.6+i*4.8+2.1,-1.12,["six","nine","zero","one"][i],ha="center",va="center",fontsize=5.4,color=TXTC)
ax.text(12,-1.66,"one text position = 4 $\\times$ 7-bit word codes",ha="center",fontsize=6.2,color=TXTC)
ax.annotate("",xy=(64,0.10),xytext=(64,-0.72),arrowprops=dict(arrowstyle="-",color=IMGC,lw=0.6))
ax.annotate("",xy=(80,0.10),xytext=(100,-0.72),arrowprops=dict(arrowstyle="-",color=IMGC,lw=0.6))
ax.add_patch(Rectangle((64,-1.52),36,0.80,fc="none",ec=IMGC,lw=0.7))
row=D["real_img"][0][15,:28]
for i,v in enumerate(row):
    ax.add_patch(Rectangle((64.9+i*1.22,-1.40),1.12,0.56,fc=IMGC if v>0.5 else "white",
                           ec=IMGC,lw=0.25,alpha=.95 if v>0.5 else .45))
ax.text(82,-1.66,"one image position = 28 raw pixels = one digit row",ha="center",fontsize=6.2,color=IMGC)

# ---------------- (c,d,e) the three directions ----------------
def panel(ax,title):
    ax.set_title(title,loc="left",fontsize=8.5,pad=5,color=INK); ax.axis("off")
def strip(ax,imgs,x0,y0,w,n=4):
    for k in range(n):
        a=ax.inset_axes([x0+k*(w+0.012),y0,w,w*1.0])
        a.imshow(imgs[k],cmap="gray_r",interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_color(MUT); s.set_linewidth(0.4)

ax=fig.add_subplot(gs[1,0]); panel(ax,"(c)  image $\\rightarrow$ text")
strip(ax,D["real_img"],0.02,0.50,0.20)
ax.text(0.02,0.34,"given the image, the model writes:",fontsize=6.4,color=MUT,transform=ax.transAxes)
ax.text(0.02,0.17,f'“{A["i2t_pred"][0]}”',fontsize=6.4,color=INK,transform=ax.transAxes)
ax.text(0.02,0.00,"97.90% exactly correct",fontsize=6.6,color=ACC,transform=ax.transAxes)

ax=fig.add_subplot(gs[1,1]); panel(ax,"(d)  text $\\rightarrow$ image")
strip(ax,D["t2i_img"],0.02,0.50,0.20)
ax.text(0.02,0.34,"drawn from the equation alone:",fontsize=6.4,color=MUT,transform=ax.transAxes)
ax.text(0.02,0.17,f'“{A["t2i_prompt"][0]}”',fontsize=6.4,color=INK,transform=ax.transAxes)
ax.text(0.02,0.00,"99.98% all four digits right",fontsize=6.6,color=ACC,transform=ax.transAxes)

ax=fig.add_subplot(gs[1,2]); panel(ax,"(e)  joint — both from noise")
strip(ax,D["joint_img"],0.02,0.50,0.20)
ax.text(0.02,0.34,"its own equation, same trajectory:",fontsize=6.4,color=MUT,transform=ax.transAxes)
ax.text(0.02,0.17,f'“{A["joint"][0]["text"]}”',fontsize=6.4,color=INK,transform=ax.transAxes)
ax.text(0.02,0.00,"96.85% mutually consistent",fontsize=6.6,color=ACC,transform=ax.transAxes)

fig.savefig(SP+"/fig_mnist_sum_overview.pdf",bbox_inches="tight")
fig.savefig(SP+"/fig_mnist_sum_overview.png",dpi=260,bbox_inches="tight")
print("wrote fig_mnist_sum_overview")
