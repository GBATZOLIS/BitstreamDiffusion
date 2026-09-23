"""Compact MNIST-Sum figure for the CoBit paper (P=28 arm only).

Two panels. (a) One real image--equation pair, the two conditional directions
drawn as arrows between its halves, and the pair's actual 121 x 28-bit flat
bitstream underneath. (b) One independently co-generated pair for the joint
direction. Every image and equation is a fixed sample from the transferred
P=28 asset bundle; every number is read from the committed evaluation JSON.

Usage: python fig_paper_compact.py [out_dir]
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[2]
ASSET_DIR = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ASSET_DIR
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1c2022"
MUTED = "#6b7477"
TEAL = "#0f6e62"
BLUE = "#2f6fc4"
ORANGE = "#d9641f"
GREY = "#9aa3a6"
PALE_TEAL = "#e6f2ef"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "mathtext.fontset": "stix",
    }
)

assets = json.load(open(ASSET_DIR / "assets.json"))
arrays = np.load(ASSET_DIR / "assets.npz")
r28 = json.load(open(ROOT / "results/mnist_sum_p28_500k_unified/results.json"))

I2T = 100 * r28["i2t"]["val_gs2"]["end_to_end"]["value"]
T2I = 100 * r28["t2i"]["val_gs2"]["all_four"]["value"]
JOINT = 100 * r28["joint"]["uncond"]["consistent"]["value"]
N = int(r28["n"])

# ---- the pair shown in (a): a real validation example -----------------------
PAIR = 0
pair_img = arrays["real_img"][PAIR]
pair_eq = assets["i2t_true"][PAIR]
assert assets["i2t_scored"][PAIR]["ok"] and assets["t2i_scored"][PAIR]["ok"]

# ---- the co-generated pair shown in (b) --------------------------------------
JOINT_IDX = 3
joint_img = arrays["joint_img"][JOINT_IDX]
joint_eq = assets["joint"][JOINT_IDX]["text"]
assert assets["joint_scored"][JOINT_IDX]["consistent"]

# ---- rebuild the exact P=28 bitstream of the pair (numpy port of the codec) --
P, TEXT_BITS, LT = 28, 7, 12
NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "twenty-one", "twenty-two",
    "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven",
    "twenty-eight", "twenty-nine", "thirty", "thirty-one", "thirty-two",
    "thirty-three", "thirty-four", "thirty-five", "thirty-six",
]
WORD_TO_CODE = {w: i for i, w in enumerate(NUMBER_WORDS)}
WORD_TO_CODE["+"], WORD_TO_CODE["="], PAD = 37, 38, 39


def bitstream(img56, equation):
    """[28, 121] {0,1}: bit (27 - row) of position i, i.e. MSB on top."""
    toks = [WORD_TO_CODE[w] for w in equation.split()]
    toks += [PAD] * (LT - len(toks))
    tpp = P // TEXT_BITS
    text_codes = (np.asarray(toks).reshape(3, tpp) << (TEXT_BITS * np.arange(tpp))).sum(1)
    quads = np.stack(
        [img56[:28, :28], img56[:28, 28:], img56[28:, :28], img56[28:, 28:]]
    ).astype(np.int64)
    img_codes = (quads.reshape(-1, P) << np.arange(P)).sum(1)  # 112 rows, one per position
    markers = [(1 << P) - 1 - i for i in range(6)]  # SOS SOT EOT SOI EOI EOS
    codes = np.concatenate(
        [markers[:2], text_codes, markers[2:4], img_codes, markers[4:]]
    ).astype(np.int64)
    assert codes.shape[0] == 121
    b = ((codes[None, :] >> np.arange(P)[:, None]) & 1).astype(float)
    return b[::-1]  # bit 27 on top, bit 0 at the bottom



# ---- canvas in inches --------------------------------------------------------
W, H = 5.5, 2.42
fig = plt.figure(figsize=(W, H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


def thumb(img, x0, y0, size, edge, lw=0.6):
    ax.imshow(img, cmap="gray_r", interpolation="nearest", extent=[x0, x0 + size, y0, y0 + size], zorder=3)
    ax.add_patch(Rectangle((x0, y0), size, size, fill=False, edgecolor=edge, lw=lw, zorder=4))


def arrow(x0, x1, y, color):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=color, zorder=5,
            shrinkA=0, shrinkB=0,
        )
    )


def eq_lines(eq):
    lhs, rhs = eq.split(" = ")
    return f"{lhs}\n$=$ {rhs}"


# ================= (a) one pair, two conditional directions, its bitstream ====
ax.text(0.02, H - 0.13, "(a)  One pair, its two conditional directions and its flat bitstream", fontsize=7.6, color=INK, va="center")

IMG_X, IMG_Y, IMG_S = 0.06, 1.30, 0.86
thumb(pair_img, IMG_X, IMG_Y, IMG_S, BLUE)
ax.text(IMG_X + IMG_S / 2, IMG_Y - 0.075, "raw $56{\\times}56$ binary pixels", ha="center", va="center", fontsize=6.4, color=BLUE)

BOX_X0, BOX_X1 = 1.94, 3.30
BOX_Y0, BOX_Y1 = 1.51, 1.91
ax.add_patch(
    FancyBboxPatch(
        (BOX_X0, BOX_Y0), BOX_X1 - BOX_X0, BOX_Y1 - BOX_Y0,
        boxstyle="round,pad=0,rounding_size=0.04", facecolor="white", edgecolor=ORANGE, lw=0.6, zorder=2,
    )
)
ax.text((BOX_X0 + BOX_X1) / 2, (BOX_Y0 + BOX_Y1) / 2, eq_lines(pair_eq), ha="center", va="center", fontsize=7.4, color=INK, linespacing=1.35)
ax.text((BOX_X0 + BOX_X1) / 2, BOX_Y0 - 0.085, "deterministic 7-bit code per symbol", ha="center", va="center", fontsize=6.4, color=ORANGE)

AX0, AX1 = IMG_X + IMG_S + 0.06, BOX_X0 - 0.06
AXC = (AX0 + AX1) / 2
Y_TOP, Y_BOT = 1.81, 1.61
arrow(AX0, AX1, Y_TOP, INK)
arrow(AX1, AX0, Y_BOT, INK)
ax.text(AXC, Y_TOP + 0.22, "image $\\rightarrow$ text", ha="center", va="center", fontsize=6.6, color=INK)
ax.text(AXC, Y_TOP + 0.10, f"{I2T:.2f}% exact", ha="center", va="center", fontsize=7.0, color=TEAL, fontweight="bold")
ax.text(AXC, Y_BOT - 0.10, f"{T2I:.2f}% all four digits", ha="center", va="center", fontsize=7.0, color=TEAL, fontweight="bold")
ax.text(AXC, Y_BOT - 0.22, "text $\\rightarrow$ image", ha="center", va="center", fontsize=6.6, color=INK)

# the actual flat bitstream of this pair, to scale in BITS
toks = [WORD_TO_CODE[w] for w in pair_eq.split()]
toks += [PAD] * (LT - len(toks))
TEXT_BITS_N, IMG_BITS_N, MK_BITS_N = LT * TEXT_BITS, 4 * 28 * 28, 6 * P     # 84, 3136, 168
TOTAL_BITS = TEXT_BITS_N + IMG_BITS_N + MK_BITS_N                            # 3388
# bit ranges in stream order: [SOS SOT] text [EOT SOI] image [EOI EOS]
b_mk1 = (0, 2 * P)
b_txt = (b_mk1[1], b_mk1[1] + TEXT_BITS_N)
b_mk2 = (b_txt[1], b_txt[1] + 2 * P)
b_img = (b_mk2[1], b_mk2[1] + IMG_BITS_N)
b_mk3 = (b_img[1], b_img[1] + 2 * P)
assert b_mk3[1] == TOTAL_BITS

SX0, SX1 = 0.06, 3.30
bx = lambda b: SX0 + (SX1 - SX0) * b / TOTAL_BITS
BAR_Y0, BAR_H = 0.86, 0.10
for (a, b), col in [(b_mk1, GREY), (b_txt, ORANGE), (b_mk2, GREY), (b_img, BLUE), (b_mk3, GREY)]:
    ax.add_patch(Rectangle((bx(a), BAR_Y0), bx(b) - bx(a), BAR_H, facecolor=col, edgecolor="white", lw=0.3, zorder=3))
LAB_Y = BAR_Y0 + BAR_H + 0.075
ax.text(SX0, LAB_Y, f"text: {TEXT_BITS_N} bits", ha="left", va="center", fontsize=6.4, color=ORANGE)
ax.text((bx(b_img[0]) + bx(b_img[1])) / 2 + 0.1, LAB_Y, f"image: {IMG_BITS_N:,} raw pixel bits", ha="center", va="center", fontsize=6.4, color=BLUE)
ax.text(SX1, LAB_Y, f"markers: {MK_BITS_N} bits", ha="right", va="center", fontsize=6.4, color=MUTED)

# zoom-ins on actual bits
SQ = 0.048
ZY = 0.46                     # bottom of the bit squares


def bit_squares(x0, bits, edge):
    for i, v in enumerate(bits):
        ax.add_patch(Rectangle((x0 + i * SQ, ZY), SQ, SQ, facecolor=INK if v else "white", edgecolor=edge, lw=0.4, zorder=4))
    return x0 + len(bits) * SQ


def code_bits(code):          # 7 bits in stream order (LSB first)
    return [(code >> k) & 1 for k in range(TEXT_BITS)]


# (i) the first three text symbols
words = pair_eq.split()[:3]
x = SX0
GAP = 0.05
group_x = []
for w in words:
    x1 = bit_squares(x, code_bits(WORD_TO_CODE[w]), ORANGE)
    group_x.append((x, x1))
    x = x1 + GAP
ZL0, ZL1 = SX0, group_x[-1][1]
ax.text(ZL1 + 0.04, ZY + SQ / 2, "$\\cdots$", ha="left", va="center", fontsize=7, color=MUTED)
for (g0, g1), w in zip(group_x, words):
    ax.text((g0 + g1) / 2, ZY - 0.075, w, ha="center", va="center", fontsize=6.4, color=INK)
ax.text((ZL0 + ZL1) / 2, ZY - 0.19, "7 bits per symbol", ha="center", va="center", fontsize=6.0, color=ORANGE)
# text-segment bits that these three symbols occupy
tz0, tz1 = bx(b_txt[0]), bx(b_txt[0] + 3 * TEXT_BITS)

# (ii) one pixel row of the top-left digit
ROW = int(np.argmax([np.sum(np.abs(np.diff(pair_img[r, :28]))) for r in range(28)]))  # busiest row
row_bits = pair_img[ROW, :28].astype(int).tolist()
ZR1 = SX1
ZR0 = ZR1 - 28 * SQ
bit_squares(ZR0, row_bits, BLUE)
ax.text((ZR0 + ZR1) / 2, ZY - 0.075, f"pixel row {ROW + 1} of the top-left digit", ha="center", va="center", fontsize=6.4, color=INK)
ax.text((ZR0 + ZR1) / 2, ZY - 0.19, "28 bits per pixel row, 28 rows per digit", ha="center", va="center", fontsize=6.0, color=BLUE)
iz0 = bx(b_img[0] + ROW * 28)
iz1 = bx(b_img[0] + (ROW + 1) * 28)
# highlight that row on the image above
ry = IMG_Y + IMG_S - (ROW + 1) * IMG_S / 56
ax.add_patch(Rectangle((IMG_X, ry), IMG_S / 2, IMG_S / 56, facecolor=BLUE, alpha=0.45, edgecolor="none", zorder=5))

# funnel lines from the bar down to each zoom
CL = "#b5bcbf"
for (u0, u1), (l0, l1) in [((tz0, tz1), (ZL0, ZL1)), ((iz0, iz1), (ZR0, ZR1))]:
    ax.plot([u0, l0], [BAR_Y0, ZY + SQ], color=CL, lw=0.5, zorder=2)
    ax.plot([u1, l1], [BAR_Y0, ZY + SQ], color=CL, lw=0.5, zorder=2)
    ax.add_patch(Rectangle((u0, BAR_Y0 - 0.012), u1 - u0, BAR_H + 0.024, fill=False, edgecolor=INK, lw=0.5, zorder=6))

ax.text(
    (SX0 + SX1) / 2, 0.12,
    f"One flat sequence of {TOTAL_BITS:,} bits, read with 1-D position only",
    ha="center", va="center", fontsize=5.9, color=MUTED,
)

# ================= (b) joint generation from noise ============================
BX = 3.55
BC = (BX + W) / 2
ax.text(BX, H - 0.13, "(b)  Joint generation from noise", fontsize=7.6, color=INK, va="center")

rng = np.random.default_rng(0)
noise = rng.integers(0, 2, size=(56, 56)).astype(float)
NS = 0.42
NY = H - 0.28 - NS
thumb(noise, BC - NS / 2, NY, NS, GREY, lw=0.5)
ax.text(BC + NS / 2 + 0.07, NY + NS / 2, "noise on every\ntext and image bit", ha="left", va="center", fontsize=6.2, color=MUTED, linespacing=1.2)

ARR_Y0, ARR_Y1 = NY - 0.03, NY - 0.20
ax.add_patch(FancyArrowPatch((BC, ARR_Y0), (BC, ARR_Y1), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=INK, zorder=5, shrinkA=0, shrinkB=0))

PB_X0, PB_X1 = BC - 0.78, BC + 0.78
PB_Y1 = ARR_Y1 - 0.02
JS = 0.66
PB_Y0 = PB_Y1 - JS - 0.32
ax.add_patch(
    FancyBboxPatch(
        (PB_X0, PB_Y0), PB_X1 - PB_X0, PB_Y1 - PB_Y0,
        boxstyle="round,pad=0,rounding_size=0.05", facecolor=PALE_TEAL, edgecolor=TEAL, lw=0.6, zorder=1,
    )
)
thumb(joint_img, BC - JS / 2, PB_Y1 - 0.06 - JS, JS, TEAL)
ax.text(BC, PB_Y0 + 0.13, joint_eq, ha="center", va="center", fontsize=7.0, color=INK)

ax.text(BC, PB_Y0 - 0.12, f"{JOINT:.2f}% mutually consistent", ha="center", va="center", fontsize=7.0, color=TEAL, fontweight="bold")
ax.text(BC, PB_Y0 - 0.24, "generated equation correct for generated image", ha="center", va="center", fontsize=6.0, color=MUTED)
ax.text(W - 0.06, 0.06, f"$n$ = {N:,} per direction", ha="right", va="center", fontsize=6.0, color=MUTED)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_mnist_sum_compact.{ext}", dpi=400, facecolor="white")
print(f"wrote {OUT / 'fig_mnist_sum_compact.pdf'}  ({W}x{H} in)  i2t={I2T:.2f} t2i={T2I:.2f} joint={JOINT:.2f} n={N}")
