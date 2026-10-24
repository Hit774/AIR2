"""Draw Figure 1 (AIR pipeline) for the paper. Usage: python architecture.py <out_dir> <preview_png>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
})

INK = "#222222"
PANEL = "#f4f6f9"
EDGE = "#9aa5b4"
RED = "#d62728"
BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
GREY = "#c9ced6"

fig, ax = plt.subplots(figsize=(7.0, 2.55))
ax.set_xlim(0, 140)
ax.set_ylim(0, 51)
ax.axis("off")


def box(x0, y0, x1, y1, fc=PANEL, ec=EDGE, lw=0.8, r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(x0, y0, x1, y1, lw=1.1, color=INK, z=5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=8, lw=lw, color=color, zorder=z))


def text(x, y, s, size=7, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=kw.pop("color", INK), zorder=6, **kw)


# Panels: (x0, x1, title)
TOP, BOT = 48.5, 9.5
panels = [
    (1, 23, "1  Atomize"),
    (27, 60, "2  Micro-LoRA self-distillation"),
    (64, 85, "3  Belief states"),
    (89, 113, "4  Relevance-aware gate"),
    (117, 139, "5  Cross-JSD score"),
]
for x0, x1, title in panels:
    box(x0, BOT, x1, TOP)
    text((x0 + x1) / 2, TOP - 2.6, title, size=7.6, weight="bold")

MID = 27.5
for (_, a1, _), (b0, _, _) in zip(panels[:-1], panels[1:]):
    arrow(a1 + 0.4, MID, b0 - 0.4, MID, lw=1.3)

# 1 Atomize: document of chunks; two evidence passages highlighted
dx0, dx1 = 5.5, 18.5
labels = ["$c_1$", "$c_2$", "$c_a$", "$c_4$", "$\\cdots$", "$c_b$", "$c_n$"]
colors = [GREY, GREY, RED, GREY, None, BLUE, GREY]
y = 41.2
for lab, col in zip(labels, colors):
    if col is None:
        text((dx0 + dx1) / 2, y - 1.3, "$\\vdots$", size=8)
    else:
        ax.add_patch(Rectangle((dx0, y - 2.8), dx1 - dx0, 2.8, fc=col, ec="white", lw=0.6, zorder=3,
                               alpha=0.9 if col != GREY else 1.0))
        text((dx0 + dx1) / 2, y - 1.4, lab, size=7, color="white" if col != GREY else INK)
    y -= 3.6
text(12, 13.6, "long document $D$", size=6.6, style="italic")
text(12, 11.3, "256-tok windows, 64 overlap", size=5.9)

# 2 Self-distillation: teacher -> KL -> student, repeated per chunk
box(29, 33.5, 58, 42.3, fc="white", ec=EDGE)
text(43.5, 39.9, "Teacher: frozen LLM $\\Theta$", size=7)
text(43.5, 36.0, "prompt $=$ chunk $c_i$ $+$ query $Q$", size=6.6)
box(29, 15.5, 58, 24.3, fc="#fff4f0", ec=RED, lw=0.9)
text(43.5, 21.9, "Student: LLM $\\Theta+\\Delta W_i$  ($r{=}2$)", size=7)
text(43.5, 18.0, "prompt $=$ query $Q$ only", size=6.6)
arrow(43.5, 33.3, 43.5, 24.6, lw=1.0)
text(51.5, 30.2, "minimize KL", size=6.6)
text(51.5, 27.6, "teacher $\\Vert$ student", size=6.6)
text(43.5, 12.2, "repeated per chunk; adapter then discarded", size=6.2, style="italic")

# 3 Belief states: Yes/No bars per chunk
groups = [("$P_1$", 0.62, 0.38), ("$P_a$", 0.97, 0.03), ("$P_b$", 0.05, 0.95), ("$P_n$", 0.80, 0.20)]
gx = 66.6
base, hmax = 17.0, 21.0
for lab, yes, no in groups:
    ax.add_patch(Rectangle((gx, base), 1.7, yes * hmax, fc=GREEN, ec="none", zorder=3))
    ax.add_patch(Rectangle((gx + 1.9, base), 1.7, no * hmax, fc=ORANGE, ec="none", zorder=3))
    text(gx + 1.8, base - 1.9, lab, size=7)
    gx += 4.7
ax.plot([65.8, 84.2], [base, base], color=INK, lw=0.6, zorder=4)
ax.add_patch(Rectangle((66.0, 39.9), 1.4, 1.4, fc=GREEN, ec="none", zorder=3))
text(67.9, 40.6, "Yes", size=6, ha="left")
ax.add_patch(Rectangle((74.0, 39.9), 1.4, 1.4, fc=ORANGE, ec="none", zorder=3))
text(75.9, 40.6, "No", size=6, ha="left")
text(74.5, 12.0, "$C_i=\\mathrm{max}_y\\,P_i(y)$", size=7)

# 4 Relevance-aware gate
text(101, 39.0, "$G_i = C_i \\cdot R_i$", size=9)
text(101, 33.4, "$R_i$: lexical $+$ semantic", size=6.6)
text(101, 30.9, "match of $c_i$ to question $q$", size=6.6)
text(101, 25.6, "keep top-$K$ chunks of", size=6.6)
text(101, 23.1, "$V=\\{\\,i : R_i \\geq \\rho\\,\\}$", size=7)
box(91.5, 13.4, 110.5, 18.6, fc="#eceef2", ec=EDGE)
text(101, 16.0, "$|V|<2\\ \\Rightarrow$ abstain", size=6.6)

# 5 Cross-JSD conflict score and outcomes
text(128, 39.0, "$\\mathcal{C}_{\\mathrm{score}}=\\max_{a\\neq b}$", size=8)
text(128, 35.2, "$\\mathrm{JSD}(P_a\\,\\Vert\\,P_b)$", size=8)
for (lab, fc, ec), yy in zip([("Conflict", "#fbe3e3", RED), ("Consistent", "#e3f3e3", GREEN),
                             ("Insufficient evidence", "#eceef2", EDGE)], [27.0, 21.2, 15.4]):
    box(119.2, yy - 2.2, 136.8, yy + 2.2, fc=fc, ec=ec, lw=0.8)
    text(128, yy, lab, size=6.8)

# Query feeds the distillation prompts and the relevance gate
box(27, 1.2, 113, 6.4, fc="white", ec=INK, lw=0.8)
text(70, 3.8, "query $Q$ (claim to verify) with raw question $q$", size=7)
arrow(43.5, 6.6, 43.5, 9.3, lw=0.9)
arrow(101, 6.6, 101, 9.3, lw=0.9)

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(sys.argv[1] + "/architecture.pdf")
fig.savefig(sys.argv[2], dpi=220)
print("saved")
