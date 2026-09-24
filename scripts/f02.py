import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7, 4))
ax.set_xlim(-1.4, 10.5); ax.set_ylim(-0.2, 10); ax.axis("off")

ax.annotate("", xy=(0, 9.9), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="k"))
ax.plot([0, 9.5], [0, 0], "k", lw=1.5)
ax.text(-0.15, 0, "0", ha="right", va="center", fontsize=12)
ax.text(-1.1, 5, "Height, $z$", rotation=90, ha="center", va="center", fontsize=12)

for y, ls, lab in [(2.0, "--", r"$h_c$"), (4.8, "-", r"$z^*$"), (7.6, "-", "")]:
    ax.plot([0, 8.2], [y, y], color="k", ls=ls, lw=0.9, zorder=1)
    ax.text(-0.15, y, lab, ha="right", va="center", fontsize=12)

for x in np.arange(1.3, 7.8, 0.9):
    ax.plot([x, x], [0, 0.35], color="saddlebrown", lw=2.5, solid_capstyle="butt", zorder=2)
    ax.fill([x - 0.25, x, x + 0.25], [0.35, 2.0, 0.35], color="forestgreen", lw=0, zorder=2)

ax.text(2.4, 3.3, "Roughness sublayer (RSL)", fontsize=12)
ax.text(2.4, 6.1, "Inertial sublayer (ISL)", fontsize=12)
ax.text(2.4, 8.7, "Outer layer", fontsize=12)
ax.annotate("", xy=(8.6, 2.0), xytext=(8.6, 7.6), arrowprops=dict(arrowstyle="<->", lw=0.9))
ax.text(8.75, 4.8, "Surface layer\n(constant flux)", va="center", fontsize=11)

fig.savefig("f02.pdf", bbox_inches="tight")
fig.savefig("f02.png", dpi=300, bbox_inches="tight")
