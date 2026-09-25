import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 10, 'axes.linewidth': 0.8, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

COLORS = {'stable': '#0173B2', 'neutral': '#029E73', 'unstable': '#D55E00',
          'data': '#CC79A7', 'ref': '#F0E442'}
KAPPA = 0.4
ZETA_MIN, ZETA_MAX = -1e4, 10.0
AS, BS, CS, DS = 1.0, 2.0 / 3.0, 5.0, 0.35


def psi_h(zeta):
    zeta = np.clip(zeta, ZETA_MIN, ZETA_MAX)
    if zeta <= 0:
        phi_inv = np.sqrt(1.0 + 7.9 * abs(zeta) ** (2.0 / 3.0))
        return 3.0 * np.log((1.0 + phi_inv) / 2.0)
    return (-2.0 / 3.0 * (zeta - CS / DS) * np.exp(-DS * zeta)
            - (1.0 + 2.0 / 3.0 * AS * zeta) ** 1.5 - BS * CS / DS + 1.0)


def mo_factor(za, zb, L):
    return (np.log(zb / za) - psi_h(zb / L) + psi_h(za / L)) / KAPPA


def profile(z1, z2, c1, c2, L, z):
    cstar = (c2 - c1) / mo_factor(z1, z2, L)
    return c1 + cstar * np.array([mo_factor(z1, zi, L) for zi in z])


def main():
    z1, z2, zref = 8.0, 24.0, 20.0
    z = np.linspace(1.0, 50.0, 200)
    rows = [(8.0, 10.0, 'Deposition scenario'), (10.0, 8.0, 'Emission scenario')]
    cases = [(50.0, '(a) Stable', 'stable', r'$L = 50$ m'),
             (1e30, '(b) Neutral', 'neutral', r'$L \rightarrow \infty$'),
             (-20.0, '(c) Unstable', 'unstable', r'$L = -20$ m')]

    xmax = 1.05 * max(profile(z1, z2, c1, c2, L, z).max()
                      for c1, c2, _ in rows for L, *_ in cases)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for r, (c1, c2, rlabel) in enumerate(rows):
        c_neutral = profile(z1, z2, c1, c2, 1e30, z)
        for k, (L, title, key, ltext) in enumerate(cases):
            ax = axes[r, k]
            ax.plot(profile(z1, z2, c1, c2, L, z), z, color=COLORS[key], lw=2.5)
            if key != 'neutral':
                ax.plot(c_neutral, z, '--', color=COLORS['neutral'], lw=1.5, alpha=0.7)
            ax.scatter([c1, c2], [z1, z2], s=70, color=COLORS['data'],
                       edgecolor='white', linewidth=0.8, zorder=5)
            ax.scatter(profile(z1, z2, c1, c2, L, [zref]), [zref], s=90, marker='D',
                       color=COLORS['ref'], edgecolor='black', linewidth=0.8, zorder=6)
            ax.set_xlim(0, xmax)
            ax.set_ylim(0, 50)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.text(0.97 if r else 0.04, 0.95, ltext, transform=ax.transAxes,
                    ha='right' if r else 'left', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            if r == 0:
                ax.set_title(title, fontweight='bold')
            else:
                ax.set_xlabel('Mole fraction (nmol mol$^{-1}$)')
            if k == 0:
                ax.set_ylabel('Height (m)')
        axes[r, 2].annotate(rlabel, xy=(1.03, 0.5), xycoords='axes fraction',
                            fontsize=10, fontweight='bold', rotation=270,
                            va='center', ha='left')

    handles = [
        Line2D([], [], color='k', lw=2.5, label='MOST profile'),
        Line2D([], [], color=COLORS['neutral'], ls='--', lw=1.5, alpha=0.7,
               label='Neutral reference'),
        Line2D([], [], marker='o', ls='', color=COLORS['data'], mec='white',
               ms=9, label='Grid levels'),
        Line2D([], [], marker='D', ls='', mfc=COLORS['ref'], mec='black',
               ms=9, label='Reference height'),
    ]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 0.01),
               ncol=4, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 0.97, 0.97])
    plt.subplots_adjust(hspace=0.12)
    fig.savefig('fD1.png')
    fig.savefig('fD1.pdf')


if __name__ == '__main__':
    main()
