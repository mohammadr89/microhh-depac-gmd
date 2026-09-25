import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from cmcrameri import cm as cmc

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.0,
    'pdf.fonttype': 42,
})

BASE = '../cases/grassland/bg_bidir_dz04'
LABEL = dict(fontsize=14, fontweight='bold')
TICK_SIZE = 12

R, M_NH3, M_AIR, P = 8.314, 17.031, 28.97, 101325

COLOR_CCOMP = cmc.batlow(0.15)
COLOR_CONC  = cmc.batlow(0.75)
COLOR_SHADE = cmc.batlow(0.55)
COLOR_A     = cmc.batlow(0.20)
COLOR_B     = cmc.batlow(0.75)

PERIODS  = [(10.0, 12.0), (13.0, 15.0)]
Z_MAX    = 100.0
XLIM_B   = (4.88, 5.05)
SUN_TIME = [6, 18]


def load_timeseries():
    with nc.Dataset(f'{BASE}/ccomp_tot.xy.nc') as ds:
        t = ds.variables['time'][:] / 3600.0
        ccomp = ds.variables['ccomp_tot'][:].mean(axis=(1, 2))
    with nc.Dataset(f'{BASE}/nh3_first_level_xy.nc') as ds:
        nh3 = ds.variables['nh3'][:, 0, :, :]
    with nc.Dataset(f'{BASE}/T_surface.xy.nc') as ds:
        T = ds.variables['T_surface'][:]
    conc = (nh3 * P * M_NH3 * 1e6 / (R * T)).mean(axis=(1, 2))
    return t, ccomp, conc


def load_profiles():
    with nc.Dataset(f'{BASE}/plume_chem.default.0010800.nc') as ds:
        z = ds.variables['z'][:]
        t = ds.variables['time'][:] / 3600.0
        rho = ds.groups['thermo'].variables['rhoref'][:]
        c = ds.groups['default'].variables['nh3'][:]
        c2 = ds.groups['default'].variables['nh3_2'][:]
    fac = (M_NH3 / M_AIR) * rho[np.newaxis, :] * 1e9
    mean, std = c * fac, np.sqrt(np.maximum(c2, 0.0)) * fac
    zm = z <= Z_MAX
    profiles = []
    for ts, te in PERIODS:
        i = (t >= ts) & (t <= te)
        profiles.append((mean[i].mean(axis=0)[zm], std[i].mean(axis=0)[zm]))
    return z[zm], profiles


def print_exceedance_periods(t, ccomp, conc):
    d = ccomp - conc
    pos = d > 0
    cross = lambda i: t[i] - d[i] * (t[i+1] - t[i]) / (d[i+1] - d[i])
    starts = [t[0]] if pos[0] else []
    ends = []
    for i in np.flatnonzero(pos[:-1] != pos[1:]):
        (starts if pos[i+1] else ends).append(cross(i))
    if pos[-1]:
        ends.append(t[-1])
    fmt = lambda h: f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"
    for s, e in zip(starts, ends):
        print(f"chi_c > chi_a: {fmt(s)}–{fmt(e)} ({e - s:.2f} h)")


def style_axis(ax):
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def panel_label(ax, text):
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha='left', va='bottom', **LABEL)


def plot_timeseries(ax, t, ccomp, conc):
    ax.fill_between(t, ccomp, conc, where=ccomp > conc, color=COLOR_SHADE,
                    alpha=1.0, interpolate=True)
    ax.plot(t, ccomp, color=COLOR_CCOMP, linewidth=2.0, alpha=0.85)
    ax.plot(t, conc, color=COLOR_CONC, linewidth=2.0, alpha=0.85)
    for ts in SUN_TIME:
        ax.plot(ts, 0, 'o', color='orange', markersize=8, zorder=11, clip_on=False)

    ax.set_xlabel('Time (h)', **LABEL)
    ax.set_ylabel('Concentration (µg m⁻³)', **LABEL)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    vals = np.concatenate([ccomp, conc])
    ax.set_ylim(0 if vals.min() >= 0 else vals.min() * 1.2,
                0 if vals.max() <= 0 else vals.max() * 1.2)
    ax.grid(True, linestyle='--', alpha=0.5)
    style_axis(ax)

    handles = [
        mlines.Line2D([], [], color=COLOR_CCOMP, linewidth=2.0, alpha=0.85,
                      label='Compensation point'),
        mlines.Line2D([], [], color=COLOR_CONC, linewidth=2.0, alpha=0.85,
                      label='NH₃ at lowest model level'),
        Rectangle((0, 0), 1, 1, fc=COLOR_SHADE, label=r'$\chi_c > \chi_a$'),
        mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                      markersize=6, label='Sunrise & sunset'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.95, fontsize=11,
              frameon=True, fancybox=False)


def plot_profile(ax, z, mean, std, color, period):
    lbl = f'{period[0]:.1f}–{period[1]:.1f} h'
    ax.fill_betweenx(z, mean - std, mean + std, color=color, alpha=0.25,
                     label=r'$\pm 1\sigma_\mathrm{h}$' + f' ({lbl})')
    ax.plot(mean, z, '-', color=color, linewidth=2.0, label=f'Mean ({lbl})')
    ax.set_xlabel('NH₃ (µg m⁻³)', **LABEL)
    ax.set_ylim(0, Z_MAX)
    ax.set_xlim(XLIM_B)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax.grid(True, linestyle='--', alpha=0.7)
    style_axis(ax)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], fontsize=TICK_SIZE - 1, framealpha=0.9)


def main():
    t, ccomp, conc = load_timeseries()
    print_exceedance_periods(t, ccomp, conc)
    z, profiles = load_profiles()

    fig = plt.figure(figsize=(22, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.12)
    gs_b = gs[1].subgridspec(1, 2, wspace=0.06)
    ax_a = fig.add_subplot(gs[0])
    ax_b1 = fig.add_subplot(gs_b[0])
    ax_b2 = fig.add_subplot(gs_b[1], sharey=ax_b1)

    plot_timeseries(ax_a, t, ccomp, conc)
    for ax, (mean, std), color, period in zip([ax_b1, ax_b2], profiles,
                                              [COLOR_A, COLOR_B], PERIODS):
        plot_profile(ax, z, mean, std, color, period)
    ax_b1.set_ylabel('z (m)', **LABEL)
    plt.setp(ax_b2.get_yticklabels(), visible=False)

    panel_label(ax_a, '(a)')
    panel_label(ax_b1, '(b)')

    for ext in ('pdf', 'png'):
        fig.savefig(f'f04.{ext}', facecolor='white', bbox_inches='tight')
    print('Saved: f04.pdf, f04.png')


if __name__ == '__main__':
    main()
