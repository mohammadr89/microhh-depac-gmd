import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         13,
    'axes.titlesize':    15,
    'axes.labelsize':    13,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'axes.linewidth':    1.0,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'grid.linewidth':    0.4,
    'grid.color':        '#cccccc',
    'lines.linewidth':   2.2,
    'figure.dpi':        300,
    'savefig.dpi':       300,
})

RES_COLORS = {
    '4m':  '#009E73',
    '8m':  '#DE8F05',
    '12m':  '#0173B2',
    '16m': '#CC3311',
}
RESOLUTIONS = ('4m', '8m', '12m', '16m')


def load_ctarget(fp):
    with nc.Dataset(fp, 'r') as ds:
        return ds['time'][:].copy(), ds['c_target'][:].copy()

def load_nh3(fp):
    with nc.Dataset(fp, 'r') as ds:
        t   = ds['time'][:].copy()
        nh3 = np.squeeze(ds['nh3'][:].copy(), axis=1)
        return t, nh3

def xy_avg_ppb(arr):
    return np.mean(arr, axis=(1, 2)) * 1e9

def to_hours(s):
    return np.asarray(s) / 3600.0

def common_times(*arrays):
    c = set(arrays[0])
    for a in arrays[1:]:
        c &= set(a)
    return np.array(sorted(c))

def subset(tarr, darr, mask):
    return darr[np.searchsorted(tarr, mask)]


def load_all(t_start=10800, t_end=86400):
    base = Path('../cases/forest')
    paths_ct  = {r: base / f'bg_bidir_dz{r[:-1].zfill(2)}/c_target.xy.nc'        for r in RESOLUTIONS}
    paths_nh3 = {r: base / f'bg_bidir_dz{r[:-1].zfill(2)}/nh3_first_level_xy.nc' for r in RESOLUTIONS}

    raw_ct, raw_nh3 = {}, {}
    for r in RESOLUTIONS:
        raw_ct[r]  = load_ctarget(paths_ct[r])
        raw_nh3[r] = load_nh3(paths_nh3[r])

    t_comm = common_times(*[raw_ct[r][0]  for r in RESOLUTIONS],
                           *[raw_nh3[r][0] for r in RESOLUTIONS])
    t_comm = t_comm[(t_comm >= t_start) & (t_comm <= t_end)]

    ct, nh3 = {}, {}
    for r in RESOLUTIONS:
        ct[r]  = xy_avg_ppb(subset(raw_ct[r][0],  raw_ct[r][1],  t_comm))
        nh3[r] = xy_avg_ppb(subset(raw_nh3[r][0], raw_nh3[r][1], t_comm))

    return to_hours(t_comm), ct, nh3


def mean_spread(d):
    stack    = np.vstack([d[r] for r in RESOLUTIONS])   # (4, T)
    envelope = stack.max(axis=0) - stack.min(axis=0)     # (T,)
    return float(envelope.mean())

def overall_mean(d):
    return float(np.mean([np.mean(d[r]) for r in RESOLUTIONS]))


def plot(hours, ct, nh3, stem='f07'):
    fig, (ax_nh3, ax_ct) = plt.subplots(1, 2, figsize=(16, 6),
                                         sharey=False,
                                         gridspec_kw={'wspace': 0.12})

    xticks = np.arange(int(hours[0]), int(hours[-1]) + 1, 3)

    spread_nh3 = mean_spread(nh3)
    pct_nh3    = spread_nh3 / overall_mean(nh3) * 100
    spread_ct  = mean_spread(ct)
    pct_ct     = spread_ct  / overall_mean(ct)  * 100

    stack_nh3  = np.vstack([nh3[r] for r in RESOLUTIONS])
    stack_ct   = np.vstack([ct[r]  for r in RESOLUTIONS])
    global_max = max(stack_nh3.max(), stack_ct.max())
    ylo        = 0.0
    yhi        = global_max + 0.40 * (global_max - min(stack_nh3.min(), stack_ct.min()))

    for res, color in RES_COLORS.items():
        ax_nh3.plot(hours, nh3[res], color=color, lw=2.2, label=res)
    ax_nh3.fill_between(hours, stack_nh3.min(axis=0), stack_nh3.max(axis=0),
                        color='#888888', alpha=0.15, zorder=0)
    ax_nh3.set_xlim(hours[0], hours[-1])
    ax_nh3.set_ylim(ylo, yhi)
    ax_nh3.set_xticks(xticks)
    ax_nh3.set_xlabel('Time of day (h)', fontsize=13)
    ax_nh3.set_ylabel('Mole fraction (nmol mol$^{-1}$)', fontsize=13)
    ax_nh3.set_title('NH₃ surface mole fraction', fontweight='bold', fontsize=15)
    ax_nh3.grid(True, alpha=0.4)
    ax_nh3.legend(title='Δz', framealpha=0.9, fontsize=11,
                  title_fontsize=11, loc='lower right')
    ax_nh3.text(0.04, 0.97,
                f'Mean spread:  {spread_nh3:.2f} nmol mol$^{{-1}}$  ({pct_nh3:.0f}% of mean)',
                transform=ax_nh3.transAxes, fontsize=12, va='top', ha='left',
                color='#222222',
                bbox=dict(boxstyle='round,pad=0.4', fc='#fff3f3', ec='#CC3311', lw=1.2))

    for res, color in RES_COLORS.items():
        ax_ct.plot(hours, ct[res], color=color, lw=2.2, label=res)

    ax_ct.fill_between(hours, stack_ct.min(axis=0), stack_ct.max(axis=0),
                       color='#888888', alpha=0.15, zorder=0)
    ax_ct.set_xlim(hours[0], hours[-1])
    ax_ct.set_ylim(ylo, yhi)
    ax_ct.set_xticks(xticks)
    ax_ct.set_xlabel('Time of day (h)', fontsize=13)
    ax_ct.set_ylabel('Mole fraction (nmol mol$^{-1}$)', fontsize=13)
    ax_ct.set_title('Reference height mole fraction', fontweight='bold', fontsize=15)
    ax_ct.grid(True, alpha=0.4)
    ax_ct.legend(title='Δz', framealpha=0.9, fontsize=11,
                 title_fontsize=11, loc='lower right')
    ax_ct.text(0.04, 0.97,
               f'Mean spread:  {spread_ct:.3f} nmol mol$^{{-1}}$  ({pct_ct:.1f}% of mean)',
               transform=ax_ct.transAxes, fontsize=12, va='top', ha='left',
               color='#222222',
               bbox=dict(boxstyle='round,pad=0.4', fc='#f0f7ff', ec='#0173B2', lw=1.2))

    fig.tight_layout()
    fig.savefig(f'{stem}.png', dpi=300, facecolor='white', edgecolor='none',
                format='png', bbox_inches='tight')
    fig.savefig(f'{stem}.pdf', facecolor='white', edgecolor='none',
                format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {stem}.png, {stem}.pdf')

def main():
    print('Loading data ...')
    hours, ct, nh3 = load_all()
    print(f'  {len(hours)} timesteps  ({hours[0]:.1f} – {hours[-1]:.1f} h)')
    plot(hours, ct, nh3)
    print('Done.')

if __name__ == '__main__':
    main()
