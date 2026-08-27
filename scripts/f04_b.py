import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import re
from cmcrameri import cm as cmc
from matplotlib.ticker import FuncFormatter

LABEL_SIZE   = 14
LABEL_WEIGHT = 'bold'
TICK_SIZE    = 12
CMAP         = cmc.batlow
COLOR_A      = CMAP(0.20)
COLOR_B      = CMAP(0.75)

XLIM_NMOL = (6.95, 7.15)   # nmol mol⁻¹
XLIM_UGM3 = (4.88, 5.05)   # µg m⁻³

ds     = nc.Dataset('../cases/grassland/bg_bidir_dz04/plume_chem.default.0010800.nc')
z      = ds.variables['z'][:]
time   = ds.variables['time'][:]
rhoref = ds.groups['thermo'].variables['rhoref'][:]   # (z,) kg m⁻³

nh3   = ds.groups['default'].variables['nh3'][:]      # (time, z) horizontal mean
nh3_2 = ds.groups['default'].variables['nh3_2'][:]    # (time, z) <c'²> central moment

# σ_h = sqrt(<c'²>)
nh3_std = np.sqrt(np.maximum(nh3_2, 0.0))

time_hours   = time / 3600.0
M_NH3, M_air = 17.031, 28.97

def to_nmol(c):
    """mol/mol → nmol/mol"""
    return c * 1e9

def to_ugm3(c):
    """mol/mol → µg m⁻³  using  c × (M_NH3/M_air) × rhoref × 1e9"""
    return c * (M_NH3 / M_air) * rhoref[np.newaxis, :] * 1e9

nh3_nmol     = to_nmol(nh3)
nh3_std_nmol = to_nmol(nh3_std)
nh3_ugm3     = to_ugm3(nh3)
nh3_std_ugm3 = to_ugm3(nh3_std)

t_min, t_max = time_hours[0], time_hours[-1]
dt_h = float(np.median(np.diff(time_hours)))

z_max = 100.0
z_mask = z <= z_max

print(f"\nTime available: {t_min:.2f} – {t_max:.2f} h  (timestep ~{dt_h*60:.0f} s)")


def parse_time_input(prompt, t_min, t_max):
    """Return (t_start, t_end, is_snapshot) from user input."""
    raw = input(prompt).strip()
    m = re.match(
        r'^([+\-]?[0-9]*\.?[0-9]+(?:[eE][+\-]?[0-9]+)?)'
        r'-'
        r'([+\-]?[0-9]*\.?[0-9]+(?:[eE][+\-]?[0-9]+)?)$',
        raw
    )
    if m:
        t_s, t_e = float(m.group(1)), float(m.group(2))
        snapshot = False
    else:
        t_s = t_e = float(raw)
        snapshot = True
    tol = 0.1
    if t_s < t_min - tol or t_s > t_max + tol or t_e < t_min - tol or t_e > t_max + tol:
        raise ValueError(f"Time {raw!r} outside available range {t_min:.2f}–{t_max:.2f} h")
    return max(t_s, t_min), min(t_e, t_max), snapshot


def get_profile(data_tz, t_start, t_end, snapshot):
    """Return mean profile over [t_start, t_end], or nearest timestep if snapshot."""
    if snapshot:
        idx = np.argmin(np.abs(time_hours - t_start))
        print(f"  Snapshot: nearest timestep = {time_hours[idx]:.4f} h")
        return data_tz[idx]
    idx = np.where((time_hours >= t_start) & (time_hours <= t_end))[0]
    if len(idx) == 0:
        raise ValueError(f"No timesteps found in {t_start:.2f}–{t_end:.2f} h")
    print(f"  Range: {len(idx)} timesteps from {time_hours[idx[0]]:.4f} to {time_hours[idx[-1]]:.4f} h")
    return data_tz[idx].mean(axis=0)


def label_period(t_s, t_e, snapshot):
    if snapshot:
        return f"{time_hours[np.argmin(np.abs(time_hours - t_s))]:.2f} h"
    return f"{t_s:.1f}–{t_e:.1f} h"


def file_tag(t_s, t_e, snapshot):
    if snapshot:
        return f"{time_hours[np.argmin(np.abs(time_hours - t_s))]:.2f}h".replace('.', 'p')
    return f"{t_s:.1f}-{t_e:.1f}h".replace('.', 'p')


tA_s, tA_e, tA_snap = 10.0, 12.0, False
tB_s, tB_e, tB_snap = 13.0, 15.0, False

print("\nPeriod A:")
mA_nmol = get_profile(nh3_nmol,     tA_s, tA_e, tA_snap)
sA_nmol = get_profile(nh3_std_nmol, tA_s, tA_e, tA_snap)
mA_ugm3 = get_profile(nh3_ugm3,     tA_s, tA_e, tA_snap)
sA_ugm3 = get_profile(nh3_std_ugm3, tA_s, tA_e, tA_snap)

print("Period B:")
mB_nmol = get_profile(nh3_nmol,     tB_s, tB_e, tB_snap)
sB_nmol = get_profile(nh3_std_nmol, tB_s, tB_e, tB_snap)
mB_ugm3 = get_profile(nh3_ugm3,     tB_s, tB_e, tB_snap)
sB_ugm3 = get_profile(nh3_std_ugm3, tB_s, tB_e, tB_snap)

z_plot = z[z_mask]
labelA = label_period(tA_s, tA_e, tA_snap)
labelB = label_period(tB_s, tB_e, tB_snap)
tagA   = file_tag(tA_s, tA_e, tA_snap)
tagB   = file_tag(tB_s, tB_e, tB_snap)


def make_figure(mA, sA, mB, sB, xlabel, xlim, outname_base, save_output=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    for ax, mean, std, color, lbl in zip(
        axes,
        [mA[z_mask], mB[z_mask]],
        [sA[z_mask], sB[z_mask]],
        [COLOR_A, COLOR_B],
        [labelA,  labelB],
    ):
        ax.fill_betweenx(z_plot, mean - std, mean + std,
                         color=color, alpha=0.25,
                         label=r'$\pm 1\sigma_\mathrm{h}$' + f' ({lbl})')
        ax.plot(mean, z_plot, '-', color=color, linewidth=2.0, label=f'Mean ({lbl})')
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight=LABEL_WEIGHT)
        ax.set_ylim(0, z_max)
        ax.set_xlim(xlim)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        handles, labels_leg = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels_leg[::-1], fontsize=TICK_SIZE - 1, framealpha=0.9)

    axes[0].set_ylabel('z (m)', fontsize=LABEL_SIZE, fontweight=LABEL_WEIGHT)
    plt.tight_layout()

    if save_output:
        plt.savefig(f'{outname_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{outname_base}.pdf', dpi=300, bbox_inches='tight')
        print(f"Saved {outname_base}.png / .pdf")
    else:
        print(f"Skipped saving output for '{outname_base}' (save_output=False)")
    # plt.show()


make_figure(mA_nmol, sA_nmol, mB_nmol, sB_nmol,
            xlabel='NH₃ (nmol mol⁻¹)', xlim=XLIM_NMOL, outname_base='f05_2_nmol',
            save_output=False)

make_figure(mA_ugm3, sA_ugm3, mB_ugm3, sB_ugm3,
            xlabel='NH₃ (µg m⁻³)', xlim=XLIM_UGM3, outname_base='f04_b')

ds.close()
