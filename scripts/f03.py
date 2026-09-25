"""
Analyse NH3 dry-deposition flux from MicroHH netCDF output (grassland scenario).
"""

import os
import traceback

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import netCDF4 as nc
import numpy as np

plt.rcParams.update({
    'font.size':          12,
    'axes.titlesize':     16,
    'axes.labelsize':     14,
    'xtick.labelsize':    12,
    'ytick.labelsize':    12,
    'legend.fontsize':    12,
    'figure.titlesize':   16,
})

EMISSION_RATE     = 4e-5   # kg/s  (4 sources x 0.01 g/s each)
SOURCE_X_POSITION = 400    # m from western boundary
X_MAX             = 7680   # m, eastern boundary

NAVY  = '#003087'   # PSBG - BG
BLACK = '#000000'   # PS onedir

GRASS_COLOR = 'lightgreen'

REPORT_DISTANCES = [1400, 2400, 3400, 4400, 5400, 6400, 7400]  # m

BASE = '../cases/grassland'
FILES = {
    'psbg':   f'{BASE}/psbg_bidir_dz04/flux_inst.xy.nc',
    'bg':     f'{BASE}/bg_bidir_dz04/flux_inst.xy.nc',
    'onedir': f'{BASE}/ps_onedir_dz04/flux_inst.xy.nc',
}


def _draw_source_marker(ax, x_pos, radius_pts=10, color='r'):
    theta = np.linspace(0, np.pi, 100)
    verts = np.column_stack([np.cos(theta), np.sin(theta)])
    verts = np.vstack([verts, [0, 0]])
    codes = (
        [mpath.Path.MOVETO]
        + [mpath.Path.LINETO] * (len(theta) - 1)
        + [mpath.Path.CLOSEPOLY]
    )
    semi_marker = mpath.Path(verts, codes)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(x_pos, 0, marker=semi_marker, markersize=radius_pts * 2,
            color=color, transform=trans, clip_on=False,
            label='Point source', zorder=5, linestyle='none')


def _select_time(flux, time, start_time, end_time):
    if start_time is None and end_time is None:
        print(f"  Using all {flux.shape[0]} timesteps")
        return flux

    st   = time[0]  if start_time is None else start_time
    et   = time[-1] if end_time   is None else end_time
    mask = (time >= st) & (time <= et)

    if not np.any(mask):
        print(f"  Warning: no data in [{st:.0f}, {et:.0f}] s -- using all timesteps.")
        return flux

    selected = flux[mask]
    sel_time = time[mask]
    print(f"  Selected {len(sel_time)} timesteps ({sel_time[0]:.0f} - {sel_time[-1]:.0f} s)")
    return selected


def _integrate_and_sort(flux, x, y):
    """
    Integrate flux [kg/m2/s] over y for every (timestep, x-column),
    average over time, sort by x, and compute cumulative deposition.
    """
    n          = flux.shape[0]
    integrated = np.zeros((n, len(x)))

    print("  Integrating over y:")
    for t in range(n):
        if t % 10 == 0 or t == n - 1:
            print(f"  Progress: {(t + 1) / n * 100:.1f}%  ({t + 1}/{n})", end="\r")
        for i in range(len(x)):
            integrated[t, i] = np.trapz(flux[t, :, i], y)  # kg/m/s
    print()

    mean_flux = np.mean(integrated, axis=0)
    sort_idx  = np.argsort(x)
    sx        = x[sort_idx]
    sf        = mean_flux[sort_idx]
    dx        = np.diff(sx, prepend=sx[0])
    cumul     = np.cumsum(-sf * dx) / EMISSION_RATE

    return sx, sf * 1e6, cumul


def analyze_single(filename, start_time=None, end_time=None):
    """Read one flux_inst.xy.nc file, integrate over y, and average over time."""
    with nc.Dataset(filename, 'r') as ds:
        x    = ds.variables['x'][:]
        y    = ds.variables['y'][:]
        time = ds.variables['time'][:]
        flux = ds.variables['flux_inst'][:]

    print(f"  {os.path.basename(filename)}: {flux.shape[0]} timesteps "
          f"({time[0]:.0f} - {time[-1]:.0f} s)")

    flux = _select_time(flux, time, start_time, end_time)
    return _integrate_and_sort(flux, x, y)


def analyze_difference(psbg_file, bg_file, start_time=None, end_time=None):
    """
    Subtract BG flux from PSBG flux to isolate the point-source contribution,
    then integrate over y and average over time.
    """
    with nc.Dataset(psbg_file, 'r') as ds:
        x         = ds.variables['x'][:]
        y         = ds.variables['y'][:]
        time      = ds.variables['time'][:]
        psbg_flux = ds.variables['flux_inst'][:]

    with nc.Dataset(bg_file, 'r') as ds:
        bg_time = ds.variables['time'][:]
        bg_flux = ds.variables['flux_inst'][:]

    print(f"  PSBG: {psbg_flux.shape[0]} timesteps ({time[0]:.0f} - {time[-1]:.0f} s)")

    if not np.array_equal(time, bg_time):
        print("  Warning: time axes of PSBG and BG files do not match exactly.")

    psbg_flux = _select_time(psbg_flux, time, start_time, end_time)
    bg_flux   = _select_time(bg_flux,   time, start_time, end_time)

    return _integrate_and_sort(psbg_flux - bg_flux, x, y)


def _style_ax(ax, x_all):
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.axvspan(min(x_all), max(x_all), color=GRASS_COLOR, alpha=0.3, label='Grassland')
    _draw_source_marker(ax, SOURCE_X_POSITION, radius_pts=10, color='r')
    ax.set_xlim(left=0, right=X_MAX)
    ax.spines['left'].set_position(('data', 0))
    ax.grid(True)


def _print_statistics(results, labels):
    for (x, flux, cumul), label in zip(results, labels):
        print(f"\n--- {label} ---")
        print(f"  Total deposition fraction : {cumul[-1]:.4f}  ({cumul[-1] * 100:.2f} %)")
        peak_idx = np.argmin(flux)
        print(f"  Peak deposition flux      : {flux[peak_idx]:.6f} ug/m/s  at x = {x[peak_idx]:.1f} m")
        print("  Cumulative deposition at key distances:")
        for dist in REPORT_DISTANCES:
            if dist <= max(x):
                idx = np.abs(x - dist).argmin()
                print(f"    {dist:5d} m : {cumul[idx] * 100:.2f} %")


def _print_difference(results, labels):
    total1 = results[0][2][-1]
    total2 = results[1][2][-1]
    diff_pct = (total2 - total1) / total1 * 100
    print(f"\n--- Difference ({labels[1]} vs {labels[0]}) ---")
    print(f"  {labels[0]}: {total1 * 100:.2f} %")
    print(f"  {labels[1]}: {total2 * 100:.2f} %")
    print(f"  Difference : {diff_pct:+.2f} %")


def plot_results(results, labels, colors, output_name='f03'):
    x_all = results[0][0]

    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax, x_all)
    for (x, _, cumul), label, color in zip(results, labels, colors):
        ax.plot(x, cumul * 100, color=color, linewidth=2, label=label)
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel('Cumulative deposition (%)')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{output_name}.png', dpi=300)
    fig.savefig(f'{output_name}.pdf')
    plt.show()

    _print_statistics(results, labels)
    _print_difference(results, labels)


def main():
    try:
        print("Computing PSBG - BG...")
        x1, f1, c1 = analyze_difference(FILES['psbg'], FILES['bg'])
        print("Computing PS onedir...")
        x2, f2, c2 = analyze_single(FILES['onedir'])
        plot_results(
            [(x1, f1, c1), (x2, f2, c2)],
            labels=['Point source, bidirectional (background subtracted)',
                    'Point source, unidirectional'],
            colors=[NAVY, BLACK])
    except Exception as exc:
        print(f"\nError: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
