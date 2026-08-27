"""
Compare NH3 background dry-deposition mass from MicroHH netCDF output,
unidirectional exchange (compensation point deactivated), background
only, no point source: grassland vs forest without target height vs
forest with target height.
"""

import os
import traceback

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

plt.rcParams.update({
    'font.size':          12,
    'axes.titlesize':     16,
    'axes.labelsize':     14,
    'axes.labelweight':   'bold',
    'xtick.labelsize':    12,
    'ytick.labelsize':    12,
    'legend.fontsize':    12,
    'figure.titlesize':   16,
    'figure.titleweight': 'bold',
})

X_MAX = 7680   # m, eastern boundary

NAVY  = '#003087'   # grassland, onedir
BLACK = '#000000'   # forest, onedir, no target height
GREEN = '#2e5339'   # forest, onedir, target height

REPORT_DISTANCES = [1400, 2400, 3400, 4400, 5400, 6400, 7400]  # m

GRASS_BASE  = '../cases/grassland'
FOREST_BASE = '../cases/forest'
FILES = {
    'grass':          f'{GRASS_BASE}/bg_onedir_dz04/flux_inst.xy.nc',
    'forest_no_target': f'{FOREST_BASE}/bg_onedir_no_target_height_dz04/flux_inst.xy.nc',
    'forest_target':    f'{FOREST_BASE}/bg_onedir_dz04/flux_inst.xy.nc',
}


def _select_time(flux, time, start_time, end_time):
    if start_time is None and end_time is None:
        print(f"  Using all {flux.shape[0]} timesteps")
        return flux, time

    st   = time[0]  if start_time is None else start_time
    et   = time[-1] if end_time   is None else end_time
    mask = (time >= st) & (time <= et)

    if not np.any(mask):
        print(f"  Warning: no data in [{st:.0f}, {et:.0f}] s -- using all timesteps.")
        return flux, time

    selected = flux[mask]
    sel_time = time[mask]
    print(f"  Selected {len(sel_time)} timesteps ({sel_time[0]:.0f} - {sel_time[-1]:.0f} s)")
    return selected, sel_time


def analyze_single(filename, start_time=None, end_time=None):
    """Read one flux_inst.xy.nc file, integrate over y, average over time,
    then convert the time-averaged rate to a cumulative deposited mass (kg)
    using the total time span covered by the selected timesteps."""
    with nc.Dataset(filename, 'r') as ds:
        x    = ds.variables['x'][:]
        y    = ds.variables['y'][:]
        time = ds.variables['time'][:]
        flux = ds.variables['flux_inst'][:]

    print(f"  {os.path.basename(filename)}: {flux.shape[0]} timesteps "
          f"({time[0]:.0f} - {time[-1]:.0f} s)")

    flux, time = _select_time(flux, time, start_time, end_time)

    n          = flux.shape[0]
    integrated = np.zeros((n, len(x)))
    print("  Integrating over y:")
    for t in range(n):
        if t % 10 == 0 or t == n - 1:
            print(f"  Progress: {(t + 1) / n * 100:.1f}%  ({t + 1}/{n})", end="\r")
        integrated[t] = np.trapz(flux[t], y, axis=0)  # kg/m/s
    print()

    mean_rate = np.mean(integrated, axis=0)          # kg/m/s
    dt        = np.mean(np.diff(time))
    duration  = dt * n                                # s
    mass_rate = mean_rate * duration                  # kg/m (over full duration)

    sort_idx = np.argsort(x)
    sx       = x[sort_idx]
    sm       = mass_rate[sort_idx]
    dx       = np.diff(sx, prepend=sx[0])
    cumul_kg = np.cumsum(-sm * dx)

    return sx, sm, cumul_kg, duration


def _style_ax(ax, x_all):
    ax.tick_params(axis='both', which='major', labelsize=12)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')

    ax.set_xlim(left=0, right=X_MAX)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.grid(True)


def _print_statistics(results, labels):
    for (x, sm, cumul_kg, duration), label in zip(results, labels):
        print(f"\n--- {label} ---")
        print(f"  Time window analysed       : {duration:.0f} s")
        net_sign = 'deposition' if cumul_kg[-1] >= 0 else 'emission'
        print(f"  Net {net_sign:<10}       : {abs(cumul_kg[-1]):.6e} kg")
        peak_idx  = np.argmax(np.abs(sm))
        peak_sign = 'deposition' if sm[peak_idx] < 0 else 'emission'
        print(f"  Peak {peak_sign} rate        : {abs(sm[peak_idx]):.6e} kg/m at x = {x[peak_idx]:.1f} m")
        print("  Cumulative deposited mass at key distances (negative = net emission):")
        for dist in REPORT_DISTANCES:
            if dist <= max(x):
                idx = np.abs(x - dist).argmin()
                print(f"    {dist:5d} m : {cumul_kg[idx]:.6e} kg")


def plot_results(results, labels, colors, output_name):
    label_font = {'fontsize': 14, 'fontweight': 'bold'}
    x_all      = results[0][0]

    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax, x_all)
    for (x, _, cumul_kg, _), label, color in zip(results, labels, colors):
        ax.plot(x, cumul_kg, color=color, linewidth=2, label=label)
    ax.set_xlabel('X Coordinate (m)', **label_font)
    ax.set_ylabel('Cumulative Deposition (kg, negative = net emission)', **label_font)
    ax.legend(loc='upper left')
    fig.tight_layout()
    # fig.savefig(f'{output_name}_cumulative.png', dpi=300)
    # fig.savefig(f'{output_name}_cumulative.pdf')
    # plt.show()

    _print_statistics(results, labels)
    _print_differences(results, labels)


def _print_differences(results, labels):
    """Pairwise differences of every case against the grassland reference (index 0)."""
    ref_total   = results[0][2][-1]
    ref_label   = labels[0]
    print(f"\n--- Differences vs {ref_label} ---")
    for (x, sm, cumul_kg, duration), label in zip(results[1:], labels[1:]):
        total   = cumul_kg[-1]
        diff_kg = total - ref_total
        print(f"\n  {label} vs {ref_label}")
        print(f"    {ref_label}: {ref_total:.6e} kg")
        print(f"    {label}: {total:.6e} kg")
        print(f"    Difference : {diff_kg:+.6e} kg")
        if ref_total == 0 or (ref_total < 0) != (total < 0):
            print("    (Percentage difference omitted: totals differ in sign or one is zero,"
                  " a % change is not meaningful here.)")
        else:
            diff_pct = diff_kg / ref_total * 100
            print(f"                 {diff_pct:+.0f} %")


def _prompt_time_window():
    start = input("Start time (s, blank = from beginning): ").strip()
    end   = input("End time (s, blank = to end): ").strip()
    return (float(start) if start else None,
            float(end)   if end   else None)


def _output_name(start_time, end_time):
    st = 'start' if start_time is None else f'{start_time:.0f}'
    et = 'end'   if end_time   is None else f'{end_time:.0f}'
    return f'grass_vs_forest_notarget_vs_target_onedir_{st}-{et}s'


def main():
    try:
        start_time, end_time = _prompt_time_window()
        print("Computing grassland, unidirectional...")
        r1 = analyze_single(FILES['grass'], start_time, end_time)
        print("Computing forest, unidirectional, no target height...")
        r2 = analyze_single(FILES['forest_no_target'], start_time, end_time)
        print("Computing forest, unidirectional, target height...")
        r3 = analyze_single(FILES['forest_target'], start_time, end_time)
        plot_results(
            [r1, r2, r3],
            labels=['Grassland, unidirectional',
                    'Forest, unidirectional, no target height',
                    'Forest, unidirectional, target height'],
            colors=[NAVY, BLACK, GREEN],
            output_name=_output_name(start_time, end_time))
    except Exception as exc:
        print(f"\nError: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
