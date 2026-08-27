import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from cmcrameri import cm as cmc
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
})

LABEL_SIZE   = 14
LABEL_WEIGHT = 'bold'
TICK_SIZE    = 12

COLOR_CCOMP = cmc.batlow(0.15)
COLOR_CONC  = cmc.batlow(0.75)
# COLOR_CONC  = '#009E73'
COLOR_SHADE = cmc.batlow(0.55)

R       = 8.314
M_NH3   = 17.031
P       = 101325


def convert_nh3_to_ugm3(nh3_molmol, temp_k):
    """Convert NH3 surface concentration from mol/mol to µg/m³ using ideal gas law."""
    return nh3_molmol * P * M_NH3 * 1e6 / (R * temp_k)


def analyze_time_series(ccomp_filename, nh3_filename, temp_filename,
                        x_range=None, y_range=None):
    try:
        print(f"Reading: {ccomp_filename}")
        with nc.Dataset(ccomp_filename, 'r') as ds:
            x_ccomp    = ds.variables['x'][:]
            y_ccomp    = ds.variables['y'][:]
            time_ccomp = ds.variables['time'][:]
            ccomp_data = ds.variables['ccomp_tot'][:]
        print(f"  ccomp_tot: {ccomp_data.shape[0]} timesteps "
              f"({time_ccomp[0]:.0f} – {time_ccomp[-1]:.0f} s)")

        print(f"Reading: {nh3_filename}")
        with nc.Dataset(nh3_filename, 'r') as ds:
            x_nh3    = ds.variables['x'][:]
            y_nh3    = ds.variables['y'][:]
            time_nh3 = ds.variables['time'][:]
            nh3_surface = ds.variables['nh3'][:, 0, :, :] 
        print(f"  nh3 surface: {nh3_surface.shape[0]} timesteps "
              f"({time_nh3[0]:.0f} – {time_nh3[-1]:.0f} s)")

        print(f"Reading: {temp_filename}")
        with nc.Dataset(temp_filename, 'r') as ds:
            temp_data = ds.variables['T_surface'][:]
        print(f"  T_surface: {temp_data.shape}")

        if nh3_surface.shape != temp_data.shape:
            raise ValueError(f"Shape mismatch: nh3 {nh3_surface.shape} vs T {temp_data.shape}")

        conc_data = convert_nh3_to_ugm3(nh3_surface, temp_data)

        def idx_range(coord, rng):
            if rng is None:
                return 0, len(coord) - 1
            return np.abs(coord - rng[0]).argmin(), np.abs(coord - rng[1]).argmin()

        xi0_c, xi1_c = idx_range(x_ccomp, x_range)
        yi0_c, yi1_c = idx_range(y_ccomp, y_range)
        xi0_n, xi1_n = idx_range(x_nh3,   x_range)
        yi0_n, yi1_n = idx_range(y_nh3,   y_range)

        x_min = x_ccomp[xi0_c]; x_max = x_ccomp[xi1_c]
        y_min = y_ccomp[yi0_c]; y_max = y_ccomp[yi1_c]

        avg_ccomp = np.mean(ccomp_data[:, yi0_c:yi1_c+1, xi0_c:xi1_c+1], axis=(1, 2))
        avg_conc  = np.mean(conc_data[:,  yi0_n:yi1_n+1, xi0_n:xi1_n+1], axis=(1, 2))

        return time_ccomp, avg_ccomp, avg_conc, (x_min, x_max, y_min, y_max)

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None


def plot_ccomp_and_conc(time_seconds, ccomp_values, conc_values, region_info,
                        panel_label=None, vlines=None):

    fig, ax = plt.subplots(figsize=(10, 6))
    time_hours = time_seconds / 3600.0

    mask_ccomp_higher = ccomp_values > conc_values
    ax.fill_between(time_hours, ccomp_values, conc_values,
                    where=mask_ccomp_higher, alpha=1.0,
                    color=COLOR_SHADE, interpolate=True)

    ax.plot(time_hours, ccomp_values, color=COLOR_CCOMP, linewidth=2.0, alpha=0.85)
    ax.plot(time_hours, conc_values,  color=COLOR_CONC,  linewidth=2.0, alpha=0.85)

    if vlines is not None:
        for vline_time in vlines:
            ax.plot(vline_time, 0, 'o', color='orange', markersize=8,
                    zorder=11, clip_on=False)

    ax.set_xlabel('Time (h)',               fontsize=LABEL_SIZE, fontweight=LABEL_WEIGHT)
    ax.set_ylabel('Concentration (µg m⁻³)', fontsize=LABEL_SIZE, fontweight=LABEL_WEIGHT)

    if panel_label is not None:
        ax.text(0.02, 0.98, f'({panel_label})', transform=ax.transAxes,
                fontsize=TICK_SIZE, verticalalignment='top', fontweight=LABEL_WEIGHT,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    all_values = np.concatenate([ccomp_values, conc_values])
    min_val    = np.min(all_values)
    max_val    = np.max(all_values)
    ax.set_ylim(0 if min_val >= 0 else min_val * 1.2,
                0 if max_val <= 0 else max_val * 1.2)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    legend_handles = [
        mlines.Line2D([], [], color=COLOR_CCOMP, linewidth=2.0, alpha=0.85,
                      label='Compensation point'),
        mlines.Line2D([], [], color=COLOR_CONC,  linewidth=2.0, alpha=0.85,
                      label='NH₃ at lowest model level'),
        Rectangle((0, 0), 1, 1, fc=COLOR_SHADE, alpha=1.0,
                  label=r'$\chi_{comp} > \chi_a$'),
        mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                      markersize=6, label='Sunrise & sunset'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', framealpha=0.95,
              fontsize=11, frameon=True, fancybox=False, shadow=False)

    plt.tight_layout()

    output_name = 'f04_a'
    plt.savefig(f'{output_name}.png', dpi=300, facecolor='white', edgecolor='none',
                format='png', bbox_inches='tight')
    plt.savefig(f'{output_name}.pdf', facecolor='white', edgecolor='none',
                format='pdf', bbox_inches='tight')
    # plt.show()
    print(f"Saved: {output_name}.png, {output_name}.pdf")

    return fig, ax


def main():
    BASE = '../cases/grassland/bg_bidir_dz04'
    ccomp_filename = f'{BASE}/ccomp_tot.xy.nc'
    nh3_filename   = f'{BASE}/nh3_first_level_xy.nc'
    temp_filename  = f'{BASE}/T_surface.xy.nc'
    panel_label    = None
    vlines         = [6, 18]

    for f in [ccomp_filename, nh3_filename, temp_filename]:
        if not os.path.exists(f):
            print(f"Error: file not found: {f}")
            return

    time_data, ccomp_data, conc_data, region_info = analyze_time_series(
        ccomp_filename, nh3_filename, temp_filename)

    if time_data is None:
        print("Error: could not process data.")
        return

    plot_ccomp_and_conc(time_data, ccomp_data, conc_data,
                        region_info, panel_label, vlines)


if __name__ == "__main__":
    main()
