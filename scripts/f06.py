import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker
import numpy as np
import os
import sys

import matplotlib
matplotlib.use('Agg')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
})

BASE = '../cases/grassland/ps_bidir_dz04'

nc_files = {
    'ra':       f'{BASE}/ra.xy.nc',
    'rb':       f'{BASE}/rb.xy.nc',
    'rc_tot':   f'{BASE}/rc_tot.xy.nc',
    'vdnh3':    f'{BASE}/vdnh3.xy.nc',
    'flux_inst':f'{BASE}/flux_inst.xy.nc',
    'nh3':      f'{BASE}/nh3_first_level_xy.nc',
    'temp':     f'{BASE}/T_surface.xy.nc',
}

missing_files = [f for f in nc_files.values() if not os.path.exists(f)]
if missing_files:
    print(f"Error: Files not found: {', '.join(missing_files)}")
    sys.exit(1)

FLUX_CONV = 1e12
NH3_THRESHOLD_UGM3 = 1e-5

R     = 8.314
M_NH3 = 17.031
P     = 101325

COLOR_NH3  = '#009E73'
COLOR_VD   = '#000000'
COLOR_FLUX = '#D55E00'


def convert_nh3_to_ugm3(nh3_molmol, temp_k):
    """Convert NH3 surface concentration from mol/mol to µg/m³ using ideal gas law."""
    return nh3_molmol * P * M_NH3 * 1e6 / (R * temp_k)


try:
    print("Opening NetCDF files...")
    data_resistance = {}
    nh3_raw = temp_3d = vd_3d = flux_3d = None
    time_hours = None

    for var, filename in nc_files.items():
        print(f"  Reading {filename}")
        ds = nc.Dataset(filename, 'r')

        if time_hours is None:
            time_hours = ds.variables['time'][:] / 3600.0

        if var == 'vdnh3':
            vd_3d = np.array(ds.variables[var][:]) * 100.0
        elif var == 'flux_inst':
            flux_3d = np.array(ds.variables[var][:]) * FLUX_CONV * -1
        elif var == 'nh3':
            nh3_raw = np.array(ds.variables['nh3'][:, 0, :, :])  # surface level
        elif var == 'temp':
            temp_3d = np.array(ds.variables['T_surface'][:])
        else:
            data_resistance[var] = np.mean(ds.variables[var][:], axis=(1, 2))

        ds.close()

    nh3_3d = convert_nh3_to_ugm3(nh3_raw, temp_3d)

    # Per-timestep plume mask
    plume_mask     = nh3_3d > NH3_THRESHOLD_UGM3
    n_plume_cells  = plume_mask.sum(axis=(1, 2))
    print(f"\n  Plume cells — min: {n_plume_cells.min()},  "
          f"max: {n_plume_cells.max()},  "
          f"mean: {n_plume_cells.mean():.1f}")

    def masked_spatial_mean(data_3d, mask_3d):
        masked = np.where(mask_3d, data_3d, np.nan)
        with np.errstate(all='ignore'):
            return np.nanmean(masked, axis=(1, 2))

    data_nh3  = masked_spatial_mean(nh3_3d,  plume_mask)
    data_vd   = masked_spatial_mean(vd_3d,   plume_mask)
    data_flux = masked_spatial_mean(flux_3d, plume_mask)

    # Resistance partitioning
    total_r = data_resistance['ra'] + data_resistance['rb'] + data_resistance['rc_tot']
    total_r = np.where(total_r == 0, 1e-10, total_r)
    ra_pct  = data_resistance['ra']     / total_r * 100
    rb_pct  = data_resistance['rb']     / total_r * 100
    rc_pct  = data_resistance['rc_tot'] / total_r * 100

    N_TICKS    = 6
    N_TICKS_VD = 6

    vmax_nh3          = 1.05 * np.nanmax(data_nh3)
    step_nh3          = np.ceil((vmax_nh3 / (N_TICKS - 1)) / 0.5) * 0.5
    vmax_nh3_rounded  = step_nh3 * (N_TICKS - 1)

    vmax_vd           = np.nanmax(data_vd)
    step_vd           = np.ceil((vmax_vd / (N_TICKS_VD - 1)) / 0.05) * 0.05
    vmax_vd_rounded   = step_vd * (N_TICKS_VD - 1)

    vmax_flux         = 1.05 * np.nanmax(data_flux)
    step_flux         = np.ceil((vmax_flux / (N_TICKS - 1)) / 0.5) * 0.5
    vmax_flux_rounded = step_flux * (N_TICKS - 1)

    FIG_H = 6
    FIG_W = 13

    ax_left_in   = 0.20 * FIG_W
    ax_bottom_in = 0.14 * FIG_H
    ax_w_in      = 0.56 * FIG_W
    ax_h_in      = 0.68 * FIG_H

    FIG_W_FINAL = ax_left_in + ax_w_in + 1.5
    FIG_SIZE    = (FIG_W_FINAL, FIG_H)
    AX_RECT     = [ax_left_in   / FIG_W_FINAL,
                   ax_bottom_in / FIG_H,
                   ax_w_in      / FIG_W_FINAL,
                   ax_h_in      / FIG_H]

    fig    = plt.figure(figsize=FIG_SIZE)
    ax_nh3 = fig.add_axes(AX_RECT)

    ax_res = ax_nh3.twinx()
    ax_res.set_zorder(ax_nh3.get_zorder() - 1)
    ax_nh3.patch.set_visible(False)

    ax_res.fill_between(time_hours, 0, ra_pct,
                        color='#1f77b4', alpha=0.3, label='Aerodynamic resistance',    zorder=1)
    ax_res.fill_between(time_hours, ra_pct, ra_pct + rb_pct,
                        color='orange',  alpha=0.3, label='Boundary layer resistance', zorder=1)
    ax_res.fill_between(time_hours, ra_pct + rb_pct, 100,
                        color='#2ca02c', alpha=0.3, label='Total canopy resistance',   zorder=1)
    ax_res.set_ylabel('Resistance partitioning (%)', fontsize=8, fontweight='bold')
    ax_res.tick_params(axis='y', labelsize=9)
    for spine in ax_res.spines.values():
        spine.set_edgecolor('black')
    ax_res.set_ylim(0, 100)
    ax_res.set_yticks(range(0, 101, 20))

    ax_nh3.plot(time_hours, data_nh3, '-', color=COLOR_NH3, linewidth=2.0, zorder=10)
    ax_nh3.set_ylabel('NH₃ concentration (µg m⁻³)', fontsize=8, fontweight='bold', color=COLOR_NH3)
    ax_nh3.tick_params(axis='y', labelcolor=COLOR_NH3, labelsize=9)
    ax_nh3.tick_params(axis='x', labelcolor='black',   labelsize=9)
    ax_nh3.spines['left'].set_edgecolor(COLOR_NH3)
    ax_nh3.spines['right'].set_edgecolor('black')
    ax_nh3.spines['bottom'].set_edgecolor('black')
    ax_nh3.spines['left'].set_position(('outward', 0))
    ax_nh3.set_ylim(0, vmax_nh3_rounded)
    ax_nh3.set_yticks(np.arange(0, vmax_nh3_rounded + step_nh3 * 0.5, step_nh3))
    ax_nh3.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

    ax_vd = ax_nh3.twinx()
    ax_vd.spines['left'].set_position(('outward', 50))
    ax_vd.spines['left'].set_edgecolor(COLOR_VD)
    ax_vd.spines['left'].set_visible(True)
    ax_vd.spines['right'].set_visible(False)
    ax_vd.yaxis.set_label_position('left')
    ax_vd.yaxis.set_ticks_position('left')
    ax_vd.patch.set_visible(False)
    ax_vd.plot(time_hours, data_vd, '-', color=COLOR_VD, linewidth=2.0, zorder=10)
    ax_vd.set_ylabel('Exchange velocity (cm s⁻¹)', fontsize=8, fontweight='bold', color=COLOR_VD)
    ax_vd.tick_params(axis='y', labelcolor=COLOR_VD, labelsize=9)
    ax_vd.set_ylim(0, vmax_vd_rounded)
    ax_vd.set_yticks(np.arange(0, vmax_vd_rounded + step_vd * 0.5, step_vd))
    ax_vd.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

    ax_flux = ax_nh3.twinx()
    ax_flux.spines['left'].set_position(('outward', 100))
    ax_flux.spines['left'].set_edgecolor(COLOR_FLUX)
    ax_flux.spines['left'].set_visible(True)
    ax_flux.spines['right'].set_visible(False)
    ax_flux.yaxis.set_label_position('left')
    ax_flux.yaxis.set_ticks_position('left')
    ax_flux.patch.set_visible(False)
    ax_flux.plot(time_hours, data_flux, '-', color=COLOR_FLUX, linewidth=2.0, zorder=10)
    ax_flux.set_ylabel(r'Deposition flux magnitude, $-F_\mathrm{inst}$ (ng m$^{-2}$ s$^{-1}$)',
                       fontsize=8, fontweight='bold', color=COLOR_FLUX)
    ax_flux.tick_params(axis='y', labelcolor=COLOR_FLUX, labelsize=9)
    ax_flux.set_ylim(0, vmax_flux_rounded)
    ax_flux.set_yticks(np.arange(0, vmax_flux_rounded + step_flux * 0.5, step_flux))
    ax_flux.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

    ax_nh3.set_xlabel('Time (h)', fontsize=8, fontweight='bold')
    ax_nh3.set_xlim(0, 24)
    ax_nh3.set_xticks(range(0, 25, 2))
    ax_nh3.grid(True, alpha=0.3)
    ax_nh3.yaxis.grid(False)

    ax_nh3.plot(6,  0, 'o', color='orange', markersize=8, zorder=11, clip_on=False)
    ax_nh3.plot(18, 0, 'o', color='orange', markersize=8, zorder=11, clip_on=False)

    for ax in [ax_nh3, ax_vd, ax_flux, ax_res]:
        ax.spines['top'].set_visible(False)

    left_handles = [
        mlines.Line2D([], [], color=COLOR_NH3,  linewidth=2.0, label='NH₃ (µg m⁻³)'),
        mlines.Line2D([], [], color=COLOR_VD,   linewidth=2.0, label='Exchange velocity (cm s⁻¹)'),
        mlines.Line2D([], [], color=COLOR_FLUX, linewidth=2.0,
                      label=r'Deposition flux magnitude, $-F_\mathrm{inst}$ (ng m$^{-2}$ s$^{-1}$)'),
        mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                      markersize=6, label='Sunrise & sunset'),
    ]
    leg_left = ax_nh3.legend(left_handles, [h.get_label() for h in left_handles],
                             loc='lower left', bbox_to_anchor=(0.0, 1.01),
                             borderaxespad=0, fontsize=7, frameon=True,
                             fancybox=False, shadow=False, ncol=1)
    leg_left.get_frame().set_edgecolor('black')

    res_handles = [
        mlines.Line2D([], [], color='#1f77b4', linewidth=8, alpha=0.3,
                      label='Aerodynamic resistance'),
        mlines.Line2D([], [], color='orange',  linewidth=8, alpha=0.3,
                      label='Boundary layer resistance'),
        mlines.Line2D([], [], color='#2ca02c', linewidth=8, alpha=0.3,
                      label='Total canopy resistance'),
    ]
    leg_right = ax_res.legend(handles=res_handles,
                              labels=[h.get_label() for h in res_handles],
                              loc='lower right', bbox_to_anchor=(1.0, 1.01),
                              borderaxespad=0, fontsize=7, frameon=True,
                              fancybox=False, shadow=False, ncol=1)
    leg_right.get_frame().set_edgecolor('black')

    output_name = 'f06'
    fname_png = f'{output_name}.png'
    fname_pdf = f'{output_name}.pdf'
    fig.savefig(fname_png, dpi=300, facecolor='white', edgecolor='none', format='png')
    fig.savefig(fname_pdf, facecolor='white', edgecolor='none', format='pdf')
    size_mb = os.path.getsize(fname_png) / (1024 * 1024)
    print(f"Saved: {fname_png} ({size_mb:.2f} MB), {fname_pdf}")
    plt.close(fig)

    i_max_flux = np.nanargmax(data_flux)
    print(f"\nVARIABLE RANGES (plume-masked):")
    print(f"  Exchange velocity (cm/s): min={np.nanmin(data_vd):.4f}, max={np.nanmax(data_vd):.4f}")
    print(f"  Flux (ng/m²/s):           min={np.nanmin(data_flux):.4f}, max={np.nanmax(data_flux):.4f}")
    print(f"  NH3 (µg/m³):              min={np.nanmin(data_nh3):.4f}, max={np.nanmax(data_nh3):.4f}")
    print(f"  Ra/Rb/Rc (%): "
          f"{np.nanmin(ra_pct):.1f}–{np.nanmax(ra_pct):.1f} / "
          f"{np.nanmin(rb_pct):.1f}–{np.nanmax(rb_pct):.1f} / "
          f"{np.nanmin(rc_pct):.1f}–{np.nanmax(rc_pct):.1f}")
    print(f"\nAt peak flux (t = {time_hours[i_max_flux]:.2f} h):")
    print(f"  Flux={data_flux[i_max_flux]:.4f} ng/m²/s, "
          f"Vd={data_vd[i_max_flux]:.4f} cm/s, "
          f"NH3={data_nh3[i_max_flux]:.4f} µg/m³")

except Exception as e:
    print(f"Error: {e}")
    raise
