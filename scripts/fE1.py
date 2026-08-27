#!/usr/bin/env python3
import netCDF4 as nc
import matplotlib.pyplot as plt
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
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
})

nc_file = '../cases/grassland/psbg_bidir_dz04/plume_chem.default.0010800.nc'

if not os.path.exists(nc_file):
    print(f"Error: File not found: {nc_file}")
    sys.exit(1)

try:
    dataset = nc.Dataset(nc_file, 'r')

    time      = dataset.variables['time'][:]
    time_hours = time / 3600.0

    rad   = dataset.groups["radiation"]
    sw_dn = rad.variables["sw_flux_dn"][:]
    sw_up = rad.variables["sw_flux_up"][:]
    lw_dn = rad.variables["lw_flux_dn"][:]
    lw_up = rad.variables["lw_flux_up"][:]

    Rn = (sw_dn - sw_up) + (lw_dn - lw_up)

    fig, ax = plt.subplots(figsize=(6, 4))

    line1 = ax.plot(time_hours, sw_dn, '--', color='#ff7f0e', linewidth=1.5, label='SW down')[0]
    line2 = ax.plot(time_hours, sw_up, '--', color='#d62728', linewidth=1.5, label='SW up')[0]
    line3 = ax.plot(time_hours, lw_dn, '--', color='#1f77b4', linewidth=1.5, label='LW down')[0]
    line4 = ax.plot(time_hours, lw_up, '--', color='#2ca02c', linewidth=1.5, label='LW up')[0]
    line5 = ax.plot(time_hours, Rn,    '-',  color='black',   linewidth=2.0, label='Net radiation')[0]

    ax.set_xlabel('Time (h)', fontsize=8)
    ax.set_ylabel('Radiation flux (W m⁻²)', fontsize=8)

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))

    all_values = np.concatenate([sw_dn, sw_up, lw_dn, lw_up, Rn])
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.2 * y_range, y_max + 0.2 * y_range)

    ax.grid(True, alpha=0.3)

    ax.legend(handles=[line1, line2, line3, line4, line5],
              labels=['SW down', 'SW up', 'LW down', 'LW up', 'Net radiation'],
              loc='upper right', fontsize=7, frameon=True,
              fancybox=False, shadow=False, bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()

    output_name  = 'fE1'
    png_filename = f'{output_name}.png'
    pdf_filename = f'{output_name}.pdf'

    plt.savefig(png_filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')
    plt.savefig(pdf_filename, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='pdf')

    plt.show()

    print(f"Time range: {time_hours[0]:.2f}–{time_hours[-1]:.2f} h")
    print(f"Number of time steps: {len(time)}")

    print("\nNet radiation zero crossings:")
    zero_crossings = []
    for i in range(len(Rn) - 1):
        if (Rn[i] <= 0 and Rn[i+1] > 0) or (Rn[i] >= 0 and Rn[i+1] < 0):
            t1, t2 = time_hours[i], time_hours[i+1]
            r1, r2 = Rn[i], Rn[i+1]
            if r2 != r1:
                t_cross = t1 - r1 * (t2 - t1) / (r2 - r1)
                zero_crossings.append(t_cross)
                direction = "positive" if r2 > r1 else "negative"
                print(f"  Crosses zero at {t_cross:.2f} h (going {direction})")

    if not zero_crossings:
        print("  No zero crossings found")

    dataset.close()

    png_size_mb = os.path.getsize(png_filename) / (1024 * 1024)
    pdf_size_mb = os.path.getsize(pdf_filename) / (1024 * 1024)
    print(f"\nSaved '{png_filename}' ({png_size_mb:.2f} MB)")
    print(f"Saved '{pdf_filename}' ({pdf_size_mb:.2f} MB)")

except Exception as e:
    print(f"Error processing NetCDF file: {e}")
    sys.exit(1)
