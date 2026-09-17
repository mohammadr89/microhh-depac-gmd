import os

import netCDF4 as nc
import numpy as np

BASE = '../cases/forest'
FLUX_FILE = f'{BASE}/bg_bidir_dz04/flux_inst.xy.nc'


def analyze_single(filename):
    with nc.Dataset(filename, 'r') as ds:
        x = ds.variables['x'][:]
        y = ds.variables['y'][:]
        time = ds.variables['time'][:]
        flux = ds.variables['flux_inst'][:]

    n = flux.shape[0]
    integrated = np.zeros((n, len(x)))
    for t in range(n):
        integrated[t] = np.trapz(flux[t], y, axis=0)  # kg/m/s

    mean_rate = np.mean(integrated, axis=0)  # kg/m/s
    dt = np.mean(np.diff(time))
    duration = dt * n  # s
    mass_rate = mean_rate * duration  # kg/m

    sort_idx = np.argsort(x)
    sx = x[sort_idx]
    sm = mass_rate[sort_idx]
    dx = np.diff(sx, prepend=sx[0])
    cumul_kg = np.cumsum(-sm * dx)

    return cumul_kg[-1], duration


def main():
    if not os.path.exists(FLUX_FILE):
        print(f"MISSING: {FLUX_FILE}")
        return

    total_kg, duration = analyze_single(FLUX_FILE)
    net_sign = 'deposition' if total_kg >= 0 else 'emission'
    print(f"Forest, bidirectional, reference height (dz=4 m)")
    print(f"  Time window analysed : {duration:.0f} s")
    print(f"  Net {net_sign}      : {abs(total_kg):.6e} kg")


if __name__ == "__main__":
    main()
