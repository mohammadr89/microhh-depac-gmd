import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import netCDF4 as nc

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
})

FILE = '../cases/grassland/psbg_bidir_dz04/ccomp_tot.xy.nc'

TARGET_TIMES_S = [6 * 3600, 12 * 3600]  # 06:00 and 12:00
OUT_NAMES      = ['f05_a', 'f05_b']


def get_nearest_index(times, t_target):
    """Return the index of the time step closest to t_target."""
    return int(np.argmin(np.abs(times - t_target)))


def plot_snapshot(ds, times, x, y, t_target, out_name):
    idx      = get_nearest_index(times, t_target)
    t_actual = times[idx]
    print(f"\nNearest time step: {t_actual:.0f} s ({t_actual/3600:.4f} h)  [index {idx}]")

    ccomp_snap = ds.variables["ccomp_tot"][idx, :, :]  # (y, x)  µg m⁻³

    ccomp_masked  = np.ma.masked_less_equal(ccomp_snap, 0)
    positive_vals = ccomp_snap[ccomp_snap > 0]
    vmin = positive_vals.min() if positive_vals.size > 0 else 0.01
    vmax = 280
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    label_font = {'fontsize': 14, 'fontweight': 'bold'}
    out_base   = out_name

    fig, ax = plt.subplots(figsize=(11, 5.5))
    pcm = ax.pcolormesh(x / 1000, y / 1000, ccomp_masked,
                        cmap="RdYlBu_r", shading="nearest", norm=norm)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Compensation point (μg m⁻³)", **label_font)
    cbar.ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    cbar.ax.yaxis.set_major_formatter(mticker.LogFormatter(base=10, labelOnlyBase=False))
    cbar.ax.tick_params(labelsize=12)
    ax.set_xlabel("x (km)", **label_font)
    ax.set_ylabel("y (km)", **label_font)
    ax.tick_params(axis='both', which='major', labelsize=12)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{out_base}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{out_base}.pdf", bbox_inches="tight")
    print(f"\nSaved → {out_base}.png, {out_base}.pdf")
    print(f"Min:  {ccomp_snap.min():.4f} μg m⁻³")
    print(f"Max:  {ccomp_snap.max():.4f} μg m⁻³")
    # plt.show()
    plt.close(fig)


def main():
    ds    = nc.Dataset(FILE)
    times = ds.variables["time"][:]
    x     = ds.variables["x"][:]
    y     = ds.variables["y"][:]
    t_min, t_max = times[0], times[-1]
    print(f"\nAvailable time range: {t_min:.0f} – {t_max:.0f} s "
          f"({t_min/3600:.2f} – {t_max/3600:.2f} h)\n")

    for t_target, out_name in zip(TARGET_TIMES_S, OUT_NAMES):
        if t_target < t_min or t_target > t_max:
            raise ValueError(f"Requested time {t_target:.0f} s outside {t_min:.0f}–{t_max:.0f} s.")
        plot_snapshot(ds, times, x, y, t_target, out_name)

    ds.close()


if __name__ == "__main__":
    main()
