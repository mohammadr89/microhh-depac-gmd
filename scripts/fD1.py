import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Colorblind-safe color palette
COLORS = {
    'stable': '#0173B2',      # Blue
    'unstable': '#D55E00',    # Orange-red  
    'neutral': '#029E73',     # Green
    'data': '#CC79A7',        # Purple-pink
    'target': '#F0E442'       # Yellow
}

# Conversion constants
P = 101325
T = 293.15
MW_NH3 = 17.031
R = 8.314

def ugm3_to_molmol(c_ugm3):
    c_mol_m3 = c_ugm3 / (MW_NH3 * 1e6)
    rho_air = P / (R * T)
    return (c_mol_m3 / rho_air) * 1e9

def stability_correction_unstable(zeta):
    x = (1 - 16 * zeta) ** 0.25
    return 2 * np.log((1 + x**2) / 2)

def stability_correction_stable(zeta):
    return -5 * zeta

def monin_obukhov_factor(z1, z2, L):
    kappa = 0.4
    
    if abs(L) < 1e-10:
        return np.log(z2/z1) / kappa
    
    zeta1 = z1 / L
    zeta2 = z2 / L
    
    if zeta1 <= 0:
        psi1 = stability_correction_unstable(zeta1)
    else:
        psi1 = stability_correction_stable(zeta1)
        
    if zeta2 <= 0:
        psi2 = stability_correction_unstable(zeta2)
    else:
        psi2 = stability_correction_stable(zeta2)
    
    return (np.log(z2/z1) - psi2 + psi1) / kappa

def calculate_concentration_profile(z1, z2, c1, c2, L, heights):
    gradient_factor = monin_obukhov_factor(z1, z2, L)
    
    if abs(gradient_factor) > 1e-10:
        c_star = (c2 - c1) / gradient_factor
    else:
        c_star = 0.0
        
    factors = [monin_obukhov_factor(z1, z, L) for z in heights]
    concentrations = c1 + c_star * np.array(factors)
    
    return concentrations, c_star

def create_combined_validation_figure(dep_c1, dep_c2, em_c1, em_c2):

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Physical parameters
    z1, z2    = 8.0, 24.0   # Measurement heights (m)
    z_target  = 20.0         # Target height (m)
    heights   = np.linspace(1, 50, 100)

    L_values     = [50, 1e30, -20]
    panel_titles = ['(a) Stable', '(b) Neutral', '(c) Unstable']
    colors       = [COLORS['stable'], COLORS['neutral'], COLORS['unstable']]

    row_params = [
        (dep_c1, dep_c2, 'deposition scenario'),
        (em_c1,  em_c2,  'emission scenario'),
    ]

    for row, (c1, c2, scenario_name) in enumerate(row_params):

        neutral_conc, _ = calculate_concentration_profile(
            z1, z2, c1, c2, 1e30, heights)

        all_conc = []
        for L in L_values:
            conc, c_star = calculate_concentration_profile(
                z1, z2, c1, c2, L, heights)
            target_factor = monin_obukhov_factor(z1, z_target, L)
            c_target = c1 + c_star * target_factor
            all_conc.extend(conc)
            all_conc.append(c_target)
        x_max = max(max(all_conc), c1, c2) * 1.05

        for col, (L, ptitle, color) in enumerate(
                zip(L_values, panel_titles, colors)):
            ax = axes[row, col]

            conc, c_star = calculate_concentration_profile(
                z1, z2, c1, c2, L, heights)
            target_factor = monin_obukhov_factor(z1, z_target, L)
            c_target = c1 + c_star * target_factor

            ax.plot(conc, heights, color=color, linewidth=2.5,
                    label=f'MO profile ({ptitle.split()[1].lower()})')

            if L < 1e20:
                ax.plot(neutral_conc, heights, '--', color=COLORS['neutral'],
                        linewidth=1.5, alpha=0.7, label='Neutral reference')

            ax.scatter([c1, c2], [z1, z2], color=COLORS['data'], s=70,
                       marker='o', label='LES data points', zorder=5,
                       edgecolor='white', linewidth=0.8)

            ax.scatter(c_target, z_target, color=COLORS['target'], s=90,
                       marker='D', label='Target height', zorder=5,
                       edgecolor='black', linewidth=0.8)

            ax.set_xlabel('Mole fraction (nmol mol⁻¹)' if row == 1 else '')
            ax.tick_params(labelbottom=(row == 1))
            ax.set_ylabel('Height (m)' if col == 0 else '')
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, 50)
            ax.grid(True, alpha=0.3, linewidth=0.5)

            if row == 0:
                ax.set_title(ptitle, fontweight='bold')

            L_text = f'L = {L:.0f} m' if L < 1e20 else 'L → ∞'
            x_pos  = 0.98 if 'emission' in scenario_name else 0.05
            h_align = 'right' if 'emission' in scenario_name else 'left'
            ax.text(x_pos, 0.95, L_text, transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8),
                    verticalalignment='top',
                    horizontalalignment=h_align, fontsize=8)

        scenario_label = scenario_name.capitalize()
        axes[row, 2].annotate(
            scenario_label,
            xy=(1.03, 0.5), xycoords='axes fraction',
            fontsize=10, fontweight='bold', rotation=270,
            va='center', ha='left')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.01),
               ncol=4, fontsize=9, frameon=False)

    plt.tight_layout(rect=[0, 0.06, 0.97, 0.97])
    plt.subplots_adjust(hspace=0.12)
    return fig


def main():

    emission_c1,   emission_c2   = ugm3_to_molmol(7.1), ugm3_to_molmol(5.7)
    deposition_c1, deposition_c2 = ugm3_to_molmol(5.7), ugm3_to_molmol(7.1)

    fig = create_combined_validation_figure(
        deposition_c1, deposition_c2,
        emission_c1,   emission_c2)

    fig.savefig('fD1.png', dpi=300, bbox_inches='tight')
    fig.savefig('fD1.pdf', dpi=300, bbox_inches='tight')

    print("Saved fD1.png and fD1.pdf")
    print(f"  Deposition row: c = {deposition_c1:.2f} → {deposition_c2:.2f} nmol mol⁻¹")
    print(f"  Emission row  : c = {emission_c1:.2f} → {emission_c2:.2f} nmol mol⁻¹")
    print("  Measurement heights : 8.0 m and 24.0 m")
    print("  Target height       : 20.0 m")
    print("  Stability conditions: L = 50 m, ∞, −20 m")

if __name__ == "__main__":
    main()

