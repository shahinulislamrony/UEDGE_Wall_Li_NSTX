# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 08:07:55 2025

@author: islam9
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors



csv_file1 = r'C:\Users\islam9\OneDrive - LLNL\Desktop\NSTX_U_PET_2025\High_FX\lambdaq_comparison.csv'
csv_file2 = r'C:\Users\islam9\OneDrive - LLNL\Desktop\NSTX_U_PET_2025\High_FX\D_diffusive_model\Power_scan_lambdaq_scan\heat_flux_profiles.csv'
output_dir = r'C:\Users\islam9\OneDrive - LLNL\Desktop\NSTX_U_PET_2025\High_FX_old\D_diffusive_model\plots'
os.makedirs(output_dir, exist_ok=True)

# === Read data ===
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

# === Variable symbols ===
var_symbols = {
    "kye": r"$\chi_{i,e}$ (m$^2$/s)",
    "lambdaq_exp_o": r"$\lambda_{q}$ (mm)",
    "lambdaq_exp_i": r"$\lambda_{q}^{exp, i}$ (mm)",
    "lambdaq_eich_o": r"$\lambda_{q}^{eich, o}$ (mm)",
    "lambdaq_eich_in": r"$\lambda_{q}^{eich, i}$ (mm)",
    "Pcore": r"$P_{core}$ (MW)",
    "Psol": r"$P_{SOL}$ (MW)",
    "Podiv": r"$P_{Odiv}$ (MW)",
    "Pindiv": r"$P_{Idiv}$ (MW)",
    "Pwall": r"$P_{wall}$ (MW)",
    "Prad": r"$P_{rad}$ (MW)",
    "Ppfr": r"$P_{PFR}$ (MW)",
    "fniy_core": r"$\phi_{i,y}^{core}$ (10$^{22}$ /s)",
    "fniy_wall": r"$\phi_{wall}$ (10$^{22}$ /s)",
    "fnix_odiv": r"$\phi_{Odiv}$ (10$^{22}$ /s)",
    "fnix_idiv": r"$\phi_{Idiv}$ (10$^{22}$ /s)"
}


x_var = 'Psol'
variables = [col for col in df1.columns if col != x_var]


for var in variables:
    plt.figure(figsize=(5, 2.75))
    plt.scatter(df1[x_var], df1[var], color='b', marker='o', label='No-Drifts')
    plt.scatter(df2[x_var], df2[var], color='r', marker='^', label='Drifts')
    plt.xlabel(var_symbols.get(x_var, x_var), fontsize=12)
    plt.ylabel(var_symbols.get(var, var), fontsize=12)
    plt.title(f"{var_symbols.get(var, var)} vs {var_symbols.get(x_var, x_var)}", fontsize=14)
    plt.grid(True)
    plt.legend()
    ymax = max(df1[var].max(), df2[var].max())
    plt.ylim([0, ymax * 1.05])
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{var}_vs_{x_var}_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.show()

print(f"Comparison plots saved to {output_dir}")

# === Selected variables subplot comparison ===
selected_vars = ["lambdaq_exp_o"]

fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=True)
for ax, var in zip(axes, selected_vars):
    ax.scatter(df1[x_var], df1[var], color='b', marker='o', label='No-Drifts')
    ax.scatter(df2[x_var], df2[var], color='r', marker='^', label='Drifts')
    ax.set_xlabel(var_symbols.get(x_var, x_var), fontsize=12)
    ax.set_ylabel(var_symbols.get(var, var), fontsize=12)
    ax.set_title(f"{var_symbols.get(var, var)} vs {var_symbols.get(x_var, x_var)}", fontsize=10)
    ax.grid(True)
    ax.set_ylim([0, 5])
    ax.legend()

plt.tight_layout()
subplot_path = os.path.join(output_dir, 'subplots_lambdaq_vs_ncore_comparison.png')
plt.savefig(subplot_path, dpi=300)
plt.show()

print(f"Subplot comparison figure saved to {subplot_path}")



fig, axs = plt.subplots(3, 1, figsize=(4, 6), sharex=True)

# Wall
axs[0].plot(df1[x_var], df1['fniy_wall'], label='No Drifts', marker='o')
axs[0].plot(df2[x_var], df2['fniy_wall'], label='Drifts', linestyle='--', marker='s')
axs[0].set_ylabel(var_symbols["fniy_wall"])
axs[0].legend(fontsize=12, loc='center')
axs[0].set_ylim([0, 0.75])
axs[0].text(0.02, 0.85, "Wall", transform=axs[0].transAxes, fontsize=14, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Odiv
axs[1].plot(df1[x_var], df1['fnix_odiv'], label='With Drifts', marker='o')
axs[1].plot(df2[x_var], df2['fnix_odiv'], label='No Drifts', linestyle='--', marker='s')
axs[1].set_ylabel(var_symbols["fnix_odiv"])
axs[1].set_ylim([0, 30])
axs[1].text(0.02, 0.85, "Odiv", transform=axs[1].transAxes, fontsize=14, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Idiv
axs[2].plot(df1[x_var], df1['fnix_idiv'], label='With Drifts', marker='o')
axs[2].plot(df2[x_var], df2['fnix_idiv'], label='No Drifts', linestyle='--', marker='s')
axs[2].set_ylabel(var_symbols["fnix_idiv"])
axs[2].set_xlabel(var_symbols[x_var])
axs[2].set_ylim([0, 20])
axs[2].text(0.02, 0.85, "Idiv", transform=axs[2].transAxes, fontsize=14, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.tight_layout()
plt.show()



fig, axs = plt.subplots(3, 1, figsize=(4, 6), sharex=True)

# Wall
axs[0].plot(df1[x_var], df1['Pwall'], label='No Drifts', marker='o')
axs[0].plot(df2[x_var], df2['Pwall'], label='Drifts', linestyle='--', marker='s')
axs[0].set_ylabel(var_symbols["Pwall"])
#axs[0].legend(fontsize=12, loc='center')
axs[0].set_ylim([0, 1.5])
axs[0].text(0.02, 0.85, "Wall", transform=axs[0].transAxes, fontsize=14, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Odiv
axs[1].plot(df1[x_var], df1['Podiv'], label='With Drifts', marker='o')
axs[1].plot(df2[x_var], df2['Podiv'], label='No Drifts', linestyle='--', marker='s')
axs[1].set_ylabel(var_symbols["Podiv"])
axs[1].set_ylim([0, 5])
axs[1].text(0.02, 0.85, "Odiv", transform=axs[1].transAxes, fontsize=14, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Idiv
axs[2].plot(df1[x_var], df1['Pindiv'], label='With Drifts', marker='o')
axs[2].plot(df2[x_var], df2['Pindiv'], label='No Drifts', linestyle='--', marker='s')
axs[2].set_ylabel(var_symbols["Pindiv"])
axs[2].set_xlabel(var_symbols[x_var])
axs[2].set_ylim([0, 2])
axs[2].text(0.02, 0.85, "Idiv", transform=axs[2].transAxes, fontsize=14, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.tight_layout()
plt.show()



fig, axs = plt.subplots(3, 2, figsize=(7, 7), sharex=True)

axs[0, 0].plot(df1[x_var], df1['fniy_wall'], label='No Drifts', marker='o')
axs[0, 0].plot(df2[x_var], df2['fniy_wall'], label='Drifts', linestyle='--', marker='s')
axs[0, 0].set_ylabel(var_symbols["fniy_wall"], fontsize=14)
axs[0, 0].set_ylim([0, 0.75])
axs[0, 0].text(0.02, 0.85, "O-Wall", transform=axs[0, 0].transAxes, fontsize=14, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
axs[0, 0].legend(fontsize=12, loc='center')

# Right: Power
axs[0, 1].plot(df1[x_var], df1['Pwall'], label='No Drifts', marker='o')
axs[0, 1].plot(df2[x_var], df2['Pwall'], label='Drifts', linestyle='--', marker='s')
axs[0, 1].set_ylabel(var_symbols["Pwall"],fontsize=14)
axs[0, 1].set_ylim([0, 1.5])
axs[0, 1].text(0.02, 0.85, "o-Wall", transform=axs[0, 1].transAxes, fontsize=14, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# --- Row 2: Odiv ---
# Left: fn
axs[1, 0].plot(df1[x_var], df1['fnix_odiv'], label='With Drifts', marker='o')
axs[1, 0].plot(df2[x_var], df2['fnix_odiv'], label='No Drifts', linestyle='--', marker='s')
axs[1, 0].set_ylabel(var_symbols["fnix_odiv"],fontsize=14)
axs[1, 0].set_ylim([0, 30])
axs[1, 0].text(0.02, 0.85, "Odiv", transform=axs[1, 0].transAxes, fontsize=14, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Right: Power
axs[1, 1].plot(df1[x_var], df1['Podiv'], label='With Drifts', marker='o')
axs[1, 1].plot(df2[x_var], df2['Podiv'], label='No Drifts', linestyle='--', marker='s')
axs[1, 1].set_ylabel(var_symbols["Podiv"],fontsize=14)
axs[1, 1].set_ylim([0, 5])
axs[1, 1].text(0.02, 0.85, "Odiv", transform=axs[1, 1].transAxes, fontsize=14, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# --- Row 3: Idiv ---
# Left: fn
axs[2, 0].plot(df1[x_var], df1['fnix_idiv'], label='With Drifts', marker='o')
axs[2, 0].plot(df2[x_var], df2['fnix_idiv'], label='No Drifts', linestyle='--', marker='s')
axs[2, 0].set_ylabel(var_symbols["fnix_idiv"],fontsize=14)
axs[2, 0].set_ylim([0, 20])
axs[2, 0].set_xlabel(var_symbols[x_var],fontsize=14)
axs[2, 0].text(0.02, 0.85, "Idiv", transform=axs[2, 0].transAxes, fontsize=14, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Right: Power
axs[2, 1].plot(df1[x_var], df1['Pindiv'], label='With Drifts', marker='o')
axs[2, 1].plot(df2[x_var], df2['Pindiv'], label='No Drifts', linestyle='--', marker='s')
axs[2, 1].set_ylabel(var_symbols["Pindiv"],fontsize=14)
axs[2, 1].set_ylim([0, 2])
axs[2, 1].set_xlabel(var_symbols[x_var],fontsize=14)
axs[2, 1].text(0.02, 0.85, "Idiv", transform=axs[2, 1].transAxes, fontsize=14, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

for ax_row in axs:
    for ax in ax_row:
        ax.grid()
plt.tight_layout()
plt.savefig("Particle_power.png",  dpi=600)
plt.show()

csv_file1 = r'C:\Users\islam9\OneDrive - LLNL\Desktop\NERSC\Drift_highFX\PET_2025\High_FX\WO_Drifts\diffusive_model\Power_scan_lambdaq_scan\lambda_vs_psol.txt'
csv_file2 = r'C:\Users\islam9\OneDrive - LLNL\Desktop\NERSC\Drift_highFX\PET_2025\High_FX\Drifts\D_diffusive_model\Power_scan_lambdaq_scan\lambda_vs_psol.txt'

# Pcore array
Pmax = 8.2
Pstart = 2.2
inc = 0.2
Pcore = np.arange(Pstart, Pmax + inc, inc)


df1 = pd.read_csv(csv_file1, delim_whitespace=True, comment='#', names=['P_SOL', 'lambda_q', 'kye'])
df2 = pd.read_csv(csv_file2, delim_whitespace=True, comment='#', names=['P_SOL', 'lambda_q', 'kye'])

# Extract columns
psol1, lambda1, kye1 = df1['P_SOL'], df1['lambda_q'], df1['kye']
psol2, lambda2, kye2 = df2['P_SOL'], df2['lambda_q'], df2['kye']


lambda_all = np.concatenate([lambda1, lambda2])
norm = mcolors.Normalize(vmin=lambda_all.min(), vmax=lambda_all.max())
norm = mcolors.Normalize(vmin=1.8, vmax=1.9)
cmap = cm.plasma

plt.figure(figsize=(5, 3))
sc1 = plt.scatter(psol1, kye1, c=lambda1, cmap=cmap, norm=norm, marker='o', label='No-drifts', edgecolor='k', s=100)
sc2 = plt.scatter(psol2, kye2, c=lambda2, cmap=cmap, norm=norm, marker='^', label='Drifts', edgecolor='k',s=100)



plt.xlabel('P$_{SOL}$ [MW]', fontsize=14)
plt.ylabel(r'$\chi_{i,e}$ [m$^2$/s]', fontsize=14)
#plt.title(r'$\lambda_q$ vs P$_{SOL}$ comparison', fontsize=15)
plt.grid(True)
cbar = plt.colorbar(sc1)
cbar.set_label(r'$\chi_{i,e}$ [m$^2$/s]')
cbar.set_label(r'$\lambda_q$ [mm]')
plt.ylim([0, 1.3])
plt.xlim([0, 10])
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(5, 3), dpi=300) 
sc1 = ax.scatter(psol1, kye1, c=lambda1, cmap=cmap, norm=norm, marker='o', label='No-drifts', edgecolor='k', s=100)
sc2 = ax.scatter(psol2, kye2, c=lambda2, cmap=cmap, norm=norm, marker='^', label='Drifts', edgecolor='k', s=100)
ax.set_xlabel('P$_{SOL}$ [MW]', fontsize=14)
ax.set_ylabel(r'$\chi_{i,e}$ [m$^2$/s]', fontsize=14)

ax.set_xlim([0, 10])
ax.set_ylim([0, 1.3])
ax.legend(loc='upper left', fontsize=11, frameon=False)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=10)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

cbar = plt.colorbar(sc1, ax=ax, pad=0.02)
cbar.set_label(r'$\lambda_q$ [mm]', fontsize=13)
cbar.ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('chi_lambda_psol.png', dpi=600, bbox_inches='tight')  
plt.show()



A = 1.7
epsilon = 1/A
R = 0.93            
Bpol = 0.549       

P_SOL = np.linspace(1, 10e6, 100)  
P_SOL= 2.44

lambda_q = 1.35 * (epsilon ** 0.42) * (R ** 0.04) * (Bpol ** -0.92) * (P_SOL ** -0.02)

plt.figure(figsize=(4, 3))
plt.plot(P_SOL, lambda_q, label=r'$\lambda_q$ scaling', color='darkblue')
plt.xlabel(r'$P_{\mathrm{SOL}}$ [MW]', fontsize=16)
plt.ylabel(r'$\lambda_q$ [mm]', fontsize=16)
plt.title(r'$\lambda_q = 1.35\, \varepsilon^{0.42}\, R^{0.04}\, B_{\mathrm{pol}}^{-0.92}\, P_{\mathrm{SOL}}^{-0.02}$', fontsize=12)
plt.grid(True)
plt.ylim([0, 2])
plt.xlim([0, 10])
plt.tight_layout()
#plt.legend()
plt.show()
