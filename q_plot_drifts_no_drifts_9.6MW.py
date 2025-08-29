# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 07:29:18 2025

@author: islam9
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os

#qpara plot for drifts (9.6MWKye0.26) and without drifts (9.6MWKye)


yyrb = np.array([-0.06506854, -0.0592725 , -0.04815738, -0.03801022, -0.02887471,
       -0.02078381, -0.01371695, -0.00760699, -0.0023902 ,  0.00980619,
        0.02988696,  0.04980043,  0.06879613,  0.08660054,  0.10325506,
        0.1195342 ,  0.13566592,  0.15188392,  0.1680489 ,  0.18418833,
        0.20069347,  0.21756767,  0.23471614,  0.25233707,  0.27125284,
        0.28119419])

q_para_drfts = np.array([1.16678510e+08, 3.73884657e+07, 3.76094880e+07, 4.55749434e+07,
       6.93144641e+07, 8.46917209e+07, 1.57773437e+08, 1.53709057e+08,
       6.28997050e+08, 5.69928134e+08, 3.09688486e+08, 2.76830852e+08,
       2.37226321e+08, 1.94244277e+08, 1.64682344e+08, 1.33436995e+08,
       1.11153118e+08, 9.33683585e+07, 7.93512056e+07, 6.67052609e+07,
       5.81971575e+07, 4.67709510e+07, 6.60515453e+07, 3.94395065e+07,
       1.14602056e+07, 1.61463928e+12])
q_perp_drift = np.array([ 705419.41966828,  705419.41966828,  645593.10254948,
        700173.43988001,  941779.62833715, 1120419.12538866,
       1875005.22763399, 1924779.65046336, 6997358.96652292,
       6366496.58715516, 4805277.18464765, 4970631.3680779 ,
       5122356.27813585, 5076216.47129682, 5088350.4223183 ,
       4917277.406667  , 4770984.79396518, 4619074.00782202,
       4495830.79556942, 4269016.41173246, 4066771.1237231 ,
       3506778.29648677, 3928138.15917255, 2934703.24676481,
       1442113.11209916, 1442113.11209916])

drift_path = r'C:\Users\islam9\OneDrive - LLNL\Desktop\NSTX_U_PET_2025\High_FX\D_diffusive_model\Power_scan_lambdaq_scan\heat_flux_profiles.csv'
data_drifts =pd.read_csv(drift_path)

# R-R_sep_mm,Ion_conduction_MW,Neutral_conduction_MW,Ion_convection_MW,Neutral_convection_MW,Electron_conduction_MW,Electron_convection_MW

r = data_drifts['R-R_sep_mm']

Ion_conduction_MW_drifts = data_drifts['Ion_conduction_MW']
Neutral_conduction_MW_drifts = data_drifts['Neutral_conduction_MW']
Ion_convection_MW_drifts = data_drifts['Ion_convection_MW']
Neutral_convection_MW_drifts = data_drifts['Neutral_convection_MW']
Electron_conduction_MW_drifts = data_drifts['Electron_conduction_MW']
Electron_convection_MW_drifts = data_drifts['Electron_convection_MW']

Convection_drifts= Electron_convection_MW_drifts  + Ion_convection_MW_drifts  + Neutral_convection_MW_drifts 
Conduction_drifts = Ion_conduction_MW_drifts  + Electron_conduction_MW_drifts + Neutral_conduction_MW_drifts 
total_drift = Convection_drifts + Conduction_drifts 

plt.figure(figsize= (4,2.5))
plt.plot(r, Conduction_drifts , label ='Drifts-Cond')
plt.plot(r, Convection_drifts , label ='Drifts-Conv')
plt.plot(r, total_drift, label='total-drift')
plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)')
plt.ylabel('q (MW)')
plt.grid()
plt.legend()
plt.ylim([0, np.max(total_drift)*1.05])
plt.show()
plt.savefig('q_conduction.png', dpi=300)


WO_drift_path = r'C:\Users\islam9\OneDrive - LLNL\Desktop\NSTX_U_PET_2025\High_FX\WO_Drifts\diffusive_model\Power_scan_lambdaq_scan\heat_flux_profiles.csv'

data = pd.read_csv(WO_drift_path)


r = data['R-R_sep_mm']

# Extract all components (without drifts)
Ion_conduction_MW_WO_drift     = data['Ion_conduction_MW']
Neutral_conduction_MW_WO_drift = data['Neutral_conduction_MW']
Ion_convection_MW_WO_drift     = data['Ion_convection_MW']
Neutral_convection_MW_WO_drift = data['Neutral_convection_MW']
Electron_conduction_MW_WO_drift = data['Electron_conduction_MW']
Electron_convection_MW_WO_drift = data['Electron_convection_MW']

# Compute conduction and convection totals
Convection_WO_drift = (
    Electron_convection_MW_WO_drift +
    Ion_convection_MW_WO_drift +
    Neutral_convection_MW_WO_drift
)

Conduction_WO_drift = (
    Ion_conduction_MW_WO_drift +
    Electron_conduction_MW_WO_drift +
    Neutral_conduction_MW_WO_drift
)

# Plot
plt.figure(figsize=(4, 2.5))
plt.plot(r, Conduction_WO_drift, label='WO-Drifts-Cond')
plt.plot(r, Convection_WO_drift, label='WO-Drifts-Conv')

# Optional: total without drifts
total_WO_drift = Conduction_WO_drift + Convection_WO_drift
plt.plot(r, total_WO_drift, label='WO-Drift Total')

plt.xlabel('r$_{omp}$ - r$_{sep}$ (mm)')
plt.ylabel('q (MW)')
plt.grid()
plt.legend()
plt.ylim([0, np.max(total_WO_drift) * 1.05])
plt.tight_layout()
plt.savefig('q_conduction.png', dpi=300)
plt.show()


plt.figure(figsize=(4.5, 3))

# With drifts
plt.plot(r, Conduction_drifts, label='Drifts - Conduction', color='tab:blue', linestyle='-')
plt.plot(r, Convection_drifts, label='Drifts - Convection', color='tab:orange', linestyle='-')
plt.plot(r, total_drift, label='Drifts - Total', color='tab:green', linestyle='-')

# Without drifts
plt.plot(r, Conduction_WO_drift, label='No Drifts - Conduction', color='tab:blue', linestyle='--')
plt.plot(r, Convection_WO_drift, label='No Drifts - Convection', color='tab:orange', linestyle='--')
plt.plot(r, total_WO_drift, label='No Drifts - Total', color='tab:green', linestyle='--')

# Labels and legend
plt.xlabel('r$_{omp}$ - r$_{sep}$ (mm)')
plt.ylabel('q (MW)')
plt.grid()

# Legend on top with 2 rows
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, fontsize=8, frameon=False)

plt.ylim([0, 1.05 * max(np.max(total_drift), np.max(total_WO_drift))])
plt.tight_layout()

# Save and show
plt.savefig('q_conduction_comparison.png', dpi=300)
plt.show()



fig, ax = plt.subplots(figsize=(4.5, 3))

l1, = ax.plot(r, Conduction_drifts, label='Conduction', color='tab:blue', linestyle='-')
l2, = ax.plot(r, Convection_drifts, label='Convection', color='tab:orange', linestyle='-')
l3, = ax.plot(r, total_drift, label='Total', color='tab:green', linestyle='-')

# -- No Drifts --
l4, = ax.plot(r, Conduction_WO_drift, label='Conduction', color='tab:blue', linestyle='--')
l5, = ax.plot(r, Convection_WO_drift, label='Convection', color='tab:orange', linestyle='--')
l6, = ax.plot(r, total_WO_drift, label='Total', color='tab:green', linestyle='--')


fig.subplots_adjust(top=0.72)


legend1 = fig.legend(
    handles=[l1, l2, l3],
    title='Drifts',
    loc='upper left',
    bbox_to_anchor=(0.1, 0.98),
    fontsize=8,
    title_fontsize=9,
  #  frameon=False
)

# -- No Drifts Legend (right)
legend2 = fig.legend(
    handles=[l4, l5, l6],
    title='No Drifts',
    loc='upper right',
    bbox_to_anchor=(0.9, 0.98),
    fontsize=8,
    title_fontsize=9,
   # frameon=False
)

ax.text(0.8, 9.6, 'Dotted lines: No Drifts', ha='center', va='top', fontsize=9)
ax.set_xlabel('r$_{omp}$ - r$_{sep}$ (mm)', fontsize=16)
ax.set_ylabel('q (MW)', fontsize=16)
ax.grid()

ax.set_ylim([0, 1.02 * max(np.max(total_drift), np.max(total_WO_drift))])
plt.savefig('q_conduction_comparison_grouped_legend_top_outside.png', dpi=300)
plt.show()



eV_to_J = 1.60218e-19  # 1 eV in Joules

def total_plasma_pressure(ne, Te, ni, Ti, u, m_i):
    Te_J = Te 
    Ti_J = Ti 
    p_e = ne * Te_J       # electron pressure [Pa]
    p_i = ni * Ti_J       # ion pressure [Pa]

    p_dyn = 0.5 * m_i * u**2 * ni  # [Pa]
    P_total = p_e + p_i + p_dyn
    
    return P_total




# Constants
eV_to_J = 1.60218e-19
m_D = 3.344e-27  # Deuterium ion mass [kg]
sep = 13         # Radial index for separatrix
bbb_ixmp = 47    # Poloidal index of OMP (e.g., midpoint of poloidal grid)

def load_fields(folder):
    ne = np.load(os.path.join(folder, 'ne.npy'))  # [m^-3]
    Te = np.load(os.path.join(folder, 'Te.npy'))  # [eV]
    ni = np.load(os.path.join(folder, 'ni.npy'))  # [m^-3]
    Ti = np.load(os.path.join(folder, 'Ti.npy'))  # [eV]
    u  = np.load(os.path.join(folder, 'up.npy'))   # [m/s]
    L  = np.load(os.path.join(folder, 'L.npy'))   # [m]
    return ne, Te, ni, Ti, u, L

def total_plasma_pressure(ne, Te, ni, Ti, u, m_i):
    Te_J = Te * eV_to_J
    Ti_J = Ti * eV_to_J
    p_e = ne * Te_J
    p_i = ni * Ti_J
    p_dyn = 0.5 * m_i * ni * u**2
    return p_e + p_i + p_dyn  # [Pa]

def plot_along_separatrix(L_drift, L_nodrift, val_drift, val_nodrift, name, unit='', scale=1.0, ylabel=None):
    plt.figure(figsize=(4, 2.75))
    plt.plot(L_drift[:, sep], val_drift[:, sep] / scale, label='With Drifts', color='tab:red')
    plt.plot(L_nodrift[:, sep], val_nodrift[:, sep] / scale, label='Without Drifts', color='tab:blue')
    
    omp_x = L_drift[bbb_ixmp, sep]
    ymax = max(np.max(val_drift[:, sep]), np.max(val_nodrift[:, sep])) / scale

    plt.axvline(omp_x, color='gray', linestyle='--')
    plt.text(omp_x * 1.02, ymax * 0.85, 'OMP', rotation=90, color='gray', fontsize=12)
    plt.xlim([0, np.max(L_drift[:, sep])])
    plt.xlabel('L$_{||}$ from Idiv [m]', fontsize=12)
    plt.ylabel(ylabel if ylabel else f'{name} [{unit}]', fontsize=12)
    plt.title(f'{name} along the Separatrix', fontsize=13)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


Drift_case = r'C:\Users\islam9\OneDrive - LLNL\Desktop\NSTX_U_PET_2025\High_FX\D_diffusive_model\Power_scan_lambdaq_scan'
No_Drift_case = r'C:\Users\islam9\OneDrive - LLNL\Desktop\NSTX_U_PET_2025\High_FX\WO_Drifts\diffusive_model\Power_scan_lambdaq_scan'

ne_drift, Te_drift, ni_drift, Ti_drift, u_drift, L_drift = load_fields(Drift_case)
ne_nodrift, Te_nodrift, ni_nodrift, Ti_nodrift, u_nodrift, L_nodrift = load_fields(No_Drift_case)

P_drift = total_plasma_pressure(ne_drift, Te_drift, ni_drift, Ti_drift, u_drift, m_D)
P_nodrift = total_plasma_pressure(ne_nodrift, Te_nodrift, ni_nodrift, Ti_nodrift, u_nodrift, m_D)


plot_along_separatrix(L_drift, L_nodrift, P_drift, P_nodrift, 'Total Pressure', unit='MPa', scale=1e6)
plot_along_separatrix(L_drift, L_nodrift, ne_drift, ne_nodrift, '$n_e$', unit='10$^{19}$ m$^{-3}$', scale=1e19)
plot_along_separatrix(L_drift, L_nodrift, Te_drift, Te_nodrift, '$T_e$', unit='eV')
plot_along_separatrix(L_drift, L_nodrift, ni_drift, ni_nodrift, '$n_i$', unit='10$^{19}$ m$^{-3}$', scale=1e19)
plot_along_separatrix(L_drift, L_nodrift, Ti_drift, Ti_nodrift, '$T_i$', unit='eV')
plot_along_separatrix(L_drift, L_nodrift, u_drift, u_nodrift, '$u_{||}$', unit='km/s', scale=1e3)


omp_x = L_drift[bbb_ixmp, sep]
Lx = L_drift[:, sep]
ne_d = ne_drift[:, sep]
Te_d = Te_drift[:, sep]
u_d  = u_drift[:, sep]

ne_nd = ne_nodrift[:, sep]
Te_nd = Te_nodrift[:, sep]
u_nd  = u_nodrift[:, sep]

# === Subplot ===
fig, axs = plt.subplots(3, 1, figsize=(4,6), sharex=True)

# 1. Electron density
axs[0].plot(Lx, ne_d/1e19, label='With Drifts', color='tab:red')
axs[0].plot(Lx, ne_nd/1e19, label='Without Drifts', color='tab:blue')
axs[0].axvline(omp_x, color='gray', linestyle='--')

axs[0].text(omp_x * 1.01, 0.9 * np.max(ne_d/1e19), 'OMP', rotation=90, color='gray')
axs[0].set_ylabel('$n_e$ [10$^{19}$ m$^{-3}$]', fontsize = 16)
axs[0].legend()
axs[0].grid(True)

# 2. Electron temperature
axs[1].plot(Lx, Te_d, color='tab:red')
axs[1].plot(Lx, Te_nd, color='tab:blue')
axs[1].axvline(omp_x, color='gray', linestyle='--')
axs[1].text(omp_x * 1.01, 0.9 * np.max(Te_d), 'OMP', rotation=90, color='gray')
axs[1].set_ylabel('$T_e$ [eV]', fontsize = 16)
axs[1].grid(True)

# 3. Parallel velocity
axs[2].plot(Lx, u_d/1e3, color='tab:red')
axs[2].plot(Lx, u_nd/1e3, color='tab:blue')
axs[2].axvline(omp_x, color='gray', linestyle='--')
axs[2].text(omp_x * 1.01, 0.9 * np.max(u_d/1e3), 'OMP', rotation=90, color='gray')
axs[2].set_ylabel('$u_{||}$ [km/s]', fontsize = 16)
axs[2].set_xlabel('L$_{||}$ from I-div [m]', fontsize = 16)
axs[2].grid(True)

# === Formatting ===
#plt.suptitle('Profiles Along the Separatrix (Radial Index {})'.format(sep), fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('ntU.png', dpi=300)
#plt.tight_layout()
plt.show()

