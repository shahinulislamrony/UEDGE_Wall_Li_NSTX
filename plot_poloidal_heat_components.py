# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 21:18:16 2025

@author: islam9
"""
import numpy as np
import matplotlib.pyplot as plt
import os

omp_cell=74; oxpt=96; ixpt=8; separatrix_cell=8;
com_nx = 106

folder = r'C:\UEDGE_run_Shahinul\PET_2025\data_analysis\compare_drft_no_drifts'

npy_files = [f for f in os.listdir(folder) if f.endswith('.npy')]

for filename in npy_files:
    filepath = os.path.join(folder, filename)
    array = np.load(filepath)
    var_name = os.path.splitext(filename)[0]
    globals()[var_name] = array
    print(f"Variable '{var_name}' created, shape: {array.shape}")

# Check loaded variable names
print("\nLoaded variables:")
for filename in npy_files:
    print(os.path.splitext(filename)[0])


###total values

q_cond_el_ion = np.load(os.path.join(folder, "q_cond_el_ion.npy"))
q_conv_el_ion  = np.load(os.path.join(folder, "q_conv_el_ion.npy"))
total_el_ion     = np.load(os.path.join(folder, "total_el_ion.npy"))
q_uedge = np.load(os.path.join(folder, "q_uedge.npy"))


q_spitzer_pol = np.load(os.path.join(folder, "q_spitzer_drfits.npy"))
q_gradB_pol   = np.load(os.path.join(folder, "q_gradB_pol_drfits.npy"))
q_ExB_pol     = np.load(os.path.join(folder, "q_ExB_pol_drfits.npy"))
q_parall_current = np.load(os.path.join(folder, "q_parall_current.npy"))
q_ion         = np.load(os.path.join(folder, "q_ion.npy"))
q_ele         = np.load(os.path.join(folder, "q_ele.npy"))
qwtot         = np.load(os.path.join(folder, "qwtot_pol_drfits.npy"))
sx         = np.load(os.path.join(folder, "sx.npy"))
vol         = np.load(os.path.join(folder, "vol.npy"))
rr        = np.load(os.path.join(folder, "rr.npy"))
L        = np.load(os.path.join(folder, "L.npy"))

feex       = np.load(os.path.join(folder, "feex.npy"))
feix        = np.load(os.path.join(folder, "feix.npy"))
#fnixmu       = np.load(os.path.join(folder, "fnix_mu.npy"))
#seec       = np.load(os.path.join(folder, "seec.npy"))
#q_limit      = np.load(os.path.join(folder, "limit.npy"))
kin_ion      = np.load(os.path.join(folder, "Ke.npy"))



q_ExB_elec     = np.load(os.path.join(folder, "q_ExB_elec.npy"))
q_gradB_cbe       = np.load(os.path.join(folder, "q_gradB_cbe.npy"))
q_para_current       = np.load(os.path.join(folder, "q_para_current.npy"))
upe     = np.load(os.path.join(folder, "upe.npy"))
term_upe      = np.load(os.path.join(folder, "term_upe.npy"))
term_v2ce    = np.load(os.path.join(folder, "term_v2ce.npy"))
term_ve2cb      = np.load(os.path.join(folder, "term_ve2cb.npy"))
vex      = np.load(os.path.join(folder, "vex.npy"))
elec_conv_tot    = np.load(os.path.join(folder, "elec_conv_tot.npy"))
elec_conv    = np.load(os.path.join(folder, "q_conv_e.npy"))
elec_cond    = np.load(os.path.join(folder, "ele_cond.npy"))

q_drifts_sum = q_para_current + q_ExB_elec  + q_gradB_cbe  + elec_conv + elec_cond


# --- Load saved arrays ---
qion_gradb       = np.load(os.path.join(folder, "qion_gradb.npy"))
qion_EB          = np.load(os.path.join(folder, "qion_EB.npy"))
qion_conv        = np.load(os.path.join(folder, "qion_conv.npy"))
q_ion            = np.load(os.path.join(folder, "q_ion.npy"))
qion_sum         = np.load(os.path.join(folder, "qion_sum.npy"))
icond            = np.load(os.path.join(folder, "icond.npy"))
feix             = np.load(os.path.join(folder, "feix.npy"))

# --- Plot ---np.save("q_parall_current.npy", q_parall_current)


q_gradB_el      = np.load(os.path.join(folder, "q_gradB_el.npy"))
q_eb_el         = np.load(os.path.join(folder, "q_eb_el.npy"))
q_conv_ele       = np.load(os.path.join(folder, "q_conv_ele.npy"))
q_parall_current = np.load(os.path.join(folder, "q_parall_current.npy"))
econd            = np.load(os.path.join(folder, "econd.npy"))
econv             = np.load(os.path.join(folder, "econv.npy"))
econd  = econd *-1

q_conv_ele =  econv  - (q_gradB_el+q_eb_el +q_para_current)

nx, ny = q_spitzer_pol.shape
poloidal_indices = np.arange(nx)
poloidal_indices = L[:,9]
mid_idx = 9


plt.figure(figsize=(5, 3.5))
plt.plot(poloidal_indices[1:-1], qion_gradb[1:-1, mid_idx] / 1e6, label='gradB', color='green')
plt.plot(poloidal_indices[1:-1], qion_EB[1:-1, mid_idx]    / 1e6, label='E×B',   color='magenta')
plt.plot(poloidal_indices[1:-1], qion_conv[1:-1, mid_idx]  / 1e6, label='2.5nTu', color='blue')
plt.plot(poloidal_indices[1:-1], icond[1:-1, mid_idx]      / 1e6, label='Cond',  color='orange')
plt.plot(poloidal_indices[1:-1], feix[1:-1, mid_idx]       / 1e6, label='feix',  color='black')
plt.plot(poloidal_indices[1:-1], qion_sum[1:-1, mid_idx]   / 1e6, label='sum',   linestyle='--', color='red')

plt.xlabel('Poloidal grid index', fontsize=16)
plt.ylabel(r'$q_\mathrm{poloidal}^{ion}$ [MW]', fontsize=16)
plt.axvline(poloidal_indices[len(poloidal_indices)//2], color='black', linestyle=':', linewidth=2)  # fallback if bbb.ixmp not saved
plt.legend(fontsize=12, ncol=2)
plt.xlim([0, 65])
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("poloidal_heat_flux_ion_reloaded.png", dpi=300)
plt.show()




fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

# --- Ion fluxes ---
ax1.plot(poloidal_indices[1:-1], qion_gradb[1:-1, mid_idx]/1e6, label='gradB', color='green')
ax1.plot(poloidal_indices[1:-1], qion_EB[1:-1, mid_idx]/1e6, label='E×B', color='magenta')
ax1.plot(poloidal_indices[1:-1], qion_conv[1:-1, mid_idx]/1e6, label='2.5nTu', color='blue')
ax1.plot(poloidal_indices[1:-1], icond[1:-1, mid_idx]/1e6, label='Cond', color='orange')
ax1.plot(poloidal_indices[1:-1], feix[1:-1, mid_idx]/1e6, label='feix', color='black')
ax1.plot(poloidal_indices[1:-1], qion_sum[1:-1, mid_idx]/1e6, label='sum', linestyle='--', color='red')
ax1.axvline(poloidal_indices[len(poloidal_indices)//2], color='black', linestyle=':', linewidth=2)

ax1.set_ylabel(r'$q_\mathrm{ion}$ [MW]', fontsize=14)
ax1.legend(fontsize=10, ncol=2)
ax1.grid(True)

sum_el = q_gradB_el +  q_eb_el + q_conv_ele + econd + q_para_current
# --- Electron fluxes ---
ax2.plot(poloidal_indices[1:-1], q_gradB_el[1:-1, mid_idx]/1e6, label='gradB', color='green')
ax2.plot(poloidal_indices[1:-1], q_eb_el[1:-1, mid_idx]/1e6, label='E×B', color='magenta')
ax2.plot(poloidal_indices[1:-1], q_conv_ele[1:-1, mid_idx]/1e6, label='2.5nTu', color='blue')
ax2.plot(poloidal_indices[1:-1], econd[1:-1, mid_idx]/1e6, label='Cond', color='orange')
ax2.plot(poloidal_indices[1:-1], feex[1:-1, mid_idx]/1e6, label='Conv', color='black')
ax2.plot(poloidal_indices[1:-1], q_parall_current[1:-1, mid_idx]/1e6, label='j∥', linestyle='--', color='cyan')
ax2.plot(poloidal_indices[1:-1], sum_el[1:-1, mid_idx]/1e6, label='Sum', linestyle='--', color='red')
ax2.axvline(poloidal_indices[len(poloidal_indices)//2], color='black', linestyle=':', linewidth=2)

ax2.set_xlabel('Poloidal grid index', fontsize=14)
ax2.set_ylabel(r'$q_\mathrm{electron}$ [MW]', fontsize=14)
ax2.legend(fontsize=10, ncol=2)
ax2.grid(True)

# --- Layout ---
plt.xlim([0, 65])
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig("poloidal_heat_flux_ions_electrons.png", dpi=300)
plt.show()




fig, axes = plt.subplots(3, 1, figsize=(4.5, 7), sharex=True)

# -------------------------------------------------------------------------
# (1) Conduction vs convection (electrons+ions total, your first block)
# -------------------------------------------------------------------------
ax = axes[0]
ax.plot(poloidal_indices[1:-1], q_cond_el_ion[1:-1], label=r'$q_{\mathrm{Conduction}}$', color='orange', linewidth=2)
ax.plot(poloidal_indices[1:-1], q_conv_el_ion[1:-1], label=r'$q_{\mathrm{Convection}}$', color='blue', linewidth=2)
ax.plot(poloidal_indices[1:-1], total_el_ion[1:-1], label=r'$q_{\mathrm{sum}}$', color='red', linewidth=2)
ax.plot(poloidal_indices[1:-1], q_uedge[1:-1], label=r'$q_{\mathrm{UEDGE}}$', color='black', linestyle='--', linewidth=2)

ax.axvline(poloidal_indices[74], color='black', linestyle=':', linewidth=2)  # vertical marker
ax.text(poloidal_indices[74], 0.6, "OMP", color='red', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel(r'$q_{\mathrm{total}}^{Pol}$ [MW]', fontsize=16)
ax.legend(fontsize=12, loc='best')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=12, length=7)
ax.tick_params(axis='both', which='minor', labelsize=10, length=4)
ax.minorticks_on()

# -------------------------------------------------------------------------
# (2) Ion heat fluxes
# -------------------------------------------------------------------------
ax1 = axes[1]
ax1.plot(poloidal_indices[1:-1], qion_gradb[1:-1, mid_idx]/1e6, label=r'$\nabla$B', color='green')
ax1.plot(poloidal_indices[1:-1], qion_EB[1:-1, mid_idx]/1e6, label='E×B', color='magenta')
ax1.plot(poloidal_indices[1:-1], qion_conv[1:-1, mid_idx]/1e6, label='2.5nTu', color='blue')
ax1.plot(poloidal_indices[1:-1], icond[1:-1, mid_idx]/1e6, label='Cond', color='orange')
ax1.plot(poloidal_indices[1:-1], feix[1:-1, mid_idx]/1e6, label='feix', color='black')
ax1.plot(poloidal_indices[1:-1], qion_sum[1:-1, mid_idx]/1e6, label='sum', linestyle='--', color='red')
ax1.axvline(poloidal_indices[74], color='black', linestyle=':', linewidth=2)  # vertical marker

ax1.set_ylabel(r'$q_\mathrm{ion}^{Pol}$ [MW]', fontsize=16)
ax1.legend(fontsize=10, ncol=2)
ax1.grid(True)

# -------------------------------------------------------------------------
# (3) Electron heat fluxes
# -------------------------------------------------------------------------
sum_el = q_gradB_el + q_eb_el + q_conv_ele + econd + q_parall_current

ax2 = axes[2]
ax2.plot(poloidal_indices[1:-1], q_gradB_el[1:-1, mid_idx]/1e6, label=r'$\nabla$B', color='green')
ax2.plot(poloidal_indices[1:-1], q_eb_el[1:-1, mid_idx]/1e6, label='E×B', color='magenta')
ax2.plot(poloidal_indices[1:-1], q_conv_ele[1:-1, mid_idx]/1e6, label='2.5nTu', color='blue')
ax2.plot(poloidal_indices[1:-1], econd[1:-1, mid_idx]/1e6, label='Cond', color='orange')
ax2.plot(poloidal_indices[1:-1], q_parall_current[1:-1, mid_idx]/1e6, label=r'Current', linestyle='--', color='cyan')
ax2.plot(poloidal_indices[1:-1], feex[1:-1, mid_idx]/1e6, label='feex', color='black')
ax2.plot(poloidal_indices[1:-1], sum_el[1:-1, mid_idx]/1e6, label='Sum', linestyle='--', color='red')
ax2.axvline(poloidal_indices[74], color='black', linestyle=':', linewidth=2)  # vertical marker


ax2.set_xlabel('L$_{||}$ from Idiv [m]', fontsize=16)
ax2.set_ylabel(r'$q_\mathrm{electron}^{Pol}$ [MW]', fontsize=16)
ax2.legend(fontsize=8, ncol=2)
ax2.grid(True)

# -------------------------------------------------------------------------
# Layout & save
# -------------------------------------------------------------------------
plt.xlim([0, 65])
plt.tight_layout()
plt.savefig("poloidal_heat_flux_3panel.png", dpi=300)
plt.show()



fig, ax = plt.subplots(figsize=(5., 3.0))

ax.plot(poloidal_indices[1:-1], q_gradB_cbe[1:-1, mid_idx] / 1e6, label=r'$q_{\nabla B}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], q_ExB_elec[1:-1, mid_idx] / 1e6, label=r'$q_{E \times B}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], q_para_current[1:-1, mid_idx] / 1e6, label=r'$q_{Current}$', linewidth=2)

ax.plot(poloidal_indices[1:-1], elec_conv[1:-1, mid_idx] / 1e6, label='q$_{2.5*nuT}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], elec_cond[1:-1, mid_idx] / 1e6, label='q$_{cond}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], q_drifts_sum[1:-1, mid_idx] / 1e6, label='q$_{sum}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], feex[1:-1, mid_idx] / 1e6, label='q$_{feex}$', linewidth=2, linestyle='--')


ymax = ax.get_ylim()[1]

#if omp_cell is not None:
##    ax.axvline(omp_cell - 0, color='red', linestyle='--', linewidth=2)
 #   ax.text(omp_cell - 1, ymax, "OMP", color='red', ha='center', va='bottom', fontweight='bold')

#if oxpt is not None:
#    ax.axvline(oxpt - 0, color='orange', linestyle='--', linewidth=2)
#    ax.text(oxpt + 4,  ymax, "Oxpt", color='orange', ha='center', va='bottom', fontweight='bold')

#if ixpt is not None:
#    ax.axvline(ixpt - 0, color='purple', linestyle='--', linewidth=2)
#    ax.text(ixpt + 4,  ymax, "Ixpt", color='purple', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Poloidal grid index', fontsize=16)
ax.axvline(poloidal_indices[74], color='black', linestyle=':', linewidth=2)  # Comment if not needed
ax.text(poloidal_indices[74], ymax, "OMP", color='red', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel(r'q$_{Poloidal}$ [MW]', fontsize=16)
ax.legend(fontsize=9, ncol=2)
#ax.set_xlim([1, 104])
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(folder, "poloidal_heat_flux.png"), dpi=300)
plt.show()


bbb_ixmp = 74
com_ixpt2 = 96
com_ixpt1 = 8

x_mm = L[:, mid_idx] 

omp_cell = L[bbb_ixmp]
oxpt     = L[com_ixpt2] 
ixpt     = L[com_ixpt1] 

# === Plot settings ===
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.2
})

fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

# ================================
# 1st subplot — conduction vs convection vs total
# ================================
ax = axes[0]
ax.plot(x_mm, q_cond_el_ion,  label=r'$q_{\mathrm{Conduction}}$', color='green', linewidth=2)
ax.plot(x_mm, q_conv_el_ion ,  label=r'$q_{\mathrm{Convection}}$', color='black', linewidth=2)
ax.plot(x_mm, total_el_ion,   label=r'$q_{\mathrm{sum}}$',        color='red', linewidth=2)
ax.plot(x_mm, q_uedge, label=r'$q_{\mathrm{UEDGE}}$',      color='blue', linestyle='--', linewidth=2)

ax.set_ylabel(r'$q_{\mathrm{poloidal}}$ [MW]', fontsize=16)
ax.axvline(poloidal_indices[74], color='black', linestyle=':', linewidth=2)  # Comment if not needed
ax.text(poloidal_indices[74], ymax, "OMP", color='red', ha='center', va='bottom', fontweight='bold')
ax.legend(fontsize=12,  loc='best')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=12, length=7)
ax.tick_params(axis='both', which='minor', labelsize=10, length=4)
ax.minorticks_on()

# Mark OMP, Oxpt, Ixpt
ymax = ax.get_ylim()[1]
marker_kwargs = dict(linestyle='--', linewidth=1.5)
#ax.axvline(omp_cell, color='red', **marker_kwargs)
#ax.text(omp_cell, ymax, "OMP", color='red', ha='center', va='bottom', fontweight='bold')

#ax.axvline(oxpt, color='orange', **marker_kwargs)
#ax.text(oxpt + 0.5, ymax, "Oxpt", color='orange', ha='center', va='bottom', fontweight='bold')

##ax.axvline(ixpt, color='purple', **marker_kwargs)
#ax.text(ixpt + 0.5, ymax, "Ixpt", color='purple', ha='center', va='bottom', fontweight='bold')

ax = axes[1]
ax.plot(x_mm, q_gradB_cbe[:, mid_idx] / 1e6, label=r'$q_{\nabla B}$', linewidth=2)
ax.plot(x_mm, q_ExB_elec[:, mid_idx] / 1e6, label=r'$q_{E \times B}$', linewidth=2)
ax.plot(x_mm, q_para_current[:, mid_idx] / 1e6, label=r'$q_{Current}$', linewidth=2)

ax.plot(x_mm, elec_conv[:, mid_idx] / 1e6, label='q$_{2.5*nuT}$', linewidth=2)
ax.plot(x_mm, elec_cond[:, mid_idx] / 1e6, label=r'$q_{cond}$', linewidth=2)
ax.plot(x_mm, q_drifts_sum[:, mid_idx] / 1e6, label='q$_{sum}$', linewidth=2)


ax.plot(x_mm, feex[:, mid_idx] / 1e6, label='q$_{feex}$', linewidth=2, linestyle='--')

ax.set_xlabel(r'$L_{||}$ from I div [m]', fontsize=16)
ax.set_ylabel(r'$q_{\mathrm{e-poloidal}}^{\mathrm{Convec}}$ [MW]', fontsize=16)
ax.legend(fontsize=10, ncol=2, loc='best')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=12, length=7)
ax.tick_params(axis='both', which='minor', labelsize=10, length=4)
ax.minorticks_on()


plt.tight_layout()
plt.savefig('poloidal_flux_subplots.png', dpi=600, bbox_inches='tight')
plt.show()



import matplotlib.pyplot as plt

# ================================
# Marker / location definitions
# ================================
bbb_ixmp = 74
com_ixpt2 = 96
com_ixpt1 = 8

x_mm = L[:, mid_idx]

omp_cell = L[bbb_ixmp]
oxpt     = L[com_ixpt2]
ixpt     = L[com_ixpt1]

# ================================
# Plot settings
# ================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.2
})

# High-contrast, colorblind-friendly palette
colors = {
    "cond":      "green",  # Teal
    "conv":      "#d95f02",  # Orange
    "sum":       "#7570b3",  # Purple
    "uedge":     "#e7298a",  # Magenta
    "gradB":     "blue",  # Green
    "ExB":       "purple",  # Gold
    "current":   "red",  # Brown
    "nuT":       "#1f78b4",  # Blue
    "cond_elec": "green",  # Light green
    "sum_drift": "#ff7f00",  # Bright orange
    "feex":      "#6a3d9a"   # Deep purple
}

fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

# ================================
# 1st subplot — conduction vs convection vs total
# ================================
ax = axes[0]
ax.plot(x_mm, q_cond_el_ion, label=r'$q_{\mathrm{Conduction}}$',
        color=colors["cond"], linewidth=2)
ax.plot(x_mm, q_conv_el_ion, label=r'$q_{\mathrm{Convection}}$',
        color=colors["conv"], linewidth=2)
ax.plot(x_mm, total_el_ion, label=r'$q_{\mathrm{sum}}$',
        color=colors["sum"], linewidth=2)
ax.plot(x_mm, q_uedge, label=r'$q_{\mathrm{UEDGE}}$',
        color=colors["uedge"], linestyle='--', linewidth=2)

ax.axvline(poloidal_indices[74], color='black', linestyle=':', linewidth=2)  # Comment if not needed
ax.text(poloidal_indices[74], ymax, "OMP", color='red', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel(r'$q_{\mathrm{poloidal}}$ [MW]', fontsize=18)
ax.legend(fontsize=16, loc='best')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=12, length=7)
ax.tick_params(axis='both', which='minor', labelsize=10, length=4)
ax.minorticks_on()


ax = axes[1]
ax.plot(x_mm, q_gradB_cbe[:, mid_idx] / 1e6, label=r'$q_{\nabla B}$',
        color=colors["gradB"], linewidth=2)
ax.plot(x_mm, q_ExB_elec[:, mid_idx] / 1e6, label=r'$q_{E \times B}$',
        color=colors["ExB"], linewidth=2)
ax.plot(x_mm, q_para_current[:, mid_idx] / 1e6, label=r'$q_{\mathrm{Current}}$',
        color=colors["current"], linewidth=2)

ax.plot(x_mm, elec_conv[:, mid_idx] / 1e6, label=r'$q_{2.5nuT}$',
        color=colors["nuT"], linewidth=2)
ax.plot(x_mm, elec_cond[:, mid_idx] / 1e6, label=r'$q_{\mathrm{cond}}$',
        color=colors["cond_elec"], linewidth=2)
ax.plot(x_mm, q_drifts_sum[:, mid_idx] / 1e6, label=r'$q_{\mathrm{sum}}$',
        color=colors["sum_drift"], linewidth=2)

ax.plot(x_mm, feex[:, mid_idx] / 1e6, label=r'$q_{\mathrm{feex}}$',
        color=colors["feex"], linewidth=2, linestyle='--')
ax.axvline(poloidal_indices[74], color='black', linestyle=':', linewidth=2) 
ax.set_xlabel(r'$L_{||}$ from I div [m]', fontsize=18)
ax.set_ylabel(r'$q_{\mathrm{poloidal}}^{\mathrm{Electron}}$ [MW]', fontsize=18)
ax.legend(fontsize=12, ncol=3, loc='best')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=12, length=7)
ax.tick_params(axis='both', which='minor', labelsize=10, length=4)
ax.minorticks_on()


plt.tight_layout()
plt.savefig('poloidal_flux_subplots.png', dpi=600, bbox_inches='tight')
plt.show()



# %%
q_spitzer_pol = q_spitzer_pol

q_gradB_pol   = q_gradB_pol
q_ExB_pol     = q_ExB_pol 
q_parall_current =q_parall_current
qwtot         = qwtot 
q_ion        =q_ion 
q_ele        =q_ele 

qwtot         = np.load(os.path.join(folder, "fetx.npy"))
# %%
total =  q_ion + q_spitzer_pol + q_gradB_pol + q_ExB_pol  + q_parall_current   +q_ele 


nx, ny = q_spitzer_pol.shape
poloidal_indices = np.arange(nx)
#poloidal_indices = L[:,9]
mid_idx = 10

fig, ax = plt.subplots(figsize=(6., 4.0))

ax.plot(poloidal_indices[1:-1], q_spitzer_pol[1:-1, mid_idx] / 1e6, label=r'$q_{Spit.}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], q_gradB_pol[1:-1, mid_idx] / 1e6, label=r'$q_{\nabla B}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], q_ExB_pol[1:-1, mid_idx] / 1e6, label=r'$q_{E \times B}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], q_parall_current[1:-1, mid_idx] / 1e6, label=r'$q_{Current}$', linewidth=2, linestyle='--')
ax.plot(poloidal_indices[1:-1], q_ion[1:-1, mid_idx] / 1e6, label='q$_{ion}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], q_ele[1:-1, mid_idx] / 1e6, label='q$_{ele}$', linewidth=2)
#ax.plot(poloidal_indices[1:-1], kin_ion[1:-1, mid_idx] / 1e6, label='q$_{0.5mu*2*fnix}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], qwtot[1:-1, mid_idx] / 1e6, label='feex+feix', linewidth=2)
#ax.plot(poloidal_indices[1:-1], total[1:-1, mid_idx] / 1e6, label='sum', linewidth=2, color ='cyan', linestyle='--')

ymax = ax.get_ylim()[1]

if omp_cell is not None:
    ax.axvline(omp_cell - 0, color='red', linestyle='--', linewidth=2)
    ax.text(omp_cell - 1, ymax, "OMP", color='red', ha='center', va='bottom', fontweight='bold')

if oxpt is not None:
    ax.axvline(oxpt - 0, color='orange', linestyle='--', linewidth=2)
    ax.text(oxpt + 4,  ymax, "Oxpt", color='orange', ha='center', va='bottom', fontweight='bold')

if ixpt is not None:
    ax.axvline(ixpt - 0, color='purple', linestyle='--', linewidth=2)
    ax.text(ixpt + 4,  ymax, "Ixpt", color='purple', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Poloidal grid index', fontsize=16)
ax.axvline(poloidal_indices[74], color='black', linestyle=':', linewidth=2)  # Comment if not needed
ax.set_ylabel(r'q$_{Poloidal}$ [MW]', fontsize=16)
ax.legend(fontsize=12, ncol=2)
ax.set_xlim([1, 104])
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(folder, "poloidal_heat_flux.png"), dpi=300)
plt.show()



mid_indices = [8, 9, 10, 11]  # mid_idx values

fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.5), sharex=True, sharey=True)
axs = axs.flatten()

for i, (ax, mid_idx) in enumerate(zip(axs, mid_indices)):
    ax.plot(poloidal_indices[1:-1], q_spitzer_pol[1:-1, mid_idx]/1e6, label=r'$q_{Spitzer}$', linewidth=2)
    ax.plot(poloidal_indices[1:-1], q_gradB_pol[1:-1, mid_idx]/1e6, label=r'$q_{\nabla B}$', linewidth=2)
    ax.plot(poloidal_indices[1:-1], q_ExB_pol[1:-1, mid_idx]/1e6, label=r'$q_{E \times B}$', linewidth=2)
    ax.plot(poloidal_indices[1:-1], q_parall_current[1:-1, mid_idx]/1e6, label=r'$q_{Current}$', linewidth=2, linestyle='--')
    ax.plot(poloidal_indices[1:-1], q_ion[1:-1, mid_idx] / 1e6, label='q$_{ion}$', linewidth=2)
    ax.plot(poloidal_indices[1:-1], qwtot[1:-1, mid_idx]/1e6, label='q$_{UEDGE}$', linewidth=2)

    #ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.axvline(poloidal_indices[74], color='black', linestyle='--', linewidth=1)
    ax.set_xlim([0, 105])
    ax.grid(True)

    # Annotation inside plot (top right)
    ax.text(
        0.3, 0.95, f"FT = {mid_idx}",
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=10, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )
    ax.tick_params(axis='both', which='major', labelsize=12) 

    # Y-axis label for left column
    if i % 2 == 0:
        ax.set_ylabel(r'q$_{Poloidal}$ [MW/m$^2$]', fontsize=16)

    # X-axis label for bottom row
    if i // 2 == 1:
        ax.set_xlabel('Poloidal grid index', fontsize=16)

    if i == 0:
        ax.legend(fontsize=8, ncol=1, loc='best', frameon=True)

ymax = ax.get_ylim()[1]

if omp_cell is not None:
    ax.axvline(omp_cell - 0, color='black', linestyle='--', linewidth=1)
    ax.text(omp_cell - 10, 0.3, "OMP", color='black', ha='center', va='bottom', fontweight='bold')

if oxpt is not None:
    ax.axvline(oxpt - 0, color='orange', linestyle='--', linewidth=1)
    ax.text(oxpt - 8, 0.3, "Oxpt", color='orange', ha='center', va='bottom', fontweight='bold')

if ixpt is not None:
    ax.axvline(ixpt - 0, color='purple', linestyle='--', linewidth=1)
    ax.text(ixpt + 8, 0.3, "Ixpt", color='purple', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("poloidal_heat_flux_subplots.png", dpi=300)
plt.show()

#L = np.load(os.path.join(folder, "L.npy"))

#poloidal_indices = L[:,9]

start, end = 73, 106

plt.figure(figsize=(5.0, 3.0))
plt.plot(poloidal_indices[start:end], q_spitzer_pol[start:end, mid_idx]/1e6, label=r'$q_{Spitzer}$', linewidth=2)
plt.plot(poloidal_indices[start:end], q_gradB_pol[start:end, mid_idx]/1e6, label=r'$q_{\nabla B}$', linewidth=2)
plt.plot(poloidal_indices[start:end], q_ExB_pol[start:end, mid_idx]/1e6, label=r'$q_{E \times B}$', linewidth=2)
ax.plot(poloidal_indices[start:end], q_ion[start:end, mid_idx] / 1e6, label='q$_{ion}$', linewidth=2)
plt.plot(poloidal_indices[start:end], q_parall_current[start:end, mid_idx]/1e6, label=r'$q_{Current}$', linewidth=2, linestyle='--')

plt.plot(poloidal_indices[start:end], qwtot[start:end, mid_idx]/1e6, label='q$_{UEDGE}$', linewidth=2)


#if omp_cell is not None:
   # plt.axvline(omp_cell - 0, color='black', linestyle='--', linewidth=1)
 #   plt.text(omp_cell + 1, 40, "OMP", color='black', ha='center', va='bottom', fontweight='bold')

#if oxpt is not None:
   # plt.axvline(oxpt - 0, color='orange', linestyle='--', linewidth=1)
   # plt.text(oxpt + 2, 20, "Oxpt", color='orange', ha='center', va='bottom', fontweight='bold')

    

plt.xlabel('Poloidal grid index', fontsize=16)
plt.ylabel('q$_{Poloidal}$ [MW]', fontsize=16)
plt.legend(fontsize=10, ncol=2)
plt.xlim([poloidal_indices[start], poloidal_indices[end-1]])
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True)
#plt.ylim([0, 40])
#plt.yscale('symlog', linthresh=1e-3)
plt.tight_layout()
plt.savefig("poloidal_heat_flux_74_106.png", dpi=300)
plt.show()



mid_idx = 9

start_left, end_left = 1, 73   # Left plot range (indices)
start_right, end_right = 73, 106  # Right plot range (indices)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(7.0, 2.75))


ax1.plot(poloidal_indices[start_left:end_left], q_spitzer_pol[start_left:end_left, mid_idx]/1e6, label=r'$q_{Spitzer}$', linewidth=2)
ax1.plot(poloidal_indices[start_left:end_left], q_gradB_pol[start_left:end_left, mid_idx]/1e6, label=r'$q_{\nabla B}$', linewidth=2)
ax1.plot(poloidal_indices[start_left:end_left], q_ExB_pol[start_left:end_left, mid_idx]/1e6, label=r'$q_{E \times B}$', linewidth=2)
ax1.plot(poloidal_indices[start_left:end_left],  q_ion[start_left:end_left, mid_idx] / 1e6, label='q$_{ion}$', linewidth=2)
ax1.plot(poloidal_indices[start_left:end_left], q_parall_current[start_left:end_left, mid_idx]/1e6, label=r'$q_{Current}$', linewidth=2, linestyle='--')
ax1.plot(poloidal_indices[start_left:end_left], qwtot[start_left:end_left, mid_idx]/1e6, label='q$_{UEDGE}$', linewidth=2)

if omp_cell is not None and start_left <= omp_cell < end_left:
    ax1.axvline(omp_cell, color='black', linestyle='--', linewidth=1)
    ax1.text(omp_cell + 1, ax1.get_ylim()[1]*1.05, "OMP", color='black', ha='center', va='bottom', fontweight='bold')

if oxpt is not None and start_left <= oxpt < end_left:
    ax1.axvline(oxpt, color='orange', linestyle='--', linewidth=1)
    ax1.text(oxpt + 1, ax1.get_ylim()[1]*1.05, "Oxpt", color='orange', ha='center', va='bottom', fontweight='bold')
    
if ixpt is not None:
    ax1.axvline(ixpt - 0, color='purple', linestyle='--', linewidth=1)
    ax1.text(ixpt + 2, 0.5, "Ixpt", color='purple', ha='center', va='bottom', fontweight='bold')

ax1.set_xlabel('Poloidal grid index', fontsize=14)
ax1.set_ylabel('q$_{Poloidal}$ [MW]', fontsize=14)
ax1.set_xlim([poloidal_indices[start_left], poloidal_indices[end_left-1]])
ax1.set_ylim([-0.3, 0.5])
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True)
ax1.legend(fontsize=9, ncol=2, loc='upper right')


ax1.set_title('Indiv to OMP')

# Right subplot: indices 73 to 106
ax2.plot(poloidal_indices[start_right:end_right], q_spitzer_pol[start_right:end_right, mid_idx]/1e6, label=r'$q_{Spitzer}$', linewidth=2)
ax2.plot(poloidal_indices[start_right:end_right], q_gradB_pol[start_right:end_right, mid_idx]/1e6, label=r'$q_{\nabla B}$', linewidth=2)
ax2.plot(poloidal_indices[start_right:end_right], q_ExB_pol[start_right:end_right, mid_idx]/1e6, label=r'$q_{E \times B}$', linewidth=2)
ax2.plot(poloidal_indices[start_right:end_right],  q_ion[start_right:end_right, mid_idx] / 1e6, label='q$_{ion}$', linewidth=2)
ax2.plot(poloidal_indices[start_right:end_right], q_parall_current[start_right:end_right, mid_idx]/1e6, label=r'$q_{Current}$', linewidth=2, linestyle='--')
ax2.plot(poloidal_indices[start_right:end_right], qwtot[start_right:end_right, mid_idx]/1e6, label='q$_{UEDGE}$', linewidth=2)

if omp_cell is not None and start_right <= omp_cell < end_right:
    ax2.axvline(omp_cell, color='black', linestyle='--', linewidth=1)
    ax2.text(omp_cell + 3, ax2.get_ylim()[1]*1.05, "OMP", color='black', ha='center', va='bottom', fontweight='bold')

if oxpt is not None and start_right <= oxpt < end_right:
    ax2.axvline(oxpt, color='orange', linestyle='--', linewidth=1)
    ax2.text(oxpt + 3, ax2.get_ylim()[1]*1.05, "Oxpt", color='orange', ha='center', va='bottom', fontweight='bold')

ax2.set_xlabel('Poloidal grid index', fontsize=14)
ax2.set_xlim([poloidal_indices[start_right], poloidal_indices[end_right-1]])
ax2.set_ylim([-0.3, 0.5])
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.grid(True)
#ax2.legend(fontsize=9, ncol=2)
ax2.set_title('OMP to Odiv')
#ax2.set_yscale('symlog', linthresh=1)

plt.tight_layout()
plt.savefig("poloidal_heat_flux_split.png", dpi=300)
plt.show()



import matplotlib.pyplot as plt

# --- Create figure with 3 rows, shared x-axis ---
fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

# -------------------------------------------------------------------------
# (1) Conduction vs convection (electrons+ions total, your first block)
# -------------------------------------------------------------------------
ax = axes[0]
ax.plot(x_mm, q_cond_el_ion, label=r'$q_{\mathrm{Conduction}}$', color=colors["cond"], linewidth=2)
ax.plot(x_mm, q_conv_el_ion, label=r'$q_{\mathrm{Convection}}$', color=colors["conv"], linewidth=2)
ax.plot(x_mm, total_el_ion, label=r'$q_{\mathrm{sum}}$', color=colors["sum"], linewidth=2)
ax.plot(x_mm, q_uedge, label=r'$q_{\mathrm{UEDGE}}$', color=colors["uedge"], linestyle='--', linewidth=2)

ax.axvline(poloidal_indices[74], color='black', linestyle=':', linewidth=2)  # vertical marker
ax.text(poloidal_indices[74], ymax, "OMP", color='red', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel(r'$q_{\mathrm{poloidal}}$ [MW]', fontsize=16)
ax.legend(fontsize=12, loc='best')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=12, length=7)
ax.tick_params(axis='both', which='minor', labelsize=10, length=4)
ax.minorticks_on()

# -------------------------------------------------------------------------
# (2) Ion heat fluxes
# -------------------------------------------------------------------------
ax1 = axes[1]
ax1.plot(poloidal_indices[1:-1], qion_gradb[1:-1, mid_idx]/1e6, label='gradB', color='green')
ax1.plot(poloidal_indices[1:-1], qion_EB[1:-1, mid_idx]/1e6, label='E×B', color='magenta')
ax1.plot(poloidal_indices[1:-1], qion_conv[1:-1, mid_idx]/1e6, label='2.5nTu', color='blue')
ax1.plot(poloidal_indices[1:-1], icond[1:-1, mid_idx]/1e6, label='Cond', color='orange')
ax1.plot(poloidal_indices[1:-1], feix[1:-1, mid_idx]/1e6, label='feix', color='black')
ax1.plot(poloidal_indices[1:-1], qion_sum[1:-1, mid_idx]/1e6, label='sum', linestyle='--', color='red')
ax1.axvline(poloidal_indices[len(poloidal_indices)//2], color='black', linestyle=':', linewidth=2)

ax1.set_ylabel(r'$q_\mathrm{ion}$ [MW]', fontsize=16)
ax1.legend(fontsize=10, ncol=2)
ax1.grid(True)

# -------------------------------------------------------------------------
# (3) Electron heat fluxes
# -------------------------------------------------------------------------
sum_el = q_gradB_el + q_eb_el + q_conv_ele + econd + q_parall_current

ax2 = axes[2]
ax2.plot(poloidal_indices[1:-1], q_gradB_el[1:-1, mid_idx]/1e6, label='gradB', color='green')
ax2.plot(poloidal_indices[1:-1], q_eb_el[1:-1, mid_idx]/1e6, label='E×B', color='magenta')
ax2.plot(poloidal_indices[1:-1], q_conv_ele[1:-1, mid_idx]/1e6, label='2.5nTu', color='blue')
ax2.plot(poloidal_indices[1:-1], econd[1:-1, mid_idx]/1e6, label='Cond', color='orange')
ax2.plot(poloidal_indices[1:-1], feex[1:-1, mid_idx]/1e6, label='Conv', color='black')
ax2.plot(poloidal_indices[1:-1], q_parall_current[1:-1, mid_idx]/1e6, label=r'$j_\parallel$', linestyle='--', color='cyan')
ax2.plot(poloidal_indices[1:-1], sum_el[1:-1, mid_idx]/1e6, label='Sum', linestyle='--', color='red')
ax2.axvline(poloidal_indices[len(poloidal_indices)//2], color='black', linestyle=':', linewidth=2)

ax2.set_xlabel('Poloidal grid index', fontsize=16)
ax2.set_ylabel(r'$q_\mathrm{electron}$ [MW]', fontsize=16)
ax2.legend(fontsize=10, ncol=2)
ax2.grid(True)

# -------------------------------------------------------------------------
# Layout & save
# -------------------------------------------------------------------------
plt.xlim([0, 65])
plt.tight_layout()
plt.savefig("poloidal_heat_flux_3panel.png", dpi=300)
plt.show()
