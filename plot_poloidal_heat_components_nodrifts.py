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

folder = r'C:\UEDGE_run_Shahinul\PET_2025\data_analysis\high_FX_no_drifts'

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





q_spitzer_pol = np.load(os.path.join(folder, "q_spitzer_nodrfits.npy"))
q_ion         = np.load(os.path.join(folder, "q_ion.npy"))
q_ele         = np.load(os.path.join(folder, "q_ele.npy"))

sx         = np.load(os.path.join(folder, "sx.npy"))
vol         = np.load(os.path.join(folder, "vol.npy"))
rr        = np.load(os.path.join(folder, "rrv.npy"))
L        = np.load(os.path.join(folder, "L.npy"))

feex       = np.load(os.path.join(folder, "feex.npy"))
feix        = np.load(os.path.join(folder, "feix.npy"))

q_e_total      = np.load(os.path.join(folder, "q_e_total.npy"))
q_i_total      = np.load(os.path.join(folder, "q_i_total.npy"))

q_spitzer_pol = q_spitzer_pol
q_ion        = q_ion 
q_ele        = q_ele 

qwtot         = np.load(os.path.join(folder, "fetx.npy"))
qwtot   = qwtot/sx
# %%
total = q_ion+ q_ele #+ q_spitzer_pol


nx, ny = q_spitzer_pol.shape
poloidal_indices = np.arange(nx)
#poloidal_indices = L[:,9]
mid_idx = 10




fig, ax = plt.subplots(figsize=(6., 4.0))
#ax.plot(poloidal_indices[1:-1], q_spitzer_pol[1:-1, mid_idx] / 1e6, label=r'$q_{Spit.}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], q_e_total[1:-1, mid_idx] / 1e6, label=r'$q_{e-tot}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], feex[1:-1, mid_idx] / 1e6, label='${feex}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], q_i_total[1:-1, mid_idx] / 1e6, label='q$_{i-tot}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], feix[1:-1, mid_idx] / 1e6, label='feix', linewidth=2)

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



fig, ax = plt.subplots(figsize=(6., 4.0))

ax.plot(poloidal_indices[1:-1], q_spitzer_pol[1:-1, mid_idx] / 1e6, label=r'$q_{Spit.}$', linewidth=2)
#ax.plot(poloidal_indices[1:-1], q_gradB_pol[1:-1, mid_idx] / 1e6, label=r'$q_{\nabla B}$', linewidth=2)
#ax.plot(poloidal_indices[1:-1], q_ExB_pol[1:-1, mid_idx] / 1e6, label=r'$q_{E \times B}$', linewidth=2)
#ax.plot(poloidal_indices[1:-1], q_parall_current[1:-1, mid_idx] / 1e6, label=r'$q_{Current}$', linewidth=2, linestyle='--')
ax.plot(poloidal_indices[1:-1], q_ion[1:-1, mid_idx] / 1e6, label='q$_{ion}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], q_ele[1:-1, mid_idx] / 1e6, label='q$_{ele}$', linewidth=2)
ax.plot(poloidal_indices[1:-1], qwtot[1:-1, mid_idx] / 1e6, label='feex+feix', linewidth=2)
ax.plot(poloidal_indices[1:-1], total[1:-1, mid_idx] / 1e6, label='sum', linewidth=2, color ='cyan', linestyle='--')

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
  #  ax.plot(poloidal_indices[1:-1], q_gradB_pol[1:-1, mid_idx]/1e6, label=r'$q_{\nabla B}$', linewidth=2)
  #  ax.plot(poloidal_indices[1:-1], q_ExB_pol[1:-1, mid_idx]/1e6, label=r'$q_{E \times B}$', linewidth=2)
  #  ax.plot(poloidal_indices[1:-1], q_parall_current[1:-1, mid_idx]/1e6, label=r'$q_{Current}$', linewidth=2, linestyle='--')
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

ax.plot(poloidal_indices[start:end], q_ion[start:end, mid_idx] / 1e6, label='q$_{ion}$', linewidth=2)
#plt.plot(poloidal_indices[start:end], q_parall_current[start:end, mid_idx]/1e6, label=r'$q_{Current}$', linewidth=2, linestyle='--')

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
