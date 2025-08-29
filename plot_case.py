# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 07:45:00 2025

@author: islam9
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from cycler import cycler
from matplotlib.collections import LineCollection
import json
import re

from matplotlib.collections import LineCollection



sxnp = np.array([3.35404901e-08, 3.39626464e-02, 3.19107170e-02, 2.95808764e-02,
       2.68852014e-02, 2.39997574e-02, 2.11190302e-02, 1.83978410e-02,
       1.57142358e-02, 6.59730369e-02, 7.17170424e-02, 6.96879881e-02,
       6.98719029e-02, 6.49659944e-02, 6.47738697e-02, 6.52471802e-02,
       6.66649569e-02, 6.90308846e-02, 6.92583421e-02, 7.18869528e-02,
       7.56178648e-02, 7.84676614e-02, 8.15351906e-02, 8.64769772e-02,
       9.79929098e-02, 9.91486145e-08])


yyrb = np.array([-0.06506854, -0.0592725 , -0.04815738, -0.03801022, -0.02887471,
       -0.02078381, -0.01371695, -0.00760699, -0.0023902 ,  0.00980619,
        0.02988696,  0.04980043,  0.06879613,  0.08660054,  0.10325506,
        0.1195342 ,  0.13566592,  0.15188392,  0.1680489 ,  0.18418833,
        0.20069347,  0.21756767,  0.23471614,  0.25233707,  0.27125284,
        0.28119419])


import os
import numpy as np
import pandas as pd
import re

def eval_Li_evap_at_T_Cel(temperature):
    """Calculate lithium evaporation flux at a given temperature in Celsius."""
    a1 = 5.055
    b1 = -8023.0
    xm1 = 6.939
    tempK = temperature + 273.15
    if tempK <= 0:
        raise ValueError("Temperature must be above absolute zero (-273.15\u00b0C).")
    vpres1 = 760 * 10 ** (a1 + b1 / tempK)  # Vapor pressure
    sqrt_argument = xm1 * tempK
    if sqrt_argument <= 0:
        raise ValueError("Invalid value for sqrt: xm1 * tempK has non-positive values.")
    fluxEvap = 1e4 * 3.513e22 * vpres1 / np.sqrt(sqrt_argument)  # Evaporation flux
    return fluxEvap

def get_available_indices(folder, prefix, suffix):
    if not os.path.exists(folder):
        print(f"Warning: Folder '{folder}' does not exist.")
        return []
    files = os.listdir(folder)
    indices = []
    pattern = re.compile(rf"{prefix}(\d+){suffix}")
    for f in files:
        m = pattern.match(f)
        if m:
            indices.append(int(m.group(1)))
    indices.sort()
    return indices

def replace_with_linear_interpolation(arr):
    arr = pd.Series(arr)
    arr_interpolated = arr.interpolate(method='linear', limit_direction='both')
    return arr_interpolated.bfill().ffill().to_numpy()

def load_data_auto(filename, row=None, col=None):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return np.nan
    try:
        if filename.endswith('.npy'):
            data = np.load(filename)
        else:
            # Force numeric, replace non-numeric with NaN
            data = pd.read_csv(filename, header=None).apply(pd.to_numeric, errors='coerce').values
    except Exception as e:
        print(f"Could not load {filename}: {e}")
        return np.nan
    try:
        if row is not None and col is not None:
            row = int(round(row))
            col = int(round(col))
            return data[row, col]
        elif row is not None:
            row = int(round(row))
            return data[row]
        else:
            return data
    except Exception as e:
        print(f"Index error in {filename} at [{row},{col}]: {e}")
        return np.nan

def safe_weighted_sum(arr, sxnp, label):
    """
    Ensures arr and sxnp are 1D, same length, and numeric before summing arr * sxnp.
    Prints warnings and attempts to auto-fix common issues.
    """
    arr = np.array(arr)
    sxnp = np.array(sxnp)
    # Remove singleton dimensions
    arr = np.squeeze(arr)
    sxnp = np.squeeze(sxnp)
    # Flatten if not 1D
    if arr.ndim > 1:
        print(f"Warning: {label} array has shape {arr.shape}, flattening.")
        arr = arr.flatten()
    if sxnp.ndim > 1:
        print(f"Warning: sxnp array has shape {sxnp.shape}, flattening.")
        sxnp = sxnp.flatten()
    # Truncate or pad if needed
    if arr.shape != sxnp.shape:
        minlen = min(arr.shape[0], sxnp.shape[0])
        print(f"Warning: {label} and sxnp shape mismatch: {arr.shape} vs {sxnp.shape}. Truncating to {minlen}.")
        arr = arr[:minlen]
        sxnp = sxnp[:minlen]
    # Convert to float if needed
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float)
        except Exception as e:
            print(f"Error converting {label} to float: {e}")
            return np.nan
    if not np.issubdtype(sxnp.dtype, np.number):
        try:
            sxnp = sxnp.astype(float)
        except Exception as e:
            print(f"Error converting sxnp to float: {e}")
            return np.nan
    # Final check
    if arr.shape != sxnp.shape:
        print(f"Final shape mismatch for {label}: {arr.shape} vs {sxnp.shape}. Returning nan.")
        return np.nan
    return np.sum(arr * sxnp)

def process_dataset(data_path, dt, sep=8, ixmp=36, sxnp=None, eval_Li_evap_at_T_Cel=None):
    dirs = {
        "q_perp": os.path.join(data_path, 'q_perp'),
        "Tsurf_Li": os.path.join(data_path, 'Tsurf_Li'),
        "q_Li_surface": os.path.join(data_path, 'q_Li_surface'),
        "C_Li_omp": os.path.join(data_path, 'C_Li_omp'),
        "n_Li3": os.path.join(data_path, 'n_Li3'),
        "n_Li2": os.path.join(data_path, 'n_Li2'),
        "n_Li1": os.path.join(data_path, 'n_Li1'),
        "n_e": os.path.join(data_path, 'n_e'),
        "T_e": os.path.join(data_path, 'T_e'),
        "Li": os.path.join(data_path, 'Gamma_Li'),
        "q_Li_surface": os.path.join(data_path, 'q_Li_surface'),
        "Li_rad": os.path.join(data_path, 'Li_rad'),
    }
    available_indices = get_available_indices(dirs["Tsurf_Li"], "T_surfit_", ".csv")
    max_value_tsurf, max_q, evap_flux_max, max_q_Li_list = [], [], [], []
    C_Li_omp, Te, n_Li_total, ne, phi_sput, evap, ad, total, n_Li3, Prad = [], [], [], [], [], [], [], [], [], []

    for i in available_indices:
        filenames = {
            "tsurf": os.path.join(dirs["Tsurf_Li"], f'T_surfit_{i}.csv'),
            "qsurf": os.path.join(dirs["q_perp"], f'q_perpit_{i}.csv'),
            "qsurf_Li": os.path.join(dirs["q_Li_surface"], f'q_Li_surface_{i}.csv'),
            "C_Li": os.path.join(dirs["C_Li_omp"], f'CLi_prof_{i}.csv'),
            "n_Li3": os.path.join(dirs["n_Li3"], f'n_Li3_{i}.csv'),
            "n_Li2": os.path.join(dirs["n_Li2"], f'n_Li2_{i}.csv'),
            "n_Li1": os.path.join(dirs["n_Li1"], f'n_Li1_{i}.csv'),
            "ne": os.path.join(dirs["n_e"], f'n_e_{i}.npy'),  
            "Te": os.path.join(dirs["T_e"], f'T_e_{i}.csv'),
            "PS": os.path.join(dirs["Li"], f'PhysSput_flux_{i}.csv'),
            "Evap": os.path.join(dirs["Li"], f'Evap_flux_{i}.csv'),
            "Ad": os.path.join(dirs["Li"], f'Adstom_flux_{i}.csv'),
            "Total": os.path.join(dirs["Li"], f'Total_Li_flux_{i}.csv'),
            "prad_Li": os.path.join(dirs["Li_rad"], f'Li_rad_{i}.csv'),
        }
        # Fallback for ne if .npy not found
        if not os.path.exists(filenames["ne"]):
            filenames["ne"] = os.path.join(dirs["n_e"], f'n_e_{i}.csv')

        max_tsurf = np.nanmax(load_data_auto(filenames["tsurf"]))
        max_q_i = np.nanmax(load_data_auto(filenames["qsurf"]))
        max_q_Li_i = np.nanmax(load_data_auto(filenames["qsurf_Li"]))
        C_Li_i = load_data_auto(filenames["C_Li"], row=sep)
        Te_i = load_data_auto(filenames["Te"], row=ixmp, col=sep)
        n_Li3_i = load_data_auto(filenames["n_Li3"], row=ixmp, col=sep)
        n_Li2_i = load_data_auto(filenames["n_Li2"], row=ixmp, col=sep)
        n_Li1_i = load_data_auto(filenames["n_Li1"], row=ixmp, col=sep)
        ne_i = load_data_auto(filenames["ne"], row=ixmp, col=sep)
        ps_arr = load_data_auto(filenames["PS"])
        evap_arr = load_data_auto(filenames["Evap"])
        ad_arr = load_data_auto(filenames["Ad"])
        total_arr = load_data_auto(filenames["Total"])
        Prad_in = load_data_auto(filenames["prad_Li"])

        phi_sput_i = safe_weighted_sum(ps_arr, sxnp, "PhysSput_flux")
        evap_i = safe_weighted_sum(evap_arr, sxnp, "Evap_flux")
        ad_i = safe_weighted_sum(ad_arr, sxnp, "Adstom_flux")
        total_i = phi_sput_i + evap_i + ad_i
        vol = np.load('vol.npy')
        Prad_i = np.sum(Prad_in*vol)
        

        max_value_tsurf.append(max_tsurf)
        max_q.append(max_q_i)
        max_q_Li_list.append(max_q_Li_i)
        C_Li_omp.append(C_Li_i)
        Te.append(Te_i)
        n_Li3.append(n_Li3_i)
        n_Li_total.append(n_Li3_i + n_Li2_i + n_Li1_i)
        ne.append(ne_i)
        phi_sput.append(phi_sput_i)
        evap.append(evap_i)
        ad.append(ad_i)
        total.append(total_i)
        Prad.append(Prad_i)

    # Interpolate missing values
    max_value_tsurf = replace_with_linear_interpolation(max_value_tsurf)
    max_q = replace_with_linear_interpolation(max_q)
    max_q_Li_list = replace_with_linear_interpolation(max_q_Li_list)
    C_Li_omp = replace_with_linear_interpolation(C_Li_omp)
    n_Li_total = replace_with_linear_interpolation(n_Li_total)
    Te = replace_with_linear_interpolation(Te)
    ne = replace_with_linear_interpolation(ne)
    phi_sput = replace_with_linear_interpolation(phi_sput)
    evap = replace_with_linear_interpolation(evap)
    ad = replace_with_linear_interpolation(ad)
    total = replace_with_linear_interpolation(total)
    nLi3 = replace_with_linear_interpolation(n_Li3)
    Prad = replace_with_linear_interpolation(Prad)

    evap_flux_max = []
    for max_tsurf_val in max_value_tsurf:
        if not np.isnan(max_tsurf_val) and eval_Li_evap_at_T_Cel is not None:
            try:
                evap_flux = eval_Li_evap_at_T_Cel(max_tsurf_val)
            except Exception as e:
                print(f"Error calculating evaporation flux: {e}")
                evap_flux = np.nan
        else:
            evap_flux = np.nan
        evap_flux_max.append(evap_flux)
    evap_flux_max = replace_with_linear_interpolation(evap_flux_max)
    q_surface = np.array(max_q) - 2.26e-19 * np.array(evap_flux_max)
    time_axis = dt * np.arange(1, len(max_q) + 1)

    return (max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis,
            max_q_Li_list, C_Li_omp, n_Li_total, Te, ne, phi_sput, evap, ad, total, n_Li3, Prad)



parent_dir = r'C:\UEDGE_run_Shahinul\PET_2025'
folders = {
    "nx_P1": os.path.join(r"C:\UEDGE_run_Shahinul\PET_2025\high_FX_with_drifts\PePi5.8_drifts_Kye0.02", "C_Li_omp"),
    "nx_P2": os.path.join(r"C:\UEDGE_run_Shahinul\PET_2025\high_FX_no_drifts", "C_Li_omp"),
}

def count_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' does not exist.")
        return 0
    return len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])

file_counts = {key: count_files_in_folder(path) for key, path in folders.items()}

nx_P1 = file_counts["nx_P1"]
nx_P2 = file_counts["nx_P2"]


datasets = [


  {
        'path': os.path.join(r'C:\UEDGE_run_Shahinul\PET_2025\high_FX_with_drifts\PePi5.8_drifts_Kye0.02'),
        'nx': nx_P1,
        'dt': 10e-3,
        'label_tsurf': 'Drifts'
    },

  
  # {
  #       'path': os.path.join(r'C:\UEDGE_run_Shahinul\PET_2025\high_FX_no_drifts'),
  #       'nx': nx_P2,
  #       'dt': 10e-3,
  #       'label_tsurf': 'No-Drifts'
  #   },


]


colors = ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'purple']

fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)

max_vals = {'q': [], 'Tsurf': [], 'phi': [], 'Prad': [], 'CLi': [], 'nLi': []}

for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis,
     max_q_Li, C_Li_omp, n_Li_total, Te, ne, phi_sput, evap, ad, total, n_Li3, Prad) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )

    color = colors[idx % len(colors)]


    axes[0,0].plot(time_axis, np.array(max_q)/1e6, linewidth=2, color=color)
    axes[1,0].plot(time_axis, max_value_tsurf, linewidth=2, color=color)
    axes[0,1].plot(time_axis, Prad/1e6, linewidth=2, color=color)
    axes[1,1].plot(time_axis, C_Li_omp*100, linewidth=2, color=color)
    axes[2,1].plot(time_axis, n_Li_total/1e18, linewidth=2, color=color)
    

    if idx == 0:
        axes[2,0].plot(time_axis, phi_sput/1e21, '--', linewidth=2, color='blue', label='P.Sput')
        axes[2,0].plot(time_axis, evap/1e21, '-.', linewidth=2, color='red', label='Evap')
        axes[2,0].plot(time_axis, ad/1e21, ':', linewidth=2, color='green', label='Ad')
        axes[2,0].plot(time_axis, total/1e21, '-', linewidth=2, color='purple', label='Total')


    max_vals['q'].append(np.max(np.array(max_q)/1e6))
    max_vals['Tsurf'].append(np.max(max_value_tsurf))
    max_vals['phi'].append(np.max(total/1e21))
    max_vals['Prad'].append(np.max(Prad/1e6))
    max_vals['CLi'].append(np.max(C_Li_omp*100))
    max_vals['nLi'].append(np.max(n_Li_total/1e18))


axes[0,0].set_ylim([0, np.max(max_vals['q'])*1.05])
axes[1,0].set_ylim([0, np.max(max_vals['Tsurf'])*1.05])
axes[2,0].set_ylim([0, np.max(max_vals['phi'])*1.05])
axes[0,1].set_ylim([0, np.max(max_vals['Prad'])*1.05])
axes[1,1].set_ylim([0, np.max(max_vals['CLi'])*1.05])
axes[2,1].set_ylim([0, np.max(max_vals['nLi'])*1.05])

# Labels
axes[0,0].set_ylabel('q$_{\\perp}^{max}$ (MW/m$^2$)', fontsize=16)
axes[1,0].set_ylabel("T$_{surf}^{max}$ ($^\\circ$C)", fontsize=16)
axes[2,0].set_ylabel("$\\phi_{Li}$ (10$^{21}$ atom/s)", fontsize=16)
axes[0,1].set_ylabel("P$_{Li-rad}$ (MW)", fontsize=16)
axes[1,1].set_ylabel("C$_{Li,sep}^{OMP}$ (%)", fontsize=16)
axes[2,1].set_ylabel("n$_{Li,sep}^{OMP}$ ($10^{18}$ m$^{-2}$)", fontsize=16)

for ax in axes[2,:]:
    ax.set_xlabel('Simulation time [s]', fontsize=16)


for row in axes:
    for ax in row:
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=12)


axes[2,0].legend(fontsize=12)


labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
for label, ax in zip(labels, axes.flatten()):
    ax.text(0.02, 0.90, label, transform=ax.transAxes, fontsize=12, fontweight='bold',
            va='top', ha='left')

plt.xlim([0, 5])
plt.tight_layout()
plt.savefig('qsurf_T_surf_Li_rad_phi_6panel.png', dpi=300)
plt.show()






import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

colors = ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'purple']

fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)

max_q_all = []
max_tsurf_all = []
max_phi_all = []
max_prad_all = []
max_CLiomp_all = []
max_nLi_all = []

for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis,
     max_q_Li, C_Li_omp, n_Li_total, Te, ne, phi_sput, evap, ad, total, n_Li3, Prad) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )

    color = colors[idx % len(colors)]
    # Main 6 plots
    axes[0,0].plot(time_axis, np.array(max_q) / 1e6,  linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)
    axes[1,0].plot(time_axis, max_value_tsurf,   linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)
    axes[0,1].plot(time_axis, Prad / 1e6,        linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)
    axes[1,1].plot(time_axis, C_Li_omp * 100,    linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)
    axes[2,1].plot(time_axis, n_Li_total / 1e18, linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)

    # 4th row, left: total for each dataset, processes only once
    axes[2,0].plot(time_axis, total/1e22, '-', linewidth=2, color=color, label=f'Total ({dataset["label_tsurf"]})')
    if idx == 0:
        axes[2,0].plot(time_axis, phi_sput/1e22, '--', linewidth=2, color='blue', label='phi_sput')
        axes[2,0].plot(time_axis, evap/1e22, '-.', linewidth=2, color='red', label='evap')
        axes[2,0].plot(time_axis, ad/1e22, ':', linewidth=2, color='green', label='ad')
        axes[2,0].plot(time_axis, (ad+phi_sput+evap)/1e22, '--', linewidth=2, color='purple', label='sum')
    
    # Store max for autoscaling if needed
    max_q_all.append(np.max(np.array(max_q) / 1e6))
    max_tsurf_all.append(np.max(max_value_tsurf))
    max_phi_all.append(np.max(total / 1e22))
    max_prad_all.append(np.max(Prad / 1e6))
    max_CLiomp_all.append(np.max(C_Li_omp * 100))
    max_nLi_all.append(np.max(n_Li_total / 1e18))

# === Dynamic Y-limits ===
axes[0,0].set_ylim([0, np.max(max_q_all) * 1.05])
axes[1,0].set_ylim([0, np.max(max_tsurf_all) * 1.05])
axes[2,0].set_ylim([0, np.max(max_phi_all) * 1.05])
axes[0,1].set_ylim([0, np.max(max_prad_all) * 1.05])
axes[1,1].set_ylim([0, np.max(max_CLiomp_all) * 1.05])
axes[2,1].set_ylim([0, np.max(max_nLi_all) * 1.05])

# === Labels ===
axes[0,0].set_ylabel('q$_{\\perp}^{max}$ (MW/m$^2$)', fontsize=14)
axes[1,0].set_ylabel("T$_{surf}^{max}$ ($^\\circ$C)", fontsize=14)
axes[2,0].set_ylabel("$\\phi_{Li}$ (10$^{22}$ atom/s)", fontsize=14)
axes[0,1].set_ylabel("P$_{Li-rad}$ (MW)", fontsize=14)
axes[1,1].set_ylabel("C$_{Li,omp}$ (%)", fontsize=14)
axes[2,1].set_ylabel("n$_{Li,total}$ ($10^{18}$ m$^{-2}$)", fontsize=14)
#axes[3,0].set_ylabel('Sources (10$^{22}$ atom/s)', fontsize=14)
#axes[3,0].set_xlabel('t$_{simulation}$ (s)', fontsize=16)
#axes[3,1].axis('off')  # Hide unused subplot

for ax in axes[3,:]:
    ax.set_xlabel('t$_{simulation}$ (s)', fontsize=16)

for row in axes:
    for ax in row:
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=12)

# Legends
for i in range(3):
    for j in range(2):
        axes[i,j].legend(fontsize=10)
# For the sources plot, only show unique labels
handles, labels = axes[3,0].get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes[3,0].legend(by_label.values(), by_label.keys(), fontsize=10, ncol=2)

plt.tight_layout()
plt.savefig('qsurf_T_surf_Li_rad_phi_8panel.png', dpi=300)
plt.show()


fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
ax1, ax2, ax3, ax4 = axes

max_q_all = []
max_tsurf_all = []
max_phi_all = []
max_prad_all = []

for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis,
     max_q_Li, C_Li_omp, n_Li_total, Te, ne, phi_sput, evap, ad, total, n_Li3, Prad) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )

    color = colors[idx % len(colors)]
    ax1.plot(time_axis, np.array(max_q) / 1e6,  linewidth=2, label=f'{dataset["label_tsurf"]}')
    ax2.plot(time_axis, max_value_tsurf,   linewidth=2, label=f'{dataset["label_tsurf"]}')
    ax3.plot(time_axis, total / 1e22,      linewidth=2, label=f'{dataset["label_tsurf"]}')
    ax4.plot(time_axis, Prad / 1e6,        linewidth=2, label=f'{dataset["label_tsurf"]}')
    

    # Store global max values
    max_q_all.append(np.max(np.array(max_q) / 1e6))
    max_tsurf_all.append(np.max(max_value_tsurf))
    max_phi_all.append(np.max(total / 1e22))
    max_prad_all.append(np.max(Prad / 1e6))

# === Dynamic Y-limits ===
ax1.set_ylim([0, np.max(max_q_all) * 1.05])
ax2.set_ylim([0, np.max(max_tsurf_all) * 1.05])
ax3.set_ylim([0, np.max(max_phi_all) * 1.05])
ax4.set_ylim([0, np.max(max_prad_all) * 1.05])


# === Subplot labels ===
#ax1.text(0.98, 0.90, "(a)", transform=ax1.transAxes, fontsize=16, va='top', ha='right', fontweight='bold')
#ax2.text(0.50, 0.50, "(b)", transform=ax2.transAxes, fontsize=16, va='center', ha='center', fontweight='bold')
#ax3.text(0.02, 0.90, "(c)", transform=ax3.transAxes, fontsize=16, va='top', ha='left', fontweight='bold')
#ax4.text(0.02, 0.90, "(d)", transform=ax4.transAxes, fontsize=16, va='top', ha='left', fontweight='bold')

# === Labels, grid, etc ===
ax1.set_ylabel('q$_{\\perp}^{max}$ (MW/m$^2$)', fontsize=16)
ax2.set_ylabel("T$_{surf}^{max}$ ($^\\circ$C)", fontsize=16)
ax3.set_ylabel("$\\phi_{Li}$ (10$^{22}$ atom/s)", fontsize=16)
ax4.set_ylabel("P$_{Li-rad}$ (MW)", fontsize=16)
ax4.set_xlabel('t$_{simulation}$ (s)', fontsize=18)
#ax2.legend(fontsize=12, ncol=2)
ax1.set_xlim([0, 5])
ax2.set_xlim([0, 5])
ax3.set_xlim([0, 5])
ax4.set_xlim([0, 5])

for ax in axes:
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig('qsurf_T_surf_Li_rad_phi.png', dpi=300)
plt.show()



ymax = 0
for dataset in datasets:
    (_, _, _, _, time_axis, _, _, _, _, _, phi_sput, evap, ad, total, _,_) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    ymax = max(ymax, np.max(phi_sput), np.max(evap), np.max(ad), np.max(total))

plt.figure(figsize=(5, 3))

for dataset in datasets:
    (_, _, _, _, time_axis, _, _, _, _, _, phi_sput, evap, ad, total, _,_) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    plt.plot(time_axis, phi_sput, color ='blue',linestyle='--', label= 'Phy. Sput', linewidth = '2')
    plt.plot(time_axis, evap, '-r', label='Evaporation', linewidth = '2')
    plt.plot(time_axis, ad, color ='green', label='Ad-atom', linestyle=':', linewidth = '3')

plt.xlabel('t$_{simulation}$ (s)', fontsize=18)
plt.ylabel('$\phi_{Li}^{Emitted}$ (atom/s)', fontsize=18)
plt.ylim([0, ymax*1.05])
plt.xlim([0, 5])
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.yscale('log')
plt.ylim([1e15, ymax*1.10])
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('li_flux.png', dpi=300)
plt.savefig('Phi_Li_combined.eps', format='eps', dpi=300)
plt.close()


fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
ax1, ax2, ax3, ax4 = axes

for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis,
     max_q_Li, C_Li_omp, n_Li_total, Te, ne, phi_sput, evap, ad, total, n_Li3, Prad) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    ax1.plot(time_axis, np.array(max_q) / 1e6,  linewidth=2, label=f'{dataset["label_tsurf"]}')
    ax2.plot(time_axis, max_value_tsurf,   linewidth=2, label=f'{dataset["label_tsurf"]}')
    ax3.plot(time_axis, total/1e22,  linewidth=2, label=f'{dataset["label_tsurf"]}')
    ax4.plot(time_axis, Prad/1e6,  linewidth=2, label=f'{dataset["label_tsurf"]}')

# Add subplot labels
ax1.text(0.98, 0.90, "(a)", transform=ax1.transAxes, fontsize=16, va='top', ha='right', fontweight='bold')
ax2.text(0.50, 0.50, "(b)", transform=ax2.transAxes, fontsize=16, va='center', ha='center', fontweight='bold')
ax3.text(0.02, 0.90, "(c)", transform=ax3.transAxes, fontsize=16, va='top', ha='left', fontweight='bold')
ax4.text(0.02, 0.90, "(d)", transform=ax4.transAxes, fontsize=16, va='top', ha='left', fontweight='bold')

ax1.set_ylabel('q$_{\perp}^{max}$ (MW/m$^2$)', fontsize=16)
ax1.set_xlim([0, 5])
ax1.set_ylim([0, 10])
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=14)
ax1.tick_params(axis='both', labelsize=14)

ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=16)
ax2.set_ylim([0, 650])
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)

ax3.set_ylabel("$\phi_{Li}$ (10$^{22}$ atom/s)", fontsize=16)
ax3.set_ylim([0, 5])
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=14)
ax3.tick_params(axis='both', labelsize=14)

ax4.set_ylabel("P$_{Li-rad}$ (MW)", fontsize=16)
ax4.set_ylim([0, 0.2])
ax4.set_xlabel('t$_{simulation}$ (s)', fontsize=18)
ax4.grid(True)
ax4.tick_params(axis='both', labelsize=14)

ax4.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('qsurf_T_surf_Li_rad_phi.png', dpi=300)
plt.show()




# 1. Three-panel plot: q_surf, T_surf, C_Li_omp
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis,
     max_q_Li, C_Li_omp, n_Li_total, Te, ne, phi_sput, evap, ad, total, n_Li3,_) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    ax1.plot(time_axis, np.array(max_q_Li) / 1e6, '-', linewidth=2, label=dataset["label_tsurf"], color=color)
    ax2.plot(time_axis, max_value_tsurf, '-', linewidth=2, label=dataset["label_tsurf"], color=color)
    ax3.plot(time_axis, C_Li_omp * 100, '-', linewidth=2, label=dataset["label_tsurf"], color=color)

ax1.set_ylabel('q$_{s}^{max}$ (MW/m$^2$)', fontsize=18)
ax1.set_xlim([0, 5])
ax1.set_ylim([0, 10])
ax1.legend(loc='best', fontsize=12, ncol=2)
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=14)

ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
ax2.set_ylim([0, 750])
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=14)

ax3.set_ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=18)
ax3.set_ylim([0, 15])
ax3.set_xlabel('t$_{simulation}$ (s)', fontsize=18)
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig('qsurf_T_surf_CLi_omp.png', dpi=300)
plt.show()

# 2. Two-panel plot: short time axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), dpi=300, sharex=True)
for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis,
     max_q_Li, C_Li_omp, n_Li_total, Te, ne, phi_sput, evap, ad, total, n_Li3,_) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    ax1.plot(time_axis, np.array(max_q_Li) / 1e6, '-', linewidth=1.5, label=dataset["label_tsurf"], color=color)
    ax2.plot(time_axis, max_value_tsurf, '-', linewidth=1.5, label=dataset["label_tsurf"], color=color)

ax1.set_ylabel('q$_{s}^{max}$ (MW/m$^2$)', fontsize=14)
ax1.set_xlim([0, 1.5])
ax1.set_ylim([0, 10])
ax1.grid(True, linestyle='--', linewidth=0.5)
ax1.tick_params(axis='both', labelsize=12)

ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=14)
ax2.set_xlabel('t$_{sim}$ (s)', fontsize=14)
ax2.set_ylim([0, 750])
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.tick_params(axis='both', labelsize=12)
ax2.legend(loc='best', fontsize=12, ncol=2)

plt.tight_layout()
plt.savefig('qsurf_T_surf.pdf', format='pdf', bbox_inches='tight')
plt.savefig('qsurf_T_surf.png', dpi=600, bbox_inches='tight')
plt.savefig('qsurf_T_surfdt.eps', format='eps', bbox_inches='tight')
plt.show()

# 3. Emission mechanisms

fig, ax = plt.subplots(figsize=(5, 3))
for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (_, _, _, _, time_axis, _, _, _, _, _, phi_sput, evap, ad, _, _, _) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    ax.plot(time_axis, evap, '-', linewidth=2, color='red', label='Evaporation')
    ax.plot(time_axis, phi_sput, '--', linewidth=2, color='blue', label='Phy. Sput.')
    ax.plot(time_axis, ad, ':', linewidth=3, color='green', label='Ad-atom')

ax.set_xlabel('t$_{simulation}$ (s)', fontsize=14)
ax.set_ylabel('$\phi_{Li}^{emitted}$ (atom/s)', fontsize=14)
ax.set_xlim([0, 5])
ax.set_ylim([1e15, 1e23])
ax.set_yscale('log')
ax.tick_params(axis='both', labelsize=12)
ax.minorticks_on()
ax.grid()
ax.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig('Phi_Li_combined.png', dpi=300)

grid_kwargs_eps = dict(which='both', linestyle='--', linewidth=0.5)
ax.grid(**grid_kwargs_eps)
plt.savefig('Phi_Li_combined.eps', format='eps', bbox_inches='tight')

plt.show()

# 4. T_surf vs C_Li_omp
plt.figure(figsize=(5, 4))
for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (max_value_tsurf, _, _, _, _, _, C_Li_omp, _, _, _, _, _, _, _,_,_) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    plt.plot(max_value_tsurf, C_Li_omp * 100, '-', linewidth=2, label=dataset["label_tsurf"], color=color)

plt.xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
plt.ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=18)
plt.axhline(3, color='black', linestyle=':', linewidth=2, label='y = 3')
plt.legend(fontsize=14)
plt.ylim([0, 15])
plt.xlim([0, 700])
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('T_surf_CLi_omp.png', dpi=300)
plt.show()

# 5. T_surf vs n_Li3
plt.figure(figsize=(5, 4))
for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (max_value_tsurf, _, _, _, _, _, _, n_Li_total, _, _, _, _, _, _, n_Li3,_) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    plt.plot(max_value_tsurf, n_Li3, '-', linewidth=2, label=dataset["label_tsurf"], color=color)

plt.xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
plt.ylabel("n$_{Li-sep}^{omp}$ (m$^{-3}$)", fontsize=18)
plt.legend(fontsize=14)
plt.ylim([0, 2e18])
plt.xlim([0, 700])
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('T_surf_nLi_omp.png', dpi=300)
plt.show()

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5, 6))  # Slightly taller for two plots

for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    # First subplot: T_surf vs C_Li_omp
    (max_value_tsurf, _, _, _, _, _, C_Li_omp, _, _, _, _, _, _, _, _, _) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    axs[0].plot(max_value_tsurf, C_Li_omp * 100, '-', linewidth=2, label=dataset["label_tsurf"], color=color)

    # Second subplot: T_surf vs n_Li3
    (_, _, _, _, _, _, _, n_Li_total, _, _, _, _, _, _, n_Li3, _) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    axs[1].plot(max_value_tsurf, n_Li3, '-', linewidth=2, label=dataset["label_tsurf"], color=color)


axs[0].set_ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=16)
axs[0].axhline(2, color='black', linestyle=':', linewidth=2, label='y = 3')
axs[0].set_ylim([0, 8])
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
axs[0].tick_params(axis='both', labelsize=12)
axs[0].minorticks_on()

axs[1].set_xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=16)
axs[1].set_ylabel("n$_{Li-sep}^{omp}$ (m$^{-3}$)", fontsize=16)
axs[1].set_ylim([0, 2e18])
axs[1].set_xlim([0, 700])
axs[1].legend(fontsize=12)
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
axs[1].tick_params(axis='both', labelsize=12)
axs[1].minorticks_on()

plt.tight_layout()
plt.savefig('T_surf_CLi_nLi_omp_combined.png', dpi=600)  # Use PDF for vector quality, or PNG with high dpi
plt.show()


fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5, 6))

max_C_Li_omp = 0
max_n_Li3 = 0
tsurf_at_2pct = []  # List of tuples (T_surf, color, label)

for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    
    # Unpack relevant variables
    (max_value_tsurf, _, _, _, _, _, C_Li_omp, _, _, _, _, _, _, _, n_Li3, _) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )

    color = colors[idx % len(colors)]
    label = dataset["label_tsurf"]
    y_cli = C_Li_omp * 100

    # === Top plot: C_Li_omp (%) vs T_surf
    axs[0].plot(max_value_tsurf, y_cli, '-', linewidth=2, label=label, color=color)
    max_C_Li_omp = max(max_C_Li_omp, np.max(y_cli))

    # Identify T_surf where C_Li_omp crosses 2%
    idx_cross = np.where(y_cli >= 2)[0]
    if len(idx_cross) > 0:
        tsurf_val = max_value_tsurf[idx_cross[0]]
        tsurf_at_2pct.append((tsurf_val, color, label))
        axs[0].plot(tsurf_val, y_cli[idx_cross[0]], 'o', color=color, markersize=6)
        axs[0].text(tsurf_val + 5, y_cli[idx_cross[0]] + 0.2, f"{label}", fontsize=9, color=color)

    # === Bottom plot: n_Li3 vs T_surf
    axs[1].plot(max_value_tsurf, n_Li3, '-', linewidth=2, label=label, color=color)
    max_n_Li3 = max(max_n_Li3, np.max(n_Li3))


# === Axis styling for top plot
axs[0].set_ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=16)
axs[0].set_ylim([0, max_C_Li_omp * 1.05])
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
axs[0].tick_params(axis='both', labelsize=12)
axs[0].minorticks_on()

# === Axis styling for bottom plot
axs[1].set_xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=16)
axs[1].set_ylabel("n$_{Li-sep}^{omp}$ (m$^{-3}$)", fontsize=16)
axs[1].set_ylim([0, max_n_Li3 * 1.05])
axs[1].set_xlim([0, 700])
axs[1].legend(fontsize=12)
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
axs[1].tick_params(axis='both', labelsize=12)
axs[1].minorticks_on()


for tsurf_val, color, label in tsurf_at_2pct:
    axs[1].axvline(tsurf_val, color=color, linestyle=':', linewidth=1.5)
    #axs[1].text(tsurf_val + 5, max_n_Li3 * 0.9, f"{label}", rotation=90, va='top', fontsize=9, color=color)


plt.tight_layout()
plt.savefig('T_surf_CLi_nLi_omp_combined.png', dpi=600)
plt.show()



# 6. Total Li emission vs n_Li3
plt.figure(figsize=(4, 2.25))
all_n_Li3 = []
for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (_, _, _, _, _, _, _, n_Li_total, _, _, _, _, _, total,n_Li3,_) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    plt.plot(total / 1e21, n_Li3, '-', linewidth=2, label=dataset["label_tsurf"], color=color)
    all_n_Li3.append(n_Li3)

ymax = max([np.nanmax(n) if np.iterable(n) else n for n in all_n_Li3]) * 1.05

plt.xlabel("$\phi_{Li}$ ($10^{21}$atom/s)", fontsize=18)
plt.ylabel("n$_{Li-sep}^{omp}$ (m$^{-3}$)", fontsize=18)
plt.legend(fontsize=11)
plt.ylim([0, ymax])
plt.xlim([0, 2])
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('Cs_Phi_Li_nLi_omp.eps', format='eps', dpi=600)
plt.savefig('Cs_Phi_Li_nLi_omp.jpg', format='jpg', dpi=600)
plt.savefig('Cs_Phi_Li_nLi_omp.png', format='png', dpi=300)
plt.show()



fig, axs = plt.subplots(2, 1, sharex=True, figsize=(4, 5)) 

for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (
        max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis,
        max_q_Li_list, C_Li_omp, n_Li_total, Te, ne, phi_sput,
        evap, ad, total, n_Li3, Prad
    ) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    axs[0].plot(max_value_tsurf, C_Li_omp * 100, '-', linewidth=2, label=dataset["label_tsurf"], color=color)
    axs[1].plot(max_value_tsurf, n_Li_total / 1e18, '-', linewidth=2, label=dataset["label_tsurf"], color=color)

axs[0].set_ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=16)
axs[0].set_ylim([0, 0.04])
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
axs[0].tick_params(axis='both', labelsize=12)
axs[0].minorticks_on()
axs[0].text(0.3, 0.9, "(a)", transform=axs[0].transAxes, fontsize=16, va='center', ha='left', fontweight='bold')
#axs[0].legend(fontsize=12)

axs[1].set_xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=16)
axs[1].set_ylabel("n$_{Li-sep}^{omp}$ ($10^{18}$ m$^{-3}$)", fontsize=16)
axs[1].set_ylim([0, 0.2])
axs[1].set_xlim([0, 550])
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
axs[1].tick_params(axis='both', labelsize=12)
axs[1].minorticks_on()
axs[1].text(0.3, 0.9, "(b)", transform=axs[1].transAxes, fontsize=16, va='center', ha='left', fontweight='bold')

plt.tight_layout()
plt.savefig('T_surf_CLi_nLiTotal_omp_combined.png', dpi=600)
plt.show()



plt.figure(figsize=(5, 4))

all_totals = []  # To store all y-data for auto-scaling

for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (_, _, _, _, time_axis, _, _, n_Li_total, _, _, _, _, _, total, n_Li3,_) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    plt.plot(time_axis, total, '-', linewidth=2, label=dataset["label_tsurf"], color=color)
    all_totals.append(total)

# Find global max for y-axis
ymax = max([t.max() for t in all_totals]) * 1.05

plt.xlabel("t$_{sim}$ (s)", fontsize=18)
plt.ylabel("$\phi_{Li}$ (atom/s)", fontsize=18)
plt.legend(fontsize=12)
plt.ylim([0, ymax])
plt.xlim([0, 5])
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('tsim_Phi_Li_omp.png', dpi=300)
plt.show()

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(4, 5))

for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (
        _, _, _, _, _, _, 
        C_Li_omp, n_Li_total, _, _, _, _, _, total, _, _
    ) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    axs[0].plot(total, C_Li_omp * 100, '-_', linewidth=2, label=dataset["label_tsurf"], color=color)
    axs[1].plot(total, n_Li_total / 1e18, '-_', linewidth=2, label=dataset["label_tsurf"], color=color)

# === Top subplot: C_Li_omp vs total ===
axs[0].set_ylabel("C$_{Li,sep}^{omp}$ (%)", fontsize=16)
axs[0].set_ylim([0, 3])
axs[0].set_xlim([0, np.max(total)*1.05])
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
axs[0].tick_params(axis='both', labelsize=12)
axs[0].minorticks_on()
axs[0].text(0.03, 0.85, "(a)", transform=axs[0].transAxes, fontsize=16, fontweight='bold')
axs[0].legend(fontsize=10, loc='upper right')

# === Bottom subplot: n_Li_total vs total ===
axs[1].set_xlabel(r"$\phi_{Li}^{total}$ (atoms/s)", fontsize=16)
axs[1].set_ylabel("n$_{Li,sep}^{omp}$ ($10^{18}$ m$^{-3}$)", fontsize=16)
axs[1].set_ylim([0, 0.5])
axs[1].set_xlim([0, np.max(total)*1.05])
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
axs[1].tick_params(axis='both', labelsize=12)
axs[1].minorticks_on()
axs[1].text(0.03, 0.85, "(b)", transform=axs[1].transAxes, fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig("PhiLi_CLi_nLiTotal_omp_vs_total.png", dpi=600)
plt.show()




