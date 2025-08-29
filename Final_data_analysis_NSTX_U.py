import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from cycler import cycler
from matplotlib.collections import LineCollection


sxnp = np.array([1.65294727e-08, 1.68072047e-02, 1.57913285e-02, 1.47307496e-02,
       1.37802350e-02, 1.28710288e-02, 1.19247238e-02, 1.09516290e-02,
       1.02117906e-02, 2.14465429e-02, 1.11099457e-02, 1.26587791e-02,
       1.45917191e-02, 1.66915905e-02, 1.94826593e-02, 2.23582531e-02,
       2.53806793e-02, 2.94667140e-02, 3.39163144e-02, 3.85707117e-02,
       4.34572856e-02, 4.70735328e-02, 5.17684150e-02, 5.64336990e-02,
       5.97825750e-02, 6.08629236e-08])

yyrb = np.array([-0.05544489, -0.0507309 , -0.04174753, -0.03358036, -0.02614719,
       -0.01935555, -0.01316453, -0.00755603, -0.00245243,  0.00497426,
        0.012563  ,  0.01795314,  0.02403169,  0.03088529,  0.03865124,
        0.04744072,  0.05723254,  0.06818729,  0.0804908 ,  0.09413599,
        0.10907809,  0.12501805,  0.14181528,  0.15955389,  0.17792796,
        0.18716496])

    
def eval_Li_evap_at_T_Cel(temperature):
    """Calculate lithium evaporation flux at a given temperature in Celsius."""
    a1 = 5.055
    b1 = -8023.0
    xm1 = 6.939
    tempK = temperature + 273.15

    if tempK <= 0:
        raise ValueError("Temperature must be above absolute zero (-273.15Â°C).")

    vpres1 = 760 * 10 ** (a1 + b1 / tempK)  # Vapor pressure
    sqrt_argument = xm1 * tempK

    if sqrt_argument <= 0:
        raise ValueError("Invalid value for sqrt: xm1 * tempK has non-positive values.")

    fluxEvap = 1e4 * 3.513e22 * vpres1 / np.sqrt(sqrt_argument)  # Evaporation flux
    return fluxEvap


def count_files_in_folder(folder_path):
    """Count the number of files in a folder."""
    return len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])


folders = {

     
      "nx_P1": r"C:\UEDGE_run_Shahinul\PET_2025\UEDGE_Wall_Li_scan\PePi5.8Kye0.01\C_Li_omp",
      "nx_P2": r"C:\UEDGE_run_Shahinul\PET_2025\UEDGE_Wall_Li_scan\PePi5.8Kye0.01_new\C_Li_omp",
      "nx_P3": r"C:\UEDGE_run_Shahinul\PET_2025\UEDGE_Wall_Li_scan\PePi6.4Kye0.04\C_Li_omp",
      "nx_P4": r"C:\UEDGE_run_Shahinul\PET_2025\UEDGE_Wall_Li_scan\PePi7.2Kye0.1\C_Li_omp",
      "nx_P5": r"C:\UEDGE_run_Shahinul\PET_2025\UEDGE_Wall_Li_scan\PePi7.8Kye0.16\C_Li_omp",
      "nx_P6": r"C:\UEDGE_run_Shahinul\PET_2025\UEDGE_Wall_Li_scan\PePi9.6Kye0.26\C_Li_omp",
}


file_counts = {key: count_files_in_folder(path) for key, path in folders.items()}


nx_P1, nx_P2, nx_P3, nx_P4, nx_P5, nx_P6 = file_counts.values()


datasets = [

      {'path': r'C:\UEDGE_run_Shahinul\PET_2025\UEDGE_Wall_Li_scan\PePi5.8Kye0.01', 'nx': nx_P1, 'dt': 10e-3, 'label_tsurf': r'P : 5.8 MW'},
      {'path': r'C:\UEDGE_run_Shahinul\PET_2025\UEDGE_Wall_Li_scan\PePi5.8Kye0.01_new', 'nx': nx_P2, 'dt': 10e-3, 'label_tsurf': r'P : 5.8 MW-NoD'},
      {'path': r'C:\UEDGE_run_Shahinul\PET_2025\UEDGE_Wall_Li_scan\PePi6.4Kye0.04', 'nx': nx_P3, 'dt': 10e-3, 'label_tsurf': r'P : 6.4 MW'},
      {'path': r'C:\UEDGE_run_Shahinul\PET_2025\UEDGE_Wall_Li_scan\PePi7.2Kye0.1', 'nx': nx_P4, 'dt': 10e-3, 'label_tsurf': r'P : 7.2 MW'},
      {'path': r'C:\UEDGE_run_Shahinul\PET_2025\UEDGE_Wall_Li_scan\PePi7.8Kye0.16', 'nx': nx_P5, 'dt': 10e-3, 'label_tsurf': r'P : 7.8 MW'},
     {'path': r'C:\UEDGE_run_Shahinul\PET_2025\UEDGE_Wall_Li_scan\PePi9.6Kye0.26', 'nx': nx_P6, 'dt': 10e-3, 'label_tsurf': r'P : 9.6 MW'},
]




def process_dataset(data_path, nx, dt, sep=8, ixmp=76, sxnp=1.0, eval_Li_evap_at_T_Cel=None):
    max_value_tsurf, max_q, evap_flux_max, max_q_Li_list = [], [], [], []
    C_Li_omp, Te, n_Li3, n_Li2, n_Li1, ne, phi_sput, evap, ad, total = [], [], [], [], [], [], [], [], [], []

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
    }

    for i in range(1, nx):
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
            "Total": os.path.join(dirs["Li"], f'Total_Li_flux_{i}.csv')
        }

        # Defaults
        max_tsurf = max_q_i = evap_flux = max_q_Li_i = np.nan
        C_Li_i = Te_i = n_Li3_i = n_Li2_i = n_Li1_i = ne_i = phi_sput_i = evap_i = ad_i = total_i = np.nan

        try:
            max_tsurf = np.max(pd.read_csv(filenames["tsurf"], header=None).values.astype(float))
            max_q_i = np.max(pd.read_csv(filenames["qsurf"], header=None).values.astype(float))
            max_q_Li_i = np.max(pd.read_csv(filenames["qsurf_Li"], header=None).values.astype(float))

            # C_Li parsing
            df_C_Li = pd.read_csv(filenames["C_Li"], header=None)
            cell = df_C_Li.iloc[sep, 0]

            if isinstance(cell, str):
                C_Li_vals = np.array([float(x) for x in cell.split(',')])
            elif isinstance(cell, (float, int, np.number)):
                C_Li_vals = np.array([cell])
            else:
                raise ValueError(f"Unexpected C_Li cell type: {type(cell)}")

            C_Li_i = C_Li_vals

            # Te and Li densities
            Te_data = pd.read_csv(filenames["Te"], header=None).apply(pd.to_numeric, errors='coerce').values
            Te_i = Te_data[ixmp, sep]

            n_Li3_data = pd.read_csv(filenames["n_Li3"], header=None).apply(pd.to_numeric, errors='coerce').values
            n_Li3_i = n_Li3_data[ixmp, sep]

            n_Li2_data = pd.read_csv(filenames["n_Li2"], header=None).apply(pd.to_numeric, errors='coerce').values
            n_Li2_i = n_Li2_data[ixmp, sep]

            n_Li1_data = pd.read_csv(filenames["n_Li1"], header=None).apply(pd.to_numeric, errors='coerce').values
            n_Li1_i = n_Li1_data[ixmp, sep]

            ne_data = np.load(filenames["ne"])
            ne_i = ne_data[ixmp, sep] if ne_data.ndim > 1 else ne_data[ixmp]

            phi_sput_i = np.sum(np.loadtxt(filenames["PS"], delimiter=',') * sxnp)
            evap_i = np.sum(np.loadtxt(filenames["Evap"], delimiter=',') * sxnp)
            ad_i = np.sum(np.loadtxt(filenames["Ad"], delimiter=',') * sxnp)
            total_i = np.sum(np.loadtxt(filenames["Total"], delimiter=',') * sxnp)

        except FileNotFoundError as e:
            print(f"[{i}] File not found: {e}")
        except Exception as e:
            print(f"[{i}] Error reading files: {e}")

        max_value_tsurf.append(max_tsurf)
        max_q.append(max_q_i)
        max_q_Li_list.append(max_q_Li_i)
        C_Li_omp.append(C_Li_i)
        Te.append(Te_i)
        n_Li3.append(n_Li3_i + n_Li2_i + n_Li1_i)
        n_Li2.append(n_Li2_i)
        n_Li1.append(n_Li1_i)
        ne.append(ne_i)
        phi_sput.append(phi_sput_i)
        evap.append(evap_i)
        ad.append(ad_i)
        total.append(total_i)

 
    def replace_with_linear_interpolation(arr):
        arr = pd.Series(arr)
        arr_interpolated = arr.interpolate(method='linear', limit_direction='both')
        return arr_interpolated.bfill().ffill().to_numpy()

    max_value_tsurf = replace_with_linear_interpolation(max_value_tsurf)
    max_q = replace_with_linear_interpolation(max_q)
    max_q_Li_list = replace_with_linear_interpolation(max_q_Li_list)

    C_Li_omp = np.array(C_Li_omp, dtype=object)
    n_Li3 = replace_with_linear_interpolation(n_Li3)
    n_Li2 = replace_with_linear_interpolation(n_Li2)
    n_Li1 = replace_with_linear_interpolation(n_Li1)
    Te = replace_with_linear_interpolation(Te)
    ne = replace_with_linear_interpolation(ne)
    phi_sput = replace_with_linear_interpolation(phi_sput)
    evap = replace_with_linear_interpolation(evap)
    ad = replace_with_linear_interpolation(ad)
    total = replace_with_linear_interpolation(total)

    if eval_Li_evap_at_T_Cel is None:
        def eval_Li_evap_at_T_Cel(T): return np.nan

    for max_tsurf in max_value_tsurf:
        try:
            evap_flux = eval_Li_evap_at_T_Cel(max_tsurf) if not np.isnan(max_tsurf) else np.nan
        except Exception as e:
            print(f"Evaporation evaluation error at T={max_tsurf}: {e}")
            evap_flux = np.nan
        evap_flux_max.append(evap_flux)

    evap_flux_max = replace_with_linear_interpolation(evap_flux_max)

    q_surface = np.array(max_q) - 2.26e-19 * np.array(evap_flux_max)
    time_axis = dt * np.arange(1, len(max_q) + 1)

    return (
        max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis,
        max_q_Li_list, C_Li_omp, n_Li3, n_Li2, n_Li1, Te, ne,
        phi_sput, evap, ad, total
    )





fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
colors = ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'purple']

for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3, n_Li2, n_Li1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    ax1.plot(time_axis, np.array(max_q) / 1e6, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])
    ax2.plot(time_axis, max_value_tsurf, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])
    ax3.plot(time_axis, C_Li_omp * 100, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])

ax1.set_ylabel('q$_{\perp}^{max}$ (MW/m$^2$)', fontsize=18)
ax1.set_xlim([0, 5])
ax1.set_ylim([0, 10])

ax1.grid(True)
ax1.tick_params(axis='both', labelsize=14)

ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
ax2.set_ylim([0, 750])
ax2.set_xlim([0, 5])
ax2.legend(loc='best', fontsize=12, ncol=2)
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=14)

ax3.set_ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=18)
ax3.set_ylim([0, 15])
ax3.set_xlim([0, 5])
ax3.set_xlabel('t$_{sim}$ (s)', fontsize=18)
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig('qsurf_T_surf_CLi_omp.png', dpi=300)
plt.show()



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), dpi=300, sharex=True)

for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3,  n_Li2, n_Li1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    
    ax1.plot(time_axis, np.array(max_q) / 1e6, linestyle='-', linewidth=1.5,
             label=f'{dataset["label_tsurf"]}', color=colors[idx])
    
    ax2.plot(time_axis, max_value_tsurf, linestyle='-', linewidth=1.5,
             label=f'{dataset["label_tsurf"]}', color=colors[idx])

# Axis labels and limits
ax1.set_ylabel('q$_{\perp}^{max}$ (MW/m$^2$)', fontsize=14)
ax1.set_xlim([0, 1.5])
ax1.set_ylim([0, 10])
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.tick_params(axis='both', labelsize=12)

ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=14)
ax2.set_xlabel('t$_{sim}$ (s)', fontsize=14)
ax2.set_ylim([0, 750])
ax2.set_xlim([0, 1.5])
ax2.legend(loc='best', fontsize=12, ncol=2)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig('qsurf_T_surf.pdf', format='pdf', bbox_inches='tight')   # preferred for publication
plt.savefig('qsurf_T_surf.png', dpi=600, bbox_inches='tight')        # high-res image
plt.savefig('qsurf_T_surfdt.eps', format='eps', bbox_inches='tight')   # optional legacy vector
plt.show()


line_styles = ['-', '--', '-.', ':']
#colors = ['tab:black', 'tab:orange', 'tab:green', 'tab:red']
colors = ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'purple']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4), dpi=300, sharex=True)

for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3,  nLi2, nLi1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )

    style = line_styles[idx % len(line_styles)]
    color = colors[idx % len(colors)]
    
    
    ax1.plot(time_axis, np.array(max_q) / 1e6, 
             linestyle=style, linewidth=1.5,
             label=f'{dataset["label_tsurf"]}', color=color)
    
  
    ax2.plot(time_axis, max_value_tsurf, 
             linestyle=style, linewidth=1.5,
             label=f'{dataset["label_tsurf"]}', color=color)


ax1.set_ylabel('q$_{\perp}^{max}$ (MW/m$^2$)', fontsize=14)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 10])
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.tick_params(axis='both', labelsize=12)


ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=14)
ax2.set_xlabel('t$_{simulation}$ (s)', fontsize=14)
ax2.set_ylim([0, 750])
ax2.set_xlim([0, 1])
ax2.legend(loc='best', fontsize=10, ncol=1)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.tick_params(axis='both', labelsize=12)


plt.tight_layout()
plt.savefig('qsurf_T_surf.pdf', format='pdf', bbox_inches='tight')
plt.savefig('qsurf_T_surf.png', dpi=600, bbox_inches='tight')
plt.savefig('qsurf_T_surfdt.eps', format='eps', bbox_inches='tight')
plt.show()



line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
markers = ['o', 's', '^', 'D', 'x', 'o']  # Only needed for last two
colors = ['r', 'g', 'b', 'k', 'm', 'c', 'y', 'purple']


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4), dpi=300, sharex=True)


num_datasets = len(datasets)

for idx, dataset in enumerate(datasets):
    # Unpack the dataset variables (your actual function here)
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3,  nLi2, nLi1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )

    style = line_styles[idx % len(line_styles)]
    color = colors[idx % len(colors)]
    label = dataset["label_tsurf"]

    # Use markers only for the last two datasets
    if idx >= num_datasets - 1:
        marker = markers[idx % len(markers)]
        markersettings = {'marker': marker, 'markevery': 10, 'markersize': 6}
    else:
        markersettings = {}

    # --- q_surf plot ---
    ax1.plot(time_axis, np.array(max_q_Li) / 1e6,
             linestyle=style, linewidth=1.5,
             color=color, label=label, **markersettings)

    # --- T_surf plot ---
    ax2.plot(time_axis, max_value_tsurf,
             linestyle=style, linewidth=1.5,
             color=color, label=label, **markersettings)

# === Axis labels and limits ===
ax1.set_ylabel('q$_{s}^{max}$ (MW/m$^2$)', fontsize=14)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 10])
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.tick_params(axis='both', labelsize=12)

ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=14)
ax2.set_xlabel('t$_{simulation}$ (s)', fontsize=14)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 600])
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.tick_params(axis='both', labelsize=12)
ax2.legend(loc='best', fontsize=9, ncol=1)

# === Save and display ===
plt.tight_layout()
plt.savefig('qsurf_T_surf.pdf', format='pdf', bbox_inches='tight')
plt.savefig('qsurf_T_surf.png', dpi=600, bbox_inches='tight')
plt.savefig('qsurf_T_surfdt.eps', format='eps', bbox_inches='tight')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.linalg import norm

# --- Select the reference dataset (smallest dt assumed to be first) ---
ref_dataset = datasets[0]
ref_results = process_dataset(ref_dataset['path'], ref_dataset['nx'], ref_dataset['dt'])
_, _, _, _, ref_time, ref_q_Li, *_ = ref_results
ref_q = np.array(ref_q_Li)

# Prepare for plotting and error tracking
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4), dpi=300, sharex=True)
num_datasets = len(datasets)
errors = []

# --- Loop through datasets ---
for idx, dataset in enumerate(datasets):
    # Process the dataset
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3, Te, ne,  nLi2, nLi1, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )

    # Plotting style
    style = line_styles[idx % len(line_styles)]
    color = colors[idx % len(colors)]
    label = dataset["label_tsurf"]

    # Use markers only for last two datasets
    if idx >= num_datasets - 2:
        marker = markers[idx % len(markers)]
        markersettings = {'marker': marker, 'markevery': 10, 'markersize': 6}
    else:
        markersettings = {}

    # --- Plot q_surf ---
    ax1.plot(time_axis, np.array(max_q_Li) / 1e6,
             linestyle=style, linewidth=1.5,
             color=color, label=label, **markersettings)

    # --- Plot T_surf ---
    ax2.plot(time_axis, max_value_tsurf,
             linestyle=style, linewidth=1.5,
             color=color, label=label, **markersettings)

    # --- Error analysis (skip ref dataset) ---
    if idx != 0:
        q = np.array(max_q_Li)
        interp_func = interp1d(time_axis, q, kind='linear', bounds_error=False, fill_value="extrapolate")
        q_interp = interp_func(ref_time)
        l2_error = norm(q_interp - ref_q) / norm(ref_q)
        errors.append((dataset['dt'], l2_error))

# --- Finalize q_surf subplot ---
ax1.set_ylabel('q$_{s}^{max}$ (MW/m$^2$)', fontsize=14)
ax1.set_xlim([0, 2])
ax1.set_ylim([0, 10])
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.tick_params(axis='both', labelsize=12)

# --- Finalize T_surf subplot ---
ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=14)
ax2.set_xlabel('t$_{simulation}$ (s)', fontsize=14)
ax2.set_xlim([0, 2])
ax2.set_ylim([0, 650])
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.tick_params(axis='both', labelsize=12)
ax2.legend(loc='best', fontsize=9, ncol=2)

# --- Save figure ---
plt.tight_layout()
plt.savefig('qsurf_T_surf.pdf', format='pdf', bbox_inches='tight')
plt.savefig('qsurf_T_surf.png', dpi=600, bbox_inches='tight')
plt.savefig('qsurf_T_surfdt.eps', format='eps', bbox_inches='tight')
plt.show()

# === Error plot (L2 norm vs. time step) ===
if errors:
    dts, errs = zip(*errors)
    plt.figure(dpi=150)
    plt.loglog(dts, errs, 'o-', label='Relative L2 Error')
    plt.xlabel('Time step Δt (s)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Time Step Convergence', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_plot.png", dpi=300)
    plt.show()

    # Print error values to console
    print("\n=== Time Step Error Analysis ===")
    for dt, err in errors:
        print(f"dt = {dt:.1e} -> Relative L2 Error = {err:.3e}")


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
colors = ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'purple']

for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3,  nLi2, nLi1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    ax1.plot(time_axis, max_value_tsurf, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])
    ax2.plot(time_axis, np.array(max_q) / 1e6, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])
    ax3.plot(time_axis, np.array(max_q_Li) / 1e6, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])

ax1.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
ax1.set_ylim([0, 750])
ax1.set_xlim([0, 5])
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=14)

ax2.set_ylabel('q$_{\perp}^{max}$ (MW/m$^2$)', fontsize=18)
ax2.set_xlim([0, 5])
ax2.set_ylim([0, 20])
ax2.legend(loc='best', fontsize=12, ncol=2)
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=14)

ax3.set_ylabel('q$_{s}^{max}$ (MW/m$^2$)', fontsize=18)
ax3.set_xlim([0, 5])
ax3.set_ylim([0, 20])
ax3.legend(loc='best', fontsize=12, ncol=2)
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=14)
ax3.set_xlabel('t$_{sim}$ (s)', fontsize=18)


plt.tight_layout()
plt.savefig('qsurf_T_surf_qperp.png', dpi=300)
plt.show()





fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
colors = ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'purple']

for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3,  nLi2, nLi1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    ax1.plot(time_axis, np.array(max_q) / 1e6, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])
    ax2.plot(time_axis, max_value_tsurf, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])
    ax3.plot(time_axis, nLi3, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])

ax1.set_ylabel('q$_{\perp}^{max}$ (MW/m$^2$)', fontsize=18)
ax1.set_xlim([0, 5])
ax1.set_ylim([0, 15])
ax1.legend(loc='best', fontsize=12, ncol=2)
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=14)

ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
ax2.set_ylim([0, 750])
ax2.set_xlim([0, 5])
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=14)

ax3.set_ylabel("n$_{Li-sep}^{omp}$ (m$^{-3}$)", fontsize=18)
ax3.set_ylim([0, 10e18])
ax3.set_xlim([0, 5])
ax3.set_xlabel('t$_{sim}$ (s)', fontsize=18)
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig('qsurf_T_surf_nLi_omp.png', dpi=300)
plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
colors = ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'purple']

for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3,  nLi2, nLi1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    ax1.plot(time_axis, phi_sput/1e22, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])
    ax2.plot(time_axis, evap/1e23, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])
    ax3.plot(time_axis, ad/1e20, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])
    
ax1.text(0.3, 0.25, '(a)', fontsize=16, fontweight='bold')  # Adjust x/y as needed
ax2.text(0.3, 0.4, '(b)', fontsize=16, fontweight='bold')
ax3.text(0.3, 0.25, '(c)', fontsize=16, fontweight='bold')

ax1.set_ylabel('$\phi_{PS}$ (10$^{22}$ atom/s)', fontsize=16)
ax1.set_xlim([0, 5])
ax1.set_ylim([0, 0.3])
#ax1.legend(loc='best', fontsize=12, ncol=2)
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=14)

ax2.set_ylabel("$\phi_{Ev}$ (10$^{23}$ atom/s)", fontsize=16)
ax2.set_ylim([0, 0.5])
#ax2.legend(loc='best', fontsize=12, ncol=2)
ax2.set_xlim([0, 5])
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=14)

ax3.set_ylabel("$\phi_{ad}$ (10$^{20}$ atom/s)", fontsize=16)
ax3.set_ylim([0, 0.3])
ax3.set_xlim([0, 5])
ax3.set_xlabel('t$_{simulation}$ (s)', fontsize=18)
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig('Phi_Li.png', dpi=300)
plt.savefig('Phi_Li.eps', format='eps', bbox_inches='tight')  
plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
colors = ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'purple']

for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3,  nLi2, nLi1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    ax1.plot(time_axis, phi_sput/1e22, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])
    ax2.plot(time_axis, evap/1e23, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])
    ax3.plot(time_axis, ad/1e20, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx])
    
ax1.text(0.3, 0.25, '(a)', fontsize=16, fontweight='bold')  # Adjust x/y as needed
ax2.text(0.3, 0.4, '(b)', fontsize=16, fontweight='bold')
ax3.text(0.3, 0.25, '(c)', fontsize=16, fontweight='bold')

ax1.set_ylabel('$\phi_{PS}$ (10$^{22}$ atom/s)', fontsize=16)
ax1.set_xlim([0, 5])
ax1.set_ylim([0, 0.3])
#ax1.legend(loc='best', fontsize=12, ncol=2)
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=14)

ax2.set_ylabel("$\phi_{Ev}$ (10$^{23}$ atom/s)", fontsize=16)
ax2.set_ylim([0, 0.5])
#ax2.legend(loc='best', fontsize=12, ncol=2)
ax2.set_xlim([0, 5])
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=14)

ax3.set_ylabel("$\phi_{ad}$ (10$^{20}$ atom/s)", fontsize=16)
ax3.set_ylim([0, 0.3])
ax3.set_xlim([0, 5])
ax3.set_xlabel('t$_{simulation}$ (s)', fontsize=18)
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig('Phi_Li.png', dpi=300)
plt.savefig('Phi_Li.eps', format='eps', bbox_inches='tight')  
plt.show()





fig, ax = plt.subplots(figsize=(4, 3))

for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3, nLi2, nLi1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    

    ax.plot(time_axis, evap, linewidth=2, linestyle='-', color='red', label='Evaporation')
    ax.plot(time_axis, phi_sput, linewidth=2, linestyle='--', color='blue', label='Phy. Sput')
    ax.plot(time_axis, ad, linewidth=3, linestyle=':', color='green', label='Ad-atom')

ax.set_xlabel('t$_{simulation}$ (s)', fontsize=16)
ax.set_ylabel('$\phi_{Li}^{Emitted}$ (atom/s)', fontsize=16)
ax.set_xlim([0, 5])
ax.set_ylim([1e15, 1e23])  # Adjusted to match log scale visibility
ax.set_yscale('log')
ax.tick_params(axis='both', labelsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(True, which='both', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('Phi_Li_combined.png', dpi=300)
plt.savefig('Phi_Li_combined.eps', format='eps', bbox_inches='tight')
plt.show()



for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3,  nLi2, nLi1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    plt.plot(max_value_tsurf, C_Li_omp * 100, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}',  color=colors[idx])

plt.xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
plt.ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=18)
plt.legend(fontsize=14)
plt.axhline(3, color='black', linestyle=':', linewidth=2, label='y = 3') 
plt.ylim([0, 15])
plt.xlim([0, 700])

plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('T_surf_CLi_omp.png', dpi=300)
plt.show()

for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3,  nLi2, nLi1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    plt.plot(max_value_tsurf, nLi3, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}',  color=colors[idx])

plt.xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
plt.ylabel("n$_{Li-sep}^{omp}$ (m$^{-3}$)", fontsize=18)
plt.legend(fontsize=14)
plt.axhline(3, color='black', linestyle=':', linewidth=2, label='y = 3')  # Reference line at y=3
plt.ylim([0, 2e18])
plt.xlim([0, 700])

plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('T_surf_nLi_omp.png', dpi=300)
plt.show()


plt.figure(figsize=(4, 2.25))
for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3,  nLi2, nLi1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    plt.plot(total/1e21, nLi3, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}',  color=colors[idx])

plt.xlabel("$\phi_{Li}$ ($10^{21}$atom/s)", fontsize=18)
plt.ylabel("n$_{Li-sep}^{omp}$ (m$^{-3}$)", fontsize=18)
plt.legend(fontsize=11)
plt.ylim([0, 1e17])
plt.xlim([0, 2])

plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('Cs_Phi_Li_nLi_omp.eps', format='eps', dpi=600)  # High-res EPS for publication
plt.savefig('Cs_Phi_Li_nLi_omp.jpg', format='jpg', dpi=600)  # High-res JPG
plt.savefig('Cs_Phi_Li_nLi_omp.png', format='png', dpi=300)  
plt.show()


plt.figure(figsize=(4, 3))

line_styles = ['-', '--', '-.', ':']

for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3,  nLi2, nLi1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    plt.plot(
        total/1e20,
        nLi3/1e16,
        linestyle=line_styles[idx % len(line_styles)],
        linewidth=2,
        label=f'{dataset["label_tsurf"]}',
        color=colors[idx]
    )

plt.xlabel(r"$\phi_{Li}$ ($10^{20}$ atom/s)", fontsize=18)
plt.ylabel(r"$n_{Li-sep}^{omp}$ ($10^{16}$ m$^{-3}$)", fontsize=18)
plt.legend(fontsize=11, loc='lower right')  # Legend at bottom right
#plt.yscale('log')
plt.ylim([0, 5])
plt.xlim([0, 5])


plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()

plt.savefig('Cs_Phi_Li_nLi_omp.eps', format='eps', dpi=600)
plt.savefig('Cs_Phi_Li_nLi_omp.jpg', format='jpg', dpi=600)
plt.savefig('Cs_Phi_Li_nLi_omp.png', format='png', dpi=300)

plt.show()


for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3,  nLi2, nLi1, Te, ne, phi_sput, evap, ad, total = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    plt.plot(time_axis, total, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}',  color=colors[idx])

plt.xlabel("t$_{sim}$ (s)", fontsize=18)
plt.ylabel("$\phi_{Li}$ (atom/s)", fontsize=18)
plt.legend(fontsize=14)
plt.axhline(3, color='black', linestyle=':', linewidth=2, label='y = 3')  # Reference line at y=3
plt.ylim([0, 1e22])
plt.xlim([0, 5])
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('tsim_Phi_Li_omp.png', dpi=300)
plt.show()




def process_dataset_profile(data_path, nx, dt, yyrb, sxnp, sep=8, ixmp=36):
    qsurf_Li_profiles = []
    qsurf_profiles = []
    total_Li_flux = []

    dirs = {
        "q_perp": os.path.join(data_path, 'q_perp'),
        "Tsurf_Li": os.path.join(data_path, 'Tsurf_Li'),
        "q_Li_surface": os.path.join(data_path, 'q_Li_surface'),
        "Li": os.path.join(data_path, 'Gamma_Li'),
    }

    for i in range(1, nx):
        filenames = {
            "tsurf": os.path.join(dirs["Tsurf_Li"], f'T_surfit_{i}.0.csv'),
            "qsurf": os.path.join(dirs["q_perp"], f'q_perpit_{i}.0.csv'),
            "qsurf_Li": os.path.join(dirs["q_Li_surface"], f'q_Li_surface_{i}.0.csv'),
            "Total": os.path.join(dirs["Li"], f'Total_Li_flux_{i}.0.csv')
        }

        try:
            qsurf_Li = np.loadtxt(filenames["qsurf_Li"])
            qsurf = np.loadtxt(filenames["qsurf"])
            total_i = np.sum(np.loadtxt(filenames["Total"]) * sxnp)
        except Exception as e:
            print(f"File not found or error: {e}")
            qsurf_Li = np.full_like(yyrb, np.nan)
            qsurf = np.full_like(yyrb, np.nan)
            total_i = np.nan

        qsurf_Li_profiles.append(qsurf_Li)
        qsurf_profiles.append(qsurf)
        total_Li_flux.append(total_i)

    qsurf_Li_profiles = np.array(qsurf_Li_profiles)
    qsurf_profiles = np.array(qsurf_profiles)
    total_Li_flux = np.array(total_Li_flux)
    total_Li_flux = total_Li_flux/1e21

    # --- Scatter plot ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    sc1 = ax[0].scatter(
        np.tile(yyrb, nx-1),
        qsurf_profiles.flatten(),
        c=np.repeat(total_Li_flux, len(yyrb)),
        cmap='plasma'
    )
    ax[0].set_xlabel('yyrb')
    ax[0].set_ylabel('q_perp')
    ax[0].set_title('q_perp Profile (colored by Total Lithium)')
    fig.colorbar(sc1, ax=ax[0], label='Total Lithium Flux')

    sc2 = ax[1].scatter(
        np.tile(yyrb, nx-1),
        qsurf_Li_profiles.flatten(),
        c=np.repeat(total_Li_flux, len(yyrb)),
        cmap='plasma'
    )
    ax[1].set_xlabel('yyrb')
    ax[1].set_ylabel('q_Li_surface')
    ax[1].set_title('q_Li_surface Profile (colored by Total Lithium)')
    fig.colorbar(sc2, ax=ax[1], label='Total Lithium Flux')

    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(2, 1, figsize=(4, 6), sharex=True)
    segments_q = [np.column_stack([yyrb, qsurf_profiles[i]/1e6]) for i in range(qsurf_profiles.shape[0])]
    lc_q = LineCollection(segments_q, cmap='plasma', array=total_Li_flux, linewidths=2)
    ax[0].add_collection(lc_q)
    ax[0].set_xlim(yyrb.min(), yyrb.max())
    ax[0].set_ylim(0, 9)
    ax[0].set_ylabel('q$_\perp$ (MW/m$^2$)', fontsize=16)
    ax[0].grid()
    ax[0].text(
    0.95, 0.95, '(a)', 
    transform=ax[0].transAxes, 
    fontsize=16, 
    va='top', ha='right'
)
    ax[0].tick_params(axis='both', labelsize=14)
   # fig.colorbar(lc_q, ax=ax[0], label='$\phi_{Li}$ (10$^{21}$ atom/s)')
    cbar = fig.colorbar(lc_q, ax=ax[0], label='$\phi_{Li}$ (10$^{21}$ atom/s)')
    cbar.ax.tick_params(labelsize=14)  # Set tick label font size
    cbar.set_label('$\phi_{Li}$ (10$^{21}$ atom/s)', fontsize=16)  # Set colorbar label font size

    segments_li = [np.column_stack([yyrb, qsurf_Li_profiles[i]/1e6]) for i in range(qsurf_Li_profiles.shape[0])]
    lc_li = LineCollection(segments_li, cmap='plasma', array=total_Li_flux, linewidths=2)
    ax[1].add_collection(lc_li)
    ax[1].set_xlim(yyrb.min(), yyrb.max())
    ax[1].set_ylim(0, 9)
    ax[1].set_xlabel('r$_{div}$ - r$_{sep}$ (m)', fontsize=16)
    ax[1].set_ylabel('q$_{s}$ (MW/m$^2$)', fontsize=16)
    ax[1].grid()
    ax[1].text(
    0.95, 0.95, '(b)', 
    transform=ax[1].transAxes, 
    fontsize=16, 
    va='top', ha='right'
)
    ax[1].tick_params(axis='both', labelsize=14)
    #fig.colorbar(lc_li, ax=ax[1], label='$\phi_{Li}$ (10$^{21}$ atom/s)')
    cbar = fig.colorbar(lc_li, ax=ax[1], label='$\phi_{Li}$ (10$^{21}$ atom/s)')
    cbar.ax.tick_params(labelsize=14)  # Set tick label font size
    cbar.set_label('$\phi_{Li}$ (10$^{21}$ atom/s)', fontsize=16)  # Set colorbar label font size
    plt.tight_layout()
    plt.savefig('q_perp_psurf.png', dpi=300)
    plt.show()

    return qsurf_profiles, qsurf_Li_profiles, total_Li_flux


qsurf_profiles, qsurf_Li_profiles, total_Li_flux = process_dataset_profile(
    dataset['path'], dataset['nx'], dataset['dt'], yyrb, sxnp
)
   
   