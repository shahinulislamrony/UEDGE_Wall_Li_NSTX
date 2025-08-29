# -*- coding: utf-8 -*-
"""
Updated on 2025-07-17
@author: islam9
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# -------- PATH CONFIG --------
DATA_PATH = r'/global/cfs/cdirs/mp2/shahinul/Nuclear_Fusion/low_FX/PePi8.0MW'

import os
import glob
import pandas as pd
import numpy as np

def read_csv(filepath):
    try:
        return pd.read_csv(filepath).values
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

# look inside the Tsurf_Li folder
files = sorted(glob.glob(os.path.join("Tsurf_Li", "T_surfit_*.csv")))

Tsurf_max_values = []

for filename in files:
    data = read_csv(filename)
    if data is not None:
        Tsurf_max_values.append(np.nanmax(data))

Tsurf_max_values = np.array(Tsurf_max_values)



if len(Tsurf_max_values) == 0:
    raise RuntimeError("No T_surfit_*.csv files found in Tsurf_Li/.")

print("? Tsurf_Li max values loaded:", Tsurf_max_values.shape, 
      "Example:", Tsurf_max_values[:5])

Li_source_odiv = Tsurf_max_values

def replace_with_linear_interpolation(arr):
    arr = pd.Series(arr).infer_objects()
    return arr.interpolate(method='linear', limit_direction='both').bfill().ffill().to_numpy()


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
    arr = np.array(arr)
    sxnp = np.array(sxnp)
    arr = np.squeeze(arr)
    sxnp = np.squeeze(sxnp)
    if arr.ndim > 1:
        print(f"Warning: {label} array has shape {arr.shape}, flattening.")
        arr = arr.flatten()
    if sxnp.ndim > 1:
        print(f"Warning: sxnp array has shape {sxnp.shape}, flattening.")
        sxnp = sxnp.flatten()
    if arr.shape != sxnp.shape:
        minlen = min(arr.shape[0], sxnp.shape[0])
        print(f"Warning: {label} and sxnp shape mismatch: {arr.shape} vs {sxnp.shape}. Truncating to {minlen}.")
        arr = arr[:minlen]
        sxnp = sxnp[:minlen]
    try:
        arr = arr.astype(float)
        sxnp = sxnp.astype(float)
    except Exception as e:
        print(f"Error converting {label} or sxnp to float: {e}")
        return np.nan
    if arr.shape != sxnp.shape:
        print(f"Final shape mismatch for {label}: {arr.shape} vs {sxnp.shape}. Returning nan.")
        return np.nan
    return np.sum(arr * sxnp)


def process_dataset(data_path, nx):
    max_value_tsurf, max_q, max_q_Li_list, C_Li_omp, total = [], [], [], [], []

    dirs = {
        "q_perp": os.path.join(data_path, 'q_perp'),
        "Tsurf_Li": os.path.join(data_path, 'Tsurf_Li'),
        "q_Li_surface": os.path.join(data_path, 'q_Li_surface'),
        "C_Li_omp": os.path.join(data_path, 'C_Li_omp'),
        "Li": os.path.join(data_path, 'Gamma_Li'),
    }

    for i in range(1, nx):
        files = {
            "tsurf": os.path.join(dirs["Tsurf_Li"], f'T_surfit_{i}.csv'),
            "qsurf": os.path.join(dirs["q_perp"], f'q_perpit_{i}.csv'),
            "qsurf_Li": os.path.join(dirs["q_Li_surface"], f'q_Li_surface_{i}.csv'),
            "C_Li": os.path.join(dirs["C_Li_omp"], f'CLi_prof_{i}.csv'),
            "PS": os.path.join(dirs["Li"], f'PhysSput_flux_{i}.csv'),
            "Evap": os.path.join(dirs["Li"], f'Evap_flux_{i}.csv'),
            "Ad": os.path.join(dirs["Li"], f'Adstom_flux_{i}.csv'),
        }

        max_tsurf = max_q_i = max_q_Li_i = C_Li_i = total_i = np.nan

        if os.path.exists(files["tsurf"]):
            try:
                df_tsurf = pd.read_csv(files["tsurf"])
                max_tsurf = np.nanmax(df_tsurf.values)
            except Exception as e:
                print(f"?? Error reading Tsurf file {files['tsurf']}: {e}")
        else:
            print(f"? Missing Tsurf file: {files['tsurf']}")

        if os.path.exists(files["qsurf"]):
            try:
                if os.path.getsize(files["qsurf"]) > 0:
                    df_qsurf = pd.read_csv(files["qsurf"])
                    max_q_i = np.nanmax(df_qsurf.values)
                else:
                    print(f"? Empty q_perp file: {files['qsurf']}")
            except Exception as e:
                print(f"?? Error reading q_perp file {files['qsurf']}: {e}")
        else:
            print(f"? Missing q_perp file: {files['qsurf']}")

        if os.path.exists(files["qsurf_Li"]):
            try:
                if os.path.getsize(files["qsurf_Li"]) > 0:
                    df_qsurf_Li = pd.read_csv(files["qsurf_Li"])
                    max_q_Li_i = np.nanmax(df_qsurf_Li.values)
                else:
                    print(f"? Empty q_Li_surface file: {files['qsurf_Li']}")
            except Exception as e:
                print(f"?? Error reading q_Li_surface file {files['qsurf_Li']}: {e}")
        else:
            print(f"? Missing q_Li_surface file: {files['qsurf_Li']}")

        if os.path.exists(files["C_Li"]):
            try:
                df_C_Li = pd.read_csv(files["C_Li"])
                sep = 8
                if df_C_Li.shape[0] > sep:
                    C_Li_i = df_C_Li.values[sep]
                else:
                    print(f"?? C_Li_omp file {files['C_Li']} has insufficient rows.")
            except Exception as e:
                print(f"?? Error reading C_Li_omp file {files['C_Li']}: {e}")
        else:
            print(f"? Missing C_Li_omp file: {files['C_Li']}")

        ps_arr = load_data_auto(files["PS"])
        evap_arr = load_data_auto(files["Evap"])
        ad_arr = load_data_auto(files["Ad"])
        phi_sput_i = safe_weighted_sum(ps_arr, sxnp, "PhysSput_flux")
        evap_i = safe_weighted_sum(evap_arr, sxnp, "Evap_flux")
        ad_i = safe_weighted_sum(ad_arr, sxnp, "Adstom_flux")
        total_i = (phi_sput_i + evap_i + ad_i)/1e22

        max_value_tsurf.append(max_tsurf)
        max_q.append(max_q_i)
        max_q_Li_list.append(max_q_Li_i)
        C_Li_omp.append(C_Li_i)
        total.append(total_i)

    max_value_tsurf = replace_with_linear_interpolation(max_value_tsurf)
    max_q = replace_with_linear_interpolation(max_q)
    max_q_Li_list = replace_with_linear_interpolation(max_q_Li_list)
    C_Li_omp = replace_with_linear_interpolation(C_Li_omp)
    total = replace_with_linear_interpolation(total)

    return max_value_tsurf, max_q, max_q_Li_list, C_Li_omp, total


def get_color(value, cmap, norm):
    return cmap(norm(value))


def plot_data(ax, x, y, color, label=""):
    ax.plot(x, y, color=color, label=label)


def process_and_plot_multiple(datasets, y, max_value_tsurf, total, output_file):
    fig, axs = plt.subplots(len(datasets), 1, figsize=(4.25, 3), sharex=True)
    if len(datasets) == 1:
        axs = [axs]

    nx = min(len(total), len(max_value_tsurf))
    for i, data_info in enumerate(datasets):
        ax = axs[i]
        print(f"?? Processing: {data_info['ylabel']}")
        max_plot_value = 0

        norm = Normalize(vmin=0, vmax=np.nanmax(total)*1.05)
        sm = ScalarMappable(cmap=data_info['cmap'], norm=norm)
        sm.set_array([])

        for j in range(1, nx):
            if data_info.get('use_npy', False):
                filename = os.path.join(data_info['directory'], f"{data_info['file_prefix']}_{j}.npy")
                if not os.path.exists(filename):
                    print(f"? Missing npy file for iteration {j}: {filename}")
                    continue
            else:
                filename_csv = os.path.join(data_info['directory'], f"{data_info['file_prefix']}_{j}.csv")
                if os.path.exists(filename_csv) and os.path.getsize(filename_csv) > 0:
                    filename = filename_csv
                else:
                    print(f"? Missing or empty csv file for iteration {j}: {filename_csv}")
                    continue

            try:
                if filename.endswith('.csv') or filename.endswith('.txt'):
                    data = pd.read_csv(filename, header=None).values
                elif filename.endswith('.npy'):
                    data = np.load(filename)
                else:
                    continue

                if data_info.get('is_sxnp', False):
                    numbers = data.flatten()[:-1] / data_info['sxnp']
                elif data_info.get('is_2D', False):
                    if data.ndim == 2 and data.shape[0] > 52:
                        numbers = data[52, :-1]
                    else:
                        numbers = data.flatten()[:-1]
                elif data_info.get('is_evap', False):
                    numbers = data.flatten()[:-1] * data_info['evap']
                else:
                    numbers = data.flatten()[:-1]

                if 'unit_scale' in data_info:
                    numbers = numbers * data_info['unit_scale']

                if len(numbers) != len(y):
                    print(f"?? Skipping iteration {j} due to length mismatch: {len(numbers)} vs {len(y)}")
                    continue

                max_plot_value = max(max_plot_value, np.nanmax(numbers))

                color = get_color(total[j], data_info['cmap'], norm)
                plot_data(ax, y, numbers, color, label=f"Iter {j}" if j % 10 == 0 else "")

                print(f"Plotting {data_info['ylabel']} iter {j}, max val: {np.nanmax(numbers):.3e}")

            except Exception as e:
                print(f"? Error loading {filename}: {e}")
                continue

        ax.set_ylabel(data_info['ylabel'], fontsize=14)
        ax.grid(True)
        #ax.set_ylim(0, max_plot_value * 1.05 if max_plot_value > 0 else 1)
        ax.set_yscale('symlog', linthresh=1e0)  # Adjust linthresh as needed
        ax.set_ylabel(data_info['ylabel'], fontsize=14)
        ax.grid(True, which='both')

        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
        cbar.set_label('$\phi_{Li}$ (10$^{22}$ atom/s)', fontsize=10)

        #ax.text(0.95, 0.95, f"({chr(97 + i)})", transform=ax.transAxes,
         #       fontsize=14, va='top', ha='right')

    axs[-1].set_xlabel(r"r$_{div}$ - r$_{sep}$ (m)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()


# -------- MAIN --------
if __name__ == "__main__":
    nx = 113
    y = np.array([
        -0.02640231, -0.02372757, -0.01875399, -0.01441994, -0.01073658,
       -0.00764067, -0.00501472, -0.00277724, -0.00087231,  0.00533494,
        0.01609611,  0.02663718,  0.03709092,  0.0481553 ,  0.05976873,
        0.07184217,  0.08440904,  0.09748994,  0.1110626 ,  0.12505532,
        0.13942652,  0.15425184,  0.16937158,  0.18483239,  0.20088217
           ])
    sxnp = np.array([
       1.89751110e-08, 1.90652313e-02, 1.66399846e-02, 1.44392237e-02,
       1.22260820e-02, 1.03981165e-02, 8.88177354e-03, 7.61129701e-03,
       6.47890467e-03, 3.92735212e-02, 4.08973113e-02, 4.01229252e-02,
       4.20196483e-02, 4.55164112e-02, 4.82451257e-02, 5.08852886e-02,
       5.40979110e-02, 5.71332801e-02, 6.04026117e-02, 6.29738525e-02,
       6.61572232e-02, 6.95500878e-02, 7.14773793e-02, 7.55555667e-02,
       8.00399311e-02
       ])
    evap = 2.44e-19

    max_value_tsurf, max_q, max_q_Li_list, C_Li_omp, total = process_dataset(DATA_PATH, nx)
    print("? max_value_tsurf sample:", max_value_tsurf[:5])

    cmap = plt.get_cmap('turbo')
    norm = Normalize(vmin=0, vmax=np.max(total) * 1.05)

    datasets = [

        dict(directory=os.path.join(DATA_PATH, 'Gamma_net'),
             file_prefix='Gamma_Li_surface',
             is_2D=False, is_sxnp=False, is_evap=False,
             sxnp=sxnp, evap=evap, cmap=cmap, norm=norm,
              unit_scale=1e-22,
             ylabel='$\Gamma_{Li}^{net}$ (10$^{22}$ m$^{-2}$s$^{-1}$)')
    ]

    process_and_plot_multiple(datasets, y, max_value_tsurf, total, "Gamma_net.png")
