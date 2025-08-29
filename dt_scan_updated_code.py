# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 14:14:06 2025

@author: islam9
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- Plotting Style ----------
mpl.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2,
    'axes.grid': True,
})

# ---------- Dataset Information ----------
folders = {
    
    "nx_P2": r"C:\UEDGE_run_Shahinul\Nuclear_Fusion\Nuclear_Fusion\high_FX\PePi5.8MW\C_Li_omp",
    "nx_P3": r"C:\UEDGE_run_Shahinul\Nuclear_Fusion\Nuclear_Fusion\high_FX\PePi6.8MW\C_Li_omp",

}

def count_files_in_folder(folder_path):
    return len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])

file_counts = {key: count_files_in_folder(path) for key, path in folders.items()}

datasets = [
    
    {'path': r'C:\UEDGE_run_Shahinul\Nuclear_Fusion\Nuclear_Fusion\high_FX\PePi5.8MW',  'nx': file_counts["nx_P2"], 'dt': 10e-3,  'label_tsurf': r'P: 5.8MW'},
    {'path': r'C:\UEDGE_run_Shahinul\Nuclear_Fusion\Nuclear_Fusion\high_FX\PePi6.8MW', 'nx': file_counts["nx_P3"], 'dt': 10e-3, 'label_tsurf': r'P: 6.8MW'},
  # {'path': r'C:\UEDGE_run_Shahinul\Wall_Li_for_NSTX_PoP_Shahinul\dt20ms', 'nx': file_counts["nx_P4"], 'dt': 20e-3, 'label_tsurf': r'$\Delta t$: 20ms'},
   # {'path': r'C:\UEDGE_run_Shahinul\Wall_Li_for_NSTX_PoP_Shahinul\dt30ms', 'nx': file_counts["nx_P5"], 'dt': 30e-3, 'label_tsurf': r'$\Delta t$: 30ms'},
   # {'path': r'C:\UEDGE_run_Shahinul\Wall_Li_for_NSTX_PoP_Shahinul\dt40ms\dt40ms', 'nx': file_counts["nx_P6"], 'dt': 40e-3, 'label_tsurf': r'$\Delta t$: 40ms'},
]

# ---------- Data Processor ----------

def process_Tsurf_and_qLi(path, nx, dt):
    tsurf_max_list = []
    q_Li_max_list = []

    tsurf_dir = os.path.join(path, 'Tsurf_Li')
    q_Li_dir = os.path.join(path, 'q_Li_surface')

    for i in range(1, nx):
        try:
            # Try both naming patterns for Tsurf
            tsurf_file_a = os.path.join(tsurf_dir, f'T_surfit_{i}.csv')
            tsurf_file_b = os.path.join(tsurf_dir, f'T_surfit_{i}.0.csv')

            if os.path.exists(tsurf_file_a):
                tsurf_file = tsurf_file_a
            elif os.path.exists(tsurf_file_b):
                tsurf_file = tsurf_file_b
            else:
                raise FileNotFoundError(f"Tsurf file not found for step {i}")

            # Try both naming patterns for q_Li
            q_Li_file_a = os.path.join(q_Li_dir, f'q_Li_surface_{i}.csv')
            q_Li_file_b = os.path.join(q_Li_dir, f'q_Li_surface_{i}.0.csv')

            if os.path.exists(q_Li_file_a):
                q_Li_file = q_Li_file_a
            elif os.path.exists(q_Li_file_b):
                q_Li_file = q_Li_file_b
            else:
                raise FileNotFoundError(f"q_Li_surface file not found for step {i}")

            tsurf_data = pd.read_csv(tsurf_file, header=None).values.astype(float)
            q_Li_data = pd.read_csv(q_Li_file, header=None).values.astype(float)

            tsurf_max_list.append(np.max(tsurf_data))
            q_Li_max_list.append(np.max(q_Li_data))

        except Exception as e:
            print(f"[{path}] Error at step {i}: {e}")
            tsurf_max_list.append(np.nan)
            q_Li_max_list.append(np.nan)

    tsurf_max = pd.Series(tsurf_max_list).interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill').to_numpy()
    q_Li_max = pd.Series(q_Li_max_list).interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill').to_numpy()

    return tsurf_max, q_Li_max

# ---------- Plotting ----------
colors = plt.cm.plasma(np.linspace(0, 1, len(datasets)))
linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
markers = ['o', 's', '^', 'D', 'v', '*']  # List of marker styles

colors = plt.cm.plasma(np.linspace(0, 1, len(datasets)))
linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
markers = ['o', 's', '^', 'D', 'v', '*']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

for idx, data in enumerate(datasets):
    tsurf_max, q_Li_max = process_Tsurf_and_qLi(data['path'], data['nx'], data['dt'])
    time = np.arange(1, data['nx']) * data['dt']

    label = data['label_tsurf']
    color = colors[idx % len(colors)]
    linestyle = linestyles[idx % len(linestyles)]
    marker = markers[idx % len(markers)]

    # For dt=10ms, remove only the 4th data point (index 3)
    if '10ms' in data['path']:
        tsurf_max = np.delete(tsurf_max, 3)
        q_Li_max = np.delete(q_Li_max, 3)
        time = np.delete(time, 3)

    stride = max(1, len(time) // 20)
    ax1.plot(time, np.array(q_Li_max) / 1e6, label=label,
             color=color, linestyle=linestyle, marker=marker,
             markevery=stride, markersize=4)
    ax2.plot(time, np.array(tsurf_max), label=label,
             color=color, linestyle=linestyle, marker=marker,
             markevery=stride, markersize=4)
    
ax1.set_ylabel(r'$q_{\mathrm{Li}}^{\mathrm{surf,max}}$ (MW/m$^2$)', fontsize=16)
ax1.set_ylim([0, 10])
ax1.set_xlim([0, 5])

ax2.set_ylabel(r'$T_{\mathrm{surf}}^{\mathrm{max}}$ ($^\circ$C)',fontsize=16)
ax2.set_ylim([0, 700])
ax2.set_xlim([0, 5])
ax2.legend(loc='best', ncol=2)

ax2.set_xlabel('Time (s)')

plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.show()

    

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

for idx, data in enumerate(datasets):
    tsurf_max, q_Li_max = process_Tsurf_and_qLi(data['path'], data['nx'], data['dt'])
    time = np.arange(1, data['nx']) * data['dt']

    # Trim initial noisy steps for dt=10ms
    if '10ms' in data['path']:
        trim_idx = 0  # Remove first 3 points
        tsurf_max = tsurf_max[trim_idx:]
        q_Li_max = q_Li_max[trim_idx:]
        time = time[trim_idx:]

    label = data['label_tsurf']
    color = colors[idx]
    linestyle = linestyles[idx % len(linestyles)]
    
    ax1.plot(time, np.array(q_Li_max) / 1e6, label=label, color=color, linestyle=linestyle)
    ax2.plot(time, tsurf_max, label=label, color=color, linestyle=linestyle)

 

# ---------- Axes Labels & Legends ----------
ax1.set_ylabel(r'$q_{\mathrm{Li}}^{\mathrm{surf,max}}$ (MW/m$^2$)', fontsize=16)
#ax1.legend(loc='best', frameon=False)
ax1.set_ylim([0, 10])
ax1.set_xlim([0, 3])

ax2.set_ylabel(r'$T_{\mathrm{surf}}^{\mathrm{max}}$ ($^\circ$C)',fontsize=16)
#ax2.set_title('(c) Surface temperature and lithium surface heat flux vs time')
ax2.set_ylim([0, 700])
ax2.set_xlim([0, 3])
ax2.legend(loc='best', ncol=2)

ax2.set_xlabel('Time (s)')

plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.show()


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from cycler import cycler

# Improved color palette and style cycler
colors = plt.get_cmap('tab10').colors[:6]
markers = ['o', 's', 'v', 'D', 'v', '*']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]

plt.rc('axes', prop_cycle=(cycler('color', colors) +
                           cycler('marker', markers) +
                           cycler('linestyle', linestyles)))

mpl.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'lines.linewidth': 2.2,
    'axes.grid': True,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'grid.alpha': 0.5,
})

# Set cycler for consistent style
plt.rc('axes', prop_cycle=(cycler('color', colors) +
                           cycler('marker', markers) +
                           cycler('linestyle', linestyles)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.25, 4.25), sharex=True, constrained_layout=True)

for idx, data in enumerate(datasets):
    tsurf_max, q_Li_max = process_Tsurf_and_qLi(data['path'], data['nx'], data['dt'])
    time = np.arange(1, data['nx']) * data['dt']

    label = data['label_tsurf']
    color = colors[idx % len(colors)]
    linestyle = linestyles[idx % len(linestyles)]
    marker = markers[idx % len(markers)]

    # Remove only the 4th data point for dt=10ms
    if '10ms' in data['path']:
        tsurf_max = np.delete(tsurf_max, 3)
        q_Li_max = np.delete(q_Li_max, 3)
        time = np.delete(time, 3)

    stride = max(1, len(time) // 40)
    ax1.plot(time, np.array(q_Li_max) / 1e6, label=label,
             color=color, linestyle=linestyle, marker=marker,
             markevery=stride, markersize=6, linewidth=2.2)
    ax2.plot(time, np.array(tsurf_max), label=label,
             color=color, linestyle=linestyle, marker=marker,
             markevery=stride, markersize=6, linewidth=2.2)

# Axis labels and limits
ax1.set_ylabel(r'$q_{\mathrm{s}}^{\mathrm{max}}$ (MW/m$^2$)', fontsize=15)
ax1.set_ylim([0, 10])
ax2.set_ylabel(r'$T_{\mathrm{surf}}^{\mathrm{max}}$ ($^\circ$C)', fontsize=15)
ax2.set_ylim([0, 700])
ax2.set_xlabel('t_${simulation}$ (s)', fontsize=15)

ax1.set_xlim([0, 3])
ax2.set_xlim([0, 3])

# Minor ticks and grid
for ax in [ax1, ax2]:
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

# Legend and panel labels
ax2.legend(loc='upper right', ncol=2)
ax1.minorticks_on() 
ax2.minorticks_on() 
ax1.grid(which='minor', linestyle=':', linewidth=0.5) 
ax2.grid(which='minor', linestyle=':', linewidth=0.5)

ax2.legend(loc='best', ncol=2) 
ax1.text(0.02, 0.90, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold') 
ax2.text(0.02, 0.90, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold')

# Adjust layout and save
plt.subplots_adjust(hspace=0.08)
plt.savefig('qsurf_T_surfdt.pdf', dpi=600, bbox_inches='tight')
plt.savefig('qsurf_T_surfdt.eps', dpi=600, bbox_inches='tight')
plt.savefig('qsurf_T_surfdt.png', dpi=300, bbox_inches='tight')
plt.show()


colors = plt.get_cmap('tab10').colors[:6]
markers = ['o', 's', 'v', 'D', 'v', '*']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]

plt.rc('axes', prop_cycle=(cycler('color', colors) +
                           cycler('marker', markers) +
                           cycler('linestyle', linestyles)))

mpl.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'lines.linewidth': 2.2,
    'axes.grid': True,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'grid.alpha': 0.5,
})

# Set cycler for consistent style
plt.rc('axes', prop_cycle=(cycler('color', colors) +
                           cycler('marker', markers) +
                           cycler('linestyle', linestyles)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.25, 4.25), sharex=True, constrained_layout=True)

for idx, data in enumerate(datasets):
    tsurf_max, q_Li_max = process_Tsurf_and_qLi(data['path'], data['nx'], data['dt'])
    time = np.arange(1, data['nx']) * data['dt']

    label = data['label_tsurf']
    color = colors[idx % len(colors)]
    linestyle = linestyles[idx % len(linestyles)]
    marker = markers[idx % len(markers)]

    # Remove only the 4th data point for dt=10ms
    if '10ms' in data['path']:
        tsurf_max = np.delete(tsurf_max, 3)
        q_Li_max = np.delete(q_Li_max, 3)
        time = np.delete(time, 3)

    stride = max(1, len(time) // 40)
    ax1.plot(time, np.array(q_Li_max) / 1e6, label=label,
             color=color, linestyle=linestyle, marker=marker,
             markevery=stride, markersize=6, linewidth=2.2)
    ax2.plot(time, np.array(tsurf_max), label=label,
             color=color, linestyle=linestyle, marker=marker,
             markevery=stride, markersize=6, linewidth=2.2)

# Axis labels and limits
ax1.set_ylabel(r'$q_{\mathrm{s}}^{\mathrm{max}}$ (MW/m$^2$)', fontsize=15)
ax1.set_ylim([0, 10])
ax2.set_ylabel(r'$T_{\mathrm{surf}}^{\mathrm{max}}$ ($^\circ$C)', fontsize=15)
ax2.set_ylim([0, 700])
ax2.set_xlabel('t_${simulation}$ (s)', fontsize=15)

ax1.set_xlim([0, 3])
ax2.set_xlim([0, 3])

# Minor ticks and grid
for ax in [ax1, ax2]:
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

# Legend and panel labels
ax2.legend(loc='upper right', ncol=2)
ax1.minorticks_on() 
ax2.minorticks_on() 
ax1.grid(which='minor', linestyle=':', linewidth=0.5) 
ax2.grid(which='minor', linestyle=':', linewidth=0.5)

ax2.legend(loc='best', ncol=2) 
ax1.text(0.02, 0.90, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold') 
ax2.text(0.02, 0.90, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold')

# Adjust layout and save
plt.subplots_adjust(hspace=0.08)
plt.savefig('qsurf_T_surfdt.pdf', dpi=600, bbox_inches='tight')
plt.savefig('qsurf_T_surfdt.eps', dpi=600, bbox_inches='tight')
plt.savefig('qsurf_T_surfdt.png', dpi=300, bbox_inches='tight')
plt.show()


from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl

# Distinct colors and line styles
colors = plt.get_cmap('tab10').colors[:6]
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]

# Only cycle through color and linestyle
plt.rc('axes', prop_cycle=(cycler('color', colors) + cycler('linestyle', linestyles)))

# Style configuration
mpl.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'lines.linewidth': 2.5,
    'axes.grid': True,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'grid.alpha': 0.5,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.25, 4.25), sharex=True, constrained_layout=True)

for idx, data in enumerate(datasets):
    tsurf_max, q_Li_max = process_Tsurf_and_qLi(data['path'], data['nx'], data['dt'])
    time = np.arange(1, data['nx']) * data['dt']

    label = data['label_tsurf']
    color = colors[idx % len(colors)]
    linestyle = linestyles[idx % len(linestyles)]

    if '10ms' in data['path']:
        tsurf_max = np.delete(tsurf_max, 3)
        q_Li_max = np.delete(q_Li_max, 3)
        time = np.delete(time, 3)

    ax1.plot(time, np.array(q_Li_max) / 1e6, label=label,
             color=color, linestyle=linestyle, linewidth=2.5)
    ax2.plot(time, np.array(tsurf_max), label=label,
             color=color, linestyle=linestyle, linewidth=2.5)

# Axis labels and formatting
ax1.set_ylabel(r'$q_{\mathrm{s}}^{\mathrm{max}}$ (MW/m$^2$)')
ax1.set_ylim([0, 10])
ax2.set_ylabel(r'$T_{\mathrm{surf}}^{\mathrm{max}}$ ($^\circ$C)')
ax2.set_ylim([0, 700])
ax2.set_xlabel('t$_{\mathrm{simulation}}$ (s)')

for ax in [ax1, ax2]:
    ax.set_xlim([0, 5])
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

# Legends and panel labels
ax2.legend(loc='best', ncol=2)
ax1.text(0.02, 0.90, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold')
ax2.text(0.02, 0.90, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold')

# Save figure
plt.savefig('qsurf_T_surfdt.pdf', dpi=600, bbox_inches='tight')
plt.savefig('qsurf_T_surfdt.eps', dpi=600, bbox_inches='tight')
plt.savefig('qsurf_T_surfdt.png', dpi=300)
plt.show()


