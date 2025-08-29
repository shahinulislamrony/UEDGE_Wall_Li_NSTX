# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:36:47 2025

@author: islam9
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# =========================
# 1. Set folder and file paths
# =========================
folder_path1 = r"C:\Users\islam9\OneDrive - LLNL\Desktop\Nuclear_Fusion\Nuclear_Fusion\low_FX\Power_scan"  # <-- Folder path 2
folder_path2 = r"C:\Users\islam9\OneDrive - LLNL\Desktop\Nuclear_Fusion\Nuclear_Fusion\high_FX\Powerscan_corrected"

csv_file1 = os.path.join(folder_path1, "simulation_results.csv")  # Updated file name
csv_file2 = os.path.join(folder_path2, "simulation_results.csv") 

# =========================
# 3. Read CSV
# =========================
df = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)
# =========================
# 4. Plot b0 vs Pcore
# =========================
plt.figure(figsize=(5,3))
plt.plot(df['b0'], df['q_core'], marker='o', linestyle='-', color='b', label='low FE')
plt.plot(df2['b0'], df2['q_core'], marker='o', linestyle='--', color='r', label='High FE')
plt.xlabel('b0')
plt.ylabel('Pcore (MW)')
plt.title('Pcore vs b0')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(5,3))
plt.plot(df['b0']/1e6, df['max_q']/1e6, marker='o', linestyle='-', color='b', label='low FE')
plt.plot(df2['b0']/1e6, df2['max_q']/1e6, marker='o', linestyle='--', color='r', label='High FE')
#plt.plot(df['b0']/1e6, df['q_int_odiv']/1e6, marker='o', linestyle='-', color='b', label='low FE')
##plt.plot(df2['b0']/1e6, df2['q_int_odiv']/1e6, marker='o', linestyle='--', color='r', label='High FE')
plt.xlabel('Pcore (MW)',fontsize=14)
plt.ylabel(r'q$_{\perp, max}^{odiv}$ (MW/m$^2$)', fontsize=14)
#plt.title('Pcore vs b0')
plt.xlim([4, 10])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()






fig, ax1 = plt.subplots(figsize=(5,3.7))

# -------------------------
# Left y-axis: max_q
# -------------------------
line1 = ax1.plot(df['b0']/1e6, df['max_q']/1e6, marker='o', linestyle='-', color='b', label='low FE')
line2 = ax1.plot(df2['b0']/1e6, df2['max_q']/1e6, marker='o', linestyle='-', color='r', label='high FE')
ax1.set_xlabel('Pcore (MW)', fontsize=14)
ax1.set_ylabel(r'q$_{\perp}^{max}$ (MW/m$^2$)', fontsize=14)
ax1.tick_params(axis='y', labelcolor='k')
ax1.set_xlim([4, 10])
ax1.set_ylim([0, 20])
ax1.grid(True)

# Separate legend for max_q
leg1 = ax1.legend(title='q$_\perp^{max}$', fontsize=10, loc='lower center',
                  bbox_to_anchor=(0.25, 1.00))
ax1.add_artist(leg1)

# -------------------------
# Right y-axis: q_int_odiv
# -------------------------
ax2 = ax1.twinx()
line3 = ax2.plot(df['b0']/1e6, df['q_int_odiv']/1e6, marker='*', linestyle=':', color='b', label='low FE')
line4 = ax2.plot(df2['b0']/1e6, df2['q_int_odiv']/1e6, marker='*', linestyle=':', color='r', label='high FE')
ax2.set_ylabel(r'q$_{Odiv}$ (MW)', fontsize=14)
ax2.tick_params(axis='y', labelcolor='k')
ax2.set_xlim([4, 10])
ax2.set_ylim([0, 8])


ax2.legend(title='$\int$ q', fontsize=10, loc='lower center', 
           bbox_to_anchor=(0.75, 1.00))

# -------------------------
# Title & layout
# -------------------------
plt.tight_layout()
plt.show()


fig, ax1 = plt.subplots(figsize=(5,3.7))

# -------------------------
# Left y-axis: max_q
# -------------------------
line1 = ax1.plot(df['b0']/1e6, df['Te_max_odiv'], marker='o', linestyle='-', color='b', label='low FE')
line2 = ax1.plot(df2['b0']/1e6, df2['Te_max_odiv'], marker='o', linestyle='-', color='r', label='high FE')
ax1.set_xlabel('Pcore (MW)', fontsize=14)
ax1.set_ylabel(r'T$_{e,max}$ (eV)', fontsize=14)
ax1.tick_params(axis='y', labelcolor='k')
ax1.set_xlim([4, 10])
ax1.set_ylim([0, 250])
ax1.grid(True)

# Separate legend for max_q
leg1 = ax1.legend(title='T$_e$', fontsize=10, loc='lower center',
                  bbox_to_anchor=(0.25, 1.00))
ax1.add_artist(leg1)

# -------------------------
# Right y-axis: q_int_odiv
# -------------------------
ax2 = ax1.twinx()
line3 = ax2.plot(df['b0']/1e6, df['nemax_odiv']/1e20, marker='*', linestyle=':', color='b', label='low FE')
line4 = ax2.plot(df2['b0']/1e6, df2['nemax_odiv']/1e20, marker='*', linestyle=':', color='r', label='high FE')
ax2.set_ylabel(r'n$_{e,max}$ (10$^{20}$ (m$^{-3}$))', fontsize=14)
ax2.tick_params(axis='y', labelcolor='k')
ax2.set_xlim([4, 10])
ax2.set_ylim([0, 5])


ax2.legend(title='n$_e$', fontsize=10, loc='lower center', 
           bbox_to_anchor=(0.75, 1.00))

# -------------------------
# Title & layout
# -------------------------
plt.tight_layout()
plt.show()




fig, ax1 = plt.subplots(figsize=(5, 3.7))

# -------------------------
# Plot all three variables for LFE
# -------------------------
line_idiv_LFE = ax1.plot(df['b0']/1e6, df['q_int_idiv']/1e6, marker='o', linestyle='-', color='b', label='Idiv')
line_odiv_LFE = ax1.plot(df['b0']/1e6, df['q_int_odiv']/1e6, marker='*', linestyle='-', color='b', label='Odiv')
line_wall_LFE = ax1.plot(df['b0']/1e6, df['q_int_wall'], marker='>', linestyle='-', color='b', label='Owall')

# Legend for LFE (only LFE lines)
leg1 = ax1.legend(handles=[line_idiv_LFE[0], line_odiv_LFE[0], line_wall_LFE[0]],
                  labels=['Idiv', 'Odiv', 'Owall'],
                  loc='upper left', bbox_to_anchor=(0.15, 1.02),
                  fontsize=10, title='LFE')
ax1.add_artist(leg1)  

# -------------------------
# Plot all three variables for HFE
# -------------------------
line_idiv_HFE = ax1.plot(df2['b0']/1e6, df2['q_int_idiv']/1e6, marker='o', linestyle='--', color='r', label='Idiv')
line_odiv_HFE = ax1.plot(df2['b0']/1e6, df2['q_int_odiv']/1e6, marker='*', linestyle='--', color='r', label='Odiv')
line_wall_HFE = ax1.plot(df2['b0']/1e6, df2['q_int_wall'], marker='>', linestyle='--', color='r', label='Owall')

# Legend for HFE (only HFE lines)
leg2 = ax1.legend(handles=[line_idiv_HFE[0], line_odiv_HFE[0], line_wall_HFE[0]],
                  labels=['Idiv', 'Odiv', 'Owall'],
                  loc='upper right', bbox_to_anchor=(0.85, 1.02),
                  fontsize=10, title='HFE')
ax1.add_artist(leg2)

# -------------------------
# Axis labels & limits
# -------------------------
ax1.set_xlabel('Pcore (MW)', fontsize=14)
ax1.set_ylabel(r'q$_{int}$ (MW)', fontsize=14)
ax1.set_xlim([4, 10])
ax1.set_ylim([0, 6])
ax1.grid(True)

plt.tight_layout()
plt.show()





