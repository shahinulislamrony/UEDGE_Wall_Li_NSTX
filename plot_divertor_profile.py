# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 06:53:48 2025

@author: islam9
"""

import numpy as np
import matplotlib.pyplot as plt

yyrb = np.array([-0.06506854, -0.0592725 , -0.04815738, -0.03801022, -0.02887471,
       -0.02078381, -0.01371695, -0.00760699, -0.0023902 ,  0.00980619,
        0.02988696,  0.04980043,  0.06879613,  0.08660054,  0.10325506,
        0.1195342 ,  0.13566592,  0.15188392,  0.1680489 ,  0.18418833,
        0.20069347,  0.21756767,  0.23471614,  0.25233707,  0.27125284,
        0.28119419])

q_para_drfit =np.array([4.74107451e+07, 3.86221581e+07, 4.01648208e+07, 4.33987383e+07,
       5.17698247e+07, 6.12370593e+07, 9.81774887e+07, 1.03754867e+08,
       3.45233544e+08, 2.98972278e+08, 1.95912421e+08, 1.59600713e+08,
       1.34963806e+08, 1.11947966e+08, 9.86749250e+07, 8.42331250e+07,
       7.36230479e+07, 6.50109409e+07, 5.76160590e+07, 4.83223747e+07,
       3.86994067e+07, 2.89673882e+07, 1.92050048e+07, 1.09847468e+07,
       2.73764969e+06, 2.57873509e+12])

Te_drift = np.array([ 1.06390515,  1.42086645,  1.99991703,  2.95492999,  4.36466035,
        6.35856826,  8.31857484, 11.34866188, 12.8492826 , 20.17474247,
       11.06682511, 10.97966791, 10.43085276,  9.73730512,  9.7083556 ,
        9.37239942,  9.47466372, 10.0506874 , 11.10171557, 12.02304658,
       13.0882951 , 14.11436588, 14.38470586, 13.65844276,  8.2068822 ,
        2.        ])

ne_drift = np.array([7.66053142e+20, 8.59450236e+20, 6.04604306e+20, 4.46188960e+20,
       3.45733585e+20, 2.75079993e+20, 2.22426347e+20, 2.18270089e+20,
       1.94190365e+20, 2.95998356e+20, 5.49238804e+20, 4.05794825e+20,
       3.65510532e+20, 3.27675493e+20, 2.88281985e+20, 2.52892187e+20,
       2.07920862e+20, 1.61582225e+20, 1.22566749e+20, 9.28433202e+19,
       6.75143108e+19, 4.74358702e+19, 3.32407455e+19, 2.61044294e+19,
       2.46688944e+19, 2.23550837e+19])

q_perp_drifts = np.array(([ 443526.509099  ,  443526.509099  ,  414126.59378641,
        409504.82789029,  437146.64933284,  520317.37147412,
        647396.09182479,  908150.76556206, 1582348.04040616,
       2746850.00969284, 2420439.41275684, 2206551.5782332 ,
       2232884.65654374, 2182279.69977122, 2240378.43584734,
       2223082.59883434, 2203556.97323866, 2183757.49380278,
       2183290.74402858, 2077558.99863416, 1865048.95911161,
       1569474.22490762, 1196619.82758673,  900031.98614973,
        658853.00516619,  658853.00516619]))



q_perp_nodrifts = np.array([  15201.34588764,   15201.34588764,   17431.98746595,
         20987.66118224,   28794.48083804,   45210.33908081,
         79042.29169718,  154864.55685036,  361944.51775713,
       2358680.86325363, 3245409.96819452, 3297186.4717285 ,
       3185196.32963552, 3186694.28452906, 3137113.15680655,
       2884876.99505407, 2633799.6693801 , 2325728.76087762,
       1876044.47938415, 1410335.01805858, 1068871.70875507,
        806812.55462884,  605366.87442619,  422496.46503104,
        189108.72587184,  189108.72587184])

q_para_nodirfts = np.array([4.07074516e+06, 4.69375130e+06, 5.68120117e+06, 6.28831977e+06,
       7.18381154e+06, 9.04980983e+06, 1.33195379e+07, 2.42574182e+07,
       5.67349798e+07, 3.39408853e+08, 2.77846190e+08, 2.32823260e+08,
       1.91025242e+08, 1.63463908e+08, 1.38608083e+08, 1.09260079e+08,
       8.38488398e+07, 6.35478888e+07, 4.81513551e+07, 3.57317331e+07,
       2.56715828e+07, 1.73124047e+07, 1.03781260e+07, 4.72925263e+06,
       1.07078674e+06, 3.77689554e+05])

q_para_nodirfts_idiv = np.array([-4.05543723e+06,  8.57060136e+06,  1.28621017e+07,  2.10039744e+07,
        3.34070928e+07,  5.34381916e+07,  7.67043192e+07,  1.09298193e+08,
        1.26201728e+08,  2.60223719e+08,  2.19160908e+08,  1.70056660e+08,
        1.28013933e+08,  8.75481733e+07,  5.33549037e+07,  3.36465139e+07,
        2.45336790e+07,  1.94144800e+07,  1.59059333e+07,  1.29090768e+07,
        9.97596470e+06,  6.93258713e+06,  3.80087758e+06,  2.32266368e+06,
        1.43790011e+06, -8.99126411e+10])


ne_nodrifts = np.array([2.32940682e+17, 2.50124845e+17, 2.82494330e+17, 6.50030585e+17,
       1.43423699e+18, 3.00855942e+18, 5.99384256e+18, 1.14466442e+19,
       2.00688807e+19, 4.16480919e+19, 9.69863484e+19, 1.46294949e+20,
       1.54094633e+20, 1.73167191e+20, 1.95379522e+20, 1.94386400e+20,
       1.49907325e+20, 7.70728393e+19, 3.98946450e+19, 2.49059574e+19,
       1.74349393e+19, 1.27414680e+19, 9.77401543e+18, 7.91725023e+18,
       9.40203106e+18, 8.52635939e+18])

Te_nodrifts = np.array([  3.90024657,   5.20885674,   8.3030272 ,   9.62427052,
        11.41835768,  13.86049878,  17.67206305,  24.57243128,
        39.89539599, 104.41042623,  62.30399801,  40.54063721,
        33.10593396,  26.52097692,  20.99135647,  17.20870185,
        17.20476096,  23.82971257,  31.08285997,  32.70898007,
        31.6027601 ,  29.33130922,  25.71966459,  19.63946989,
         7.40495815,   2.        ])


Li_flux_omp_drift = np.array([ 4.54984144e+06,  3.70855911e+16,  2.73681274e+16, -1.02547995e+16,
       -3.47829207e+16, -8.86995827e+16, -1.48733342e+17, -1.56560961e+17,
       -7.89406082e+16,  3.07123140e+16,  3.57595331e+15, -1.24452167e+16,
       -2.72491239e+16, -4.87351356e+16, -6.60313095e+16, -8.59176586e+16,
       -1.03541623e+17, -1.27279535e+17, -1.43521252e+17, -1.62584190e+17,
       -1.84788441e+17, -2.10403822e+17, -2.51345792e+17, -2.99122599e+17,
       -2.94608921e+17,  1.10272016e+18])

Sput_odiv_drift = np.array([0.00000000e+00, 0.00000000e+00, 1.08531499e+17, 1.24164909e+18,
       4.34465150e+18, 7.11151016e+18, 1.33388023e+19, 1.16528236e+19,
       2.46848999e+19, 1.11961859e+20, 1.37184983e+20, 1.32600674e+20,
       1.30638169e+20, 1.20028492e+20, 1.23814504e+20, 1.25173768e+20,
       1.28688258e+20, 1.31830819e+20, 1.28462226e+20, 1.23155146e+20,
       1.12816767e+20, 9.65358452e+19, 7.89762648e+19, 7.15064277e+19,
       7.16510200e+19, 0.00000000e+00])

Sput_idiv_drift = np.array([ 0.00000000e+00, -4.95438200e+17, -2.44807589e+18, -4.96808715e+18,
       -9.05906482e+18, -1.35792785e+19, -1.97410787e+19, -2.42216492e+19,
       -2.25302079e+19, -1.07395235e+20, -9.44897408e+19, -8.35916511e+19,
       -6.91303300e+19, -5.14960636e+19, -2.90006673e+19, -8.75140718e+18,
       -9.62951569e+17, -7.53943051e+14,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -3.52260554e+17,
       -2.77468955e+18,  0.00000000e+00])

Sput_idiv_no_drifts = np.array([ 0.00000000e+00, -2.32922820e+16, -4.95393732e+16, -1.12317688e+17,
       -2.34778024e+17, -4.65598025e+17, -8.78849419e+17, -1.67852408e+18,
       -3.64475957e+18, -7.89219593e+19, -8.14387259e+19, -7.61328612e+19,
       -6.39564335e+19, -4.90482659e+19, -3.67228844e+19, -3.03306276e+19,
       -2.75929100e+19, -2.60607794e+19, -2.51577962e+19, -2.45303911e+19,
       -2.39446004e+19, -2.10552463e+19, -1.67033408e+19, -1.30820247e+19,
       -5.06522802e+18,  0.00000000e+00])

Sput_odiv_nodrift = np.array([0.00000000e+00, 1.25057190e+16, 5.74018968e+16, 1.20613375e+17,
       2.41666280e+17, 4.57643411e+17, 8.23459924e+17, 1.45173307e+18,
       2.46836733e+18, 2.94118631e+19, 7.61152517e+19, 1.12617336e+20,
       1.28712245e+20, 1.40558687e+20, 1.58787494e+20, 1.60524317e+20,
       1.48234984e+20, 1.14108503e+20, 7.83013808e+19, 5.91730509e+19,
       4.84230206e+19, 3.98987613e+19, 3.36191142e+19, 2.82908434e+19,
       1.69654276e+19, 0.00000000e+00])

Li_flux_omp_nodrift = np.array([ 0.00000000e+00,  9.47213075e+15,  1.23054277e+16,  8.30880739e+15,
        5.90685138e+13, -3.05181306e+16, -8.81086979e+16, -1.20607137e+17,
       -9.88450723e+16, -3.33967198e+16, -4.19604715e+16, -4.88895196e+16,
       -5.90331807e+16, -6.96537881e+16, -8.42164805e+16, -9.85709625e+16,
       -1.13360364e+17, -1.27641441e+17, -1.35033558e+17, -1.40891296e+17,
       -1.38018025e+17, -1.19457263e+17, -8.78413761e+16, -4.56518180e+16,
       -8.09698739e+15, -7.89348901e+09])

sx_omp = np.array([3.43358829e-08, 3.43815158e-02, 1.87070952e-02, 1.44941516e-02,
       1.65363286e-02, 1.82911823e-02, 1.72481801e-02, 1.27113465e-02,
       7.66102414e-03, 2.27583152e-03, 2.69148354e-03, 2.95265968e-03,
       3.36247799e-03, 3.75970685e-03, 4.34026796e-03, 4.91485100e-03,
       5.57735222e-03, 6.37263341e-03, 7.09224997e-03, 8.06059632e-03,
       9.24916712e-03, 1.04068879e-02, 1.18167963e-02, 1.34740497e-02,
       1.64416632e-02, 1.64517453e-08])

nLi_omp_no_drifts = np.array([3.00000000e+12, 5.05752784e+14, 1.52475748e+15, 2.30736244e+15,
       3.06246935e+15, 3.88898744e+15, 4.67078602e+15, 5.23627851e+15,
       5.55625385e+15, 5.68621703e+15, 5.75003246e+15, 5.81325570e+15,
       5.87339207e+15, 5.92789419e+15, 5.97184273e+15, 5.99794355e+15,
       5.99729694e+15, 5.95561981e+15, 5.86012485e+15, 5.70933638e+15,
       5.49220563e+15, 5.20178022e+15, 4.85715898e+15, 4.50149612e+15,
       4.19968732e+15, 4.16223247e+15])

nLi_omp_drifts = np.array([3.00000000e+12, 1.92804274e+14, 5.40286104e+14, 7.63465025e+14,
       9.54660304e+14, 1.13526976e+15, 1.28397685e+15, 1.40092568e+15,
       1.49456098e+15, 1.54231913e+15, 1.57154393e+15, 1.60789884e+15,
       1.65207969e+15, 1.70586381e+15, 1.76802148e+15, 1.83760682e+15,
       1.91107041e+15, 1.98073289e+15, 2.03829624e+15, 2.08681968e+15,
       2.12152599e+15, 2.12961901e+15, 2.11304133e+15, 2.07750448e+15,
       2.01830322e+15, 2.00030303e+15])


yyc = np.array([-0.01547339, -0.01356486, -0.01061946, -0.00878095, -0.00706681,
       -0.0051432 , -0.0031861 , -0.00153701, -0.00041696,  0.00012713,
        0.00040105,  0.00070917,  0.00105611,  0.00144915,  0.00189595,
        0.00240621,  0.00298113,  0.00363187,  0.00436896,  0.0052026 ,
        0.0061501 ,  0.00722318,  0.00844005,  0.00982061,  0.01144899,
        0.01234483])

q_para_onlyGradB = np.array([ 3.04327127e+06,  4.13589808e+06,  6.93891480e+06,  1.06730578e+07,
        1.67110992e+07,  2.42887415e+07,  3.26915872e+07,  4.31885418e+07,
        6.17414165e+07,  1.93136369e+08,  1.61736818e+08,  1.31457006e+08,
        1.06511573e+08,  8.75627360e+07,  7.41026990e+07,  6.18017122e+07,
        5.33077593e+07,  4.74615504e+07,  4.45475065e+07,  4.21302582e+07,
        3.82678287e+07,  3.06952350e+07,  2.08626346e+07,  1.25816163e+07,
        3.67814687e+06, -1.17489571e+06])

q_para_onlyGradB_idiv = np.array([-2.64031258e+07,  1.86723651e+06,  5.75634537e+06,  1.25020387e+07,
        1.93528059e+07,  3.13859878e+07,  3.85648002e+07,  5.87454180e+07,
        6.14465794e+07,  4.80605350e+08,  4.58215496e+08,  3.56953037e+08,
        2.38609897e+08,  1.38318758e+08,  7.25978106e+07,  4.10585175e+07,
        2.86711505e+07,  2.22088351e+07,  1.77282319e+07,  1.39813499e+07,
        1.06535093e+07,  7.54080128e+06,  4.44413184e+06,  2.57989921e+06,
        1.30816424e+06,  7.37175151e+05])

q_para_idiv = np.array([-4.05543723e+06,  8.57060136e+06,  1.28621017e+07,  2.10039744e+07,
        3.34070928e+07,  5.34381916e+07,  7.67043192e+07,  1.09298193e+08,
        1.26201728e+08,  2.60223719e+08,  2.19160908e+08,  1.70056660e+08,
        1.28013933e+08,  8.75481733e+07,  5.33549037e+07,  3.36465139e+07,
        2.45336790e+07,  1.94144800e+07,  1.59059333e+07,  1.29090768e+07,
        9.97596470e+06,  6.93258713e+06,  3.80087758e+06,  2.32266368e+06,
        1.43790011e+06, -8.99126411e+10])





plt.figure(figsize=(5, 3))
plt.plot(yyc[1:-1], q_para_drfit[1:-1]/1e6, '--r', label='full-drifts')
plt.plot(yyc[1:-1], q_para_onlyGradB[1:-1]/1e6, '-k', label='$\\nabla$B-drifts')
plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=16)
plt.ylabel('q$_{||}^{Odiv}$ (MW/m$^2$)', fontsize=16)
plt.grid()
plt.legend(fontsize=12)
plt.xlim([0, 0.01])
plt.ylim([0, np.max(q_para_drfit[1:-1]/1e6)*1.05])
plt.tight_layout()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('q_para_compare.png', dpi=300)
plt.show()


plt.figure(figsize=(5, 3))
plt.plot(yyc[1:-1], q_para_idiv[1:-1]/1e6, '--r', label='full-drifts')
plt.plot(yyc[1:-1], q_para_onlyGradB_idiv[1:-1]/1e6, '-k', label='$\\nabla$B-drifts')
plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=16)
plt.ylabel('q$_{||}^{Idiv}$ (MW/m$^2$)', fontsize=16)
plt.grid()
plt.legend(fontsize=12)
plt.xlim([0, 0.01])
plt.ylim([0, np.max(q_para_onlyGradB_idiv[1:-1]/1e6)*1.05])
plt.tight_layout()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('q_para_compare_idiv.png', dpi=300)
plt.show()




fig, axs = plt.subplots(2, 1, figsize=(4, 5), sharex=True)  # Taller figure for two plots

# --- First subplot: Odiv ---
axs[0].plot(yyc[1:-1], q_para_drfit[1:-1]/1e6, '-r', label='full-drifts')
axs[0].plot(yyc[1:-1], q_para_onlyGradB[1:-1]/1e6, '--k', label='$\\nabla$B-drifts')
axs[0].plot(yyc[1:-1], q_para_nodirfts[1:-1]/1e6, ':g', linewidth=2, label='No-drifts')
axs[0].set_ylabel('q$_{||}^{Odiv}$ (MW/m$^2$)', fontsize=16)
axs[0].legend(fontsize=14)
axs[0].grid()
axs[0].set_xlim([0, 0.01])
axs[0].set_ylim([0, np.max(q_para_drfit[1:-1]/1e6)*1.05])
axs[0].tick_params(axis='both', labelsize=12)

# --- Second subplot: Idiv ---
axs[1].plot(yyc[1:-1], q_para_idiv[1:-1]/1e6, '-r', label='full-drifts')
axs[1].plot(yyc[1:-1], q_para_onlyGradB_idiv[1:-1]/1e6, '--k', label='$\\nabla$B-drifts')
axs[1].plot(yyc[1:-1], q_para_nodirfts_idiv[1:-1]/1e6, ':g', linewidth=2, label='No-drifts')
axs[1].set_xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=16)
axs[1].set_ylabel('q$_{||}^{Idiv}$ (MW/m$^2$)', fontsize=16)
#axs[1].legend(fontsize=12)
axs[1].grid()
axs[1].set_xlim([0, 0.01])
axs[1].set_ylim([0, np.max(q_para_onlyGradB_idiv[1:-1]/1e6)*1.05])
axs[1].tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('q_para_compare_subplots.png', dpi=300)
plt.show()



fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)  # Side-by-side plots

# --- Left subplot: Outer divertor (Odiv) ---
axs[0].plot(yyc[1:-1], q_para_drfit[1:-1]/1e6, '-r', label='full-drifts')
axs[0].plot(yyc[1:-1], q_para_onlyGradB[1:-1]/1e6, '--k', label='$\\nabla$B-drifts')
axs[0].plot(yyc[1:-1], q_para_nodirfts[1:-1]/1e6, ':g', linewidth=3, label='No-drifts')
axs[0].set_xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=16)
axs[0].set_ylabel('q$_{||}$ (MW/m$^2$)', fontsize=16)
axs[0].set_xlim([0, 0.01])
axs[0].tick_params(axis='both', labelsize=12)
axs[0].grid()
axs[0].set_ylim([0, 500])
axs[0].legend(fontsize=13)
axs[0].text(0.95, 0.92, '(a) Odiv', transform=axs[0].transAxes, fontsize=14, ha='right', va='top')

# --- Right subplot: Inner divertor (Idiv) ---
axs[1].plot(yyc[1:-1], q_para_idiv[1:-1]/1e6, '-r', label='full-drifts')
axs[1].plot(yyc[1:-1], q_para_onlyGradB_idiv[1:-1]/1e6, '--k', label='$\\nabla$B-drifts')
axs[1].plot(yyc[1:-1], q_para_nodirfts_idiv[1:-1]/1e6, ':g', linewidth=2, label='No-drifts')
axs[1].set_xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=16)
axs[1].set_xlim([0, 0.01])
axs[1].set_ylim([0, 500])
axs[1].tick_params(axis='both', labelsize=12)
axs[1].grid()
axs[1].text(0.95, 0.92, '(b) Idiv', transform=axs[1].transAxes, fontsize=14, ha='right', va='top')

plt.tight_layout()
plt.savefig('q_para_compare_subplots.png', dpi=300)
plt.show()




Li_nodrift = np.sum(Sput_odiv_nodrift + Sput_idiv_no_drifts)
Li_drifts = np.sum(Sput_idiv_drift + Sput_odiv_drift)

plt.figure(figsize=(5,3))
plt.plot(yyc[1:-1], (Li_flux_omp_drift[1:-1]/sx_omp[1:-1])/1e20, '--r', label ='W/ Drifts')
plt.plot(yyc[1:-1], Li_flux_omp_nodrift[1:-1]/sx_omp[1:-1]/1e20, '-k', label ='W/O drifts')

plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=14)
plt.ylabel('$\Gamma_{Li}^{OMP}$ (10$^{20}$/m$^{2}$s)', fontsize=16)
plt.grid()
plt.legend(fontsize=12)

# Add Li total sputtering as annotations
plt.gcf().text(0.10, 1.00, f'$\phi_{{Li}}$ (no-drifts) = {Li_nodrift/1e20:.2f}e20', fontsize=11, color='k')
plt.gcf().text(0.6, 1.00, f'$\phi_{{Li}}$ (drifts) = {Li_drifts/1e20:.2f}e20', fontsize=11, color='r')
plt.tight_layout()
plt.savefig('Li_flux.png', dpi=300)
plt.show()



plt.figure(figsize=(5,3))
plt.plot(yyc[1:-1], nLi_omp_drifts[1:-1], '--r', label ='drifts')
plt.plot(yyc[1:-1], nLi_omp_no_drifts[1:-1], '-k', label ='No-drifts')
plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=14)
plt.ylabel('n$_{Li}^{OMP}$ (/m$^{3}$)', fontsize=16)
plt.grid()
plt.legend(fontsize=12)
plt.ylim([0, np.max(nLi_omp_no_drifts[1:-1])*1.05])
plt.gcf().text(0.10, 1.00, f'$\phi_{{Li}}$ (no-drifts) = {Li_nodrift/1e20:.2f}e20', fontsize=11, color='k')
plt.gcf().text(0.6, 1.00, f'$\phi_{{Li}}$ (drifts) = {Li_drifts/1e20:.2f}e20', fontsize=11, color='r')
plt.tight_layout()
plt.savefig('Li_den.png', dpi=300)
plt.show()




fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

axs[0].plot(yyc[1:-1], (Li_flux_omp_drift[1:-1]/sx_omp[1:-1])/1e20, '--r', label='W/ Drifts')
axs[0].plot(yyc[1:-1], (Li_flux_omp_nodrift[1:-1]/sx_omp[1:-1])/1e20, '-k', label='W/O drifts')

axs[0].set_ylabel('$\Gamma_{Li}^{OMP}$ ($10^{20}$/m$^{2}$s)', fontsize=14)
axs[0].legend(fontsize=11, loc='upper right')
axs[0].grid(True)
axs[0].text(0.02, 0.85, '(a)', transform=axs[0].transAxes, fontsize=13, fontweight='bold')
axs[0].text(0.02, 1.08, f'$\phi_{{Li}}$ (no-drifts) = {Li_nodrift/1e20:.2f}e20',
            transform=axs[0].transAxes, fontsize=10.5, color='k')
axs[0].text(0.55, 1.08, f'$\phi_{{Li}}$ (drifts) = {Li_drifts/1e20:.2f}e20',
            transform=axs[0].transAxes, fontsize=10.5, color='r')

axs[1].plot(yyc[1:-1], nLi_omp_drifts[1:-1], '--r', label='Drifts')
axs[1].plot(yyc[1:-1], nLi_omp_no_drifts[1:-1], '-k', label='No-drifts')
axs[1].set_xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=14)
axs[1].set_ylabel('n$_{Li}^{OMP}$ (/m$^{3}$)', fontsize=14)
#axs[1].legend(fontsize=11, loc='upper right')
axs[1].grid(True)
axs[1].set_ylim([0, np.max(nLi_omp_no_drifts[1:-1]) * 1.05])
axs[1].text(0.02, 0.85, '(b)', transform=axs[1].transAxes, fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('Li_flux_density_subplot.png', dpi=300)
plt.show()




plt.figure(figsize=(5,3))
plt.plot(yyc[1:-1], q_para_drfit[1:-1]/1e6, '--r', label ='drifts')
plt.plot(yyc[1:-1], q_para_nodirfts[1:-1]/1e6, '-k', label ='No-drifts')
plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=14)
plt.ylabel('q$_{||}^{Odiv}$ (MW/m$^2$)',fontsize=16)
plt.grid()
plt.legend()
plt.ylim([0, np.max(q_para_drfit[1:-1]/1e6)*1.05])
plt.savefig('q_para.png', dpi=300)
plt.show()

sxnp = np.array([3.35404901e-08, 3.39626464e-02, 3.19107170e-02, 2.95808764e-02,
       2.68852014e-02, 2.39997574e-02, 2.11190302e-02, 1.83978410e-02,
       1.57142358e-02, 6.59730369e-02, 7.17170424e-02, 6.96879881e-02,
       6.98719029e-02, 6.49659944e-02, 6.47738697e-02, 6.52471802e-02,
       6.66649569e-02, 6.90308846e-02, 6.92583421e-02, 7.18869528e-02,
       7.56178648e-02, 7.84676614e-02, 8.15351906e-02, 8.64769772e-02,
       9.79929098e-02, 9.91486145e-08])

plt.figure(figsize=(5,3))
plt.plot(yyrb[1:-1], (Sput_odiv_drift[1:-1]/sxnp[1:-1])/1e20, '--r', label ='drifts')
plt.plot(yyrb[1:-1], (Sput_odiv_nodrift[1:-1]/sxnp[1:-1])/1e20, '-k', label ='No-drifts')
plt.xlabel('r$_{div}$ - r$_{sep}$ (m)', fontsize=14)
plt.ylabel('$\Gamma_{Sput}^{Odiv}$ (/m$^2$)',fontsize=16)
plt.grid()
plt.legend()
plt.ylim([0, np.max((Sput_odiv_nodrift[1:-1]/sxnp[1:-1])/1e20,)*1.05])
plt.savefig('Sput_odiv.png', dpi=300)
plt.show()

plt.figure(figsize=(5,3))
plt.plot(yyrb[1:-1], q_perp_drifts[1:-1]/1e6, '--r', label ='drifts')
plt.plot(yyrb[1:-1], q_perp_nodrifts[1:-1]/1e6, '-k', label ='No-drifts')
plt.xlabel('r$_{div}$ - r$_{sep}$ (m)', fontsize=14)
plt.ylabel('q$_{\perp}^{Odiv}$ (MW/m$^2$)',fontsize=16)
plt.grid()
plt.legend()
plt.ylim([0, np.max(q_perp_nodrifts[1:-1]/1e6)*1.05])
plt.savefig('q_perp.png', dpi=300)
plt.show()


plt.figure(figsize=(5,3))
plt.plot(yyrb[1:-1], ne_drift[1:-1]/1e20, '--r', label ='drifts')
plt.plot(yyrb[1:-1], ne_nodrifts[1:-1]/1e20, '-k', label ='No-drifts')
plt.xlabel('r$_{div}$ - r$_{sep}$ (m)', fontsize=14)
plt.ylabel('n$_{e}^{Odiv}$ (10$^{20}$ m$^{-3}$)',fontsize=16)
plt.grid()
plt.legend()
plt.ylim([0, np.max(ne_drift[1:-1]/1e20)*1.05])
plt.savefig('n_e.png', dpi=300)
plt.show()


plt.figure(figsize=(5,3))
plt.plot(yyrb[1:-1], Te_drift[1:-1], '--r', label ='drifts')
plt.plot(yyrb[1:-1], Te_nodrifts[1:-1], '-k', label ='No-drifts')
plt.xlabel('r$_{div}$ - r$_{sep}$ (m)', fontsize=14)
plt.ylabel('T$_{e}^{Odiv}$ (eV)',fontsize=16)
plt.grid()
plt.legend()
plt.ylim([0, np.max(Te_nodrifts)*1.05])
plt.savefig('T_e.png', dpi=300)
plt.show()




fig, axs = plt.subplots(2, 2, figsize=(7, 5), sharex=True)
plt.subplots_adjust(hspace=0.25, wspace=0.3)

x = yyrb[1:-1]
x_label = r'$r_\mathrm{div} - r_\mathrm{sep}$ (m)'

def format_ticks(ax):
    ax.tick_params(axis='both', labelsize=12)


axs[0, 0].plot(x, (Sput_odiv_drift[1:-1]/sxnp[1:-1]) / 1e20, '--r', label='W/ Drifts')
axs[0, 0].plot(x, (Sput_odiv_nodrift[1:-1]/sxnp[1:-1]) / 1e20, '-k', label='W/O drifts')
axs[0, 0].set_ylabel(r'$\Gamma_{Li-source}^{Odiv}$ (10$^{20}$/m$^2$s)', fontsize=14)
axs[0, 0].legend(fontsize=12)
axs[0, 0].grid(True)
axs[0, 0].set_ylim([0, np.max((Sput_odiv_nodrift[1:-1]/sxnp[1:-1]) / 1e20) * 1.05])
axs[0, 0].text(0.02, 0.95, '(a)', transform=axs[0, 0].transAxes,
              fontsize=12, fontweight='bold', va='top')
format_ticks(axs[0, 0])

# Top-right: q_perp
axs[0, 1].plot(x, q_perp_drifts[1:-1] / 1e6, '--r', label='Drifts')
axs[0, 1].plot(x, q_perp_nodrifts[1:-1] / 1e6, '-k', label='No-drifts')
axs[0, 1].set_ylabel(r'$q_{\perp}^{Odiv}$ (MW/m$^2$)', fontsize=14)
#axs[0, 1].legend(fontsize=10)
axs[0, 1].grid(True)
axs[0, 1].set_ylim([0, np.max(q_perp_nodrifts[1:-1] / 1e6) * 1.05])
axs[0, 1].text(0.02, 0.95, '(b)', transform=axs[0, 1].transAxes,
              fontsize=12, fontweight='bold', va='top')
format_ticks(axs[0, 1])

# Bottom-left: ne
axs[1, 0].plot(x, ne_drift[1:-1] / 1e20, '--r', label='Drifts')
axs[1, 0].plot(x, ne_nodrifts[1:-1] / 1e20, '-k', label='No-drifts')
axs[1, 0].set_xlabel(x_label, fontsize=14)
axs[1, 0].set_ylabel(r'$n_e^{Odiv}$ ($10^{20}$ m$^{-3}$)', fontsize=14)
#axs[1, 0].legend(fontsize=10)
axs[1, 0].grid(True)
axs[1, 0].set_ylim([0, np.max(ne_drift[1:-1] / 1e20) * 1.05])
axs[1, 0].text(0.12, 0.95, '(c)', transform=axs[1, 0].transAxes,
              fontsize=12, fontweight='bold', va='top')
format_ticks(axs[1, 0])

# Bottom-right: Te
axs[1, 1].plot(x, Te_drift[1:-1], '--r', label='Drifts')
axs[1, 1].plot(x, Te_nodrifts[1:-1], '-k', label='No-drifts')
axs[1, 1].set_xlabel(x_label, fontsize=14)
axs[1, 1].set_ylabel(r'$T_e^{Odiv}$ (eV)', fontsize=14)
#axs[1, 1].legend(fontsize=10)
axs[1, 1].grid(True)
axs[1, 1].set_ylim([0, np.max(Te_nodrifts) * 1.05])
axs[1, 1].text(0.02, 0.95, '(d)', transform=axs[1, 1].transAxes,
              fontsize=12, fontweight='bold', va='top')

format_ticks(axs[1, 1])



plt.tight_layout()
plt.savefig("Odiv_profiles_comparison.png", dpi=600)
plt.show()

