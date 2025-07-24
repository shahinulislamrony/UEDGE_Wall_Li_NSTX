from uedge import *
import uedge_mvu.plot as mp
import uedge_mvu.utils as mu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from runcase import *

setGrid()
setPhysics(impFrac=0,fluxLimit=True)
setDChi(kye=1.0, kyi=1.0, difni=0.5,nonuniform = True)
setBoundaryConditions(ncore=6.2e19, pcoree=2.5e6, pcorei=2.5e6, recycp=0.98)
setimpmodel(impmodel=True)
#setgaspuff()

bbb.cion=3
bbb.oldseec=0
bbb.restart=1
bbb.nusp_imp = 3
bbb.icntnunk=0

hdf5_restore("./final.hdf5")
bbb.ftol=1e20
bbb.ftol=1e20;bbb.issfon=0; bbb.exmain()
mu.paws("Completed reading a converged solution and save the output")


ind = len(com.yyc)

fig = plt.figure(dpi=100)
ax = fig.add_subplot()
plt.plot(com.yyc[0:ind-1], bbb.kye_use[bbb.ixmp,0:ind-1], label=r'$\chi_e$')
plt.plot(com.yyc[0:ind-1], bbb.kyi_use[bbb.ixmp,0:ind-1], label=r'$\chi_i$')
plt.plot(com.yyc[0:ind-1], bbb.dif_use[bbb.ixmp,0:ind-1,0], label=r'$D_i$')
#plt.plot(com.yyc[0:ind-1], bbb.dif_use[bbb.ixmp,0:ind-1,1], label=r'$D_n$')
plt.xlabel('r$_{OMP}$ - r$_{sep}$(m)')
plt.ylabel('m$^2$/s')
plt.legend ()
plt.grid()
plt.show()
