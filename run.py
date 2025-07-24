import sys
import os
#sys.path.append("/home/khrabryi1/.local/lib/python3.7/site-packages")
from uedge import *
from uedge.hdf5 import *
#from parse import *

#aph.aphdir = "/home/khrabryi1/UEsrc/speciesdata"
#api.apidir = "/home/khrabryi1/UEsrc/speciesdata"

exec(open("in.py").read())

#from rdcontdt import contdt

#savefn = 'pf_50x31_alb0.99_recyc0.99_ncore1.9e+20_isimpon6'
#savefn1 = 'pf_50x31_alb1.0_recyc0.99_ncore1.3e+20_isimpon6'
savefn1 = 'pf_Li.hdf5'

bbb.recycp[0] = 0.99#vals["recycling"]
bbb.recycw[0] = 0.99#vals["recycling"]

wall_src_range = 0
bbb.albdsi[wall_src_range] = 1.#vals["albedo"]
bbb.albdso[wall_src_range] = 1.#vals["albedo"]

bbb.ncore[0] = 1.3e20#vals["ncore"]

bbb.dtreal = 1e-8 # 1e-9

if os.path.exists(savefn1):
    #os._exit(1)
    hdf5_restore(savefn1)
else:
    #hdf5_restore(fname)
    print (savefn1 + ' does not exit')

bbb.ftol = 1e20
bbb.oldseec=0
bbb.exmain()
bbb.ftol = 1e-5
exec(open('./set_target_T.py').read())

'''
if bbb.iterm != 1:
    for bbb.dtreal in [0.1 * bbb.dtreal, 0.01 * bbb.dtreal, 1e-3 * bbb.dtreal, 1e-4 * bbb.dtreal, 1e-5 * bbb.dtreal, 1e-6 * bbb.dtreal, 1e-7 * bbb.dtreal ]:
        bbb.exmain()
        if bbb.iterm == 1:
            break
'''

#bbb.dt_tot = 0.
#bbb.itermx = 1000

'''
fnrm = contdt(savefn)

if bbb.dt_tot >= bbb.t_stop or fnrm < bbb.ftol_min:
    print('Writing file')
    print('../'+new_internal_folder+'/'+fname)
    hdf5_save('../'+new_internal_folder+'/'+fname)
    os.remove(savefn)
    os._exit(1)
'''
