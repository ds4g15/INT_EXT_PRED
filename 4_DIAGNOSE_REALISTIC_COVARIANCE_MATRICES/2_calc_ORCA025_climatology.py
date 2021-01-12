import sys
import numpy as np
import netCDF4 as nc
import datetime as dt


'''
Calculate climatology from ORCA025 run so as to remove from outputs later.

DESCRIPTION:
-------
This script takes an input argument (the number of the five-day period, 1 to 73)
and creates an average of every instance of this period from a 20 year run with
5-day averaged output. The results are stored as .npy files.

ARGUMENTS: 
---------
Pentad (a number from 1 to 73) corresponding to the 5-day period of interest
from a 365-day year.

OUTPUT:
-------
The script produces the files
T_mean_pentad_???.npy
S_mean_pentad_???.npy
U_mean_pentad_???.npy
V_mean_pentad_???.npy
W_mean_pentad_???.npy

where ??? is the number of the 5-day period (001..073). 
'''

pd=int(sys.argv[1])
inds=np.arange(0,1460,73)+pd
################################################################################
# GET ORCA025 DATA
# Get correct date string for output number "N":
print(str(pd).zfill(3)+': '+'loading model output')
def get_ORCA025(N):
    YEAR     =str((N//73)+331).zfill(4)
    STARTDATE=dt.datetime.strftime(\
              dt.datetime.strptime('19010101','%Y%m%d')+\
              dt.timedelta(days=    (N%73)*5),  '%m%d')
    ENDDATE  =dt.datetime.strftime(\
              dt.datetime.strptime('19010101','%Y%m%d')+\
              dt.timedelta(days=4+((N%73)*5)),  '%m%d')

    outdir='/hpcdata/scratch/ds4g15/ORCAV3.5RUNS/CLIMF_025/CLIMF_025_output/'


    NCT=nc.Dataset((outdir+'CLIMF_025_5d_gridT_'+YEAR+STARTDATE+\
                                             '-'+YEAR+ENDDATE+'.nc'))
    NCU=nc.Dataset((outdir+'CLIMF_025_5d_gridU_'+YEAR+STARTDATE+\
                                             '-'+YEAR+ENDDATE+'.nc'))
    NCV=nc.Dataset((outdir+'CLIMF_025_5d_gridV_'+YEAR+STARTDATE+\
                                             '-'+YEAR+ENDDATE+'.nc'))
    NCW=nc.Dataset((outdir+'CLIMF_025_5d_gridW_'+YEAR+STARTDATE+\
                                             '-'+YEAR+ENDDATE+'.nc'))
################################################################################
    print((str(N).zfill(4)+': '+\
                    outdir+'CLIMF_025_5d_gridT_'+YEAR+STARTDATE+\
                                             '-'+YEAR+ENDDATE+'.nc'))
    T=NCT.variables['votemper'][0,:,:,:]
    print((str(N).zfill(4)+': '+\
                    outdir+'CLIMF_025_5d_gridT_'+YEAR+STARTDATE+\
                                             '-'+YEAR+ENDDATE+'.nc'))
    S=NCT.variables['vosaline'][0,:,:,:]
    print((str(N).zfill(4)+': '+\
                    outdir+'CLIMF_025_5d_gridU_'+YEAR+STARTDATE+\
                                             '-'+YEAR+ENDDATE+'.nc'))
    U=NCU.variables['vozocrtx'][0,:,:,:]
    print((str(N).zfill(4)+': '+\
                   outdir+'CLIMF_025_5d_gridV_'+YEAR+STARTDATE+\
                                            '-'+YEAR+ENDDATE+'.nc'))
    V=NCV.variables['vomecrty'][0,:,:,:]
    print((str(N).zfill(4)+': '+\
                   outdir+'CLIMF_025_5d_gridW_'+YEAR+STARTDATE+\
                                            '-'+YEAR+ENDDATE+'.nc'))
    W=NCW.variables['vovecrtz'][0,:,:,:]
    return T,S,U,V,W

################################################################################

ORCA025_mean_T=np.zeros((75,1021,1442))
ORCA025_mean_S=np.zeros((75,1021,1442))
ORCA025_mean_U=np.zeros((75,1021,1442))
ORCA025_mean_V=np.zeros((75,1021,1442))
ORCA025_mean_W=np.zeros((75,1021,1442))
for n in inds:
    T,S,U,V,W=get_ORCA025(n)
    ORCA025_mean_T[:,:,:]=\
        ORCA025_mean_T[:,:,:]+T/20.
    ORCA025_mean_S[:,:,:]=\
        ORCA025_mean_S[:,:,:]+S/20.
    ORCA025_mean_U[:,:,:]=\
        ORCA025_mean_U[:,:,:]+U/20.
    ORCA025_mean_V[:,:,:]=\
        ORCA025_mean_V[:,:,:]+V/20.
    ORCA025_mean_W[:,:,:]=\
        ORCA025_mean_W[:,:,:]+W/20.
print('Saving, T')
np.save('T_mean_pentad_'+str( '%0.3i' % pd)+'.npy',ORCA025_mean_T)
print('Saving, S')
np.save('S_mean_pentad_'+str( '%0.3i' % pd)+'.npy',ORCA025_mean_S)
print('Saving, U')
np.save('U_mean_pentad_'+str( '%0.3i' % pd)+'.npy',ORCA025_mean_U)
print('Saving, V')
np.save('V_mean_pentad_'+str( '%0.3i' % pd)+'.npy',ORCA025_mean_V)
print('Saving, W')
np.save('W_mean_pentad_'+str( '%0.3i' % pd)+'.npy',ORCA025_mean_W)


