import sys
import numpy as np
import netCDF4 as nc
################################################################################

'''Build a North Atlantic Ocean Heat Content (J) cost function for the
NEMOTAM configuration between 40 and 65N from the surface to 2000m
'''

################################################################################
O2G=nc.Dataset('mesh_mask.nc') # generate during trajectory run using nn_msh=1
O2M=nc.Dataset('subbasins.nc') #ORCA2 basin masks, in ORCA2INPUT 
                               #(https://doi.org/10.5281/zenodo.1471702)
latt =O2G.variables['gphit'][0,:]
lont =O2G.variables['glamt'][0,:]
e1t  =O2G.variables['e1t'][0,:]
e2t  =O2G.variables['e2t'][0,:]
e3t  =O2G.variables['e3t'][0,:]
atlmsk=O2M.variables['atlmsk_nomed'][:]
tmask=O2G.variables['tmask'][0,:]
gdept=O2G.variables['gdept'][0,:]
gdept_0=O2G.variables['gdept_0'][0,:]

Tinit=np.zeros((31,149,182))
vinit=np.zeros((31,149,182))

Tinit[np.where( (latt>40) & (latt<65) &\
         (np.cumsum(e3t,axis=0)<2000) &\
                          (atlmsk==1) )]=1
Tinit=(Tinit*e1t*e2t*e3t)/np.sum(Tinit*e1t*e2t*e3t)

################################################################################

with nc.Dataset(('OHC_SP.nc'),'w') as NC:
    NC.createDimension('z',31)
    NC.createDimension('y',149)
    NC.createDimension('x',182)
    Tinit_nc=NC.createVariable('Tinit',np.float64(),('z','y','x'))
    vinit_nc=NC.createVariable('vinit',np.float64(),('z','y','x'))
    Tinit_nc[:]=Tinit[:]
    vinit_nc[:]=vinit[:]


