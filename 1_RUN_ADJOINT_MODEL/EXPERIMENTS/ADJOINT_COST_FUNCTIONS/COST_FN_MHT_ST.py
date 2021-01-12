import sys
import numpy as np
import netCDF4 as nc
################################################################################

'''
Build a North Atlantic meridional volume transport cost function for the
NEMOTAM configuration at 25N and from the surface to the bottom

MHT = v'T+T'v+v'T', where v'T' assumed small and neglected

'''

LAT=25 #Latitude of MHT evaluation
O2G=nc.Dataset('mesh_mask.nc') # generate during trajectory run using nn_msh=1
O2M=nc.Dataset('subbasins.nc') #ORCA2 basin masks, in ORCA2INPUT 
                               #(https://doi.org/10.5281/zenodo.1471702)

# V grid
latv =O2G.variables['gphiv'][:]
lonv =O2G.variables['glamv'][:]
e1v  =O2G.variables['e1v'][:]
e2v  =O2G.variables['e2v'][:]
e3v  =O2G.variables['e3v'][:]
vmask=O2G.variables['vmask'][:]
gdepv=O2G.variables['gdepv'][:]

# T grid
latt =O2G.variables['gphit'][:]
lont =O2G.variables['glamt'][:]
e1t  =O2G.variables['e1t'][:]
e2t  =O2G.variables['e2t'][:]
e3t  =O2G.variables['e3t'][:]
atlmsk=O2M.variables['atlmsk'][:]
tmask=O2G.variables['tmask'][:]
gdept=O2G.variables['gdept'][:]
gdept_0=O2G.variables['gdept_0'][0,:]

vinit=np.zeros((31,149,182))
Tinit=np.zeros((31,149,182))
T_bar=np.ones((31,149,182)) # Note, will have to get these from the background
v_bar=np.ones((31,149,182)) # model state ONLINE, so are just ones here.

cp=4187 #J/K/kg
rho0=1025 #kg/m3
################################################################################
### FIND LATITUDE INDEX:
# Find model j index closest to lat on average when zonally avg'd over Atlantic

J=np.nanargmin(np.sum(e1v*atlmsk*(latv-LAT)**2,axis=2)/np.sum(e1v*atlmsk,axis=2))
print('Chosen latitude: '+str(LAT))
print('Nearest j index in Atlantic: '+str(J))

vinit[:,J,:]=(T_bar*vmask*atlmsk*e1v*e3v)[0,:,J,:]*cp*rho0*1e-15
Tinit[:,J,:]=(v_bar*tmask*atlmsk*e1t*e3t)[0,:,J,:]*cp*rho0*1e-15

# SAVE TO NETCDF
with nc.Dataset(('MHT_ST.nc'),'w') as NC:
    NC.createDimension('z',31)
    NC.createDimension('y',149)
    NC.createDimension('x',182)
    vinit_nc=NC.createVariable('vinit',np.float64(),('z','y','x'))
    Tinit_nc=NC.createVariable('Tinit',np.float64(),('z','y','x'))
    vinit_nc[:]=vinit[:]
    Tinit_nc[:]=Tinit[:]
