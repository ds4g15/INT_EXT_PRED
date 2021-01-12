import sys
import numpy as np
import netCDF4 as nc
################################################################################

'''Build a North Atlantic meridional volume transport (Sv) cost function for the
NEMOTAM configuration at 25N and from the surface to the depth of the 
climatological stream function maximum.
NOTE: will need to create a climatology from the 60 year trajectory
'''
LAT=25 # Latitude of MHT evaluation
O2G=nc.Dataset('mesh_mask.nc') # generate during trajectory run using nn_msh=1
O2M=nc.Dataset('subbasins.nc') #ORCA2 basin masks, in ORCA2INPUT 
                               #(https://doi.org/10.5281/zenodo.1471702)

latv =O2G.variables['gphiv'][:]
lonv =O2G.variables['glamv'][:]
e1v  =O2G.variables['e1v'][:]
e2v  =O2G.variables['e2v'][:]
e3v  =O2G.variables['e3v'][:]
NA   =O2M.variables['Natlmsk_nomed'][:]
atlmsk=O2M.variables['atlmsk'][:]
vmask=O2G.variables['vmask'][:]
gdepv=O2G.variables['gdepv'][:]
gdept_0=O2G.variables['gdept_0'][0,:]

vinit=np.zeros((31,149,182))
Tinit=np.zeros((31,149,182))

################################################################################
# Find model j index closest to lat on average when zonally avg'd over Atlantic
J=np.nanargmin(np.sum(e1v*atlmsk*(latv-LAT)**2,axis=2)/np.sum(e1v*atlmsk,axis=2))
print('Chosen latitude: '+str(LAT))
print('Nearest j index in Atlantic: '+str(J))
CLIM=nc.Dataset('ORCA2_60y_traj_climatology.nc')
MVT=np.sum(np.cumsum(CLIM.variables['vn'][:,:,J,:]*\
                         (atlmsk*e1v*e3v)[:,:,J,:],axis=1),axis=2)
K=np.argmax(np.mean(MVT,axis=0))
#K=22
print('Depth of climatological MVT maximum at this latitude: '+str(gdept_0[K]))
print('(Depth level '+str(K)+')')

vinit[0:K,J,:]=(vmask*atlmsk*e1v*e3v)[0,0:K,J,:]*1e-6

# SAVE TO NETCDF
with nc.Dataset(('MVT_ST.nc'),'w') as NC:
    NC.createDimension('z',31)
    NC.createDimension('y',149)
    NC.createDimension('x',182)
    vinit_nc=NC.createVariable('vinit',np.float64(),('z','y','x'))
    Tinit_nc=NC.createVariable('Tinit',np.float64(),('z','y','x'))
    vinit_nc[:]=vinit[:]
    Tinit_nc[:]=Tinit[:]
