import os
import sys
import time
import numpy as np
import netCDF4 as nc
import datetime as dt
import scipy.sparse as sp
import matplotlib.pyplot as plt

'''
Calculate the spatial residual mean internal buoyancy fluxes from the ORCA025
output after averaging on the ORCA2 grid.

DESCRIPTION
-----
Based on Eq. (31) from our study (from manuscript draft 2020-10-27).

a:Average ORCA025 T,S,u,v,w variables to ORCA2 grid using transformation matrix &
calculate spatial fluctuations on the ORCA025 grid after subtracting this avg.

b:Get advection of spatial buoyancy fluctuations by spatial velocity fluctuations

c:Average this residual buoyancy flux onto the ORCA2 grid and save: RHS of (30)

ARGUMENTS
--------
N: output number, between 0 and 1459. Integer corresponding to which ORCA025
output to work on (based on 5d average output for 20 years = 73*20 outputs)

OUTPUTS:
-------
Produces two .npy files per call:
dTdt_YYYYMMDD-YYYYMMDD.npy
dSdt_YYYYMMDD-YYYYMMDD.npy
which correspond to the internal subgrid temperature (K/s) and salinity (psu/s) 
fluxes over the 5d period corresponding to the ORCA025 output provided.
'''

# Get correct date string for output number "N":
N        =int(sys.argv[1])
################################################################################
### FIRST STEP: 
###       Load climatological 5d period corresponding to this 5d period & remove

pd=np.mod(N,73)
print('loading climatology for pentad '  +str('%0.3i' % pd))
print('T');T_bar=np.load('T_mean_pentad_'+str('%0.3i' % pd)+'.npy')
print('S');S_bar=np.load('S_mean_pentad_'+str('%0.3i' % pd)+'.npy')
print('U');U_bar=np.load('U_mean_pentad_'+str('%0.3i' % pd)+'.npy')
print('V');V_bar=np.load('V_mean_pentad_'+str('%0.3i' % pd)+'.npy')
print('W');W_bar=np.load('W_mean_pentad_'+str('%0.3i' % pd)+'.npy')

################################################################################
YEAR     =str((N//73)+331).zfill(4)
STARTDATE=dt.datetime.strftime(\
          dt.datetime.strptime('19010101','%Y%m%d')+\
          dt.timedelta(days=(N%73)*5)    ,  '%m%d')
ENDDATE  =dt.datetime.strftime(\
          dt.datetime.strptime('19010101','%Y%m%d')+\
          dt.timedelta(days=4+((N%73)*5)),  '%m%d')

print(str(N).zfill(4)+': '+'loading model output')
outdir='/hpcdata/scratch/ds4g15/ORCAV3.5RUNS/CLIMF_025/CLIMF_025_output/'

print((str(N).zfill(4)+': '+\
                outdir+'CLIMF_025_5d_gridT_'+YEAR+STARTDATE+\
                                         '-'+YEAR+ENDDATE+'.nc'))
print((str(N).zfill(4)+': '+\
                outdir+'CLIMF_025_5d_gridU_'+YEAR+STARTDATE+\
                                         '-'+YEAR+ENDDATE+'.nc'))
print((str(N).zfill(4)+': '+\
                outdir+'CLIMF_025_5d_gridV_'+YEAR+STARTDATE+\
                                         '-'+YEAR+ENDDATE+'.nc'))
print((str(N).zfill(4)+': '+\
                outdir+'CLIMF_025_5d_gridW_'+YEAR+STARTDATE+\
                                         '-'+YEAR+ENDDATE+'.nc'))

TNC=nc.Dataset((outdir+'CLIMF_025_5d_gridT_'+YEAR+STARTDATE+\
                                         '-'+YEAR+ENDDATE+'.nc'))
UNC=nc.Dataset((outdir+'CLIMF_025_5d_gridU_'+YEAR+STARTDATE+
                                         '-'+YEAR+ENDDATE+'.nc'))
VNC=nc.Dataset((outdir+'CLIMF_025_5d_gridV_'+YEAR+STARTDATE+
                                         '-'+YEAR+ENDDATE+'.nc'))
WNC=nc.Dataset((outdir+'CLIMF_025_5d_gridW_'+YEAR+STARTDATE+\
                                         '-'+YEAR+ENDDATE+'.nc'))

T=TNC.variables['votemper'][:]-T_bar #T anomaly from climatology
S=TNC.variables['vosaline'][:]-S_bar #S anomaly from climatology
U=UNC.variables['vozocrtx'][:]-U_bar #U anomaly from climatology
V=VNC.variables['vomecrty'][:]-V_bar #V anomaly from climatology
W=WNC.variables['vovecrtz'][:]-W_bar #W anomaly from climatology

################################################################################
### SECOND STEP:
###        Load grids for the different model configurations

print(str(N).zfill(4)+': '+'loading grid information')
mesh_025=nc.Dataset('mesh_mask_025.nc') #obtain by running ORCA025 with nn_msh=1
mesh_2  =nc.Dataset('mesh_mask_2.nc')   #obtain by running ORCA2   with nn_msh=1
    
#Load grid dimensionality for each variable
e1t=mesh_025.variables['e1t'][:];
e2t=mesh_025.variables['e2t'][:];
e3t=mesh_025.variables['e3t'][:];

e1u=mesh_025.variables['e1u'][:];
e2u=mesh_025.variables['e2u'][:];
e3u=mesh_025.variables['e3u'][:];

e1v=mesh_025.variables['e1v'][:];
e2v=mesh_025.variables['e2v'][:];
e3v=mesh_025.variables['e3v'][:];

e3w=mesh_025.variables['e3w'][:]; #'e1w' and 'e2w' correspond to e1t and e2t

#Calculate grid volumes:
dVT=e1t*e2t*e3t
dVU=e1u*e2u*e3u
dVV=e1v*e2v*e3v
dVW=e1t*e2t*e3w

tmask2=mesh_2.variables['tmask'][:]
umask2=mesh_2.variables['umask'][:]
vmask2=mesh_2.variables['vmask'][:]

################################################################################
### THIRD STEP:
###       Load transformation matrices and remove ORCA2 (LRM) avg 
###                                        from ORCA025 (HRM) T,S,u,v,w output
###       See Eq. (31) in our manuscript, 2020-10-27 draft

print(str(N).zfill(4)+': '+'Calculating ORCA2 averages and ORCA025 residuals:')
print(str(N).zfill(4)+': '+'Loading sparse conversion matrices')

CONVT       = sp.load_npz('conversion_matrices/ORCA025_to_ORCA2_centre_t.npz')
CONVU       = sp.load_npz('conversion_matrices/ORCA025_to_ORCA2_centre_u.npz')
CONVV       = sp.load_npz('conversion_matrices/ORCA025_to_ORCA2_centre_v.npz')
CONVW       = sp.load_npz('conversion_matrices/ORCA025_to_ORCA2_centre_w.npz')

###############
print(str(N).zfill(4)+': '+'Project ORCA2 land masks to ORCA025 grid')
tmask2_025 = (CONVT.T).dot( tmask2.flatten() ).reshape(1,75,1021,1442)
umask2_025 = (CONVU.T).dot( umask2.flatten() ).reshape(1,75,1021,1442) 
vmask2_025 = (CONVV.T).dot( vmask2.flatten() ).reshape(1,75,1021,1442) 


def ORCA025_residual(CONV,VAR,dV):
    '''
    Function to calculate ORCA2 average of an ORCA025 variable (VAR)
    using the conversion/transformation matrix (CONV) and remove it,
    leaving a spatial fluctuation (VAR_025) on the ORCA025 grid.
    '''
    VAR_mean_2    = (CONV  ).dot( (VAR*dV).reshape(1021*1442*75,1) )\
                  / (CONV  ).dot(     (dV).reshape(1021*1442*75,1) ) 
    VAR_025  = (CONV.T).dot( (VAR_mean_2) ) #ORCA2 mean projected on 025 grid
    VAR_025 = (VAR - VAR_025.reshape(1,75,1021,1442)) #ORCA025 residual
    return VAR_025

print(str(N).zfill(4)+': '+'calculating residual T');
T_resid_025 = ORCA025_residual(CONVT,T,dVT)*tmask2_025
print(str(N).zfill(4)+': '+'calculating residual S');
S_resid_025 = ORCA025_residual(CONVT,S,dVT)*tmask2_025
print(str(N).zfill(4)+': '+'calculating residual U');
U_resid_025 = ORCA025_residual(CONVU,U,dVU)*umask2_025
print(str(N).zfill(4)+': '+'calculating residual V');
V_resid_025 = ORCA025_residual(CONVV,V,dVV)*vmask2_025
print(str(N).zfill(4)+': '+'calculating residual W');
W_resid_025 = ORCA025_residual(CONVW,W,dVW)*tmask2_025
###############


################################################################################
### FOURTH STEP:
###        Use the spatial fluctuations calculated in step 3 to determine 
###        advection of spatial buoy. fluctuations by spatial velo. fluctuations
###        by differencing of T'v'dA across ORCA025 cells.
###        (Note T points need simple interpolation onto grid edge u,v,w points)

### i. ZONAL HEAT/SALT FLUX:
print(str(N).zfill(4)+': '+'Calculating heat and salt flux due to residuals')
print(str(N).zfill(4)+': '+'zonal heat and salt flux due to residuals')

# Get grid indices 
#(note NOT 1442: 1440 and 1441 are same as 0 and 1 by wrapped grid construction)

kk025,jj025,ii025=np.meshgrid(np.arange(75),\
                              np.arange(1021),\
                              np.arange(1440),indexing='ij')

# Calculate heat flux. Use mod to wrap around grid edge:
HFU=      U_resid_025[0,kk025,jj025,       ii025        ]*\
                  e2u[0,      jj025,       ii025        ]*\
                  e3u[0,kk025,jj025,       ii025        ]*\
     0.5*(T_resid_025[0,kk025,jj025,       ii025        ]+\
          T_resid_025[0,kk025,jj025,np.mod(ii025+1,1440)])\
                                                         -\
          U_resid_025[0,kk025,jj025,np.mod(ii025-1,1440)]*\
                  e2u[0,      jj025,np.mod(ii025-1,1440)]*\
                  e3u[0,kk025,jj025,np.mod(ii025-1,1440)]*\
     0.5*(T_resid_025[0,kk025,jj025,np.mod(ii025-1,1440)]+\
          T_resid_025[0,kk025,jj025,       ii025       ])

SFU=      U_resid_025[0,kk025,jj025,       ii025        ]*\
                  e2u[0,      jj025,       ii025        ]*\
                  e3u[0,kk025,jj025,       ii025        ]*\
     0.5*(S_resid_025[0,kk025,jj025,       ii025        ]+\
          S_resid_025[0,kk025,jj025,np.mod(ii025+1,1440)])\
                                                         -\
          U_resid_025[0,kk025,jj025,np.mod(ii025-1,1440)]*\
                  e2u[0,      jj025,np.mod(ii025-1,1440)]*\
                  e3u[0,kk025,jj025,np.mod(ii025-1,1440)]*\
     0.5*(S_resid_025[0,kk025,jj025,np.mod(ii025-1,1440)]+\
          S_resid_025[0,kk025,jj025,       ii025       ])

#######################
### ii. MERIDIONAL HEAT/SALT FLUX:

print(str(N).zfill(4)+': '+'meridional heat and salt flux due to residuals')
# Set no flux from upper edge of the grid:
V_resid_025[0,:,1020,:]=0;

# Note: for meridional heat flux, top V entry is 0, so no heat flux into top 
# cell from North regardless of T values. "mod" used to make sure /some/ T value
#  is used North of the cell. For the bottom, entire cell is on land, so any V 
# and T values can be used South of the cell. For simplicity, the values at the 
# north of the cell are used again (hence "max")

HFV=      V_resid_025[0,kk025,           jj025        ,ii025]*\
                  e1v[0,                 jj025        ,ii025]*\
                  e3v[0,kk025,           jj025        ,ii025]*\
     0.5*(T_resid_025[0,kk025,np.mod(    jj025+1,1020),ii025]+\
          T_resid_025[0,kk025,           jj025,        ii025])\
                                                             -\
          V_resid_025[0,kk025,np.maximum(jj025-1,   0),ii025]*\
                  e1v[0      ,np.maximum(jj025-1,   0),ii025]*\
                  e3v[0,kk025,np.maximum(jj025-1,   0),ii025]*\
     0.5*(T_resid_025[0,kk025,np.maximum(jj025-1,   0),ii025]+
          T_resid_025[0,kk025,           jj025,        ii025])

SFV=      V_resid_025[0,kk025,           jj025        ,ii025]*\
                  e1v[0,                 jj025        ,ii025]*\
                  e3v[0,kk025,           jj025        ,ii025]*\
     0.5*(S_resid_025[0,kk025,np.mod(    jj025+1,1020),ii025]+
          S_resid_025[0,kk025,           jj025,        ii025])\
                                                             -\
          V_resid_025[0,kk025,np.maximum(jj025-1,   0),ii025]*\
                  e1v[0      ,np.maximum(jj025-1,   0),ii025]*\
                  e3v[0,kk025,np.maximum(jj025-1,   0),ii025]*\
     0.5*(S_resid_025[0,kk025,np.maximum(jj025-1,   0),ii025]+
          S_resid_025[0,kk025,           jj025        ,ii025])

####################
### iii. VERTICAL HEAT/SALT FLUX:

print(str(N).zfill(4)+': '+'vertical heat and salt flux due to residuals')

# Set no flux through surface:
W_resid_025[0,0,:,:]=0
################################################################################
# T point used "above" surface is arbitrary, so "max" used to pick surface 
# value again. Bottom cell is in the bed, so any T or W can be used, hence "mod"

HFW=      W_resid_025[0,           kk025      ,jj025,ii025]*\
                  e1t[0                       ,jj025,ii025]*\
                  e2t[0                       ,jj025,ii025]*\
     0.5*(T_resid_025[0,np.maximum(kk025-1,0) ,jj025,ii025]+\
          T_resid_025[0,           kk025      ,jj025,ii025])\
                                                          -\
          W_resid_025[0,np.mod(    kk025+1,74),jj025,ii025]*\
                  e1t[0,                       jj025,ii025]*\
                  e2t[0,                       jj025,ii025]*\
     0.5*(T_resid_025[0,np.mod(    kk025+1,74),jj025,ii025]+\
          T_resid_025[0,           kk025      ,jj025,ii025])

SFW=      W_resid_025[0,           kk025      ,jj025,ii025]*\
                  e1t[0,                       jj025,ii025]*\
                  e2t[0,                       jj025,ii025]*\
     0.5*(S_resid_025[0,np.maximum(kk025-1,0) ,jj025,ii025]+\
          S_resid_025[0,           kk025      ,jj025,ii025])\
                                                          -\
          W_resid_025[0,np.mod(    kk025+1,74),jj025,ii025]*\
                  e1t[0,                       jj025,ii025]*\
                  e2t[0,                       jj025,ii025]*\
     0.5*(S_resid_025[0,np.mod(    kk025+1,74),jj025,ii025]+\
          S_resid_025[0,           kk025      ,jj025,ii025])


################################################################################
### iv. GET TOTAL FLUX FROM COMBINED ZONAL, MERIDIONAL AND VERTICAL:

print(str(N).zfill(4)+': '+'Calculating total temperature/salinity change rate due to heat flux')

dTdt_025                = np.zeros((75,1021,1442))
dTdt_025[:,:,   0:1440] = HFW+HFV+HFU
dTdt_025[:,:,1440:1442] = dTdt_025[:,:,0:2] #Looping at the grid edge
dTdt_025                = dTdt_025/dVT

dSdt_025                = np.zeros((75,1021,1442))
dSdt_025[:,:,   0:1440] = SFW+SFV+SFU
dSdt_025[:,:,1440:1442] = dSdt_025[:,:,0:2] #Looping at the grid edge
dSdt_025                = dSdt_025/dVT
                    
################################################################################
### FOURTH STEP: 
###        Average the residual fluxes back onto the ORCA2 grid and save.
###        RHS in Eq. (31), from manuscript draft 2020-10-27

print(str(N).zfill(4)+': '+'Averaging T,S flux to ORCA2 grid')
dTdt_2 = CONVT.dot( (dTdt_025*dVT).reshape(1021*1442*75,1) ) \
       / CONVT.dot(            dVT.reshape(1021*1442*75,1) )       
dSdt_2 = CONVT.dot( (dSdt_025*dVT).reshape(1021*1442*75,1) ) \
       / CONVT.dot(            dVT.reshape(1021*1442*75,1) )       

dTdt_2 = np.reshape(np.array(dTdt_2),(1,31,149,182))
dSdt_2 = np.reshape(np.array(dSdt_2),(1,31,149,182))

print(str(N).zfill(4)+': '+'saving')
print(  ('dTdt_'+YEAR+STARTDATE+'-'+YEAR+ENDDATE+'.npy'))
np.save(('dTdt_'+YEAR+STARTDATE+'-'+YEAR+ENDDATE+'.npy'),dTdt_2)
print(  ('dSdt_'+YEAR+STARTDATE+'-'+YEAR+ENDDATE+'.npy'))
np.save(('dSdt_'+YEAR+STARTDATE+'-'+YEAR+ENDDATE+'.npy'),dSdt_2)

