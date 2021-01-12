import sys
import numpy as np
import netCDF4 as nc
import scipy.signal as sig

'''
Calculate e-folding decorrelation times of subgrid buoyancy fluxes.

DESCRIPTION:
-----------
This script takes the time series of subgrid heat and salt fluxes created by
   3_calc_ORCA2_subgrid_buoy_fluxes.py 
and combined into single files (dTdt.npy;dSdt.npy) by 
   4_calc_Sigma_int.py 
and determines the time at which the correlation drops below 1/e by location.

OUTPUT:
-------
The array of e-folding times (in years) is saved as a .npy file (31x149x182):
e_folding_time_T.npy
e_folding_time_S.npy

The index of the original time series array corresponding to the e-folding
time is saved in a separate file:

e_folding_index_T.npy
e_folding_index_S.npy
'''


### OPTIONS:
load_data=1 # Load subgrid heat and salt fluxes 
corr_calc=1 # Create a (2920x31x149x182) array of lag autocorrelation
corr_load=0 # Load a previously created array
calc_time=1 # calculate e-folding decorrelation time
################################################################################

if load_data:
    # Load single subgrid heat/salt flux time series made by 4_calc_Sigma_int.py
    # remove climatology and detrend

    ### HEAT FLUXES:
    print('loading data, dTdt')
    dTdt_anom=np.load('dTdt.npy') 
    print('remove seasonal cycle, dTdt...')
    dTdt_anom=(dTdt_anom-np.tile(np.mean(\
               dTdt_anom.swapaxes(0,3).reshape(182,31,149,20,73),\
                                             axis=3),20).swapaxes(0,3))
    dTdt_anom[~np.isfinite(dTdt_anom)]=0
    print('detrend dTdt...')
    dTdt_anom=sig.detrend(dTdt_anom,axis=0)

    ### SALT FLUXES:
    print('loading data, dSdt...')
    dSdt_anom=np.load('dSdt.npy')
    print('remove seasonal cycle, dSdt...')
    dSdt_anom=(dSdt_anom-np.tile(np.mean(\
               dSdt_anom.swapaxes(0,3).reshape(182,31,149,20,73),\
                                         axis=3),20).swapaxes(0,3))
    dSdt_anom[~np.isfinite(dSdt_anom)]=0
    print('detrend dSdt...')
    dSdt_anom=sig.detrend(dSdt_anom,axis=0)

    ### GRID VARIABLES:
    print('loading grid')
    MESH=nc.Dataset('mesh_mask_2.nc')
    lon=MESH.variables['glamt'][0,:]       #Latitude
    lat=MESH.variables['gphit'][0,:]       #Longitude
    e3t=MESH.variables['e3t'][0,:]         #Layer height
    t=np.linspace(0,20,1460)               #Array of times (yr)

if corr_calc:    
    # Create an autocorrelation series for the (1460x31x149x182) time series of
    # subgrid fluxes [shape (2920x31x149x182)] and save:
    if dTdt_swi:
        print('calculating normalised autocorrelation, dTdt...')
        dTdt_cor=sig.fftconvolve(dTdt_anom,\
                         np.flip(dTdt_anom,0),mode='full',axes=0)\
                   /(1460*np.var(dTdt_anom,axis=0))
        print('Saving results, dTdt...')
        np.save('dTdt_detrend_deseason_cor.npy',dTdt_cor)

    if dSdt_swi:
        print('dSdt...')
        dSdt_cor=sig.fftconvolve(dSdt_anom,\
                         np.flip(dSdt_anom,0),mode='full',axes=0)\
                   /(1460*np.var(dSdt_anom,axis=0))
        print('Saving results, dSdt...')
        np.save('dSdt_detrend_deseason_cor.npy',dSdt_cor)

elif corr_load:
    # Load a previously calculated autocorrelation series:
    print('loading pre-calculated correlations, T')
    dTdt_cor=np.load('dTdt_detrend_deseason_cor.npy')
    print('loading pre-calculated correlations, S')
    dSdt_cor=np.load('dSdt_detrend_deseason_cor.npy')

if calc_time:
    # Find first index where correlation drops below 1/e
    print('calculating decorrelation time')
    print('dTdt')
    t2=t.reshape(1460,1,1)*np.ones((31,149,182))
    t2[np.signbit(dTdt_cor[1459:,:,:] - np.exp(-1.))]=np.nan
    IT=np.argmin(t2,axis=0) 
    decor_time_T=t[IT]

    print('dSdt')
    t2=t.reshape(1460,1,1)*np.ones((31,149,182))
    t2[np.signbit(dSSSdt_cor[1459:,:,:] - np.exp(-1.))]=np.nan
    IT=np.argmin(t2,axis=0)
    decor_time_S=t[IT]
    
    print('saving decorrelation time and indices')
    np.save('e_folding_time_T.npy',decor_time_T)
    np.save('e_folding_time_S.npy',decor_time_S)
    np.save('e_folding_index_T.npy',IT)
    np.save('e_folding_index_S.npy',IS)


sys.exit()
################################################################################
