import scipy.signal as sig
import netCDF4 as nc
import numpy as np
import sys

'''
Calculate the Root Mean Square Logarithmic Error (RMSLE) when fitting the
subgrid buoyancy flux time series to Ornstein-Uhlenbeck (O-U) and Gaussian
White Noise (GWN) processes.

DESCRIPTION:
-----------
This script takes the time series of subgrid heat and salt fluxes created by
   3_calc_ORCA2_subgrid_buoy_fluxes.py 
and combined into single files (dTdt.npy;dSdt.npy) by 
   4_calc_Sigma_int.py 
and estimates their power spectral density (PSD) using the welch method with a 
sampling frequency of 5 days (the output frequency).

It then calculates (from their variance and e-folding decorrelation time) the
parameters lambda and sigma corresponding to the PSD of an O-U process with the
same properties.

It finally calculates the RMSLE of fitting the theoretical O-U PSD function to
the diagnosed PSD from the time series, as well as the equivalent PSD for a 
WGN process (with constant PSD).

OUTPUTS:
--------
The RMSLE at each location (31x149x182) is saved as a .npy file for O-U & WGN, 
for both heat and salt fluxes:

RMSLE_T_int_OU.npy
RMSLE_S_int_OU.npy
RMSLE_T_int_WN.npy
RMSLE_S_int_WN.npy

'''
################################################################################
### OPTIONS:
load_data =1 # Load the subgrid flux time series
calc_PSDs =1 # Calculate the PSD of the time series & the equivalent O-U PSD 
calc_RMSLE=1 # Calculate the RMSLE between the theoretical and real PSDs
save_data =1 # Save the RMSLE array
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


################################################################################

if calc_PSDs:
    #CALCULATE TIME SERIES POWER SPECTRAL DENSITY
    print('Calculating PSDs of subgrid heat fluxes')
    WT=sig.welch(dTdt_anom,fs=1./(86400*5),nperseg=1460/5,axis=0) 
    print('Calculating PSDs of subgrid salt fluxes')
    WS=sig.welch(dSdt_anom,fs=1./(86400*5),nperseg=1460/5,axis=0)

    # CALCULATE THEORETICAL (ORNSTEIN-UHLENBECK) POWER SPECTRAL DENSITY FN:
    print('calculating parameters of theoretical Ornstein-Uhlenbeck PSD:')
    print('lambda:')
    eft_T=np.load('e_folding_time_T.npy')*(86400*365)
    eft_S=np.load('e_folding_time_S.npy')*(86400*365) # from years to s

    print('sigma:')
    sig2_T=2*np.var(dTdt_anom,axis=0)/(eft_T)
    sig2_S=2*np.var(dSdt_anom,axis=0)/(eft_S)

    print('Calculating theoretical O-U PSD from parameters')
    PSDOU_thy_T=(2*sig2_T[None,:]) / \
                ( (1./eft_T[None,:])**2 + (2*np.pi*WT[0].reshape(-1,1,1,1))**2)
    PSDOU_thy_S=(2*sig2_S[None,:]) / \
                ( (1./eft_S[None,:])**2 + (2*np.pi*WS[0].reshape(-1,1,1,1))**2)

if calc_RMSLE:
    print('Calculating RMSLE for an O-U process')
    ERR_T_OU=np.sqrt((1./len(WT[0]))*\
             np.sum(  (np.log10(PSDOU_thy_T)-np.log10(WT[1]))**2,axis=0))\
            /np.mean(np.log10(WT[1]),axis=0)

    ERR_S_OU=np.sqrt((1./len(WS[0]))*\
             np.sum(  (np.log10(PSDOU_thy_S)-np.log10(WS[1]))**2,axis=0))\
            /np.mean(np.log10(WS[1]),axis=0)

    print('Calculating RMSL for a White Gaussian Noise process')
    ERR_T_WN=np.sqrt((1./len(WT[0]))*\
             np.sum(  (np.log10(np.mean(WT[1]))-np.log10(WT[1]))**2,axis=0))\
            /np.mean(np.log10(WT[1]),axis=0)

    ERR_S_WN=np.sqrt((1./len(WS[0]))*\
             np.sum(  (np.log10(np.mean(WS[1]))-np.log10(WS[1]))**2,axis=0))\
            /np.mean(np.log10(WS[1]),axis=0)


if save_data:
    print('Saving O-U RMSLE array, T')
    np.save('RMSLE_T_int_OU.npy',ERR_T_OU)
    print('Saving O-U RMSLE array, S')
    np.save('RMSLE_S_int_OU.npy',ERR_S_OU)
    print('Saving WGN RMSLE array, T')
    np.save('RMSLE_T_int_WN.npy',ERR_T_WN)
    print('Saving WGN RMSLE array, S')
    np.save('RMSLE_S_int_WN.npy',ERR_S_WN)

################################################################################
sys.exit()
