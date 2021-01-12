import scipy.interpolate as ip
import scipy.signal as sig
import netCDF4 as nc
import numpy as np
import sys

'''
Calculate the Root Mean Square Logarithmic Error (RMSLE) when fitting the
surface buoyancy and momentum flux time series to Ornstein-Uhlenbeck (O-U) 
and Gaussian White Noise (GWN) processes.

DESCRIPTION:
-----------
This script takes the time series of anomalous surface fluxes from IPSL_CM5A
and estimates their power spectral density (PSD) using the welch method with a 
sampling frequency of 1 days (the output frequency).

It then calculates (from their variance and e-folding decorrelation time) the
parameters lambda and sigma corresponding to the PSD of an O-U process with the
same properties.

It finally calculates the RMSLE of fitting the theoretical O-U PSD function to
the diagnosed PSD from the time series, as well as the equivalent PSD for a 
WGN process (with constant PSD).

OUTPUTS:
--------
The RMSLE at each location (149x182) is saved as a .npy file for O-U & WGN, 
for all variables:

RMSLE_T_OU_ext.npy
RMSLE_S_OU_ext.npy
RMSLE_u_OU_ext.npy
RMSLE_v_OU_ext.npy

RMSLE_T_WN_ext.npy
RMSLE_S_WN_ext.npy
RMSLE_u_WN_ext.npy
RMSLE_v_WN_ext.npy

'''
MESH=nc.Dataset('/noc/soes/physics/fs_scratch/ds4g15/essential_files/ORCA2_mesh.nc')

load_grid =0
load_IPSL =0
calc_PSDs =0
calc_ERR  =0
save_data =1

if load_grid:
    print('loading grid data')
    alpha=2.1e-4;beta=7.5e-4;rho0=1026;T0=15;S0=35;#seawater reference values
    cp=4e3 #specific heat capacity

    GRID =nc.Dataset('mesh_mask_2.nc')    
    e3t=GRID.variables['e3t'][0,0,:,:];    tmask=GRID.variables['tmask'][:]
    e3u=GRID.variables['e3u'][0,0,:,:];    umask=GRID.variables['umask'][:]
    e3v=GRID.variables['e3v'][0,0,:,:];    vmask=GRID.variables['vmask'][:]

    t=np.linspace(0,20,7300)

################################################################################
    
if load_IPSL:
    print('loading model output')

    #### Freshwater flux
    SBCNC=nc.Dataset('piCtrlDaily2_18500101_18691231_1D_SBC.nc')
    FWF=-SBCNC.variables['wfo'][:]

    #### Heat flux
    # Interpolate onto ORCA2 grid and save OR load if done already:
    if intp_hflx:
        print('loading atmospheric fields')
        ATMNC=nc.Dataset('piCtrlDaily2_18500101_18691231_1D_histday.nc')
        bils=ATMNC.variables['bils'][:,:,:]   # Heat flux (W/m2)
        latL=ATMNC.variables['lat'  ][:]      # LDM lat/lon
        lonB=np.arange(-195,195,3.75)         #
        latO= GRID.variables['gphit'][0,:,:]  # ORCA2 lat/lon
        lonO= GRID.variables['glamt'][0,:,:]  # 

        print('interpolating onto ocean grid')
        HFO2=np.zeros((np.shape(bils)[0],149,182))
        X,Y=np.meshgrid(lonB,latL)
        for N in np.arange(np.shape(bils)[0]):
            bilsB=np.zeros((96,104))
            bilsB[:,0:4  ]=bils[N,:,92:]
            bilsB[:,100: ]=bils[N,:,0:4]
            bilsB[:,4:100]=bils[N,:, : ]
            HFO2[N,:,:]=ip.griddata((X.ravel(),Y.ravel()),\
                                           bilsB.ravel(),(lonO,latO))  
        np.save('bils_ORCA2_interp.npy',HFO2)
    else:
        HFO2=np.load('bils_ORCA2_interp.npy')

    ##### Zonal wind stress
    print('loading zonal wind stress')
    TAUXNC=nc.Dataset('piCtrlDaily2_18500101_18691231_1D_grid_U.nc')
    TAUX=TAUXNC.variables['tauuo'][:]

    ##### Meridional wind stress
    print('loading meridional wind stress')
    TAUYNC=nc.Dataset('piCtrlDaily2_18500101_18691231_1D_grid_V.nc')                   
    TAUY=TAUYNC.variables['tauvo'][:]

    ##### PROJECT TO SURFACE VARIABLE CHANGES
    print('Projecting HF and FWF to rate of SST and SSS change')
    dTdt=(tmask[0,0,:,:]*HF    )/(cp*rho0*e3t)
    dSdt=(tmask[0,0,:,:]*FWF*S0)/(   rho0*e3t)
    dudt=(umask[0,0,:,:]*TAUX  )/(        e3u)
    dvdt=(umask[0,0,:,:]*TAUY  )/(        e3v)

    print('removing seasonal cycle') 
    dTdt_anom=dTdt-\
               np.tile(np.mean(dTdt.reshape(20,365,149,182),axis=0),[20,1,1])
    dSdt_anom=dSdt-\
               np.tile(np.mean(dSdt.reshape(20,365,149,182),axis=0),[20,1,1])
    dudt_anom=dudt-\
               np.tile(np.mean(dudt.reshape(20,365,149,182),axis=0),[20,1,1])
    dvdt_anom=dvdt-\
               np.tile(np.mean(dvdt.reshape(20,365,149,182),axis=0),[20,1,1])
################################################################################

if calc_PSDs:
    #CALCULATE REAL POWER SPECTRAL DENSITY
    print('Calculating power spectral density of time series of: heat flux...')
    WT=sig.welch(dTdt_anom,fs=1./(86400.),nperseg=7300/5,axis=0) #samp freq. is 1d
    print('freshwater flux...')
    WS=sig.welch(dSdt_anom,fs=1./(86400.),nperseg=7300/5,axis=0)
    print('zonal momentum flux...')
    WU=sig.welch(dUdt_anom,fs=1./(86400.),nperseg=7300/5,axis=0)
    print('meridional momentum flux...')
    WV=sig.welch(dVdt_anom,fs=1./(86400.),nperseg=7300/5,axis=0)

    # CALCULATE THEORETICAL POWER SPECTRAL DENSITY
    print('calculating fit, OU')
    print('loading decorr time (in seconds)')
    eft_T=np.load('e_folding_time_HF.npy' )*(86400*365)
    eft_S=np.load('e_folding_time_FWF.npy')*(86400*365)
    eft_U=np.load('e_folding_time_ZMF.npy')*(86400*365)
    eft_V=np.load('e_folding_time_MMF.npy')*(86400*365)

    print('Calculating variance of underlying stochastic noise, sig2')
    sig2_T=2*np.var(dTdt_anom,axis=0)/(eft_T)
    sig2_S=2*np.var(dSdt_anom,axis=0)/(eft_S)
    sig2_U=2*np.var(dudt_anom,axis=0)/(eft_U)
    sig2_V=2*np.var(dvdt_anom,axis=0)/(eft_V)

    print('Calculating theoretical PSD from parameters')
    PSDOU_thy_T=(2*sig2_T[None,:]) / \
                ( (1./eft_T[None,:])**2 + (2*np.pi*WT[0].reshape(-1,1,1))**2)
    PSDOU_thy_S=(2*sig2_S[None,:]) / \
                ( (1./eft_S[None,:])**2 + (2*np.pi*WS[0].reshape(-1,1,1))**2)
    PSDOU_thy_U=(2*sig2_U[None,:]) / \
                ( (1./eft_U[None,:])**2 + (2*np.pi*WU[0].reshape(-1,1,1))**2)
    PSDOU_thy_V=(2*sig2_V[None,:]) / \
                ( (1./eft_V[None,:])**2 + (2*np.pi*WV[0].reshape(-1,1,1))**2)
    
if calc_RMSLE:
    print('Calculating error, OU')
    ERR_SST_OU=tmask[0,0,:]*np.sqrt((1./len(WT[0]))*\
             np.sum(  (np.log10(PSDOU_thy_T)-np.log10(WT[1]))**2,axis=0))/\
             np.mean(  np.log10(WT[1])  ,axis=0)

    ERR_SSS_OU=tmask[0,0,:]*np.sqrt((1./len(WS[0]))*\
             np.sum(  (np.log10(PSDOU_thy_S)-np.log10(WS[1]))**2,axis=0))/\
             np.mean(  np.log10(WS[1])  ,axis=0)             

    ERR_SSU_OU=umask[0,0,:]*np.sqrt((1./len(WU[0]))*\
             np.sum(  (np.log10(PSDOU_thy_U)-np.log10(WU[1]))**2,axis=0))/\
             np.mean(  np.log10(WU[1])  ,axis=0)             

    ERR_SSV_OU=vmask[0,0,:]*np.sqrt((1./len(WV[0]))*\
             np.sum(  (np.log10(PSDOU_thy_V)-np.log10(WV[1]))**2,axis=0))/\
             np.mean(  np.log10(WV[1])  ,axis=0)             

    print('RMSLE, WN')
    ERR_SST_WN=tmask[0,0,:]*np.sqrt((1./len(WT[0]))*\
             np.sum(  (np.log10(np.mean(WT[1],axis=0))-np.log10(WT[1]))**2,axis=0))/\
             np.mean(  np.log10(WT[1])  ,axis=0)           
    ERR_SSS_WN=tmask[0,0,:]*np.sqrt((1./len(WS[0]))*\
             np.sum(  (np.log10(np.mean(WS[1],axis=0))-np.log10(WS[1]))**2,axis=0))/\
             np.mean(  np.log10(WS[1])  ,axis=0)           

    ERR_SSU_WN=umask[0,0,:]*np.sqrt((1./len(WU[0]))*\
             np.sum(  (np.log10(np.mean(WU[1],axis=0))-np.log10(WU[1]))**2,axis=0))/\
             np.mean(  np.log10(WU[1])  ,axis=0)                                       

    ERR_SSV_WN=vmask[0,0,:]*np.sqrt((1./len(WV[0]))*\
             np.sum(  (np.log10(np.mean(WV[1],axis=0))-np.log10(WV[1]))**2,axis=0))/\
             np.mean(  np.log10(WV[1])  ,axis=0)                          


if save_data:
    print('Saving O-U RMSLE array, HF')
    np.save('RMSLE_T_ext_OU',ERR_SST_OU.data)
    print('Saving O-U RMSLE array, FWF')
    np.save('RMSLE_S_ext_OU',ERR_SSS_OU.data)
    print('Saving O-U RMSLE array, ZMF')
    np.save('RMSLE_u_ext_OU',ERR_SSU_OU.data)
    print('Saving O-U RMSLE array, MMF')
    np.save('RMSLE_v_ext_OU',ERR_SSV_OU.data)

    print('Saving WGN RMSLE array, HF')
    np.save('RMSLE_T_ext_WN',ERR_SST_WN.data)
    print('Saving WGN RMSLE array, FWF')
    np.save('RMSLE_S_ext_WN',ERR_SSS_WN.data)
    print('Saving WGN RMSLE array, ZMF')
    np.save('RMSLE_u_ext_WN',ERR_SSU_WN.data)
    print('Saving WGN RMSLE array, MMF')
    np.save('RMSLE_v_ext_WN',ERR_SSV_WN.data)
    
sys.exit()
################################################################################
