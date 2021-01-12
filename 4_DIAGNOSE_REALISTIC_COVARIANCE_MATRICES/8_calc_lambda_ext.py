import sys
import numpy as np
import netCDF4 as nc
import scipy.signal as sig
import scipy.interpolate as ip

'''
Calculate decorrelation times for anomalous fluxes from the IPSL_CM5A run

DESCRIPTION:
-----------
This script loads net heat and freshwater flux, zonal and meridional wind stress
 from the IPSL_CM5A pre-industrial control run. 
Heat flux is interpolated from the LDM to the ORCA2 grid.
Anomalies are calculated from a climatology. 

The fluxes are then linearly mapped to a change in SST,SSS, u and v per second, 
and the normalised autocorrelation is calculated. This autocorrelation is
interpolated onto a higher-frequency time array (equivalent 10 outputs per day)
and the e-folding time scale is estimated at each location. These arrays
are saved.

OUTPUT:
--------
Saves arrays of each location and its e-folding time (in years):

   e_folding_time_HF.npy
   e_folding_time_FWF.npy
   e_folding_time_ZMF.npy
   e_folding_time_MMF.npy
   
'''

load_grid=1 # Load necessary ORCA2 grid variables (as in 7_calc_Sigma_ext.py)
load_IPSL=0 # Load outputs from IPSL model (as in 7_calc_Sigma_ext.py)
corr_calc=0 # calculate normalised autocorrelation of flux anomaly time series
calc_time=1 # calculate decorrelation time
save_npy =0 # Save array of decorrelation times (in years) to .npy

################################################################################
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
            bilsB[:,0:4]=bils[N,:,92:]
            bilsB[:,100:]=bils[N,:,0:4]
            bilsB[:,4:100]=bils[N,:,:]
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

if corr_calc:    
    print('calculating normalised autocorrelation, heat flux')
    dTdt_cor=sig.fftconvolve(dTdt_anom,np.flip(dTdt_anom,0),mode='full',axes=0)\
               /(7300*np.var(dTdt_anom,axis=0))    
    print('freshwater flux')
    dSdt_cor=sig.fftconvolve(dSdt_anom,np.flip(dSdt_anom,0),mode='full',axes=0)\
               /(7300*np.var(dSdt_anom,axis=0))
    print('zonal momentum flux')
    dudt_cor=sig.fftconvolve(dudt_anom,np.flip(dudt_anom,0),mode='full',axes=0)\
               /(7300*np.var(dudt_anom,axis=0))
    print('meridional momentum flux')
    dvdt_cor=sig.fftconvolve(dvdt_anom,np.flip(dvdt_anom,0),mode='full',axes=0)\
               /(7300*np.var(dvdt_anom,axis=0))

    print('Saving autocorrelation series:')
    np.save( 'HF_anom_correlation.npy',dTdt_cor)
    np.save('FWF_anom_correlation.npy',dSdt_cor)
    np.save('ZMF_anom_correlation.npy',dudt_cor)
    np.save('MMF_anom_correlation.npy',dvdt_cor)

else:
    print('loading autocorrelation series, heat flux:')
    dTdt_cor=np.load( 'HF_anom_correlation.npy')
    print('freshwater flux:')
    dSdt_cor=np.load('FWF_anom_correlation.npy')
    print('zonal momentum flux')
    dudt_cor=np.load('ZMF_anom_correlation.npy')
    print('meridional momentum flux')
    dvdt_cor=np.load('MMF_anom_correlation.npy')
    
if calc_time:
    # Create 1y time array with 10 pts per day:
    ti=np.linspace(0,1,3650);
    print('calculating decorrelation time')
    print('HF')
    # Extend time array to every spatial location
    ti2=ti.reshape(3650,1,1)*np.ones((1,149,182))
    #Interpolate correlation onto higher-frequency time array:
    f_SST=ip.interp1d(t,dTdt_cor[7299:,:,:],axis=0)
    # Determine first index where correlation < 1/e everywhere
    ti2[np.signbit(f_SST(ti) - np.exp(-1.))]=np.nan
    ITi=np.argmin(ti2,axis=0)
    decor_time_SST=ti[ITi]

    print('FWF')
    ti2=ti.reshape(3650,1,1)*np.ones((1,149,182))
    f_SSS=ip.interp1d(t,dSdt_cor[7299:,:,:],axis=0)
    ti2[np.signbit(f_SSS(ti) - np.exp(-1.))]=np.nan
    ITi=np.argmin(ti2,axis=0)
    decor_time_SSS=ti[ITi]

    print('ZMF')
    ti2=ti.reshape(3650,1,1)*np.ones((1,149,182))
    f_SSU=ip.interp1d(t,dudt_cor[7299:,:,:],axis=0)
    ti2[np.signbit(f_SSU(ti) - np.exp(-1.))]=np.nan
    ITi=np.argmin(ti2,axis=0)
    decor_time_SSU=ti[ITi]

    print('MMF')
    ti2=ti.reshape(3650,1,1)*np.ones((1,149,182))
    f_SSV=ip.interp1d(t,dvdt_cor[7299:,:,:],axis=0)
    ti2[np.signbit(f_SSV(ti) - np.exp(-1.))]=np.nan
    ITi=np.argmin(ti2,axis=0)
    decor_time_SSV=ti[ITi]

if save_npy:
    np.save('e_folding_time_HF.npy' ,decor_time_SST) #e-folding time (yrs)
    np.save('e_folding_time_FWF.npy',decor_time_SSS)
    np.save('e_folding_time_ZMF.npy',decor_time_SSU)
    np.save('e_folding_time_MMF.npy',decor_time_SSV)    

    
sys.exit()
################################################################################
