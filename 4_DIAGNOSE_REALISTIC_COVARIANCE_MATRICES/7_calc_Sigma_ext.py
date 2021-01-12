import numpy as np
import netCDF4 as nc
import scipy.sparse as sp
import scipy.interpolate as ip

'''
 Calculate anomalous surface flux covariance matrices for the IPSL_CM5A run

DESCRIPTION:
-----------
This script loads net heat and freshwater flux, zonal and meridional wind stress
 from the IPSL_CM5A pre-industrial control run. 
Heat flux is interpolated from the LDM to the ORCA2 grid.
Anomalies are calculated from a climatology. 

The fluxes are then linearly mapped to a change in SST,SSS, u and v per second, 
and the covariance is calculated. To calculate the covariance matrix, 
land points are temporarily deleted to save memory. 

The covariance matrices are separated into 3 parts (self- and cross-covariance)
restructured to include land points again, before being saved as a sparse array
in .npz file format.

OUTPUT:
------
Sparse arrays of shape (27118,27118):
   covariance_matrices/COV_SST_SST.npz [Units K^2/s^2]
   covariance_matrices/COV_SSS_SSS.npz [Units psu^2/s^2]
   covariance_matrices/COV_SST_SSS.npz [Units (K psu)/s^2]

   covariance_matrices/COV_SSU_SSU.npz [Units m^2/s^4]
   covariance_matrices/COV_SSV_SSV.npz [Units m^2/s^4]
   covariance_matrices/COV_SSU_SSV.npz [Units m^2/s^4]
'''
################################################################################
### OPTIONS##
load_grid = 1 #Load ORCA2 grid information
load_IPSL = 1 #Load IPSL-CM5A model output (heat and FW fluxes)
intp_hflx = 1 #Interpolate heat flux from atmosphere to ocean grid
calc_covm = 1 #Calculate global covariance matrix
sprs_mtrx = 1 #Generate (149*182x149*182) sparse-format matrix
save_npzf = 1 #Save sparse matrix in .npz format
################################################################################


if load_grid:
    print('loading grid data')
    alpha=2.1e-4;beta=7.5e-4;rho0=1026;T0=15;S0=35;#seawater reference values
    cp=4e3 #specific heat capacity

    GRID =nc.Dataset('mesh_mask_2.nc')    
    e3t=GRID.variables['e3t'][0,0,:,:];    tmask=GRID.variables['tmask'][:]
    e3u=GRID.variables['e3u'][0,0,:,:];    umask=GRID.variables['umask'][:]
    e3v=GRID.variables['e3v'][0,0,:,:];    vmask=GRID.variables['vmask'][:]
    
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

################################################################################    
if calc_covm:
    print('removing seasonal cycle') 
    dTdt_anom=dTdt-\
               np.tile(np.mean(dTdt.reshape(20,365,149,182),axis=0),[20,1,1])
    dSdt_anom=dSdt-\
               np.tile(np.mean(dSdt.reshape(20,365,149,182),axis=0),[20,1,1])
    dudt_anom=dudt-\
               np.tile(np.mean(dudt.reshape(20,365,149,182),axis=0),[20,1,1])
    dvdt_anom=dvdt-\
               np.tile(np.mean(dvdt.reshape(20,365,149,182),axis=0),[20,1,1])
    
    tmaskflat=np.reshape(tmask[0,0,:,:],(149*182) )
    umaskflat=np.reshape(umask[0,0,:,:],(149*182) )
    vmaskflat=np.reshape(vmask[0,0,:,:],(149*182) )

    print('flattening time series into 2d arrays and deleting land for memory')
    dTdt_anom_flat_noland=np.delete(np.reshape(dTdt_anom,\
                                     (np.shape(dTdt_anom)[0],182*149)),\
                                      np.where(tmaskflat==0),1)
    dSdt_anom_flat_noland=np.delete(np.reshape(dSdt_anom,\
                                     (np.shape(dSdt_anom)[0],182*149)),\
                                      np.where(tmaskflat==0),1)
    dudt_anom_flat_noland=np.delete(np.reshape(dudt_anom,\
                                     (np.shape(dudt_anom)[0],182*149)),\
                                      np.where(umaskflat==0),1)
    dvdt_anom_flat_noland=np.delete(np.reshape(dvdt_anom,\
                                     (np.shape(dvdt_anom)[0],182*149)),\
                                      np.where(vmaskflat==0),1)    

    print('calculating global covariance matrix of all variables')
    COVB=np.cov(dTdt_anom_flat_noland.T,dSdt_anom_flat_noland.T)
    COVM=np.cov(dudt_anom_flat_noland.T,dvdt_anom_flat_noland.T)

    print('extracting individual covariance matrices for different variables')
    tlen=len(tmaskflat[tmaskflat>0]); #number of ORCA2 gridpoints
    ulen=len(umaskflat[umaskflat>0]);
    vlen=len(vmaskflat[vmaskflat>0])
    
    COV_TT=COVB[0:tlen,0:tlen];COV_TT[~np.isfinite(COV_TT)]=0;
    COV_TS=COVB[0:tlen,tlen: ];COV_TS[~np.isfinite(COV_TS)]=0;
    COV_SS=COVB[tlen: ,tlen: ];COV_SS[~np.isfinite(COV_SS)]=0;

    COV_UU=COVM[0:ulen,0:ulen];COV_UU[~np.isfinite(COV_UU)]=0;
    COV_UV=COVM[0:ulen,ulen: ];COV_UV[~np.isfinite(COV_UV)]=0;
    COV_VV=COVM[ulen: ,ulen: ];COV_VV[~np.isfinite(COV_VV)]=0;


if sprs_mtrx:
    Oindt=np.argwhere(tmaskflat>0).ravel() #Get non-land indices
    Oindv=np.argwhere(vmaskflat>0).ravel()
    Oindu=np.argwhere(umaskflat>0).ravel()    

    print('Make index arrays of rows and columns')
    row_indt=   np.tile(Oindt,tlen)
    row_indvv=  np.tile(Oindv,vlen)
    row_induu=  np.tile(Oindu,ulen)
    row_induv=np.repeat(Oindu,vlen)

    col_indt=  np.repeat(Oindt,tlen)
    col_indvv= np.repeat(Oindv,vlen)
    col_induu= np.repeat(Oindu,ulen)
    col_induv=   np.tile(Oindv,ulen)

    print('reshaping compressed covariance matrix')
    COV_TT=np.reshape(COV_TT,(tlen**2,))
    COV_SS=np.reshape(COV_SS,(tlen**2,))
    COV_TS=np.reshape(COV_TS,(tlen*tlen,))

    COV_VV=np.reshape(COV_VV,(vlen**2,))
    COV_UU=np.reshape(COV_UU,(ulen**2,))
    COV_UV=np.reshape(COV_UV,(ulen*vlen,))            

################################################################################
    print('creating sparse (csr) matrices with full dimensionality')
    print('Buoyancy fluxes')
    COV_TT_sp=sp.coo_matrix((COV_TT,(row_indt,col_indt)),\
                                shape=(182*149,182*149)).tocsr()
    COV_SS_sp=sp.coo_matrix((COV_SS,(row_indt,col_indt)),\
                                shape=(182*149,182*149)).tocsr()
    COV_TS_sp=sp.coo_matrix((COV_TS,(row_indt,col_indt)),\
                                shape=(182*149,182*149)).tocsr()

    print('Momentum fluxes')
    COV_VV_sp=sp.csr_matrix((COV_VV,(row_indvv,col_indvv)),\
                                  shape=(182*149,182*149))
    COV_UU_sp=sp.csr_matrix((COV_UU,(row_induu,col_induu)),\
                                  shape=(182*149,182*149))
    COV_UV_sp=sp.csr_matrix((COV_UV,(row_induv,col_induv)),\
                                  shape=(182*149,182*149))
if save_npzf:
    print('Saving sparse matrices...')
    print('Heat flux')
    sp.save_npz('covariance_matrices/COV_SST_SST',COV_TT_sp)
    print('Salt flux')
    sp.save_npz('covariance_matrices/COV_SSS_SSS',COV_SS_sp)
    print('Covariance')
    sp.save_npz('covariance_matrices/COV_SST_SSS',COV_TS_sp)    

    print('Zonal momentum flux')
    sp.save_npz('COV_SSU_SSU',COV_UU_sp)
    print('Meridional momentum flux')
    sp.save_npz('COV_SSU_SSV',COV_UV_sp)
    print('Covariance')
    sp.save_npz('COV_SSV_SSV',COV_VV_sp)    
