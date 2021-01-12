from scipy.ndimage.filters import convolve1d
import scipy.signal as sig
import scipy.sparse as sp
import datetime as dt
import numpy as np
import time
import sys

'''
This script produces a covariance matrix of subgrid buoyancy fluxes based on
the time series constructed in 3_calc_ORCA2_subgrid_buoy_fluxes.py.

The covariance in space is calculated between throughout the local vertical &
in an "N" point neighbourhood in the horizontal. It is constructed cumulatively
in a loop, rather than loading the entire time series at once, using einstein
summation. Covariance matrices for subgrid heat fluxes, salt fluxes, and their
covariance are calculated separately.

The results are then cast into sparse matrices of size (nijk,nijk), which is 
(840658,840658) for ORCA2. This allows linear algebraic operations to be carried
out on ORCA2 vectors, while saving space as the matrix is largely zero due to
the small neighbourhoods of horizontal covariance. These are saved to .npz files
'''
################################################################################
# OPTION SWITCHES #
rebuild_series      =0 # Make 1 time series from separate subgrid flux snapshots
load_series         =1 # Load a previously built series if already rebuilt
calculate_covariance=1 # Run the main loop described above and save to .npy
load_covariance     =0 # Load a previously calculated covariance matrix as .npy
build_sparse_matrix =1 # convert the calculated covariance matrix to sparse fmt
save_sparse_matrix  =1 # Save the sparse fmt covariance matrix as a .npz file

# PARAMETERS:

N=3;            #Size of correlation neighbourhood, NxN
NN=int((N-1)/2) #Number of points around the centre point for this n'hood.
################################################################################
if rebuild_series:
    # Take the 1460 subgrid flux .npy files produced by 
    # 3_calc_ORCA2_subgrid_buoy_fluxes.py and assemble into a single array:

    outdir='/path/to/CLIMF_025_TS_fluxes/fluxes/'
    dTdt=np.zeros((1460,31,149,182)) # Internal temperature flux time series
    dSdt=np.zeros((1460,31,149,182)) # Internal salinity flux time series
    i=0
    for Y in np.arange(20):
        for D in np.arange(73):
            print('Rebuilding time series, year '+str(Y)+' of 20, '+\
                                           'day '+str(D)+' of 73. ')
            YEAR=str(Y+331).zfill(4)
            STARTDATE=dt.datetime.strftime(\
                      dt.datetime.strptime('19010101','%Y%m%d')+\
                      dt.timedelta(days=(D*5))       ,  '%m%d')
            ENDDATE  =dt.datetime.strftime(\
                      dt.datetime.strptime('19010101','%Y%m%d')+\
                      dt.timedelta(days=((D*5)+4))   ,  '%m%d')

            dTdt[i,:]=(np.load((outdir+'dTdt_'+YEAR+STARTDATE+\
                                           '-'+YEAR+ENDDATE+'.npy')))
            dSdt[i,:]=(np.load((outdir+'dSdt_'+YEAR+STARTDATE+\
                                           '-'+YEAR+ENDDATE+'.npy')))
            i=i+1

    np.save('dTdt.npy',dTdt)
    np.save('dSdt.npy',dSdt)

elif load_series:
    print('loading ORCA2 residual flux time series')
    print('dTdt (K/s)')
    dTdt=np.load('dTdt.npy')
    print('dSdt (psu/s)')
    dSdt=np.load('dSdt.npy')


print('Removing seasonal cycle')
# note: np.tile only works along final axis, \
    #        so need to swap axes and swap back
print('dTdt...')
dTdt=(dTdt-np.tile(np.mean(dTdt.swapaxes(0,3).reshape(182,31,149,20,73),\
                           axis=3),20).swapaxes(0,3))
print('dSdt...')
dSdt=(dSdt-np.tile(np.mean(dSdt.swapaxes(0,3).reshape(182,31,149,20,73),\
                        axis=3),20).swapaxes(0,3))
    
################################################################################
####################            CREATE COVARIANCE MATRIX       #################

if calculate_covariance:
    T0=time.time()
    TT_cov=np.zeros((31,N,N,31,149,182))
    SS_cov=np.zeros((31,N,N,31,149,182))
    TS_cov=np.zeros((31,N,N,31,149,182))

    # Extend top and btm of the (j) grid by NN zero pts to accomodate n-hood:
    print('padding meridional boundaries')
    dTdt=np.pad(dTdt,((0,0),(0,0),(NN,NN),(0,0)),'constant')
    dSdt=np.pad(dSdt,((0,0),(0,0),(NN,NN),(0,0)),'constant')
    
    # Number of outputs (1460):
    lent=np.shape(dTdt)[0]

    # Calculate covariance:
    # Loop over outputs, & the rows (R) & columns (C) of the (e.g. 3x3) n-hood.
    # Build covariance matrix cumulatively over time using einstein summation.
    # The covariance throughout the local vertical (31x31) and within the local
    # neighbourhood (e.g. 3x3) is calculated.
    # "roll" (zonal) and "pad" (meridional) are used to calculate cov over
    # neighbouring verticals.

    for tt in np.arange(lent): 
        print(('time-step '+str(tt)+' of 3'))
        for C in np.arange(-NN,NN+1):
            for R in np.arange(NN,-NN-1,-1):
                TS_cov[:,C+NN,np.abs(R-NN),:,:,:]=
                TS_cov[:,C+NN,np.abs(R-NN),:,:,:]+\
                      (np.einsum('ijk,kjl->lijk',dTdt[tt,:,NN:-NN,:],\
                      (np.roll(dSdt[tt,:,NN+C:149+NN+C],R,axis=2)).T)/lent)
                TT_cov[:,C+NN,np.abs(R-NN),:,:,:]=\
                TT_cov[:,C+NN,np.abs(R-NN),:,:,:]+\
                      (np.einsum('ijk,kjl->lijk',dTdt[tt,:,NN:-NN,:],\
                      (np.roll(dTdt[tt,:,NN+C:149+NN+C],R,axis=2)).T)/lent)
                SS_cov[:,C+NN,np.abs(R-NN),:,:,:]=\
                SS_cov[:,C+NN,np.abs(R-NN),:,:,:]+\
                      (np.einsum('ijk,kjl->lijk',dSdt[tt,:,NN:-NN,:],\
                      (np.roll(dSdt[tt,:,NN+C:149+NN+C],R,axis=2)).T)/lent)

    # Save results as .npy file temporarily before final processing:
    print('saving results to .npy')
    print('Cov(T,S)')
    np.save(('TS_cov_3d.npy'),TS_cov)
    print('Cov(T,T)')
    np.save(('TT_cov_3d.npy'),TT_cov)
    print('Cov(S,S)')
    np.save(('SS_cov_3d.npy'),SS_cov)

    TS_cov[~np.isfinite(TS_cov)]=0
    TT_cov[~np.isfinite(TT_cov)]=0
    SS_cov[~np.isfinite(SS_cov)]=0

elif load_covariance:
    # Load an existing .npy file for processing
    print('loading raw covariance matrices')
    print('Cov(T,S)')
    TS_cov=np.load(('TS_cov_3d.npy'))
    TS_cov[~np.isfinite(TS_cov)]=0
    print('Cov(T,T)')
    TT_cov=np.load(('TT_cov_3d.npy'))
    TT_cov[~np.isfinite(TT_cov)]=0
    print('Cov(S,S)')
    SS_cov=np.load(('SS_cov_3d.npy'))
    SS_cov[~np.isfinite(SS_cov)]=0

################################################################################
##### CREATE SPARSE COVARIANCE MATRIX (SHAPE 840658x840658)

if build_sparse_matrix:
    # Step one, build arrays of indices:
    print('building sparse matrix')
    print('creating original index meshgrid')
    K2,J2,I2,K1,J1,I1=np.meshgrid(\
        np.arange(31),np.arange(-NN,NN +1),np.arange(-NN,NN +1),\
            np.arange(31),np.arange(149),np.arange(182),indexing='ij')
    I2=I2+I1;
    J2=J2+J1;
    for X in np.arange(-NN,1):
        I2[I2==X]=182+X
        J2[J2==X]=0
    for X in np.arange(0,NN):
        I2[I2==182+X]=X
        J2[J2==149+X]=0

    print('creating sparse matrix index vector')
    II=np.ravel_multi_index((K1,J1,I1),(31,149,182))
    JJ=np.ravel_multi_index((K2,J2,I2),(31,149,182))

    # Step two: populate matrix:
    print('Cov_TS')
    Cov_TS=sp.coo_matrix((TS_cov.flatten(),\
                             (II.flatten(),JJ.flatten())),\
                               shape=(31*149*182,31*149*182)).tocsr()
    print('Cov_SS')
    Cov_SS=sp.coo_matrix((SS_cov.flatten(),\
                             (II.flatten(),JJ.flatten())),\
                               shape=(31*149*182,31*149*182)).tocsr()
    print('Cov_TT')
    Cov_TT=sp.coo_matrix((TT_cov.flatten(),\
                             (II.flatten(),JJ.flatten())),\
                               shape=(31*149*182,31*149*182)).tocsr()

if save_sparse_matrix:
    print('Cov(T,S)...');sp.save_npz(('TS_cov'),Cov_TS)
    print('Cov(S,S)...');sp.save_npz(('SS_cov'),Cov_SS)
    print('Cov(T,T)...');sp.save_npz(('TT_cov'),Cov_TT)
################################################################################
print('done')
sys.exit()
