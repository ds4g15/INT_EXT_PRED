import sys
import time
import pickle
import numpy as np
import scipy.sparse as sp
import scipy.interpolate as ip

adj_dir=sys.argv[1] # Where adjoint output is found
cost_fn=sys.argv[2] # e.g. 'MVT_ST'
                    #       if you have 'your_dir/OHC_SP_328500_output.nc'
                    #       set adj_dir='your_dir'; cost_fn='OHC_SP'

### OPTIONS:
load_npz=1 #Load sparse spatial covariance matrices of stochastic fluxes
load_adj=1 #Load adjoint sesitivity fields
calc_var=1 #Calculate response variance
save_ncf=1 #Save response as netCDF files

''' 
Calculate the variance in response to prescribed stochastic fluxes following
an Ornstein-Uhlenbeck process, isolating source locations and variables.

DESCRIPTION:
------------
This script [following Equations (28) and (29) of our manuscript (draft
2020-10-27) calculates the variance in response to prescribed surface and
subgrid fluxes following an Ornstein-Uhlenbeck process.

It loads the covariance matrices and decorrelation time arrays of the fluxes
(Sigma and lambda, respectively in our equations) and estimates the variance
arising from the projection of these matrices onto adjoint sensitivity fields.

The script is based on two adjoint runs for each metric: one lasting one year
(5475 time steps) with sensitivity fields output every 5 days (75 time steps)
and one lasting sixty years (328500 time steps)with output every year. This is
to capture the faster growth which occurs during the first year adequately.

The variance accumulated during the first year from the first run is used with
the variance accumulated during the remaining 59 years from the second run.
The output is further interpolated so as to avoid crude numerical integration.

There are two functions to calculate error accumulation: one simply interpolates
the output and supplies the interpolated version to the other function. This
function then calculates the double integral numerically.

OUTPUT:
-------
The final result is a netCDF file

[COST_FN]_variance_sources.nc

containing the following variables:
var_SST_SST (variance due to surface heat flux covariance 
var_SSS_SSS (variance due to surface freshwater flux covariance 
var_SST_SSS (variance due to surface heat flux and FW flux cross-covariance)

and so on for zonal (SSU) and meridional (SSV) momentum fluxes, and internal
subgrid heat (T) and salt (S) fluxes.

The surface variables have shape (61,149,182) and the internal variables have
shape (61,31,149,182) where 61 is the number of years (including 0 at time 0)
'''


################################################################################
def variance_sources_interp(var1_ad,var2_ad,cov12,tau1,tau2,dt,\
                            nk=31,nperstep=1,n_overlap=0):

    '''
    Interpolate adjoint sensitivity fields to approximate higher frequency 
    adjoint output for less coarse numerical integration (effectively trapezium
    instead of rectangular).

    Feed the "higher frequency output" to variance_sources() to approximate the
    variance accumulated over the time of interest. Note that the final result
    is in the same shape as the input array (not finer temporal resolution)
    despite the interpolation, as the accumulated error is a sum over the 
    interpolated time frame.
    
    ARGUMENTS: 
    ----------
    var1_ad,var2_ad,cov12,tau1,tau2,dt,nk:
                            << See help for variance_sources() >>
    nperstep: number of emulated adjoint outputs per actual adjoint output.
              If adjoint output is every 5 days and daily frequency is desired
              for integration, set nperstep=5
    n_overlap: amount of overlap (in number of outputs) to ensure that temporal
               correlation of the O-U process is not impacted by interpolation

    RETURNS:
    --------
    var_map         : an array of response variance accumulated over each time
                      with the same shape as var1_ad
    
    '''
    nijk=nk*149*182
    nt  =np.shape(var1_ad)[0];
    t   =np.arange(0,nt*Dt,Dt);

    # Create interpolation function of adjoint outputs:
    f_var1_ad   =ip.interp1d(t,var1_ad,axis=0)
    if var1_ad is var2_ad: f_var2_ad=f_var1_ad
    else: f_var2_ad=ip.interp1d(t,var2_ad,axis=0)

    # Go through each step and split it into smaller (interpolated) pieces:
    int_map=np.zeros((nt,nk,149,182))
    for stp in np.arange(nt-1):

        # Interpolated time array:
        ti=np.linspace(stp*Dt,(stp+1)*Dt,nperstep) 
        dti=np.diff(ti)[0]; 

        # Add an overlap to account for correlation with times outside array:
        ti=np.linspace((stp   *Dt)-(dti*n_overlap*np.sign(stp)),\
                       (stp+1)*Dt,\
                       nperstep +(    n_overlap*np.sign(stp)))
        #("np.sign" used so no overlap for step 0)
        dti=np.diff(ti)[0];

        # Interpolate the adjoint output onto this time array:
        var1i_ad   =f_var1_ad(ti)
        if var1_ad is var2_ad: var2i_ad=var1i_ad
        else: var2i_ad=f_var2_ad(ti)

        # Calculate the variance contribution at each interpolated time-step
        VI=internal_map(var1i_ad,var2i_ad,cov12,tau1,tau2,dti,nk=nk)

        # Then extract just non-overlapping part: 
        var_map[stp+1,:]=VI[-nperstep:,:].reshape(-1,nk,149,182).sum(axis=0)
    return var_map

################################################################################
def variance_sources(var1_ad,var2_ad,cov_12,tau1,tau2,dt,nk=31):
    '''
    Calculate covariance between adjoint sensitivity variables (var1_ad,var2_ad)
    in response to ornstein-uhlenbeck process with spatial covariance Sigma
    (cov_12) and temporal decorrelation times (tau1,tau2).

    ARGUMENTS:
    ----------
    var1_ad,var2_ad : adjoint sensitivity fields of metric to var1 and var2
    cov_12          : spatial covariance matrix of stochastic noise
    tau1,tau2       : temporal decorrelation time array of var1,var2
    dt              : time between sensitivity field outputs (for integration)
    nk              : number of vertical levels under consideration (1 or 31)

    RETURNS:
    --------
    var_map         : an array of response variance accumulated over each time
                      with the same shape as var1_ad
    '''

    if np.shape(cov_12) != (182*149*nk , 182*149*nk): 
        raise ValueError('Incorrect covariance matrix shape')

    #### INITIALISE VARIABLES:
    nijk=nk*149*182                       # no. of gridpoints
    nt  =np.shape(var1_ad)[0];            # no. of outputs
    tau=0.5*(tau1+tau2).reshape(1,nijk)   # Decorrelation times    [1,nijk]
    var1_ad=np.reshape(var1_ad,(nt,nijk)) # adjoint output        [nt,nijk]
    var2_ad=np.reshape(var2_ad,(nt,nijk)) # adjoint output        [nt,nijk]
    tt=np.flip(np.arange(nt)*dt,0)        # time array            [nt,    ]
    int_var=np.zeros((nt,nt,nijk));       # double integrand   [nt,nt,nijk]


    ### CONSTRUCT INTEGRAND OF DOUBLE INTEGRAL:
    # Step through outputs:
    for i in np.arange(nt):
        # O-U temporal correlation with other outputs:
        COU=np.exp( -np.abs(tt[i] - tt).reshape(nt,1)/tau ) 
        # From Eq. (28,29) [in draft, 2020-10-27]:
        int_var[i,:,:]= (COU*var1_ad[None,:,:])*(cov_12.dot(var2_ad[i,:].T))

    ### CALCULATE DOUBLE INTEGRAL:
    # As a single line:
    #var_map=np.diagonal(np.nancumsum(np.nancumsum(\
    #                    int_var*dt*dt,axis=0),axis=1),\
    #                                          axis1=0,axis2=1)

    # As a (less memory intensive) loop:
    var_map=np.zeros((nt,nijk))
    var_map[0,:]=int_var[0,0,:]*dt*dt
    for i in np.arange(1,nt):
        var_map[i,:]=\
                      (int_var[  i, 0:i, :].sum(axis=0)*dt*dt)+\
                      (int_var[0:i,   i, :].sum(axis=0)*dt*dt)+\
                      (int_var[  i,   i, :]            *dt*dt)
    var_map=var_map.reshape(nt,nk,149,182)
    return var_map


################################################################################
### LOAD FLUX COVARIANCE MATRICES AND DECORRELATION TIME ARRAYS:

if load_npz:
    # EXTERNAL
    print('Loading surface flux covariance matrices')
    extdir='covariance_matrices/'
    COV_SST_SST=sp.load_npz((extdir+'COV_SST_SST.npz')) # Surface heat flux
    COV_SSS_SSS=sp.load_npz((extdir+'COV_SSS_SSS.npz')) # Surface FW flux
    COV_SST_SSS=sp.load_npz((extdir+'COV_SST_SSS.npz')) # covariance
    COV_SSU_SSU=sp.load_npz((extdir+'COV_SSU_SSU.npz')) # surface zonal wind
    COV_SSV_SSV=sp.load_npz((extdir+'COV_SSV_SSV.npz')) # surface merid. wind
    COV_SSU_SSV=sp.load_npz((extdir+'COV_SSU_SSV.npz')) # covariance
    print('loading surface flux decorrelation times')
    # Add 1e-32 to avoid division by zero:
    tau_SST=np.load(extdir+'/e_folding_time_SST.npy')*365*86400 + 1e-32 
    tau_SSS=np.load(extdir+'/e_folding_time_SSS.npy')*365*86400 + 1e-32
    tau_SSU=np.load(extdir+'/e_folding_time_SSU.npy')*365*86400 + 1e-32
    tau_SSV=np.load(extdir+'/e_folding_time_SSV.npy')*365*86400 + 1e-32
    
    # INTERNAL
    print('Loading subgrid buoyancy flux covariance matrices')
    intdir='covariance_matrices/'
    COV_T_T=sp.load_npz((intdir+'TT_cov.npz')) #Subgrid heat flux
    COV_S_S=sp.load_npz((intdir+'SS_cov.npz')) #Subgrid salt flux
    COV_T_S=sp.load_npz((intdir+'TS_cov.npz')) #covariance
    print('Loading subgrid buoyancy flux decorrelation times')
    tau_T=np.load(intdir+'/e_folding_time_T.npy')*365*86400 + 1e-32
    tau_S=np.load(intdir+'/e_folding_time_S.npy')*365*86400 + 1e-32


if load_adj:
    NC01y=nc.Dataset(adj_dir+'/'cost_fn+'_5475_output.nc')   #1  year run
    NC60y=nc.Dataset(adj_dir+'/'cost_fn+'_328500_output.nc') #60 year run
    
    #Load adjoint sensitivity fields, 1y run with 5d output:
    t_ad01=np.flip(NC01y.variables['t_ad'][:],0)
    s_ad01=np.flip(NC01y.variables['s_ad'][:],0)
    u_ad01=np.flip(NC01y.variables['u_ad'][:],0)
    v_ad01=np.flip(NC01y.variables['v_ad'][:],0)
    fr_i01=np.flip(NC01y.variables['ice_fraction'][:],0)
    IM01=(1-fr_i01) #ice mask

    #Load adjoint sensitivity fields,60y run with 1y output:
    t_ad60=np.flip(NC60y.variables['t_ad'][:],0)
    s_ad60=np.flip(NC60y.variables['s_ad'][:],0)
    u_ad60=np.flip(NC60y.variables['u_ad'][:],0)
    v_ad60=np.flip(NC60y.variables['v_ad'][:],0)
    fr_i60=np.flip(NC60y.variables['ice_fraction'][:],0)
    IM60=(1-fr_i60) # ice mask

################################################################################
if calc_var:
    # ONE-YEAR RUN, 5d OUTPUT:
    # External, interpolated to daily output:
    var_SST_SST_01y=variance_sources_interp(\
                                        t_ad01*IM01,t_ad01*IM01,COV_SST_SST,\
                                        tau_SST,tau_SST,dt=75,nk=1,nperstep=5)
    var_SSS_SSS_01y=variance_sources_interp(\
                                        s_ad01*IM01,s_ad01*IM01,COV_SSS_SSS,\
                                        tau_SSS,tau_SSS,dt=75,nk=1,nperstep=5)
    var_SST_SSS_01y=variance_sources_interp(\
                                        t_ad01*IM01,s_ad01*IM01,COVV_SST_SSS,\
                                        tau_SST,tau_SST,dt=75,nk=1,nperstep=5)
    var_SSU_SSU_01y=variance_sources_interp(\
                                        u_ad01*IM01,u_ad01*IM01,COVV_SSU_SSU,\
                                        tau_SSU,tau_SSU,dt=75,nk=1,nperstep=5)
    var_SSV_SSV_01y=variance_sources_interp(\
                                        v_ad01*IM01,v_ad01*IM01,COVV_SSV_SSV,\
                                        tau_SSV,tau_SSV,dt=75,nk=1,nperstep=5)
    var_SSU_SSV_01y=variance_sources_interp(\
                                        u_ad01*IM01,v_ad01*IM01,COVV_SSU_SSV,\
                                        tau_SSU,tau_SSV,dt=75,nk=1,nperstep=5)
    # Internal, no interpolation:
    var_T_T_01y    =variance_sources(\
                                 t_ad01,t_ad01,COV_T_T,tau_T,tau_T,dt=75,nk=31)
    var_S_S_01y    =variance_sources(\
                                 s_ad01,s_ad01,COV_S_S,tau_S,tau_S,dt=75,nk=31)
    var_T_S_01y    =variance_sources(\
                                 t_ad01,s_ad01,COV_T_S,tau_T,tau_T,dt=75,nk=31)
    ############################################################################
    # SIXTY-YEAR RUN, YEARLY OUTPUT:
    # External, interpolated to daily output::
    var_SST_SST_60y=variance_sources_interp(\
                                        t_ad60*IM60,t_ad60*IM60,COV_SST_SST,\
                                        tau_SST,tau_SST,dt=5475,nk=1,nperstep=5)
    var_SSS_SSS_60y=variance_sources_interp(\
                                        s_ad60*IM60,s_ad60*IM60,COV_SSS_SSS,\
                                        tau_SSS,tau_SSS,dt=5475,nk=1,nperstep=5)
    var_SST_SSS_60y=variance_sources_interp(\
                                        t_ad60*IM60,s_ad60*IM60,COVV_SST_SSS,\
                                        tau_SST,tau_SST,dt=5475,nk=1,nperstep=5)
    var_SSU_SSU_60y=variance_sources_interp(\
                                        u_ad60*IM60,u_ad60*IM60,COVV_SSU_SSU,\
                                        tau_SSU,tau_SSU,dt=5475,nk=1,nperstep=5)
    var_SSV_SSV_60y=variance_sources_interp(\
                                        v_ad60*IM60,v_ad60*IM60,COVV_SSV_SSV,\
                                        tau_SSV,tau_SSV,dt=5475,nk=1,nperstep=5)
    var_SSU_SSV_60y=variance_sources_interp(\
                                        u_ad60*IM60,v_ad60*IM60,COVV_SSU_SSV,\
                                        tau_SSU,tau_SSV,dt=5475,nk=1,nperstep=5)
    # Internal, interpolated to 5d output:
    var_T_T_60y    =variance_sources(\
                                 t_ad60,t_ad60,COV_T_T,tau_T,tau_T,dt=5475,\
                                 nperstep=73,n_overlap=17,nk=31)
    var_S_S_60y    =variance_sources(\
                                 s_ad60,s_ad60,COV_S_S,tau_S,tau_S,dt=5475,\
                                 nperstep=73,n_overlap=17,nk=31)
    var_T_S_60y    =variance_sources(\
                                 t_ad60,s_ad60,COV_T_S,tau_T,tau_T,dt=5475,\
                                 nperstep=73,n_overlap=17,nk=31)

    ##### COMBINE NESTED CALCULATIONS:
    # Initialise combined array:
    var_SST_SST=np.zeros((61,149,182))
    # Take first year from 1 year adjoint run:
    var_SST_SST[1 ,:,:]=var_SST_SST_01y.sum(axis=0)
    # Take remainder from 60y adjoint run:
    var_SST_SST[2:,:,:]=var_SST_SST_60y[2:,:]

    var_SSS_SSS=np.zeros((61,149,182))
    var_SSS_SSS[1 ,:,:]=var_SSS_SSS_01y.sum(axis=0)
    var_SSS_SSS[2:,:,:]=var_SSS_SSS_60y[2:,:]

    var_SST_SSS=np.zeros((61,149,182))
    var_SST_SSS[1 ,:,:]=var_SST_SSS_01y.sum(axis=0)
    var_SST_SSS[2:,:,:]=var_SST_SSS_60y[2:,:]

    var_SSU_SSU=np.zeros((61,149,182))
    var_SSU_SSU[1 ,:,:]=var_SSU_SSS_01y.sum(axis=0)
    var_SSU_SSU[2:,:,:]=var_SSU_SSS_60y[2:,:]

    var_SSV_SSV=np.zeros((61,149,182))
    var_SSV_SSV[1 ,:,:]=var_SSV_SSV_01y.sum(axis=0)
    var_SSV_SSV[2:,:,:]=var_SSV_SSV_60y[2:,:]

    var_SSU_SSV=np.zeros((61,149,182))
    var_SSU_SSV[1 ,:,:]=var_SSU_SSV_01y.sum(axis=0)
    var_SSU_SSV[2:,:,:]=var_SSU_SSV_60y[2:,:]

    var_T_T=np.zeros((61,31,149,182))
    var_T_T[1 ,:,:]=var_T_T_01y.sum(axis=0)
    var_T_T[2:,:,:]=var_T_T_60y[2:,:]

    var_S_S=np.zeros((61,31,149,182))
    var_S_S[1 ,:,:]=var_S_S_01y.sum(axis=0)
    var_S_S[2:,:,:]=var_S_S_60y[2:,:]

    var_T_S=np.zeros((61,31,149,182))
    var_T_S[1 ,:,:]=var_T_S_01y.sum(axis=0)
    var_T_S[2:,:,:]=var_T_S_60y[2:,:]

################################################################################
if save_ncf:
    print('creating output file')
    NCO=nc.Dataset((adj_dir+'/'+cost_fn+'_variance_sources.nc'),'w')
    NCO.createDimension('t',None)
    NCO.createDimension('z',31)
    NCO.createDimension('y',149)
    NCO.createDimension('x',182)
    
    print('external, buoyancy')
    ETT=NCO.createVariable('var_SST_SST',np.float64(),('t','y','x'));
    ETS=NCO.createVariable('var_SST_SSS',np.float64(),('t','y','x'));
    ESS=NCO.createVariable('var_SSS_SSS',np.float64(),('t','y','x'));
    ETT[:]=var_SST_SST
    ETS[:]=var_SST_SSS
    ESS[:]=var_SSS_SSS

    print('external, mechanical')
    EUU=NCO.createVariable('var_SSU_SSU',np.float64(),('t','y','x'));
    EUV=NCO.createVariable('var_SSU_SSV',np.float64(),('t','y','x'));
    EVV=NCO.createVariable('var_SSV_SSV',np.float64(),('t','y','x'));
    EUU[:]=var_SSU_SSU
    EUV[:]=var_SSU_SSV
    EVV[:]=var_SSV_SSV

    print('internal, buoyancy')
    ITT=NCO.createVariable('var_T_T',np.float64(),('t','z','y','x'));
    ITS=NCO.createVariable('var_T_S',np.float64(),('t','z','y','x'));
    ISS=NCO.createVariable('var_S_S',np.float64(),('t','z','y','x'));
    ITT[:]=var_T_T
    ITS[:]=var_T_S
    ISS[:]=var_S_S
