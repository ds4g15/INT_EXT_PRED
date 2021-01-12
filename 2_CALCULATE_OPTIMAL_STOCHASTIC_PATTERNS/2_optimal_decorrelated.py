import sys
import pickle
import numpy as np
import netCDF4 as nc
import scipy.sparse as sp
import scipy.sparse.linalg as la
################################################################################
'''
Script to calculate the FULL-DEPTH Optimal Stochastic Perturbation (OSP) covariance
matrix based on an eigendecomposition of the integrated outer product (IOP),
which is calculated within the script due to the simpler (diagonal) case of
full-depth. 

Based on Equation (24) of our manuscript
(current draft, 2021-01-10)

ARGUMENTS:
-------
cfstr: (argument 1) the adjoint run name 
       (e.g. MVT_SP_yr for year averaged subpolar MVT)
'''
################################################################################

cfstr=sys.argv[1] # CF: {MVT,MHT,OHC}_S{T,P}_{mn,yr,dc}
av=cfstr[-2:];#e.g. av='yr'
cf=cfstr[0:6];#e.g. cf='MVT_ST'
load_data=1
load_grid=1
calc_OSPs=1
save_OSPs=1
eps=1 # setting to 1 means final OSPs are unscaled
################################################################################

print('Creating average variance norms')
if load_grid:
    O2=nc.Dataset('ORCA2_mesh.nc')
    alpha0=2.1e-4;beta0=7.5e-4;rho0=1026;T0=15;S0=35; # EOS reference values
    dVT=O2.variables['e3t'][0,:,:,:]*\
        O2.variables['e1t'][0,  :,:]*\
        O2.variables['e2t'][0,  :,:]
    dVU=O2.variables['e3u'][0,:,:,:]*\
        O2.variables['e1u'][0,  :,:]*\
        O2.variables['e2u'][0,  :,:]
    dVV=O2.variables['e3v'][0,:,:,:]*\
        O2.variables['e1v'][0,  :,:]*\
        O2.variables['e2v'][0,  :,:]

    DT=24*(3600) #Time scale over which flux acts (1d)
#Norm on cov. matrix: global average flux variance in 1d
    S_inv_T=dVT.sum()/(DT*dVT)
    S_inv_S=dVT.sum()/(DT*dVT)
    S_inv_U=dVU.sum()/(DT*dVU)
    S_inv_V=dVV.sum()/(DT*dVV)

################################################################################
print('Calculating optimal stochastic noise (external, abs(corr)=1), '+\
       cf+', '+av+' avg.')
# Note: "lam" is the integrated sum of the squared output (hence the sum of 
# eigenvalues of int[0][t]M|F><F|M dT)

if load_data:
    # load data from 75-step, 5475-step and 328500-step adjoint run for nested
    # integral:
    print('loading 5-day output')
    ncf=(av+'/'+cf+'_75_output.nc')
    NC=nc.Dataset(ncf)
    dt=24*3600./15 # output frequency for integrated outer product
    print('T');D75T=np.sum(dt*S_inv_T*NC.variables['t_ad'][1:,]**2,axis=0) 
    lam_T_75=D75T.sum(); 
    M_T_75=(eps**2/lam_T_75)*(D75T*S_inv_T) #OSP

    print('S');D75S=np.sum(dt*S_inv_S*NC.variables['s_ad'][1:,]**2,axis=0) 
    lam_S_75=D75S.sum(); #AF
    M_S_75=(eps**2/lam_S_75)*(D75S*S_inv_S) #OSP

    print('U');D75U=np.sum(dt*S_inv_U*NC.variables['u_ad'][1:,]**2,axis=0) 
    lam_U_75=D75U.sum() #AF
    M_U_75=(eps**2/lam_U_75)*(D75U*S_inv_U) #OSP

    print('V');D75V=np.sum(dt*S_inv_V*NC.variables['v_ad'][1:,]**2,axis=0) 
    lam_V_75=D75V.sum() #AF
    M_V_75=(eps**2/lam_V_75)*(D75V*S_inv_V) #OSP
    ############################################################################
    print('loading 365-day output')
    ncf=(av+'/'+cf+'_5475_output.nc')
    NC=nc.Dataset(ncf)
    dt=24*3600*5 # output frequency for integration
    print('T');
    D5475T=D75T+np.sum(dt*S_inv_T*NC.variables['t_ad'][1:,]**2,axis=0) #Diag
    lam_T_5475=D5475T.sum(); #Amplification factor
    M_T_5475=(eps**2/lam_T_5475)*(D5475T*S_inv_T) #OSP
    print('S');
    D5475S=D75S+np.sum(dt*S_inv_S*NC.variables['s_ad'][1:,]**2,axis=0) #Diag
    lam_S_5475=D5475S.sum() #AF
    M_S_5475=(eps**2/lam_S_5475)*(D5475S*S_inv_S) #OSP
    print('U');
    D5475U=D75U+np.sum(dt*S_inv_U*NC.variables['u_ad'][1:,]**2,axis=0) #Diag
    lam_U_5475=D5475U.sum() #AF
    M_U_5475=(eps**2/lam_U_5475)*(D5475U*S_inv_U) #OSP
    print('V');
    D5475V=D75V+np.sum(dt*S_inv_V*NC.variables['v_ad'][1:,]**2,axis=0) #Diag
    lam_V_5475=D5475V.sum() #AF
    M_V_5475=(eps**2/lam_V_5475)*(D5475V*S_inv_V) #OSP

    ############################################################################
    print('loading 60y output');
    ncf=(av+'/'+cf+'_328500_output.nc')
    NC=nc.Dataset(ncf)
    dt=24*3600*365 # output frequency for integration
    print('T');
    D328500T=D5475T+np.sum(dt*S_inv_T*NC.variables['t_ad'][1:,]**2,axis=0) #Diag
    lam_T_328500=D328500T.sum(); #Amplification factor
    M_T_328500=(eps**2/lam_T_328500)*(D328500T*S_inv_T) #OSP
    print('S');
    D328500S=D5475S+np.sum(dt*S_inv_S*NC.variables['s_ad'][1:,]**2,axis=0) #Diag
    lam_S_328500=D328500S.sum() #AF
    M_S_328500=(eps**2/lam_S_328500)*(D328500S*S_inv_S) #OSP
    print('U');
    D328500U=D5475U+np.sum(dt*S_inv_U*NC.variables['u_ad'][1:,]**2,axis=0) #Diag
    lam_U_328500=D328500U.sum() #AF
    M_U_328500=(eps**2/lam_U_328500)*(D328500U*S_inv_U) #OSP
    print('V');
    D328500V=D5475V+np.sum(dt*S_inv_V*NC.variables['v_ad'][1:,]**2,axis=0) #Diag
    lam_V_328500=D328500V.sum() #AF
    M_V_328500=(eps**2/lam_V_328500)*(D328500V*S_inv_V) #OSP

    
if save_OSPs:

    print('creating output dictionary')
    out_dict={};
    out_dict['eps']=eps
    out_dict['OSP_unscaled_TSUV_328500']=\
        np.stack([M_T_328500,M_S_328500,M_U_328500,M_V_328500],axis=0)
    out_dict['eigenvalues_TSUV_328500']=\
        np.array([lam_T_328500,lam_S_328500,lam_U_328500,lam_V_328500])

if True:
    print('saving...')
    pickle_filename=('OSPs/'+cf+'_'+av+'_optimal_noise_decorrelated.pickle')
    with open(pickle_filename,'wb') as handle:
        pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



