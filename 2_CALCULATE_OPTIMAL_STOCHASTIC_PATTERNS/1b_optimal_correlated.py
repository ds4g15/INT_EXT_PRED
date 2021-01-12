import sys
import pickle
import numpy as np
import netCDF4 as nc
import scipy.sparse as sp
import scipy.sparse.linalg as la

################################################################################
''' 
Script to calculate the SURFACE Optimal Stochastic Perturbation (OSP)
covariance matrix based on an eigendecomposition of the integrated outer product
produced by 1a_integrated_outer_product_ext.py.

Based on Equation (22) of our manuscript
(current draft, 2021-01-10)

Uses nesting to combine integrals of short run/high output frequency with 
longer runs/lower output frequency. See comment in header of job_JOBNAME.sh

ARGUMENTS:
-------
cfstr: (argument 1) the adjoint run name 
       (e.g. MVT_SP_yr for year averaged subpolar MVT)
'''

################################################################################

cfstr=sys.argv[1] # CF: {MVT,MHT,OHC}_S{T,P}_{mn,yr,dc}
av=cfstr[-2:];#av='yr'
cf=cfstr[0:6];#cf='MVT_ST'

load_IOPs=1
load_grid=1
calc_eigs=1
calc_OSPs=1
save_OSPs=1
eps=1 # setting to 1 means final OSPs are unscaled
################################################################################
print('Creating average variance norms')
if load_grid:
    O2=nc.Dataset('mesh_mask.nc')
    alpha0=2.1e-4;beta0=7.5e-4;rho0=1026;T0=15;S0=35; # EOS reference values
    # Grid volume:
    dVT=O2.variables['e3t'][0,0,:,:]*\
        O2.variables['e1t'][0,  :,:]*\
        O2.variables['e2t'][0,  :,:]
    dVU=O2.variables['e3u'][0,0,:,:]*\
        O2.variables['e1u'][0,  :,:]*\
        O2.variables['e2u'][0,  :,:]
    dVV=O2.variables['e3v'][0,0,:,:]*\
        O2.variables['e1v'][0,  :,:]*\
        O2.variables['e2v'][0,  :,:]

    dt=24*(3600)
#Norm on cov. matrix: global average flux variance in 1d
    S_T=dt*(dVT)/dVT.sum(); S_inv_T_sp=sp.diags((1./S_T).reshape(149*182,),0)
    S_S=dt*(dVT)/dVT.sum(); S_inv_S_sp=sp.diags((1./S_S).reshape(149*182,),0)
    S_U=dt*(dVU)/dVU.sum(); S_inv_U_sp=sp.diags((1./S_U).reshape(149*182,),0)
    S_V=dt*(dVV)/dVV.sum(); S_inv_V_sp=sp.diags((1./S_V).reshape(149*182,),0)

################################################################################
print('Calculating optimal stochastic noise (external, abs(corr)=1), '+cf+', '+\
          av+' avg.')

#################### 5-day:
if load_IOPs:
    print('loading 5-day output');
    print('T');IOP_T_75=sp.load_npz('IOPs/'+cf+'_'+av+'_IOP_T_75.npz')
    print('S');IOP_S_75=sp.load_npz('IOPs/'+cf+'_'+av+'_IOP_S_75.npz')
    print('U');IOP_U_75=sp.load_npz('IOPs/'+cf+'_'+av+'_IOP_U_75.npz')
    print('V');IOP_V_75=sp.load_npz('IOPs/'+cf+'_'+av+'_IOP_V_75.npz')

if calc_eigs:
    print('calculating OSP for 5 day growth');
    print('T');lam_T_75,L_T_75=la.eigsh(S_inv_T_sp.dot(IOP_T_75),which='LM',k=1)
    print('S');lam_S_75,L_S_75=la.eigsh(S_inv_S_sp.dot(IOP_S_75),which='LM',k=1)
    print('U');lam_U_75,L_U_75=la.eigsh(S_inv_U_sp.dot(IOP_U_75),which='LM',k=1)
    print('V');lam_V_75,L_V_75=la.eigsh(S_inv_V_sp.dot(IOP_V_75),which='LM',k=1)

if calc_OSPs:
    L0_T_75=eps/(np.sqrt(np.sum(S_T*(L_T_75.reshape(149,182)**2)))) \
        * L_T_75.reshape(149,182)
    L0_S_75=eps/(np.sqrt(np.sum(S_S*(L_S_75.reshape(149,182)**2)))) \
        * L_S_75.reshape(149,182)
    L0_U_75=eps/(np.sqrt(np.sum(S_U*(L_U_75.reshape(149,182)**2)))) \
        * L_U_75.reshape(149,182)
    L0_V_75=eps/(np.sqrt(np.sum(S_V*(L_V_75.reshape(149,182)**2)))) \
        * L_V_75.reshape(149,182)

#################### 1yr
if load_IOPs:
    print('loading 1 year output');
    # Nesting: replaces first five days with integral from above   ↓  
    print('T');
    IOP_T_5475=sp.load_npz('IOPs/'+cf+'_'+av+'_IOP_T_5475.npz')+IOP_T_75
    print('S');
    IOP_S_5475=sp.load_npz('IOPs/'+cf+'_'+av+'_IOP_S_5475.npz')+IOP_S_75
    print('U');
    IOP_U_5475=sp.load_npz('IOPs/'+cf+'_'+av+'_IOP_U_5475.npz')+IOP_U_75
    print('V');
    IOP_V_5475=sp.load_npz('IOPs/'+cf+'_'+av+'_IOP_V_5475.npz')+IOP_V_75

if calc_eigs:
    print('calculating OSP for 1 yr growth');
    print('T');
    lam_T_5475,L_T_5475=la.eigsh(S_inv_T_sp.dot(IOP_T_5475),which='LM',k=1)
    print('S');
    lam_S_5475,L_S_5475=la.eigsh(S_inv_S_sp.dot(IOP_S_5475),which='LM',k=1)
    print('U');
    lam_U_5475,L_U_5475=la.eigsh(S_inv_U_sp.dot(IOP_U_5475),which='LM',k=1)
    print('V');
    lam_V_5475,L_V_5475=la.eigsh(S_inv_V_sp.dot(IOP_V_5475),which='LM',k=1)

if calc_OSPs:
    L0_T_5475=eps/(np.sqrt(np.sum(S_T*(L_T_5475.reshape(149,182)**2)))) \
              * L_T_5475.reshape(149,182)
    L0_S_5475=eps/(np.sqrt(np.sum(S_S*(L_S_5475.reshape(149,182)**2)))) \
              * L_S_5475.reshape(149,182)
    L0_U_5475=eps/(np.sqrt(np.sum(S_U*(L_U_5475.reshape(149,182)**2)))) \
              * L_U_5475.reshape(149,182)
    L0_V_5475=eps/(np.sqrt(np.sum(S_V*(L_V_5475.reshape(149,182)**2)))) \
              * L_V_5475.reshape(149,182)

################################################################################
#################### 60 yr
if load_IOPs:
    print('loading 60 year output');
    # Nesting: replaces first year with integral from above             ↓
    print('T');
    IOP_T_328500=sp.load_npz('IOPs/'+cf+'_'+av+'_IOP_T_328500.npz')+IOP_T_5475
    print('S');
    IOP_S_328500=sp.load_npz('IOPs/'+cf+'_'+av+'_IOP_S_328500.npz')+IOP_S_5475
    print('U');
    IOP_U_328500=sp.load_npz('IOPs/'+cf+'_'+av+'_IOP_U_328500.npz')+IOP_U_5475
    print('V');
    IOP_V_328500=sp.load_npz('IOPs/'+cf+'_'+av+'_IOP_V_328500.npz')+IOP_V_5475

if calc_eigs:
    print('calculating OSP for 1 yr growth');
    print('T');
    lam_T_328500,L_T_328500=la.eigsh(S_inv_T_sp.dot(IOP_T_328500),which='LM',k=1)
    print('S');
    lam_S_328500,L_S_328500=la.eigsh(S_inv_S_sp.dot(IOP_S_328500),which='LM',k=1)
    print('U');
    lam_U_328500,L_U_328500=la.eigsh(S_inv_U_sp.dot(IOP_U_328500),which='LM',k=1)
    print('V');
    lam_V_328500,L_V_328500=la.eigsh(S_inv_V_sp.dot(IOP_V_328500),which='LM',k=1)

if calc_OSPs:
    L0_T_328500=eps/(np.sqrt(np.sum(S_T*(L_T_328500.reshape(149,182)**2)))) \
              * L_T_328500.reshape(149,182)
    L0_S_328500=eps/(np.sqrt(np.sum(S_S*(L_S_328500.reshape(149,182)**2)))) \
              * L_S_328500.reshape(149,182)
    L0_U_328500=eps/(np.sqrt(np.sum(S_U*(L_U_328500.reshape(149,182)**2)))) \
              * L_U_328500.reshape(149,182)
    L0_V_328500=eps/(np.sqrt(np.sum(S_V*(L_V_328500.reshape(149,182)**2)))) \
              * L_V_328500.reshape(149,182)

################################################################################

if save_OSPs:
    print('creating output dictionary')
    out_dict={};
    out_dict['eps']=eps
    out_dict[ 'eigenvalues_TSUV_328500']=\
        np.array([lam_T_328500,lam_S_328500,lam_U_328500,lam_V_328500])
    out_dict['eigenvectors_TSUV_328500']=\
        np.hstack([L_T_328500,L_S_328500,L_U_328500,L_V_328500])
    out_dict['OSP_unscaled_TSUV_328500']=\
        np.stack([L0_T_328500,L0_S_328500,L0_U_328500,L0_V_328500],axis=0)

    
    print('saving...')
    pickle_filename=('OSPs/'+cf+'_'+av+'_optimal_noise_correlated.pickle')
    with open(pickle_filename,'wb') as handle:
        pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


