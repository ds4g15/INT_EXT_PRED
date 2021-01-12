import sys
import pickle
import numpy as np
import netCDF4 as nc
import scipy.sparse as sp
import scipy.sparse.linalg as la
################################################################################
'''
Script to calculate the integrated outer product (IOP) int_{t0}^{t1} M|F><F|M dt
of the surface layer based on adjoint output. 

Used by script
   1b_optimal_correlated.py
to calculate OSPs.

Based on the integral term of Equation (22) of our manuscript
(current draft, 2021-01-10)

ARGUMENTS:
-------
cfstr: (argument 1) the adjoint run name 
       (e.g. MVT_SP_yr for year averaged subpolar MVT)
ln   : (argument 2) the length of the adjoint run (either 75,5475 or 328500)
'''

################################################################################
# OPTIONS:
calc_IOP =True #Calculate Integrated Outer Product from scratch
save_npz =True # Saves the result of the most costly section ( ~10GB )

cfstr=sys.argv[1] # CF: {MVT,MHT,OHC}_S{T,P}_{mn,yr,dc}
ln=int(sys.argv[2]) # run length of adjoint (75, 5475 or 328500 time steps;
                     # see comment in job_JOBNAME.sh header)
av=cfstr[-2:]
cf=cfstr[0:6]

if   ln==75    :
    dt=24*3600./15;nout=75
elif ln==5475  :
    dt=24*3600*5  ;nout=73    
elif ln==328500:
    dt=24*3600*365;nout=60

################################################################################
print('Averaging period: '+av)
print('Cost function/location: '+cf)
################################################################################
# NESTED INTEGRAL: dt = 1 step -> 1 day -> 1 year

# Initialise empty sparse matrices (nijxnij), nij=no. of surface points 
H_T=sp.csr_matrix((27118,27118))
H_S=sp.csr_matrix((27118,27118))
H_U=sp.csr_matrix((27118,27118))
H_V=sp.csr_matrix((27118,27118))

#### 
if (calc_IOP):    
    ncf=(av+'/'+cf+'_'+str(ln)+'_output.nc') #adjoint output file
    NC=nc.Dataset(ncf)

    for t in np.arange(1,nout):
        # load adjoint outputs to sparse vector at each ouput snapshot
        t_ad=sp.csr_matrix( NC.variables['t_ad'][t,0,:,:].reshape(149*182,)  )
        s_ad=sp.csr_matrix( NC.variables['s_ad'][t,0,:,:].reshape(149*182,)  )
        u_ad=sp.csr_matrix( NC.variables['u_ad'][t,0,:,:].reshape(149*182,)  )
        v_ad=sp.csr_matrix( NC.variables['v_ad'][t,0,:,:].reshape(149*182,)  )
        
        # calculate outer product of sparse vector and add to total*dt
        H_T = H_T + ( (t_ad.T).dot(t_ad) * dt)
        H_S = H_S + ( (s_ad.T).dot(s_ad) * dt)
        H_U = H_U + ( (u_ad.T).dot(u_ad) * dt)
        H_V = H_V + ( (v_ad.T).dot(v_ad) * dt)

################################################################################
if save_npz:
    print('Saving integrated outer product, T')
    sp.save_npz(cf+'_'+av+'_IOP_T_'+str(ln)+'.npz',H_T) 
    print('Saving integrated outer product, S')
    sp.save_npz(cf+'_'+av+'_IOP_S_'+str(ln)+'.npz',H_S)
    print('Saving integrated outer product, U')
    sp.save_npz(cf+'_'+av+'_IOP_U_'+str(ln)+'.npz',H_U) 
    print('Saving integrated outer product, V')
    sp.save_npz(cf+'_'+av+'_IOP_V_'+str(ln)+'.npz',H_V) 
################################################################################
