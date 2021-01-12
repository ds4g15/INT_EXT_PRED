from __future__ import division
import netCDF4 as nc
import scipy.sparse as sp
import sys
import numpy as np
import shapely.geometry as ge
import matplotlib.pyplot as plt
import time

################################################################################
''' Build a transformation matrix for projecting back and forth between 
ORCA025 and ORCA2.

METHOD:
Loops over every ORCA025 point to determine which ORCA2 grid cell it lies within
This cell is given a "weight" of 1.
This is done in three stages. First the horizontal ORCA2 weights are determined,
then the vertical ORCA2 weights are determined, then these are combined.

OPTIONS: 
Ch,Cv,Vh,Vv: horizontal and vertical centre and vertex point variables
The location on the Arakawa C grid must be provided. For instance, if the ORCA025
point is a "u" point, this corresponds in the vertical to a "t" point and in the
horizontal to a "u" point. The vertex points in this case are the "v" points.
A schematic is provided below the options.

OUTPUT:
Produces a .npz file containing a sparse matrix of shape (840658,110421150)
populated with either 1s and 0s to be used for projecting the ORCA025 variable
of interest onto the ORCA2 grid, or vice versa.

'''
################################################################################
# OPTIONS:
load_grid=True
Ch       ='t' #horizontal centre point  (u,v,t or f)
Cv       ='w' #vertical   centre point  (t or w)
Vh       ='f' #horizontal vertex points (u,v,t or f)
Vv       ='t' #vertical   vertex points (t or w)

#  HORIZONTAL LAYOUT | VERTICAL LAYOUT
#   f-v-f            |      w
#   | | |            |      |
#   u-T-u            |[u/v]-T-[u/v]
#   | | |            |      |
#   f-v-f            |      w

SAVE_SP  =False   #save sparse matrix
################################################################################
# Load data
if load_grid:
    mesh_025=nc.Dataset('mesh_mask_025.nc')
    mesh_2  =nc.Dataset('mesh_mask_2.nc')

    # Horizontal grid info (lat,lon at centres)
    clats025=mesh_025.variables[('gphi'+Ch)][:];
    clons025=mesh_025.variables[('glam'+Ch)][:];
    clats2  =mesh_2.variables[  ('gphi'+Ch)][:];
    clons2  =mesh_2.variables[  ('glam'+Ch)][:];
    cmask025=mesh_025.variables[(Ch+'mask')][:];
    cmask2  =mesh_2.variables[  (Ch+'mask')][:];
    
    # Horizontal grid info (lat, lon at vertices)
    vlats025=mesh_025.variables[('gphi'+Vh)][:];
    vlons025=mesh_025.variables[('glam'+Vh)][:];
    vlats2  =mesh_2.variables[  ('gphi'+Vh)][:];
    vlons2  =mesh_2.variables[  ('glam'+Vh)][:];

    # Vertical  grid info (depth at centres and vertices)
    gdepc_025  =mesh_025.variables[('gdep'+Ch)][:];
    gdepv_025  =mesh_025.variables[('gdep'+Vv)][:];
    gdepc_2    =mesh_2.variables[  ('gdep'+Ch)][:];
    gdepv_2    =mesh_2.variables[  ('gdep'+Vv)][:];
        
    cmask025nan=np.ones(np.shape(cmask025),dtype='float64');
cmask025nan[cmask025==0]=np.nan
################################################################################

# Location of (SW,NW,NE,SE) vertices relative to (i,j) for centre point:
if   Ch=='t':
    diSW,diNW,diNE,diSE  ,  djSW,djNW,djNE,djSE  =  -1,-1,+0,+0  ,  -1,+0,+0,-1
elif Ch=='u':
    diSW,diNW,diNE,diSE  ,  djSW,djNW,djNE,djSE  =  +0,+0,+1,+1  ,  -1,+0,+0,-1
elif Ch=='v':
    diSW,diNW,diNE,diSE  ,  djSW,djNW,djNE,djSE  =  -1,-1,+0,+0  ,  +0,+1,+1,+0

# Location of Top, Bottom vertices relative to k for center point
if   Cv=='w':
    dkT,dkB=-1,0
else: 
    dkT,dkB= 0,+1


################################################################################
################################## FUNCTIONS ###################################
################################################################################

def equirectangular(lon,lat,lon0,lat0):
    ''' equirectangular projection of (lon,lat) arrays to (x,y)
    '''
    y=(lat-lat0)
    x=(lon-lon0)*np.cos(np.radians(lat0))
    return x,y

def distance(origin, destination):
    """
    Calculate the Haversine distance.
    ----------
    origin : tuple of float (lat,long)
    destination : tuple of float (lat,long)
        (lat, long)
    Returns distance_in_km : float
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) * np.sin(dlat / 2) +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) * np.sin(dlon / 2))
    a[np.abs(1-a)<1e-6]=1 # bugfix error with numbers larger but close to 1
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = radius * c
    return d

################################################################################

def ORCA2_hweights(jj025,ii025):
    
    celat025=clats025[0,jj025,ii025];celon025=clons025[0,jj025,ii025]
    ############################################################################
    # FIRST STEP: find the nearest ORCA2 centre point (to the ORCA025 centre pt)
    #             and get the lat,lon of the corners of its 9 surrounding cells

    # Find nearest ORCA2 centre point to the ORCA025 cell centre
    ce2=np.unravel_index(np.argmin(distance((clats2  ,  clons2),\
                                            (celat025,celon025))),(1,149,182))
    # Create grid of 9 neighbouring ORCA2 centre points
    jj2,ii2=np.meshgrid(np.arange(ce2[1]-1,ce2[1]+2),\
                        np.arange(ce2[2]-1,ce2[2]+2))
    
    ############# DEAL WITH ORCA2 GRID EDGES AND LAND
    # Wrap around the east edge (lon[180]=lon[0])
    ii2=np.mod(ii2,180)
    # Delete cells which have centres or vertices off the meridional edge
    if np.any(jj2+djNE>148):
        ii2=np.delete(ii2,np.unique(np.where(jj2+djNE>148)[1]),1);                
        jj2=np.delete(jj2,np.unique(np.where(jj2+djNE>148)[1]),1);
    elif np.any(jj2+djSW<1):
        ii2=np.delete(ii2,np.unique(np.where(jj2+djSW<1)[1]),1)
        jj2=np.delete(jj2,np.unique(np.where(jj2+djSW<1)[1]),1)        

    if np.all(cmask2[0,:,jj2,ii2]==0):
         #all points are on land
        WGT=np.zeros(np.shape(ii2),dtype='float64')
        return WGT,jj2,ii2

    #############

    # Get the vertices of the 9 ORCA2 cells to create polygons
    # Positions of vertices relative to centre pts depend on variable (T,u,v)
    CElats2=clats2[0,jj2     ,       ii2           ]; #CEnter point
    CElons2=clons2[0,jj2     ,       ii2           ];
    SWlats2=vlats2[0,jj2+djSW,np.mod(ii2+diSW,180) ]; #SouthWest point
    SWlons2=vlons2[0,jj2+djSW,np.mod(ii2+diSW,180) ];
    NWlats2=vlats2[0,jj2+djNW,np.mod(ii2+diNW,180) ]; #NorthWest point
    NWlons2=vlons2[0,jj2+djNW,np.mod(ii2+diNW,180) ];
    NElats2=vlats2[0,jj2+djNE,np.mod(ii2+diNE,180) ]; #NorthEast point
    NElons2=vlons2[0,jj2+djNE,np.mod(ii2+diNE,180) ];
    SElats2=vlats2[0,jj2+djSE,np.mod(ii2+diSE,180) ]; #SouthEast point
    SElons2=vlons2[0,jj2+djSE,np.mod(ii2+diSE,180) ];    

    #########################################################################
    ############ SECOND STEP: Correct "illegal" crosses of lon=+/-180

    # This "if" block conducts checks to see if any vertices in the 9-cell 
    # ORCA2 region lie on a different side of prime/antimeridian to the ORCA025
    # cell centre point, and "moves" them to the same side:

    # First check: if any longitudes have opposite sign
    if np.any(np.sign(np.hstack((SWlons2.flatten(),NWlons2.flatten(),\
                                 NElons2.flatten(),SElons2.flatten())))\
                                                        !=np.sign(celon025)):
        # Go through each of the 9 cells to find and fix the offenders
        for k in np.arange(len( NElats2.flatten() )):       
            idx=np.unravel_index(k,np.shape(NElats2) )   
            #Array of vertices for this cell: 
            O2vlat=np.array([SWlats2[idx],NWlats2[idx],\
                             NElats2[idx],SElats2[idx]]) 
            O2vlon=np.array([SWlons2[idx],NWlons2[idx],\
                             NElons2[idx],SElons2[idx]])
            #Calculate distance from vertices to antimeridian:
            D1=distance((O2vlat,O2vlon),(O2vlat,180*np.ones(np.shape(O2vlon)))) 
            #Calculate distance from from vertices to prime meridian:
            D2=distance((O2vlat,O2vlon),(O2vlat,   np.zeros(np.shape(O2vlon)))) 
            if np.sum(D1) < np.sum(D2):
                if ( np.sign(celon025)<0 ): 
                    #ORCA2 cell crosses antimeridian, ORCA025 pt is in E. hemis.
                    # Check which ORCA2 vertices in W. Hem. and move to E.
                    if SWlons2[idx]>=0: SWlons2[idx]=SWlons2[idx]-360
                    if NWlons2[idx]>=0: NWlons2[idx]=NWlons2[idx]-360
                    if NElons2[idx]>=0: NElons2[idx]=NElons2[idx]-360
                    if SElons2[idx]>=0: SElons2[idx]=SElons2[idx]-360                    
                else:                       
                    #ORCA2 cell crosses antimeridian, ORCA025 pt is in W. hemis.
                    # Check which ORCA2 vertices in E. Hem. and move to W.
                    if SWlons2[idx]<0: SWlons2[idx]=SWlons2[idx]+360   
                    if NWlons2[idx]<0: NWlons2[idx]=NWlons2[idx]+360
                    if NElons2[idx]<0: NElons2[idx]=NElons2[idx]+360
                    if SElons2[idx]<0: SElons2[idx]=SElons2[idx]+360
                    
    ############################################################################
    # THIRD STEP: Project lat,lon of 9 ORCA2 cells around ORCA025 point to x,y
    #             using equirectangular projection centred on ORCA025 pt (0,0)
    #             then create polygons corresponding to these coordinates,
    #             before checking which polygon the ORCA025 point lies inside

    
    # Project ORCA2 center points
    cx2 , cy2=equirectangular(CElons2,CElats2,celon025,celat025)
    
    # Project ORCA2 corner points
    SWx2,SWy2=equirectangular(SWlons2,SWlats2,celon025,celat025)
    NWx2,NWy2=equirectangular(NWlons2,NWlats2,celon025,celat025)
    NEx2,NEy2=equirectangular(NElons2,NElats2,celon025,celat025)
    SEx2,SEy2=equirectangular(SElons2,SElats2,celon025,celat025)
    
    # Go through each of the 9 neighbouring cells and determine if ORCA025 point
    # is inside (weight=1) or not (weight=0):
    WGT=np.zeros(np.shape(SWx2))
    cx025,cy025=0,0 #Cartesian coordinates of orca025 point are just origin
    O025pt=ge.Point(cx025,cy025)        

    class doublebreak(Exception): pass #set custom exception to exit double loop
    try:
        for J in np.arange(np.shape(SWx2)[0]):
            for I in np.arange(np.shape(SWx2)[1]):
                O2poly=ge.Polygon([(SWx2[J,I],SWy2[J,I]),(NWx2[J,I],NWy2[J,I]),\
                                   (NEx2[J,I],NEy2[J,I]),(SEx2[J,I],SEy2[J,I])])
                if O2poly.intersects(O025pt):
                    WGT[J,I]=1
                    raise doublebreak
    except doublebreak:
        'do nothing'

                                                                     
    return WGT,jj2,ii2

################################################################################
def ORCA2_vweights(kk025,jj025,ii025,jj2,ii2):
    # Find nearest ORCA2 depth to provided ORCA025 depth:
    k2_nearest=np.argmin(np.abs(gdepc_2[  0,    :,jj2  ,ii2]-\
                                gdepc_025[0,kk025,jj025,ii025])) 
    # Make a 3-cell neighbourhood:
    kk2=np.arange(k2_nearest-1,k2_nearest+2) 
    ########################################
    ### DEAL WITH EDGE ISSUES:
    if Cv=='w': 
        #Set illegal indices to placeholder value:
        kk2[np.where( (kk2<1) | (kk2>30) )]=101 
        #Check if bottom of ORCA025 cell is above top of uppermost ORCA2 cell
        if gdepv_025[0,kk025,jj025,ii025]<gdepv_2[0,0,jj2,ii2]: 
            kk2=np.array([0,0,0]);WGT=np.array([1,1,1])
            return kk2,WGT        
    else:
        #Set illegal indices to placeholder value:
        kk2[np.where( (kk2<0) | (kk2>29 ))]=101 
        #Check if bottom of ORCA025 cell is below bottom of lowermost ORCA2 cell
        if gdepv_025[0,kk025,jj025,ii025]>gdepv_2[0,-1,jj2,ii2]: 
            kk2=np.array([30,30,30]);WGT=np.array([0,0,0])
            return kk2,WGT
    ########################################
    #Define a polygon for ORCA025 cell (with arbitrary x values)
    CEL025=ge.Polygon([    (0 , gdepv_025[0,kk025+dkT,jj025,ii025]), \
                           (1 , gdepv_025[0,kk025+dkT,jj025,ii025]), \
                           (1 , gdepv_025[0,kk025+dkB,jj025,ii025]), \
                           (0 , gdepv_025[0,kk025+dkB,jj025,ii025])])        

    # Loop neighbouring ORCA2 polygons to determine intrsctn w/ ORCA025 polygon
    WGT=np.zeros(np.shape(kk2))
    for i in np.arange(len(kk2)):
        if (kk2[i]==101): continue
        CELO2=ge.Polygon([    (0 , gdepv_2[0,kk2[i]+dkT,jj2,ii2]) ,\
                              (1 , gdepv_2[0,kk2[i]+dkT,jj2,ii2]) ,\
                              (1 , gdepv_2[0,kk2[i]+dkB,jj2,ii2]) ,\
                              (0 , gdepv_2[0,kk2[i]+dkB,jj2,ii2]) ])
        if CELO2.intersects(CEL025):
            WGT[i]=1
            break
        #

    kk2[kk2==101]=30 #Set all invalid k indices to k=30 (i.e. in the bed)
    return kk2,WGT

################################################################################

def ORCA2_3dweight(kk025,jj025,ii025,mode='centre'):

    #Check if point is on land, send to a known land point in ORCA2 if so:
    if cmask025[0,kk025,jj025,ii025]==0:
        WGT=np.array([0]);
        KK2=np.array([0]);
        JJ2=np.array([0]);
        II2=np.array([0]);
        return WGT,KK2,JJ2,II2

    ############################################################################
    #################### HORIZONTAL PART

    # If not on land, calculate the local ORCA2 cell in the horizontal:
    WGT_h,jj2,ii2=ORCA2_hweights(jj025,ii025)

    # If no local ORCA2 cell found, send to a known land point in ORCA2:
    if np.all(WGT_h==0):
        WGT=np.array([0]);
        KK2=np.array([0]);
        JJ2=np.array([0]);
        II2=np.array([0]);
        return WGT,KK2,JJ2,II2

    ############################################################################
    #################### VERTICAL PART
    WGT=np.repeat(WGT_h[None,:,:],3,0) #Repeat the <=3x3 matrix in the 3rd dim
    KK2=np.zeros(np.shape(WGT))        #index matrix for vertical
    
    # Loop over local horizontal neighbourhood to determine each vertical
    for n in np.arange(np.shape(WGT)[2]):
        for m in np.arange(np.shape(WGT)[1]):

            kk2,WGT_v=ORCA2_vweights(kk025,jj025,ii025,jj2[n,m],ii2[n,m]) 
            # Multiply so 0 in horizontal if 0 in vertical:
            WGT[0:len(kk2),n,m]=WGT[0:len(kk2),n,m]*WGT_v[:]
            KK2[0:len(kk2),n,m]=kk2[:]            

    ############################################################################
    #################### CLEAN OUTPUT AND RETURN

    # As ORCA2_vweights may return outputs (kk2,WGT_v) with length < 3
    # delete any un-needed rows in the WGT & KK2 matrices (from len(kk2) to 3):
    # as they are 3d repeats of a 2d array, they may have weights of 1 in
    # these rows:
    if len(kk2)!=np.shape(WGT)[0]:      
        WGT=np.delete(WGT,np.arange(len(kk2),np.shape(WGT)[0]),0) 
        KK2=np.delete(KK2,np.arange(len(kk2),np.shape(KK2)[0]),0)

    # Make all index arrays same shape, and force integer type:
    JJ2=np.repeat(jj2[None,:,:],len(kk2),0)
    II2=np.repeat(ii2[None,:,:],len(kk2),0)
    KK2=np.array(KK2,dtype='int');
    JJ2=np.array(JJ2,dtype='int');
    II2=np.array(II2,dtype='int')
    
    # Final land check with final weight array:
    if np.all(cmask2[0,KK2,JJ2,II2]*WGT==0):
        WGT=np.array([0]);
        KK2=np.array([0]);
        JJ2=np.array([0]);
        II2=np.array([0]);

    return WGT,KK2,JJ2,II2


################################################################################
#                                 # MAIN LOOP #                                #
################################################################################
T0=time.time()

#Make an empty matrix of ORCA025 location (rows) v. local weight array (cols)
O2weights=np.zeros((1021*1442*75,27)) 
O2K=np.copy(O2weights);
O2J=np.copy(O2weights);
O2I=np.copy(O2weights);

O025idx=np.zeros((1021*1442*75,27),dtype='int') 
O2idx  =np.zeros((1021*1442*75,27),dtype='int') 

for kk025 in np.arange(75):
    for jj025 in np.arange(1021):
        print('k='+str(kk025)+', j='+str(jj025)+' time: '+str(time.time()-T0)+'s')
        for ii025 in np.arange(1442):

            WGT,KK2,JJ2,II2=ORCA2_3dweight(kk025,jj025,ii025)

            #Check if this ORCA025 point actually corresponds to an ORCA2 pt:
            if np.any(WGT!=0):
                # Get the vector index corresponding to k,j,i:
                idx=np.ravel_multi_index((kk025,jj025,ii025),(75,1021,1442))
                
                # Fill the corresponding row with the local weight array:
                O2weights[idx,0:len(WGT.flatten())]=WGT.flatten()

                # Store the corresponding ORCA2 indices
                O2idx[    idx,0:len(WGT.flatten())]=\
                    np.ravel_multi_index((KK2,JJ2,II2),(31,149,182)).flatten()
                # Store the corresponding ORCA025 indices
                O025idx[  idx,0:len(WGT.flatten())]=\
                    np.ravel_multi_index((kk025,jj025,ii025),(75,1021,1442))
                # These indices and weights used to make sparse matrix below

################################################################################
#                              # SAVE TO NPZ                                   #
################################################################################
# Create a sparse matrix of size (75*1021*1442,31*182*149) where each entry is
# the local weight
# As our case only has weights of 1 or 0, every ORCA025 point corresponds to one
# ORCA2 cell only (or land)

# After the above loop, O2weights therefore has at most 1 non-zero entry per row
# for (75*1021*1442) rows and 27 columns. These are used in conjunction with 
# the O2idx and O025idx arrays, also (75*1021*1442,27), to construct a sparse
# matrix (31*149*182,75*1021*1442) where every value in O2idx is mapped to a row
# index and every value in O025idx is mapped to a column index, and the 
# corresponding entry value is taken from O2weights.

# By nature of this method, the same row and column will be mapped to 27 times
# coo_matrix handles this by adding the 27 entries (a one and 26 zeros at most)

print('creating sparse matrix')
M=sp.coo_matrix((O2weights.flatten(),(O2idx.flatten(),O025idx.flatten())),\
                                         shape=(31*149*182,75*1021*1442))
print('eliminating zeros')
M.eliminate_zeros()
print('converting to CSR and saving')
M=M.tocsr()
if Cv=='w':
    sp.save_npz(('ORCA025_to_ORCA2_centre_'+Cv),M)
else:
    sp.save_npz(('ORCA025_to_ORCA2_centre_'+Ch),M)
