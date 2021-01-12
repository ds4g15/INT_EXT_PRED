#! /bin/bash
# Run the nonlinear ORCA2 trajectory before running the adjoint offline
np=32
runname=TRAJ60

################################################################################
#### MAKE EXPERIMENT DIRECTORY:
ORCA2INPUTDIR=/path/to/ORCA2INPUT
NEMOGCMDIR=/path/to/NEMOGCM

for FILE  in {ahmcoef,bathy_level.nc,bathy_meter.nc,bathy_updated.nc,chlorophyll.nc,coordinates.nc,data_1m_potential_temperature_nomask.nc,data_1m_salinity_nomask.nc,geothermal_heating.nc,K1rowdrg.nc,M2rowdrg.nc,mask_itf.nc,ncar_precip.15JUNE2009_orca2.nc,ncar_rad.15JUNE2009_orca2.nc,q_10.15JUNE2009_orca2.nc,runoff_core_monthly.nc,t_10.15JUNE2009_orca2.nc,u_10.15JUNE2009_orca2.nc,v_10.15JUNE2009_orca2.nc};
do 
    ln -s ${ORCA2INPUTDIR}/${FILE} .; 
done
ln -s ${NEMOGCMDIR}/CONFIG/TAM_PRED/BLD/bin/nemo.exe .
################################################################################
ln -sf namelist.${runname} namelist
ln -sf namelist_ice.${runname} namelist_ice

mkdir TRAJ60
mkdir TRAJ60/tmpdir

# your command to run nemo.exe

mv ocean.output ocean.output_TRAJ60

for filename in `cat date.file | cut -d ' ' -f 2`grid_?_0000.nc `cat date.file | cut -d ' ' -f 2`icemod_0000.nc; do \
    ${NEMOGCMDIR}/TOOLS/REBUILD_NEMO/rebuild_nemo ${filename/_0000.nc} ${np}; \
    if [ -e ${filename/_0000.nc}.nc ]; 
    then rm ${filename/_0000.nc/}_????.nc;
    fi;
done

