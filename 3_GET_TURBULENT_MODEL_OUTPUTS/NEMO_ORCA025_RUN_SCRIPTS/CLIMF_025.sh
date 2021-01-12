#! /bin/bash
################################################################################
# Run NEMO-ORCA025 with climatological forcing for 1 (of 20) years.

# This script should be modified in line with the local system
# e.g. loading modules, SLURM commands, etc.
# Set $NEMOGCMDIR to the location of your NEMOGCM directory

# Script runs for 1 year (18000 timesteps with our config), then 
# updates namelist for the subsequent year.
# It should be made to submit itself again at the end.
################################################################################

runname=CLIMF_025
run_length=18000 #18000 #1 year @ 100 p/d 

ln -sf namelist.${runname} namelist
ln -sf namelist_ice.${runname} namelist_ice

nn_it000=`cat namelist | grep "first time step" | cut -d ' ' -f 12`
nn_itend=`cat namelist | grep "last  time step" | cut -d ' ' -f 12`
echo "First time step: $((10#${nn_it000}))"
echo "Final time step: $((10#${nn_itend}))"

if [ ! -e ${runname}_$((10#${nn_it000} - 10#1))_restart.nc ]; 
then 
    echo "Restart file not found. Not continuing.";
    exit 1
else echo "restart found, submitting run"
 fi

if [ ! -d ${runname}_output ]; then mkdir ${runname}_output; fi

# your command to execute ./nemo.exe


#Rebuild restart:
${NEMOGCMDIR}/TOOLS/REBUILD_NEMO/rebuild_nemo \
    "${runname}"_"${nn_itend}"_restart ${np}

if [ -e "${runname}"_"${nn_itend}"_restart.nc ];then
    rm "${runname}"_"${nn_itend}"_restart_????.nc;
fi
#Rebuild restart_ice:
${NEMOGCMDIR}/TOOLS/REBUILD_NEMO/rebuild_nemo \
    "${runname}"_"${nn_itend}"_restart_ice ${np}
if [ -e "${runname}"_"${nn_itend}"_restart_ice.nc ];then
    rm "${runname}"_"${nn_itend}"_restart_ice_????.nc;
fi

#Rebuild output:
for X in "${runname}"_5d_grid?_????????-????????_0000.nc; do
    ${NEMOGCMDIR}/TOOLS/REBUILD_NEMO/rebuild_nemo "${X/_0000.nc}" ${np}
    if [ -e ${X/_0000.nc}.nc ];then
	rm ${X/_0000.nc}_????.nc
    fi
done

# Get rid of unwanted output
rm ${runname}_*icemod_????.nc
rm ${runname}_*flxT_????.nc

#Update namelist
sed -i -e "s@nn_it000      =  ${nn_it000}@nn_it000      =  $((10#${nn_it000}+10#${run_length}))@g" namelist."${runname}"

sed -i -e "s@nn_itend      =  ${nn_itend}@nn_itend      =  $((10#${nn_itend}+10#${run_length}))@g" namelist."${runname}"

sed -i -e "s@${runname}_$((10#${nn_it000}-1))_restart@${runname}_${nn_itend}_restart@g" namelist."${runname}"

sed -i -e "s@${runname}_$((10#${nn_it000}-1))_restart_ice@${runname}_${nn_itend}_restart_ice@g" namelist_ice."${runname}"

mv ocean.output ocean.output_${runname}_${nn_it000}
mv ${runname}.out ${runname}_${nn_it000}.out

