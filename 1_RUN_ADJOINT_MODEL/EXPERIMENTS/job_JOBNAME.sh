#!/bin/bash
PREFIX=JOBNAME
np=32 #No. of processors 
echo ${runname}
################################################################################
# This is a skeleton script to run the adjoint experiments, and should be
# modified to match the local system (e.g. SLURM commands, module loading etc.)

# Three adjoint runs are executed, one of 75 timestep length with output every
# timestep, one of 5475 timestep (1y) length with output every 75 timesteps,
# one of 328500 timestep (60y) length with output every year. The three are
# "nested" so that there is higher output frequency at earlier stages.

################################################################################
#### MAKE EXPERIMENT DIRECTORY:
ORCA2INPUTDIR=/path/to/ORCA2INPUT
NEMOGCMDIR=/path/to/NEMOGCM

for FILE  in {ahmcoef,bathy_level.nc,bathy_meter.nc,bathy_updated.nc,\
              chlorophyll.nc,coordinates.nc,\
              data_1m_potential_temperature_nomask.nc,\
              data_1m_salinity_nomask.nc,geothermal_heating.nc,K1rowdrg.nc,\
              M2rowdrg.nc,mask_itf.nc,ncar_precip.15JUNE2009_orca2.nc,\
              ncar_rad.15JUNE2009_orca2.nc,q_10.15JUNE2009_orca2.nc,\
              runoff_core_monthly.nc,t_10.15JUNE2009_orca2.nc,\
              u_10.15JUNE2009_orca2.nc,v_10.15JUNE2009_orca2.nc};
do 
    ln -s ${ORCA2INPUTDIR}/${FILE} .; 
done
ln -s ${NEMOGCMDIR}/CONFIG/TAM_PRED/BLD/bin/nemo_tam.exe .

################################################################################
# MAKE COST FUNCTION
echo "creating adjoint cost function"
python2.7 -u COST_FN_JOBNAME.py #Create cost function for adjoint

################################################################################
# FIRST RUN (75 steps)
## PARAMETERS
runname=${PREFIX}_75
outfreq=1   #27375=1p5y, 5475=1pa, 15=365pa, 1=15pd,
runlen=75
endtime=328500
cfwindow=5475

DT=$((10#${outfreq}*5760))
echo "1st run: length= ${runlen}, output freq. (timesteps)= ${outfreq}, dt=${DT}"
## EDIT NAMELIST
cp namelist.TEMPLATE namelist.${runname}
starttime=$((10#${endtime}-10#${runlen}+1))
cftime=$((10#${endtime}-10#${cfwindow}))
offset=$((10#${endtime}-10#${runlen}))
sed -i -e "s#RUNNAME#${runname}#g" namelist.${runname} #< Run name
sed -i -e "s#STARTTIME#${starttime}#g" namelist.${runname} #< nn_it000
sed -i -e "s#OFFSET#${offset}#g" namelist.${runname} #< nn_ittrjoffset
sed -i -e "s#OUTFREQ#${outfreq}#g" namelist.${runname} #< nn_ittrjfrq_tan
sed -i -e "s#CFTIME#${cftime}#g" namelist.${runname} # < nn_itcf000

## RUN NEMO
rm namelist
ln -sf namelist.${runname} namelist
ln -sf JOBNAME.nc ${runname}_init.nc
if [ ! -d ${runname}_output ]; then mkdir ${runname}_output; fi
if [ ! -d ../outputs ]; then mkdir ../outputs; fi

# your command to run ./nemo_tam.exe

## stitch outputs after run
for filename in ${runname}_output/ad_output_????????_0000.nc; do \ 
    ${NEMOGCMDIR}/TOOLS/REBUILD_NEMO/rebuild_nemo ${filename/_0000.nc} ${np}; \
	#check stitch has been successful before blindly deleting everything
    if [ -e ${filename/_0000.nc}.nc ]; 
    then rm ${filename/_0000.nc/}_????.nc;
    fi;
done;
ncrcat -d t,0,,1 ${runname}_output/ad_output_????????.nc ${runname}_output/${runname}_output.nc

################################################################################
# SECOND RUN (5475 steps)
## PARAMETERS
runname=${PREFIX}_5475
outfreq=75   #27375=1p5y, 5475=1pa, 15=365pa, 1=15pd,
runlen=5475

endtime=328500
cfwindow=5475

DT=$((10#${outfreq}*5760))
echo "2nd run: length= ${runlen}, output freq. (timesteps)= ${outfreq}, dt=${DT}"
## EDIT NAMELIST
cp namelist.TEMPLATE namelist.${runname}
starttime=$((10#${endtime}-10#${runlen}+1))
cftime=$((10#${endtime}-10#${cfwindow}))
offset=$((10#${endtime}-10#${runlen}))
sed -i -e "s#RUNNAME#${runname}#g" namelist.${runname} #< Run name
sed -i -e "s#STARTTIME#${starttime}#g" namelist.${runname} #< nn_it000
sed -i -e "s#OFFSET#${offset}#g" namelist.${runname} #< nn_ittrjoffset
sed -i -e "s#OUTFREQ#${outfreq}#g" namelist.${runname} #< nn_ittrjfrq_tan
sed -i -e "s#CFTIME#${cftime}#g" namelist.${runname} # < nn_itcf000

## RUN NEMO
rm namelist
ln -sf namelist.${runname} namelist
ln -sf JOBNAME.nc ${runname}_init.nc
if [ ! -d ${runname}_output ]; then mkdir ${runname}_output; fi
if [ ! -d ../outputs ]; then mkdir ../outputs; fi

# your command to run ./nemo_tam.exe

## stitch outputs after run
for filename in ${runname}_output/ad_output_????????_0000.nc; do \ 
    ${NEMOGCMDIR}/TOOLS/REBUILD_NEMO/rebuild_nemo ${filename/_0000.nc} ${np}; \
	#check stitch has been successful before blindly deleting everything
    if [ -e ${filename/_0000.nc}.nc ]; 
    then rm ${filename/_0000.nc/}_????.nc;
    fi;
done;
ncrcat -d t,0,,1 ${runname}_output/ad_output_????????.nc ${runname}_output/${runname}_output.nc

################################################################################
# THIRD RUN (328500 steps)
## PARAMETERS
runname=${PREFIX}_328500
outfreq=5475   #27375=1p5y, 5475=1pa, 15=365pa, 1=15pd,
runlen=328500

endtime=328500
cfwindow=5475

DT=$((10#${outfreq}*5760))
echo "3rd run: length= ${runlen}, output freq. (timesteps)= ${outfreq}, dt=${DT}"
## EDIT NAMELIST
cp namelist.TEMPLATE namelist.${runname}
starttime=$((10#${endtime}-10#${runlen}+1))
cftime=$((10#${endtime}-10#${cfwindow}))
offset=$((10#${endtime}-10#${runlen}))

sed -i -e "s#RUNNAME#${runname}#g" namelist.${runname} #< Run name
sed -i -e "s#STARTTIME#${starttime}#g" namelist.${runname} #< nn_it000
sed -i -e "s#OFFSET#${offset}#g" namelist.${runname} #< nn_ittrjoffset
sed -i -e "s#OUTFREQ#${outfreq}#g" namelist.${runname} #< nn_ittrjfrq_tan
sed -i -e "s#CFTIME#${cftime}#g" namelist.${runname} # < nn_itcf000

## RUN NEMO
rm namelist
ln -sf namelist.${runname} namelist
ln -sf JOBNAME.nc ${runname}_init.nc
if [ ! -d ${runname}_output ]; then mkdir ${runname}_output; fi
if [ ! -d ../outputs ]; then mkdir ../outputs; fi

# [your command to run NEMOTAM goes here e.g. ./nemo_tam.exe]

## stitch outputs after run
for filename in ${runname}_output/ad_output_????????_0000.nc; do \ 
    ${NEMOGCMDIR}/TOOLS/REBUILD_NEMO/rebuild_nemo ${filename/_0000.nc} ${np}; \
    #check stitch has been successful before blindly deleting everything
    if [ -e ${filename/_0000.nc}.nc ]; 
    then rm ${filename/_0000.nc/}_????.nc;
    fi;
done;
/home/acc/shared/bin/ncrcat -d t,0,,1 ${runname}_output/ad_output_????????.nc ${runname}_output/${runname}_output.nc
