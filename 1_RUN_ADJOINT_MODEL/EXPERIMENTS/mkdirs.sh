# create directories in which to run the adjoint simulations
mkdir year_average
mkdir month_average
mkdir decade_average
WD=`pwd`

for AVG in {year_average,month_average,decade_average};do
    cd "${AVG}"
    for CF in {MVT_ST,MVT_SP,MHT_SP,MHT_ST,OHC_ST,OHC_SP};do
	cp -rp "${WD}"/job_JOBNAME.sh ${CF}/
	cp -rp "${WD}"/namelist.TEMPLATE ${CF}/
	cp "${WD}"/ADJOINT_COST_FUNCTIONS/COST_FN_${CF}.py ${CF}/
	cd "${CF}"
	rename JOBNAME "${CF}" *.*
	sed -i -e "s#JOBNAME#${CF}#g" *
	cd "${WD}/${AVG}"
    done
    cd "${WD}"
done
    
