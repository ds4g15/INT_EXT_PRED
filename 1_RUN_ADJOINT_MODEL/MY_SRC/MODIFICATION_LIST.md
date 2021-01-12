
Each feature or model change has its own ID. In the source code, a modification associated with this change begins with
`!!!<ID>` and ends with `!!! /<ID>`. This allows modifications to be found, tweaked and removed without damaging other functionality. 
All changes and their purpose are listed here and were either added by S. Mueller or myself. Asterisks indicate modifications which are essential to run NEMOTAM.

- 20191004B : single adjoint output variables (e.g. tn+tb) and specification of TAM output variables
- 20191004C : Add "sshn" to nonlinear trajectory
- 20191004D: Expanding trajectory and TAM filenames to allow at least 8-digit time-steps
- 20191004E*: correct transition of b->n for kdir==1 [see here](http://forge.ipsl.jussieu.fr/nemo/attachment/ticket/1443/trj_tam.F90.diff)
- 20191004F: corrected 'stpr2 - zstp -1' to 'stpr2 - zstp +1' in trj_tam.F90
- 20191004G: adjust output of interpolation coefficients 
- 20191004H: switch to allow adjoint output writing 
- 20191004I: essential modifications to dynzdf_imp_tam.F90 [see here](http://forge.ipsl.jussieu.fr/nemo/attachment/ticket/1362/dynzdf_imp_tam.F90.diff)
- 20191004J*: addition of adjoint time-stepping loop 
- 20191004K*: proper initialisation of TAM variables 
- 201910004L: ability to read cost function/perturbation from netCDF file
- 20191004N*: building {t,u,v,f}msk_ variables dom_ 
- 20191004O*: force flux SBC (http://forge.ipsl.jussieu.fr/nemo/attachment/ticket/1738/sbcmod_tam.F90.diff)[20160524]
- 20191004R: trajectory offsetting
- 20191004S: Introduction of weighted-mean avection scheme 
- 20191004T: Introduction of trajectory-upstream scheme and manual weightings
- 20191004W: Option to include EIV
- 20191123A: Enable uniform eddy viscosity (no equatorial refinement)
- 20191119A: Shut down ability of buoyancy anomalies to feed back on density (passive ocean)
- 20191129A: Output ice_fraction with trajectory and adjoint fields