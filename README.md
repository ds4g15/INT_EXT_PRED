# INT_EXT_PRED
## Overview
This repository provides the necessary source code, run files and diagnostic scripts to recreate the experiments in our manuscript "The active and passive roles of the ocean in generating basin-scale heat content variability" Also required are model outputs from the IPSL-CM5a run used to create our stochastic representation.
There are five directories:

### 1_RUN_ADJOINT_MODEL
This contains two directories:
- `MY_SRC` which contains modifications to the default NEMO v3.4 source code (found [here](https://forge.ipsl.jussieu.fr/nemo/svn/NEMO/releases/release-3.4) ) while files necessary to run the model are found [here](https://doi.org/10.5281/zenodo.1471702) )
- `EXPERIMENTS` which contains scripts to run a 60 year trajectory, and template scripts (`namelist.TEMPLATE`,`job_JOBNAME.sh`) for running the adjoint. A selection of python scripts which produce the cost functions propagated in our experiments can be found in `ADJOINT_COST_FUNCTIONS`. `mkdirs.sh` makes a separate directory for each of our experiments and edits the template files accordingly. The script `job_JOBNAME.sh` creates symbolic links to all necessary files within its directory, then runs NEMOTAM three times: once with 75 timesteps (5d) and output every timestep, once with 5475 timesteps (1y) and output every 75 timesteps, once with 328500 timesteps (60y) and outputs every 5475 timesteps. This is so higher frequency output is available when error growth is fastest.

### 2_CALCULATE_OPTIMAL_STOCHASTIC_PATTERNS
This contains three python scripts:
- `1a_integrated_outer_product_ext.py` and `1b_optimal_correlated.py` are used for calculating the surface fully-correlated OSPs of Section 2b-2, calculating the integral and eigenvectors of Eq. (23), respectively, with the former producing sparse matrices (saved in .npz format) to be used by the latter, which produces a .pickle file.
- `2_optimal_decorrelated.py` calculates the full-depth everywhere uncorrelated OSPs of Section 2b-2 in one script, which produces a .pickle file.

### 3_GET_TURBULENT_MODEL_OUTPUTS
This includes two .txt files and a directory:
- `IPSL_OUTPUT.txt` contains instructions for obtaining the coupled model outputs used in our study
- `NEMOv3.5_SRC.txt` contains instructions for obtaining the eddying model source code used in our study
- `NEMO_ORCA025_RUN_SCRIPTS` contains namelist files and a shell script for reproducing our two-decade run
- `NEMO_ORCA025_RUN_SCRIPTS/MY_SRC` contains modifications to the NEMO v3.5 source code used to run our simulations

### 4_DIAGNOSE_REALISTIC_COVARIANCE_MATRICES
This contains nine python scripts which can be used with the coupled and eddying model outputs from `3_GET_TURBULENT_MODEL_OUTPUTS` to produce the temporal decorrelation (lambda) and spatial covariance (Sigma) matrices of Eq. (30), along with their root mean square logarithmic error (as shown in Fig. 2). Scripts (1-3) are included to project between the HRM and LRM grid, remove the climatology of the buoyancy fluxes, and calculate the final fluctuating term of Eq. (38). Scripts (4-9) allow the calculation and error evaluation of the stochastic representation.

### 5_CALCULATE_REALISTIC_RESPONSE
This contains a single script `1_calculate_response_variance.py` which takes the adjoint output of `1_RUN_ADJOINT_MODEL` and stochastic representations of `4_DIAGNOSE_REALISTIC_COVARIANCE_MATRICES` and calculates the error growth due to the different sources diagnosed from the higher fidelity models and projected onto the adjoint sensitivity fields.
