# Survival-LCS

## Instructions for running Survival-LCS



<!-- sim_full_pipeline_LCS.py

There's two main files that will need a few tweaks- updating file names and modifying any run parameters, as follows:

 - sim_full_pipeline_LCS.py
	Line 22 - update system path
	Line 59 - update home directory for output files 
	Line 60 - specify which models to include
	Line 63 - specify censoring proportions
	Line 64 - specify the number of features
	Line 65 - specify the minor allele frequencies (used by gametes)
	Line 73 - set simulated = True for simulated runs
	Line 75 - set lcs_run = True to run the survival-ExSTraCS algorithm in addition to the data simulation
	Line 134 - optional, edit number of iterations, cv folds. Can also set to default.

 - survival_LCS_pipeline.py
	Line 60 - modify any default parameters as needed (optional)


You'll need to create the following folders and subfolders, INSIDE of the home directory for output files:
cv_sim_data (with subfolders: cv_me, cv_epi, cv_het, cv_add)
pickled_cv_models (with subfolders: me, epi, het, add)
sim_lcs_output (with subfolders: me, epi, het, add)


Simulated data from Gametes is in the "simulated_datasets" folder. The script "survival_data_siumulator.py" calls "importGametes.py" to parse the gametes model files and generate survival data.  -->

