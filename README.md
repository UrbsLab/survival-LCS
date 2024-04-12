# Survival-LCS: A Rule-Based Machine Learning Approach to Survival Analysis

Survival analysis is a critical aspect of modeling time-to-event data
in fields such as epidemiology, engineering, and econometrics. Traditional 
survival methods rely heavily on assumptions and are
limited in their application to real-world datasets. To overcome
these challenges, we introduce the survival learning classifier system 
(survival-LCS) as a more flexible approach. The survival-LCS
extends the capabilities of ExSTraCS, a rule-based machine learning 
algorithm optimized for biomedical applications, to handle
survival (time-to-event) data. In addition to accounting for right
censored observations, survival-LCS handles multiple feature types
and missing data, and makes no assumptions about baseline hazard
or survival distributions.

As proof of concept, we evaluated the survival-LCS on simulated
genetic survival datasets of increasing complexity derived from the
GAMETES software. The four genetic models included univariate,
epistatic, additive, and heterogeneous models, simulated across a
range of censoring proportions, minor allele frequencies, and number 
of features. The results of this sensitivity analysis demonstrated
the ability of survival-LCS to identify complex patterns of association 
in survival data. Using the integrated Brier score as the key
performance metric, survival-LCS demonstrated reliable survival
time and distribution predictions, potentially useful for clinical
applications such as informing self-controls in clinical trials


## Description of Experiment Files

1. sim_full_pipeline_HPCNotebook1.ipynb -  File to run and save all experimnets and survival-LCS model files on an HPC Cluster.
2. sim_full_pipeline_HPCNotebook1_coxchecks.ipynb - File to run and save all 100 Feature Dataset Cox experimnets on an HPC Cluster.
3. sim_full_pipeline_HPCNotebook1_permutations.ipynb - File to run and save all 100 Feature Dataset Permutation experimnets on an HPC Cluster.
4. sim_full_pipeline_HPCNotebook1_permutations_more_parellelized.ipynb - File to run and save all 100 Feature Dataset Permutation experimnets on an HPC Cluster in a more parellelized manner.
5. sim_full_pipeline_HPCNotebook1_permutations_more_parellelized_errors.ipynb - File to run and save all 100 Feature Dataset Permutation experimnets that might've failed/thrown an error.
6. ManualPermutationTesting.ipynb - File to generate all results/p-values of all 100 Feature Dataset models for permutation testing.
7. ManualWilcoxonTesting.ipynb - File to generate all results/p-values for Wilcoxon significance for all models/setting.
8. Table_3_to_6_GECCO.ipynb -  Files to generate Tables 3-6 on the Manuscript.
9. Table_3_to_6_Cox_GECCO.ipynb -  Files to generate IBS Values for Cox Runs for Bolding Tables 3-6 on the Manuscript.
10. Table_8_GECCO.ipynb - File to generate Table 8 of the Manuscript.
11. Figure_4_GECCO.py - File to generate Figure 4 of Manuscript.
12. Figure_6_GECCO.ipynb - File to generate Figure 6 of Manuscript.
13. Figure_7_GECCO.ipynb - File to generate Figure 7 of Manuscript.
14. Figure_8_GECCO.ipynb - File to generate Figure 8 of Manuscript.

Files with '_copy' are just copies of files to run parallel analysis, all other files are base-code and runner files to run survival-LCS and the experiments. Ouptut folder has all the corresponding outputs for reference and to complete the manuscript.

## Running Survival-LCS Experiments
### Setup/Requirements
#### Python Pacakges

The analysis needs the following python pacakges to run:

```numpy==1.21.2
pandas 
matplotlib 
scikit-learn==1.22.2
skrebate 
fastcluster 
seaborn 
networkx 
pygame 
pytest-shutil 
eli5 
scikit-survival==0.21.0
ipython
notebook
dask
dask-jobqueue
```

These can be install using the following command:

```pip install -r requirements.txt```

#### HPC Requirements
The analysis takes extesnive computational requirements and while it can be run on a local machine it is recommend to run it on an HPC Cluster such as a SLURM based cluster. Our scripts are set up to run it on the SLURM HPC Cluster at Cedars-Sinai.

#### Simulated Dataset Models
Simulated Dataset model files are need to run the experiment and should be put in `pipeline/simulated_datasets` folder. These files are available upon request.

### Instructions to Run
If you're able to ssh into your HPC and Open VS-Code-Server on your local VS-Code, the process is simple.

1. Clone the respostiory on you HPC instance.
2. Install the requirements to set up the environment.
3. Open sim_full_pipeline_HPCNotebook1.ipynb and run all cells to run the base survival-LCS experiments.
4. Open sim_full_pipeline_HPCNotebook1_coxchecks.ipynb and run all cells to run the base CPH Model experiments.
5. Open sim_full_pipeline_HPCNotebook1.ipynb and run all cells to run the base Permutation Testing experiments.
6. Run respective Jupyter Notebooks to generate output tables and figures.

## Citation

ACM Reference Format:
Alexa Woodward, Harsh Bandhey, Jason H. Moore, and Ryan J. Urbanowicz.
2024. Survival-LCS: A Rule-Based Machine Learning Approach to Survival
Analysis. In Genetic and Evolutionary Computation Conference (GECCO ’24),
July 14–18, 2024, Melbourne, VIC, Australia. ACM, New York, NY, USA.


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

