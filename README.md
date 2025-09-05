# Survival-LCS: Rule-Based Survival Analysis Without Proportional Hazard Assumptions

This repository contains the code, scripts, and analysis files supporting the publication:

**Alexa Woodward, Harsh Bandhey, Jason H. Moore, and Ryan J. Urbanowicz.  
1.    Survival-LCS: Rule-Based Survival Analysis Without Proportional Hazard Assumptions.  
ACM Transactions on Evolutionary Learning and Optimization, August 2025.**

Survival-LCS extends ExSTraCS to model survival (time-to-event) data without relying on proportional hazard assumptions. This work builds on earlier evaluations and introduces broader hyperparameter resources, additional survival distributions (Random, Gamma, Gaussian, Weibull), and comprehensive benchmarking across simulated datasets to assess robustness and interpretability.

## Abstract

Survival analysis is crucial in modeling time-to-event data across biomedical research, epidemiology, and engineering. Traditional methods often rely on restrictive assumptions and face challenges in handling the complexities of real-world datasets. To address these limitations, we introduce the Survival Learning Classifier System (Survival-LCS), an extension of the ExSTraCS algorithm specifically designed for survival analysis. Survival-LCS supports right-censored data, diverse feature types, and missing data while eliminating the need for baseline hazard or survival distribution assumptions, providing a flexible and robust approach to survival modeling.

We extend the evaluation of Survival-LCS by incorporating a wider range of baseline distributions and testing its performance on an expanded set of simulated datasets generated with GAMETES software. These datasets include various genetic architectures, epistatic, additive, heterogeneous, and univariate models, alongside varying censoring proportions, minor allele frequencies, and feature dimensions. This comprehensive sensitivity analysis reveals Survival-LCS’s capability to detect complex, non-linear survival patterns without underlying proportional hazard assumptions. Using Integrated Brier Scores as a key metric, we assess its predictive accuracy for survival times under different distributions. Our findings explore challenges to the algorithm with data distributions, and the potential of Survival-LCS to overcome traditional limitations, offering significant applications in various domains of survival analysis.

---


## Repository Structure

### Runner Scripts
- **sim_cv_dataset_creation.py** – Runner to create datasets for all experiments across different survival distributions.   
- **sim_run_survivalLCS.py** – Runner for base Survival-LCS experiments.  
- **sim_run_survivalLCS_perm.py** – Runner for permutation testing for significance assessment.  
- **sim_run_coxModelRun.py** – Runner for baseline Cox proportional hazards experiments.  


### Base Code Files
- **survivalLCSRun.py** – Basecode for Survival-LCS runs.  
- **survivalLCSPermRun.py** – Basecode for permutation and Wilcoxon test runs.  
- **survivalCoxRun.py** – Basecode for Cox model runs.  
- **survivalLCSOtherOutputRun.py** – Generate additional outputs beyond main runs.  
- **survival_data_simulator.py** – Basecode to enerate simulated datasets across different survival distributions.  
- **cvPartitioner.py** – Partitioner code datasets into folds for CV.
- **sim_utils.py** – Utility functions for dataset handling and simulation.

### Analysis & Visualization
- **get_other_results.py** – Collect runtime and performance results.  
- **get_runtime_graph.py** – Generate runtime graphs for the manuscript.  
#### Notebooks
- **ComprehensiveModelFigure.ipynb** – Generate comprehensive model visualization figure.  
- **DatasetDistributions.ipynb** – Visualize dataset distributions.  
- **SurvivalAnalysisDistributions.ipynb** – Visualize survival distribution comparisons.  
- **NetworkGraph_gefx_generator.ipynb** – Generate network graph outputs.  
- **RuleTables.ipynb** – Generate rule-based model tables.  
- **ManualWilcoxonTesting.ipynb** – Wilcoxon test results post-runs.  
- **TablesCoxModels.ipynb** – Generate manuscript tables for Cox models.  
- **TablesPermutaionTests.ipynb** – Generate permutation test results tables.  
- **TablessurvivalLCS.ipynb** – Generate manuscript tables for Survival-LCS runs.  
- **GerRuntimeGraph.ipynb** / **Copy1** – Runtime visualization notebooks.

### Utilities & Commands
- **run_commands_test.txt** – Example command lines for testing runs.  
- **run_get_other_results.sh** – Script to execute `get_other_results.py`.  
- **run_get_runtime_graph.sh** – Script to generate runtime graphs.  
- **run_sim_cv_dataset_creation.sh** – Script to run dataset creation.
- **zip_all_csv_pngs.py** – Archive CSV/PNG outputs.  

### Supporting Code
- **importGametes.py** – Parse GAMETES genetic model files for simulations.  
- **requirements.txt** – Python dependencies for reproducing the analyses.  
- **.gitignore** – Standard Git ignore file.


---

## Simulated Datasets

The analysis requires simulated genetic survival datasets generated with **GAMETES**.  

- A compressed archive (`simulated_datasets.zip`) is provided separately.  
- Unzip this file into the repository home folder.  
- These files include GAMETES models files to generate different kinds of genetic dataset (univariate, additive, epistatic, heterogeneous) across varying censoring proportions, allele frequencies, and feature sets.  
- They are used by both `sim_cv_dataset_creation.py` and other run scripts.  

If you do not already have access to the dataset archive, please request it from the authors.

---

## Folder Setup

Before running experiments, create the following folders **inside the designated home directory for outputs**:

- `cv_sim_data/`  
  - subfolders: `cv_me`, `cv_epi`, `cv_het`, `cv_add`  
- `pickled_cv_models/`  
  - subfolders: `me`, `epi`, `het`, `add`  
- `sim_lcs_output/`  
  - subfolders: `me`, `epi`, `het`, `add`  

Additionally, create a **separate pipeline folder for each survival distribution**:  

```
randomspline_pipeline/
gamma_pipeline/
gaussian_pipeline/
weibull_pipeline/
```

All runs and outputs for that distribution should be contained in its respective folder.  
 **Note:** The correct format is `<distribution>_pipeline/` (e.g., `gamma_pipeline/`), **not** `pipeline/<distribution>`.

---

## Running Survival-LCS TELO Experiments

### Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
```

### HPC Requirements
<!-- These experiments are **computationally intensive** and strongly recommended for HPC execution.  

- **Cluster Type**: SLURM-based cluster (e.g., Cedars-Sinai).  
- **Job Submission**: Scripts are designed to be submitted with `sbatch`. See `run_*.sh` scripts for examples.  
- Local machines can run small-scale tests, but large-scale experiments require HPC capacity.   -->

The analysis takes extesnive computational requirements and while it can be run on a local machine it is recommend to run it on an HPC Cluster such as a SLURM based cluster. Our scripts are set up to run it on the SLURM HPC Cluster at Cedars-Sinai.

### Workflow

1. **Baseline Dataset Generation**  
   - Open `sim_cv_dataset_creation.py`.  
   - Update the `distribution_type` parameter:  
     ```python
     distribution_type = "randomspline"  # or "gamma", "gaussian", "weibull"
     ```  
   - Run the script (or `run_sim_cv_dataset_creation.sh`) to generate datasets with CV folds.  

2. **Configuring Output Directory**  
   - For **all run scripts** (`sim_run_survivalLCS.py`, `sim_run_survivalLCS_perm.py`, `sim_run_coxModelRun.py`, output-generation scripts), set:  
     ```python
     outputdir = homedir + "random_pipeline/"     # update for each distribution
     ```  
   - Ensures outputs are separated by distribution.  

3. **Configuring Run Parameters**  
   - Edit the top of the `sim_run_*.py` files to adjust parameters if needed.

4. **Running Models**  
   - `sim_run_survivalLCS.py` - Survival-LCS runs.  
   - `sim_run_coxModelRun.py` - Cox model comparisons.  
   - `sim_run_survivalLCS_perm.py` - permutation testing.  
   - Repeat for each `<distribution>_pipeline/`.  

5. **Post-Processing and Outputs**  
   - Run `get_other_results.py` to generate other visualizations such as distribution graphs.
   - Run `get_runtime_graph.py` to generate collate runtime/performance outputs.  
   - Other Outputs include:
     - **Tables** (via `Tables*.ipynb`)  
     - **Figures** (via `ComprehensiveModelFigure.ipynb`, `SurvivalAnalysisDistributions.ipynb`, etc.)  
     - **Rule Tables & Networks** (via `RuleTables.ipynb`, `NetworkGraph_gefx_generator.ipynb`)  
     - **Runtime Graphs** (via notebooks or scripts)  
   - For each baseline survival distribution (Random, Gamma, Gaussian, Weibull), repeat these runs for each dedicated `<distribution>_pipeline` folder as `outputdir`.  
   - Outputs (CSV/PNG files) are stored in the respective output directories and can be archived with `zip_all_csv_pngs.py`.

### Worked Example: Randomspline Distribution End-to-End

Below is a step-by-step example for running the Randomspline distribution experiments. Adjust similarly for Gamma, Gaussian, or Weibull.

```

# 1. Create the pipeline folder
mkdir random_pipeline

# 2. Generate baseline CV datasets
# Edit sim_cv_dataset_creation.py -> set distribution_type = "randomspline"
python sim_cv_dataset_creation.py

# 3. Run baseline Survival-LCS experiments
# Edit sim_run_survivalLCS.py -> set outputdir = homedir + "random_pipeline/"
python sim_run_survivalLCS.py

# 4. Run Cox model experiments
# Edit sim_run_coxModelRun.py -> set outputdir = homedir + "random_pipeline/"
python sim_run_coxModelRun.py

# 5. Run permutation tests
# Edit sim_run_survivalLCS_perm.py -> set outputdir = homedir + "random_pipeline/"
python sim_run_survivalLCS_perm.py

# 6. Generate additional outputs
# Edit get_other_results.py -> set outputdir = homedir + "random_pipeline/"
bash run_get_other_results.sh

# 7. Generate runtime graphs
# Edit get_runtime_graph.py -> set outputdir = homedir + "random_pipeline/"
bash run_get_runtime_graph.sh

# 8. Get other graphs from notebooks
# Edit outputdir = homedir + "random_pipeline/"
# Use VSCode to ssh-tunnel and run Notebooks on Cluster.

# 9. (Optional) Zip CSV/PNG outputs
python zip_all_csv_pngs.py
```

After completion, all results for the Randomspline distribution will be stored in:

random_pipeline/

Repeat the same process for gamma_pipeline/, gaussian_pipeline/, and weibull_pipeline/ by updating distribution_type and outputdir.

---

## Citation

If you use this code, please cite:

**Alexa Woodward, Harsh Bandhey, Jason H. Moore, and Ryan J. Urbanowicz.  
2025. Survival-LCS: Rule-Based Survival Analysis Without Proportional Hazard Assumptions.  
ACM Transactions on Evolutionary Learning and Optimization.**
