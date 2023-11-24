
#  Rule-based machine learning to address heterogeneity in high-dimensional survival data

### Alexa Woodward, MS - University of Pennsylvania, Perelman School of Medicine, DBEI/GGEB

survival-ExSTraCS is written in Python 3. First, you need to download this repository to local. To run, you will also need to first install....

```
pip3 install pandas matplotlib scikit-learn scikit-ExSTraCS skrebate fastcluster seaborn networkx pygame pytest-shutil
```

Started on: 2021-09-01

Advisors: Jason Moore, PhD, FACMI, Department of Computational Biomedicine, Cedars Sinai Medical Center & Li Shen, PhD at University of Pennsylvania, Deparment of Biostatistics, Epidemiology and Informatics (DBEI)
Mentor: Ryan Urbanowicz, PhD (DBEI)

Collaborators: Sharon Diskin, PhD, at the Children's Hospital of Philadelphia (CHOP)

Summary: Current approaches to analyzing genetic and other large data sets fail to suitably model heterogeneity – a phenomenon where different mechanisms give rise to the same disease; these methods similarly struggle with capturing interactions and other complex patterns of association. A full characterization of this complexity is necessary to fully understand and predict risk and outcomes in common diseases. Using both simulated and real-world genetic survival data, this proposal will develop and evaluate a machine learning method for identifying important variables and making predictions in the context of complex associations.

- This project will utilize two main methods - relief-based algorithms (RBAs) for feature selection, and a learning classifier system (LCS) to investigate epistasis and heterogeneity in survival data. Currently, while LCSs excel at handling heterogeneous data, none are currently suited for survival data. The aim of this project is to create a survival LCS, "sLCS" using ExSTraCS as the framework. LCSs are a type of rule-based machine learning algorithms that are well suited to complex problem spaces, but are not widely used in practice. Other strengths of this approach includes interpretability - important for clinical applications of machine learning. See below for links to background information. 

  - RBAs [here](https://www.sciencedirect.com/science/article/pii/S1532046418301412)
  - Scikit-rebate [here](https://epistasislab.github.io/scikit-rebate/) 
  - GAMETES paper [here](https://biodatamining.biomedcentral.com/articles/10.1186/1756-0381-5-16)
  - GAMETES software [here](https://github.com/UrbsLab/GAMETES)
  - ExSTraCS [here](https://github.com/UrbsLab/scikit-ExSTraCS.git)

### Repository structure: 

```
data/
docs/
code/
README.md
LICENSE
.gitignore
```

Limited project data is available in `data/`, as most data for this project is not publically available. Data has been made available through the Diskin Lab at CHOP. Summary data will be provided when applicable. The lab notebook for this project can be found in `docs/`, with dated "README" files corresponding to daily or weekly notes. README files will be updated monthly. Other documents include the data sharing agreement, manuscript sections, or to-do lists. All analysis code, including code to produce figures or tables will be kept in `code/`.
