####
# survival-ExSTraCS

The survival-ExSTraCS package includes a sklearn-compatible Python implementation of survival-ExSTraCS, inspired by ExSTraCS 2.0 and other versions of this algorithm. Survival-ExSTraCS is a Michigan-Style Learning Classifier System (LCS), modified to handle time-to-event or survival analysis problems. Like ExSTraCS, it allows users to incorporate expert knowledge in the form of attribute weights, attribute tracking, rule compaction, and a rule specificity limit, that makes it particularly adept at solving highly complex problems.


LCSs are a type of rule-based machine learning (RBML) algorithms that have key advantages for approach heterogeneous problem spaces. The rules that an LCS generates are also highly interpretable and human-readable, in contrast to many current machine learning models. They can handle a variety of problems, including supervised and reinforcement learning, classidication and regression, missing data, and other nuances. To date however, no LCS has been shown to handle time-to-event outcomes. 

Time-to-event/survival outcomes include both a time (continuous) and censoring (binary) components. 

This version of survival-ExSTraCS is suitable for supervised survival analysis problems. It supports binary, categorical, and continuous variables as well as missing data.


## Usage
For more information on how to use survival-ExSTraCS, please refer to the [survival-ExSTraCS User Guide](https://github.com/alexa-woodward/survival-ExSTraCS/) Jupyter Notebook inside this repository. [draft]


## License
Please see the repository [license](https://github.com/alexa-woodward/survival-ExSTraCS/LICENSE) for the licensing and usage information for survival-ExSTraCS.

We have licensed survival-ExSTraCS to make it as widely usable as possible.

