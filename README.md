
#  Survival-LCS: A Rule-Based Machine Learning Approach to Survival Analysis

### Alexa Woodward, Harsh Bandhey, Jason H. Moore, and Ryan J. Urbanowicz

## About

Survival analysis is a critical aspect of modeling time-to-event data in fields such as epidemiology, engineering, and econometrics. Traditional survival methods rely heavily on assumptions and are limited in their application to real-world datasets. To overcome these challenges, we introduce the survival learning classifier system (Survival-LCS) as a more flexible approach. Survival-LCS extends the capabilities of ExSTraCS, a rule-based machine learning algorithm optimized for biomedical applications, to handle survival (time-to-event) data. In addition to accounting for right-censored observations, Survival-LCS handles multiple feature types and missing data, and makes no assumptions about baseline hazard or survival distributions.

## Repository Orientation
This main branch includes the scikit-learn compatable implementation of Survival-LCS, however the algorithm and analysis code that pairs with the 2024 GECCO paper cited below is found at: 
[Survival-LCS/tree/gecco](https://github.com/UrbsLab/survival-LCS/tree/gecco).

Further the algorithm and analysis code that pairs with the 2025 expanded TELO paper is found at: 
[Survival-LCS/tree/telo](https://github.com/UrbsLab/survival-LCS/tree/telo).

Please note that we do not plan to continue development of this repository further.  Rather we plan to take some of the algorithmic concepts introduced in Survival-LCS to expand our new [HEROS](https://github.com/UrbsLab/heros) to survival analysis tasks. 

## Installation

```
pip install pip@git+https://github.com/UrbsLab/survival-LCS
```

## Citation

Alexa Woodward, Harsh Bandhey, Jason H. Moore, and Ryan J. Urbanowicz. 2024. Survival-LCS: A Rule-Based Machine Learning Approach to Survival Analysis. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '24). Association for Computing Machinery, New York, NY, USA, 431–439. https://doi.org/10.1145/3638529.3654154
