# OptAB

This Github repository encompasses the implementation of "OptAB - an optimal antibiotic selection framework for Sepsis" as presented by Wendland, Schenkel-Häger and Kschischo [1].
OptAB is to the best of our knowledge the first completely data-driven online-updateable optimal antibiotic selection model based on Artificial Intelligence for Sepsis patients accounting for antibiotic-induced side-effects. OptAB is able to handle the special characteristics of patient data including irregular measurements, numerous missing values and time-dependent confounding. 
OptAB performs an iterative optimal antibiotic selection for real-world Sepsis patients focussing on minimizing the Sepsis-related organ failure score (SOFA-Score) as treatment success while accounting for nephrotoxicity and hepatotoxicity as severe antibiotic side-effects. 
Our code builds upon the Neural CDE implementation from [2] and the Treatment-Effect Controlled differential equation [3]. 

Important: The MIMIC-IV data utilized in our analysis is not included in this repository - To get access to the MIMIC-IV dataset contact the MIMIC-IV team via physionet (https://physionet.org/content/mimiciv/2.2/).
The Amsterdamumcdb data utilized in our analysis is not included in this repository - To get access to the Amsterdamumcdb dataset contact the Amsterdamumcdb team via https://amsterdammedicaldatascience.nl/amsterdamumcdb/.


# Contents of the Repository

* Supplementary information to our Paper "OptAB - an optimal antibiotic selection framework for Sepsis" including additional plots and details about the treatment optimization and training of OptAB in the "Supplementary_information" folder.
* Scripts for executing OptAB on MIMIC-IV (and Amsterdamumcdb) along with explanations in the "code" folder.
* Scripts to create the plots showcased in our paper "OptAB - an optimal antibiotic selection framework for Sepsis" located in the "code" folder.
* The final Encoder and Decoder torch models of OptAB trained on MIMIC-IV .

# Package dependencies

* The package dependencies to run the MIMIC-IV preprocessing files are located in the Code/Mimic_Preprocessing folder.
* The python package dependencies to run the Amsterdamumcdb preprocessing files are located in the Code/Mimic_Preprocessing folder. The SOFA-Scores of the Amsterdamumcdb patients are computed via R using the ricu package [6].
* The package dependencies to run OptAB are located in the Code/OptAB folder.

# How to run OptAB

* To conduct hyperparameteroptimization execute the hypopt_enc.py and subsequently the hypopt_dec.py file.
* Train OptAB by executing the training_encoder.py and training_decoder.py files.
* For optimal treatment selection utilize the script compute_optimal_treatments.
* The script compute_treatment_influences creates predictions based on pre-defined treatments.
* To create the plots of our paper run the plot functions.

Remark: Comprehensive descriptions of the code can be found in the utils_paper.py file.

## MIMIC-IV preprocessing

* Before extracting the SOFA-Scores and Sepsis patients a postgres database of the MIMIC-IV data has to be created as described here: https://github.com/MIT-LCP/mimic-code [4]
* To extract SOFA-Scores and Sepsis patients execute the preprocessing.ipynb. The code in this notebook is based on the OpenSEP Pipeline (https://github.com/mhofford/I2_Sepsis) [5]

# References

[1] Wendland, P., Schenkel-Häger, C., & Kschischo, M. "OptAB - an optimal antibiotic selection framework for Sepsis" presented in Wendland, Schenkel-Häger and Kschischo". (2024)

[2] Seedat, N., Imrie, F., Bellot, A. & Qian, Z. "Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations". In Proceedings of the 39th International Conference on Machine Learning, vol. 162 of Proceedings of Machine Learning Research,19497–19521 (PMLR, 2022). URL https://proceedings.mlr.press//v162/seedat22b/seedat22b.pdf

[3] Kidger, P., Morrill, J., Foster, J. & Lyons, T. "Neural Controlled Differential Equations for Irregular Time Series". In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M. F. & Lin, H. (eds.) Advances in Neural Information Processing Systems, vol. 33 (Curran Associates, Inc., 2020)

[4] Johnson, A. E. W. et al. "MIMIC-IV, a freely accessible electronic health record dataset". Sci. Data 10, 1 (2023). URL https://www.nature.com/articles/s41597-022-01899-x

[5] Hofford, M. R. et al. "OpenSep: a generalizable open source pipeline for SOFA score calculation and Sepsis-3 classification". JAMIA Open 5,ooac105 (2022). URL https://academic.oup.com/jamiaopen/article/5/4/ooac105/6955623.

[6] Bennett, N. et al. "ricu: R's Interface to Intensive Care Data". arXiv:2108.00796. URL https://arxiv.org/abs/2108.00796