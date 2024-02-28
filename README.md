# Tutorial on machine learning for microbiomics

This repository contains python implementations (tested on version 3.10.0) supporting the tutorial described in:

_Dudek, N.K., Chakhvadze, M., Kobakhidze, S., Kantidze, O., Gankin, Y. Supervised machine learning for microbiomics: bridging the gap between current and best practices. In prep for submission (2024)._

In this tutorial, we'll develop an ML classifier that can predict whether an individual has schizophrenia based on the composition of their fecal microbiota. This work is motivated by the following questions:
1. Does the microbiome have predictive value for the diagnosis of schizophrenia? If so:
2. What methodological ML parameters (i.e., learning algorithm, features) yield the highest predictive results?
3. Which microbial taxa provide value for predicting whether an individual has schizophrenia?

The data used to develop the tutorial was sourced from [Zhu, Feng, et al. "Metagenome-wide association of gut microbiome features for schizophrenia." Nature communications 11.1 (2020): 1612.](https://www.nature.com/articles/s41467-020-15457-9)

### How to run

1. Clone the GitHub repo:

    ```bash
    git clone https://github.com/quantori/tutorial-on-ML-for-microbiomics.git
    ```

2. Install requirements:

    ```bash 
    pip install -r requirements.txt
    ```
   
3. If you want to train models and run SHAP from scratch (which may take several hours) you are ready to go. If you would like to load pre-computed trained models and SHAP values (recommended; default), you need to download `[results.pickle.gz](https://mmlformicrobiomics.blob.core.windows.net/trainingmml/results.pickle.gz)` (116.56 MB), `[X_test_for_shap.pickle.gz](https://mmlformicrobiomics.blob.core.windows.net/trainingmml/X_test_for_shap.pickle.gz)` (4.1 MB), and `[shap_dict.pickle.gz](https://mmlformicrobiomics.blob.core.windows.net/trainingmml/shap_dict.pickle.gz)` (4.36 MB), drop them into the cloned repo, and gunzip them.
4. Run the Jupyter Notebook: [see instructions](https://jupyter.org/install)
