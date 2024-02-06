# Best practices for machine learning in microbiomics

This repository contains python implementations (tested on version 3.10.0) supporting the tutorial described in:

_Dudek, N.K., Chakhvadze, M., Kobakhidze, S., Kantidze, O., Gankin, Y. Supervised machine learning in microbiomics: bridging the gap between current and best practices. In prep for submission (2024)._

In this tutorial, we develop machine learning (ML) classifiers that can predict whether an individual has schizophrenia based on the composition of their fecal microbiota. The data used to develop the tutorial was sourced from [Zhu, Feng, et al. "Metagenome-wide association of gut microbiome features for schizophrenia." Nature communications 11.1 (2020): 1612.](https://www.nature.com/articles/s41467-020-15457-9).

### Installation

pip install requirements.txt

### How to run

1. Clone the github repo.
2. Install requirements.
3. If you want to train models and run SHAP from scratch, which may take several hours, you are ready to go. If you would like to use pre-computed trained models and SHAP values (recommended; default), you need to download results.pickle (693.1 MB) and shap_dict.pickle (31.5 MB) and drop them into the cloned repo.
4. Run the ipnyb notebook. 
