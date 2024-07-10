# ACE: ACE: An Attention-based Model for Cardinality Estimation of Set-Valued Queries
## Introduction
This repo contains the source code of ACE

## Quick Start
The link of three datasets with generated query workloads will be provided later. Running the code mainly includes 2 steps: (1) **Featurization** represents the underlying data numerically and outputs the distilled matrix. (2) **Estimation** utilizes the data matrix and the queried element embeddings to predicate the cardinality. 

### Before running
You might need to change the folder path in `./utils/preprocess.py` (dataset) or `./atte_estimator.py` (query)

### Featurization (offline)
To train a model to represent data, you could run:
```bash
python3 data_feature.py --d [dataset] --r [distill ratio] --dis_dep [distill depth]
```
This command will split the dataset into training, validation and testing data. Then the trained aggregation and distillation models are stored in the folder `./save_model`.

### Estimation (online)
After getting the trained featurization models, you could train the estimator by running:
```bash
python3 atte_estimator.py --d [dataset] --qt [query type] --qf [frequency]
```
Here, the query workloads are divided into three classes based on their type: superset, subset and overlap. Additionally, each query workload are also partitioned into three sub-classes based on the frequency of its comprised elements: regular(considering all elements), high(only considering high-frequency element) and low(only considering high-frequency element).

This command will generate the reperesentation of the underlying data by using the learned models and train the attention-based estimator.
