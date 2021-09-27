# Credit Risk Analysis

## Resources
- Jupyter Lab
- VS Code
- `imbalanced-learn` library
- `scikit-learn` library
## Overview
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, different techniques will need to be used to train and evaluate models with unbalanced classes.  This project will explore six different Machine Learning models to predict credit risk:

- `RandomOverSampler`
- `SMOTE`
- `ClusterCentroids`
- `SMOTEENN`
- `BalancedRandomForestClassifier`
- `EasyEnsembleClassifier`
## Results

| ML Model                        | Balanced Accuracy Score | Precision | Sensitivity (Recall) |
| ------------------------------- | ----------------------- |-----------| -------------------- |          
| `RandomOverSampler`             | 0.666323                | 0.99      | 0.63                 |
| `SMOTE`                         | 0.662306                | 0.99      | 0.69                 |
| `ClusterCentroids`              | 0.544733                | 0.99      | 0.40                 |
| `SMOTEENN`                      | 0.676041                | 0.99      | 0.58                 |
| `BalancedRandomForestClassifier`| 0.995989                | 1.00      | 1.00                 |
| `EasyEnsembleClassifier`        | 0.942400                | 0.99      | 0.94                 |

### Over & Under Sampling
In each of the four algorithms that employ over- or under-sampling, 
#### `RandomOverSampler`
![](Images/oversampling.PNG)

#### `SMOTE`
![](Images/smote.PNG)

#### `ClusterCentroids`
![](Images/undersampling_clustercentroid.PNG)

#### `SMOTEENN`
![](Images/combination_smoteenn.PNG)

### `BalancedRandomForestClassifier`
![](Images/ensemble_randomforest.PNG)

### `EasyEnsembleClassifier`
![](Images/easyensembleclassifier.PNG)

There is a bulleted list that describes the balanced accuracy score and the precision and recall scores of all six machine learning models (15 pt)
## Summary
There is a summary of the results (2 pt)
There is a recommendation on which model to use, or there is no recommendation with a justification (3 pt)
