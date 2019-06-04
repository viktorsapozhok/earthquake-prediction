# LANL Earthquake Prediction

This repository presents an approach used for solving [Kaggle "Earthquake Prediction"](https://www.kaggle.com/c/LANL-Earthquake-Prediction/overview/description) problem.

[This kernel](https://www.kaggle.com/artgor/even-more-features) (slightly modified for adding some spectral features) has been used for feature engineering. 
The initial training set (/data/train.csv) contains 4194 rows (one row for each segment) and 1496 columns (features).
Genetic algorithm with CatboostRegressor for fitness evaluation is used to implement a feature selection. 
Based on the GA's results, [15 features](https://github.com/viktorsapozhok/earthquake-prediction/blob/master/src/earthquake/submission.py) has been included in the model.

CatboostRegressor with default parameters was used for training the model.

### Feature engineering

### Feature selection

### Training

### Results

Cross-validation MAE: `2.042`, public score: `1.509`, private score: `2.425` (31 place). 
   
## Links:

* LANL Earthquake Prediction, Kaggle competition: https://www.kaggle.com/c/LANL-Earthquake-Prediction
* EDA, feature engineering: https://www.kaggle.com/artgor/even-more-features