# LANL Earthquake Prediction

This repository presents an approach used for solving [Kaggle LANL Earthquake Prediction Challenge](https://www.kaggle.com/c/LANL-Earthquake-Prediction/overview/description).

For feature engineering we used [this kernel](https://www.kaggle.com/artgor/even-more-features) (slightly modified for adding some spectral features). 
The initial training set `/data/train.csv` contains 4194 rows (one row for each segment) and 1496 columns (features).
We applied genetic algorithm with CatboostRegressor for fitness evaluation to implement a feature selection. 
Based on the GA's results, we selected [15 features](https://github.com/viktorsapozhok/earthquake-prediction/blob/master/src/earthquake/submission.py) and
trained the model using CatboostRegressor with default parameters.

### Feature engineering

The initial acoustic signal is decomposed into segments with 150000 rows per segment,  
which suggests that the training dataset has 4194 rows. Features are calculated as aggregations over segments.
For more details see, for example, 
[here](https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction) and 
[here](https://www.kaggle.com/artgor/even-more-features).     

### Feature selection

### Training

### Results

Cross-validation MAE: `2.042`, public score: `1.509`, private score: `2.425` (31 place). 
   
## Links:

* LANL Earthquake Prediction, Kaggle competition: https://www.kaggle.com/c/LANL-Earthquake-Prediction
* Feature Engineering: https://www.kaggle.com/artgor/even-more-features
* LANL Earthquake EDA and Prediction: https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction