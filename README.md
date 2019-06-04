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

Before we start feature selection process, we calculate feature importance as it is explained 
[here](https://explained.ai/rf-importance/index.html) and train the model on the 15 most important features.

```python
from src.earthquake import utils
import config

# load training set
data = utils.read_csv(config.path_to_train)
# create list of features
features = [column for column in data.columns if column not in ['target', 'seg_id']]
# display importance
utils.feature_importance(data[features], data['target'], n_best=15)
```

List of 15 most important features.

```
  Imp | Feature
 0.11 | mfcc_5_avg
 0.09 | mfcc_15_avg
 0.07 | percentile_roll_std_5_window_50
 0.06 | percentile_roll_std_10_window_100
 0.06 | mfcc_4_avg
 0.03 | percentile_roll_std_20_window_500
 0.03 | percentile_roll_std_25_window_500
 0.02 | percentile_roll_std_25_window_100
 0.02 | percentile_roll_std_20_window_1000
 0.02 | percentile_roll_std_20_window_10
 0.02 | percentile_roll_std_25_window_1000
 0.01 | percentile_roll_std_10_window_500
 0.01 | percentile_roll_std_10_window_50
 0.01 | percentile_roll_std_50_window_50
 0.01 | percentile_roll_std_40_window_1000
```

### Training

## Results

`Cross-validation MAE`: 2.042, `public score`: 1.509, `private score`: 2.425 (31 place). 
   
## Links:

* LANL Earthquake Prediction, Kaggle competition: https://www.kaggle.com/c/LANL-Earthquake-Prediction
* Feature Engineering: https://www.kaggle.com/artgor/even-more-features
* LANL Earthquake EDA and Prediction: https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction
* "Beware Default Random Forest Importances": https://explained.ai/rf-importance/index.html 
