# Large-Scale-Wave-Energy-Fram

## Project Overview
This project aims to predict wave energy output using machine learning techniques. A linear regression model is applied to a large dataset to estimate Total_Power based on various input parameters.

## Dataset
- File: WEC_Perth_49.csv

- Target Variable: Total_Power

- Features: All columns except Total_Power

- Size: Large dataset

## Dependencies
Ensure you have the following Python libraries installed: *pip install numpy pandas scikit-learn matplotlib seaborn*

## Code Execution Steps

### 1.Import Required Libraries
   
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

### 2.Load Dataset
*df = pd.read_csv("WEC_Perth_49.csv")*

### 3.Prepare Data
*X = df.drop("Total_Power", axis=1)*

*y = df["Total_Power"]*

### 4.Split Dataset
*X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)*

### 5.Train Model
*model = LinearRegression()*
*model.fit(X_train, y_train)*

### 6.Make Predictions
*y_pred = model.predict(X_test)*

### 6.Evaluate Model
*mae = mean_absolute_error(y_test, y_pred)*
*mse = mean_squared_error(y_test, y_pred)*
*r2 = r2_score(y_test, y_pred)*

## Model Performance
- MAE: 167.18

- MSE: 67,666.5

- R² Score: 0.99999

## Conclusion
The linear regression model demonstrates outstanding performance in predicting wave energy output, as indicated by a high R² score. While the low error metrics suggest accurate predictions, it is essential to validate the model further through cross-validation and residual analysis. If necessary, exploring more complex models or feature engineering could enhance predictive accuracy and robustness.



