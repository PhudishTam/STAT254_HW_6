import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from preprocess import preprocess_data,scale_features,tune_hyperparameters,evaluate_model

df = pd.read_csv("SeoulBikeData.csv", encoding='unicode_escape')
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=60)
X_train, X_test = scale_features(X_train, X_test)
r2_score_train,mean_absolute_error_score_train,root_mean_square_error_train,r2_score_test,mean_absolute_error_score_test,root_mean_square_error_test = evaluate_model(LinearRegression(),X_train,y_train,X_test,y_test)
print(f"The r2 score on training is : {r2_score_train}")
print(f"The mean absolute error training is : {mean_absolute_error_score_train}")
print(f"The root mean square error training is : {root_mean_square_error_train}")
print(f"The r2 score on testing is : {r2_score_test}")
print(f"The mean absolute error testing is : {mean_absolute_error_score_test}")
print(f"The root mean square error testing is : {root_mean_square_error_test}")