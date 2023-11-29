import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from dask.distributed import Client, LocalCluster
from dask_ml.model_selection import GridSearchCV
from preprocess import preprocess_data,scale_features,tune_hyperparameters,evaluate_model

if __name__ == "__main__":
    df = pd.read_csv("SeoulBikeData.csv", encoding='unicode_escape')
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=60)
    X_train, X_test = scale_features(X_train, X_test)
    cluster = LocalCluster(n_workers=32, threads_per_worker=3)
    client = Client(cluster)
    t0 = time.time()
    estimator = xgb.XGBRegressor()
    param_test_1 = {'learning_rate': [1,0.1],'max_depth':[45,50,55,60],'n_estimators': [200,220,240],'reg_alpha': [15,16,17,18],'reg_lambda': [25,27,29,31]}
    best_params, best_score = tune_hyperparameters(estimator, param_test_1, X_train, y_train)
    print(f"Best paramaeters: {best_params}")
    print(f"The root mean square error training is: {np.sqrt(best_score)}")
    best_model = xgb.XGBRegressor(learning_rate = best_params['learning_rate'],max_depth = best_params['max_depth'],n_estimators = best_params['n_estimators'],reg_alpha = best_params['reg_alpha'] , reg_lambda = best_params['reg_lambda'] )
    r2_score_train,mean_absolute_error_score_train,root_mean_square_error_train,r2_score_test,mean_absolute_error_score_test,root_mean_square_error_test = evaluate_model(best_model,X_train,y_train,X_test,y_test)
    print(f"Time it took: {time.time() - t0}")
    print(f"The r2 score on training is : {r2_score_train}")
    print(f"The mean absolute error training is : {mean_absolute_error_score_train}")
    print(f"The root mean square error training is : {root_mean_square_error_train}")
    print(f"The r2 score on testing is : {r2_score_test}")
    print(f"The mean absolute error testing is : {mean_absolute_error_score_test}")
    print(f"The root mean square error testing is : {root_mean_square_error_test}")
    cluster.close() 

    