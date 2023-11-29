import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
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
    estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,subsample=0.8,random_state=10)
    param_test1 = {'n_estimators': range(20,500, 20)}
    best_params_n_estimators, best_score_n_estimators = tune_hyperparameters(estimator, param_test1, X_train, y_train)
    print(f"Best paramaeters: {best_params_n_estimators}")
    print(f"The root mean sqaure error training is: {np.sqrt(best_score_n_estimators)}")
    estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=best_params_n_estimators['n_estimators'],min_samples_leaf = 50, subsample=0.8,random_state=10)
    param_test2 = {'max_depth': range(5, 20, 2), 'min_samples_split': range(200, 1500, 200)}
    best_params_max_depth_min_samples_split, best_score_max_depth_min_samples_split = tune_hyperparameters(estimator, param_test2, X_train, y_train)
    print(f"Best parameters: {best_params_max_depth_min_samples_split}")
    print(f"The root mean sqaure error training is: {np.sqrt(best_score_max_depth_min_samples_split)}")
    estimator = GradientBoostingRegressor(n_estimators=best_params_n_estimators['n_estimators'],max_depth = best_params_max_depth_min_samples_split["max_depth"],min_samples_split = best_params_max_depth_min_samples_split["min_samples_split"],random_state=10)
    param_test3 = {'min_samples_leaf': range(10, 70, 5),'subsample': [0.6,0.7,0.75,0.8,0.85,0.9]}
    best_params_min_samples_leaf_subsample, best_score_min_samples_leaf_subsample = tune_hyperparameters(estimator, param_test3, X_train, y_train)
    # feature_importances = best_model.feature_importances_
    # features_and_importances = zip(X_train.columns, feature_importances)
    # sorted_features_and_importances = sorted(features_and_importances, key=lambda x: x[1], reverse=True)
    # print("Feature Importances:")
    # for feature, importance in sorted_features_and_importances:
    #     print(f"{feature}: {importance}")
    print(f"Best parameters: {best_params_min_samples_leaf_subsample}")
    print(f"The The root mean sqaure error training is: {np.sqrt(best_score_min_samples_leaf_subsample)}")
    best_model = GradientBoostingRegressor(n_estimators=best_params_n_estimators['n_estimators'],max_depth = best_params_max_depth_min_samples_split["max_depth"],min_samples_split = best_params_max_depth_min_samples_split["min_samples_split"],min_samples_leaf=best_params_min_samples_leaf_subsample["min_samples_leaf"], 
    subsample = best_params_min_samples_leaf_subsample['subsample'], random_state=10)
    r2_score_train,mean_absolute_error_score_train,root_mean_square_error_train,r2_score_test,mean_absolute_error_score_test,root_mean_square_error_test = evaluate_model(best_model,X_train,y_train,X_test,y_test)
    print(f"Time it took: {time.time() - t0}")
    print(f"The r2 score on training is : {r2_score_train}")
    print(f"The mean absolute error training is : {mean_absolute_error_score_train}")
    print(f"The root mean square error training is : {root_mean_square_error_train}")
    print(f"The r2 score on testing is : {r2_score_test}")
    print(f"The mean absolute error testing is : {mean_absolute_error_score_test}")
    print(f"The root mean square error testing is : {root_mean_square_error_test}")
    cluster.close() 
