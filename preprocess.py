import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dask_ml.model_selection import GridSearchCV

def preprocess_data(df):
    df['Month'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.month
    df['Week_Type'] = df['Date'].apply(lambda x: "Weekend" if datetime.strptime(x, '%d/%m/%Y').weekday() > 4 else "Weekday")
    
    X = df.drop(['Date', 'Rented Bike Count'], axis=1)
    y = df['Rented Bike Count']
    y = np.sqrt(y)  

    cat_columns = ['Seasons', 'Holiday', 'Functioning Day', 'Week_Type']
    for col in cat_columns:
        X[col] = X[col].astype('category')

    X = pd.get_dummies(X, columns=cat_columns, dtype=int)
    return X, y

def scale_features(X_train, X_test):
    numeric_columns = ['Temperature(°C)', 'Humidity(%)', 'Dew point temperature(°C)',
                       'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Visibility (10m)',
                       'Snowfall (cm)', 'Wind speed (m/s)']
    scaler = StandardScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    return X_train, X_test

def tune_hyperparameters(estimator, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)   
    root_mean_square_error_train = -grid_search.best_score_
    return grid_search.best_params_, root_mean_square_error_train

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_train_root = model.predict(X_train)
    y_pred_train = (y_pred_train_root)**2
    r2_score_train = r2_score(y_train**2, y_pred_train)
    mean_absolute_error_score_train = mean_absolute_error(y_train**2, y_pred_train)
    root_mean_square_error_train = np.sqrt(mean_squared_error(y_train**2, y_pred_train))
    y_pred_test_root = model.predict(X_test)
    y_pred_test = (y_pred_test_root)**2
    r2_score_test = r2_score(y_test**2, y_pred_test)
    mean_absolute_error_score_test = mean_absolute_error(y_test**2, y_pred_test)
    mean_squared_error_test = mean_squared_error(y_test**2, y_pred_test)
    root_mean_square_error_test = np.sqrt(mean_squared_error_test)
    return (r2_score_train, mean_absolute_error_score_train, root_mean_square_error_train,
            r2_score_test, mean_absolute_error_score_test, root_mean_square_error_test)
