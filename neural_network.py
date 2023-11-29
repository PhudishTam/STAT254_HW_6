import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
from dask_ml.model_selection import GridSearchCV
from preprocess import preprocess_data,scale_features,evaluate_model
from torch import nn
from skorch import NeuralNetRegressor

df = pd.read_csv("SeoulBikeData.csv", encoding='unicode_escape')
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=60)
X_train, X_test = scale_features(X_train, X_test)
device = "cuda" if torch.cuda.is_available() else "cpu"
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype= torch.float32)
y_train = torch.tensor(y_train.values,dtype= torch.float32).unsqueeze(dim = 1)
y_test = torch.tensor(y_test.values,dtype= torch.float32).unsqueeze(dim = 1)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

class NeuralNetwork(nn.Module):
    def __init__(self, input_features = 20, out_features = 1, hidden_units_1 = 10, hidden_units_2 = 10, hidden_units_3 = 10):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units_1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units_1, out_features=hidden_units_2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units_2, out_features=hidden_units_3),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units_3, out_features=out_features)
        )
    def forward(self, x):
        return self.linear_layer_stack(x)


t0 = time.time()
net = NeuralNetRegressor(
    NeuralNetwork,
    criterion=torch.nn.MSELoss,
    optimizer=torch.optim.Adam,
    max_epochs=10,
    device=device,
    verbose=0
)
params = {
    'module__hidden_units_1': [50,60,70,90],
    'module__hidden_units_2': [30,40,50,70],
    'module__hidden_units_3': [10,20,30,40],
}
best_model = GridSearchCV(net, params, cv=5, scoring='neg_mean_squared_error',n_jobs = -1)
X_train = X_train.cpu()
y_train = y_train.cpu()
X_test = X_test.cpu()
y_test = y_test.cpu()
r2_score_train,mean_absolute_error_score_train,root_mean_square_error_train,r2_score_test,mean_absolute_error_score_test,root_mean_square_error_test = evaluate_model(best_model,X_train,y_train,X_test,y_test)
print(f"Time it took: {time.time() - t0}")
print(f"The best parameters: {best_model.best_params_}")
print(f"The root mean sqaure error training is : {np.sqrt(-best_model.best_score_)}")
print(f"The r2 score on training is : {r2_score_train}")
print(f"The mean absolute error training is : {mean_absolute_error_score_train}")
print(f"The root mean square error training is : {root_mean_square_error_train}")
print(f"The r2 score on testing is : {r2_score_test}")
print(f"The mean absolute error testing is : {mean_absolute_error_score_test}")
print(f"The root mean square error testing is : {root_mean_square_error_test}")