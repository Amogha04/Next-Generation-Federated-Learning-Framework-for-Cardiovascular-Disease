import flwr as fl
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "../../data/heart.csv")
df = pd.read_csv(data_path)

X = df.drop("target", axis=1)
y = df["target"]

X = pd.get_dummies(X)

# Split into 3 hospitals manually
client_id = 1

X_split = np.array_split(X, 3)
y_split = np.array_split(y, 3)

X_local = X_split[client_id]
y_local = y_split[client_id]

scaler = StandardScaler()
X_local = scaler.fit_transform(X_local)

model = LogisticRegression(max_iter=2000)

class HeartClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        if not hasattr(model, "coef_"):
            model.fit(X_local, y_local)
        return [model.coef_, model.intercept_]

    def set_parameters(self, parameters):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        model.classes_ = np.array([0,1])

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.fit(X_local, y_local)
        return self.get_parameters(config), len(X_local), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        y_pred_prob = model.predict_proba(X_local)
        loss = log_loss(y_local, y_pred_prob)
        return loss, len(X_local), {}

fl.client.start_numpy_client(
    server_address="127.0.0.1:9090",
    client=HeartClient(),
)