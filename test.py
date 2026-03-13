import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)

X = np.column_stack([
    np.random.uniform(10, 800, 1000),   # distance
    np.random.uniform(0.1, 30, 1000),   # poids
    np.random.randint(1, 5, 1000)       # priorité
])

y = 1 + X[:,0]/200 + X[:,1]/15 + np.random.normal(0, 0.5, 1000)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()

X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_tr_s, y_tr)

preds = model.predict(X_te_s)

mae = mean_absolute_error(y_te, preds)
r2 = r2_score(y_te, preds)

print("MAE :", mae)
print("R2 :", r2)

preds = model.predict(X_te_s)

mae = mean_absolute_error(y_te, preds)
r2 = r2_score(y_te, preds)

print("MAE :", mae)
print("R2 :", r2)

joblib.dump(model, "livraison_model_v1.joblib")
joblib.dump(scaler, "scaler_v1.joblib")

mlflow.set_experiment("demo_livraison_model")

with mlflow.start_run(run_name="rf_livraison_v1"):

    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    mlflow.sklearn.log_model(model, "model")