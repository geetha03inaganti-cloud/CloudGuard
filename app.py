from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Cloud Outage Prediction API running"}

@app.post("/predict")
def predict(data: dict):
    X = np.array([[ 
        data["CPU_Usage"],
        data["Memory_Usage"],
        data["Disk_IO"],
        data["Network_IO"]
    ]])

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)

    return {"outage_prediction": int(prediction[0])}
