from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

class_names = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}


class IrisFeatures(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(data: IrisFeatures):
    if len(data.features) != 4:
        raise HTTPException(
            status_code=400, detail="Exactly 4 features required"
        )

    prediction = model.predict([data.features])
    pred_id = int(prediction[0])
    return {
        "prediction_id": pred_id,
        "class_name": class_names.get(pred_id, "Unknown"),
    }

