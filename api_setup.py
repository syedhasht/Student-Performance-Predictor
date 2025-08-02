from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("student_model.pkl")
encoders = joblib.load("label_encoder.pkl")  # dictionary

class StudentData(BaseModel):
    gender: str
    race: str
    parental_education: str
    lunch: str
    test_preparation: str

@app.post("/predict")
def predict(data: StudentData):
    try:
        # Transform inputs using encoders
        gender = encoders['gender'].transform([data.gender])[0]
        race = encoders['race/ethnicity'].transform([data.race])[0]
        parental = encoders['parental level of education'].transform([data.parental_education])[0]
        lunch = encoders['lunch'].transform([data.lunch])[0]
        test_prep = encoders['test preparation course'].transform([data.test_preparation])[0]

        features = np.array([[gender, race, parental, lunch, test_prep]])
        prediction = model.predict(features)

        return {"pass": bool(prediction[0])}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
