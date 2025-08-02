This is a simple machine learning project that predicts whether a student will pass or fail based on some background information. It uses a trained Random Forest model and is deployed using FastAPI. The idea is to get hands-on experience with model training, encoding, deployment, and building an API for predictions.

## Project Overview

The model takes 5 inputs:
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course

Based on these, it predicts whether the student is likely to pass (based on an average score threshold).

The project can be  tested on:
http://127.0.0.1:8000/docs

## Dataset

The dataset used is `StudentsPerformance.csv`, which includes students’ scores in math, reading, and writing along with demographic details.

I added two new columns:
- `average_score` – the mean of math, reading, and writing scores
- `pass` – 1 if average score is >= 50, else 0

## Model

I trained a Random Forest classifier using scikit-learn. The categorical columns were label-encoded. Features like scores were dropped after calculating the average and target column.

Once trained, I saved the model and encoders using `joblib` so they can be reused during prediction in the API.

## API

The API is built using FastAPI.

### Endpoint

- `POST /predict`: Takes student data in JSON format and returns whether they will pass or not.

### Example Input

```json
{
  "gender": "female",
  "race": "group B",
  "parental_education": "bachelor's degree",
  "lunch": "standard",
  "test_preparation": "none"
}

Example Output
{
  "pass": true
}




