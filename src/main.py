from fastapi import FastAPI
import pickle
import os
import pandas as pd
from joblib import load
from data_model import Water
app=FastAPI(
  title='Water Potability Prediction',
  description='Predicting Water Potability'
)

with open('C:/Users/bored/Music/Water_Potability/model.joblib'):
  model = load('C:/Users/bored/Music/Water_Potability/model.joblib')

@app.get('/')
def index():
  return "The Water Potability Website 🤞"

@app.post("/predict")
def model_predict(water: Water):
    sample = pd.DataFrame({
        'ph' : [water.ph],
        'Hardness' : [water.Hardness],
        'Solids' :[water.Solids],
        'Chloramines' :[water.Chloramines],
        'Sulfate' : [water.Sulfate],
        'Conductivity' :[water.Conductivity],
        'Organic_carbon' :[water.Organic_carbon],
        'Trihalomethanes' : [water.Trihalomethanes],
        'Turbidity' :[water.Turbidity]
    })

    predicted_value = model.predict(sample)


    if predicted_value == 1:
        return "Water is Consumable"
    else:
        return "Water is not Consumable"