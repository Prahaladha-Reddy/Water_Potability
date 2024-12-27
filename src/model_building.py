import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

train=pd.read_csv('C:/Users/bored/Music/Water_Potability/data/processed/train.csv')

test=pd.read_csv('C:/Users/bored/Music/Water_Potability/data/processed/test.csv')

x_train=train.drop(columns=['Potability'],axis=1)
y_train=train['Potability']

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train) 


from joblib import dump
dump(model, 'C:/Users/bored/Music/Water_Potability/model.joblib')


