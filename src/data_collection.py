import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv('C:/Users/bored/Music/Water_Potability/water_potability.csv')

train,test=train_test_split(data,test_size=0.20,random_state=42)

data_path=os.path.join('data','raw')

os.makedirs(data_path)

train.to_csv(os.path.join(data_path,'train.csv'),index=False)
test.to_csv(os.path.join(data_path,'test.csv'),index=False)
