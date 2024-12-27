import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import yaml

data=pd.read_csv('C:/Users/bored/Music/Water_Potability/water_potability.csv')

test_size=yaml.safe_load(open('params.yaml'))['data_collection']['test_size']

train,test=train_test_split(data,test_size=test_size,random_state=42)

data_path=os.path.join('data','raw')

os.makedirs(data_path)

train.to_csv(os.path.join(data_path,'train.csv'),index=False)
test.to_csv(os.path.join(data_path,'test.csv'),index=False)
