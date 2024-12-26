import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

train=pd.read_csv('C:/Users/bored/Music/Water_Potability/data/processed/train.csv')

test=pd.read_csv('C:/Users/bored/Music/Water_Potability/data/processed/test.csv')

x_train=train.iloc[:,0:-1].values
y_train=train.iloc[:,-1].values

clf=RandomForestClassifier()

clf.fit(x_train,y_train)

pickle.dump(clf,open('model.pkl','wb'))

