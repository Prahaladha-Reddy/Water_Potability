import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

test=pd.read_csv('C:/Users/bored/Music/Water_Potability/data/processed/test.csv')


x_test=test.iloc[:,0:-1].values
y_test=test.iloc[:,-1].values

model=pickle.load(open('model.pkl','rb'))

y_pred=model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
f1_score=f1_score(y_test,y_pred)
recal=recall_score(y_test,y_pred)


metrices_dict={
  'acc':accuracy,
  'precision':precision,
  'f1_score':f1_score,
  'recall':recal
}

with open('metrics.json','w') as file:
  json.dump(metrices_dict,file,indent=4)