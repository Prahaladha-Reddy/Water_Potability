import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


train_data=pd.read_csv('C:/Users/bored/Music/Water_Potability/src/data/raw/train.csv')
test_data=pd.read_csv('C:/Users/bored/Music/Water_Potability/src/data/raw/test.csv')

def fill_missing_with_median(df):
  for column in df.columns:
    if df[column].isnull().any():
      median_value=df[column].median()
      df[column].fillna(median_value,inplace=True)
  return df
train_processed=fill_missing_with_median(train_data)

test_processed=fill_missing_with_median(test_data)

path=os.path.join('data','processed')

os.makedirs(path,exist_ok=True)
train_processed.to_csv(os.path.join(path, 'train.csv'), index=False)
test_processed.to_csv(os.path.join(path, 'test.csv'), index=False)
