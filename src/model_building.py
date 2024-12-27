import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import yaml

params = yaml.safe_load(open('params.yaml'))
test_size = params['data_collection']['test_size']
random_search_params = params['model_building']['random_search']
param_grid = random_search_params['param_grid']


train = pd.read_csv('C:/Users/bored/Music/Water_Potability/data/processed/train.csv')
test = pd.read_csv('C:/Users/bored/Music/Water_Potability/data/processed/test.csv')

x_train = train.drop(columns=['Potability'], axis=1)
y_train = train['Potability']

x_test = test.drop(columns=['Potability'], axis=1)
y_test = test['Potability']

rf = RandomForestClassifier()

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=random_search_params['n_iter'],
    cv=random_search_params['cv'],
    verbose=2,
    random_state=random_search_params['random_state'],
    n_jobs=-1
)

print("Starting RandomizedSearchCV...")
random_search.fit(x_train, y_train)
print("RandomizedSearchCV completed.")


best_params = random_search.best_params_
print(f"Best Parameters: {best_params}")


best_model = random_search.best_estimator_

dump(best_model, 'C:/Users/bored/Music/Water_Potability/model.joblib')
print("Model saved successfully.")
