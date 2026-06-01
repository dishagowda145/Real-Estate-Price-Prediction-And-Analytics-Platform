import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import pickle
import os

print("Loading data...")
data = pd.read_csv("kc_house_data.csv")

labels = data['price']
conv_dates = [1 if str(values).startswith('2014') else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.10, random_state=2)

clf = ensemble.GradientBoostingRegressor(
    n_estimators=400, 
    max_depth=5, 
    min_samples_split=2, 
    learning_rate=0.1, 
    loss='squared_error',
    verbose=1
)

print("Training model...")
clf.fit(x_train, y_train)
print(f"Test score: {clf.score(x_test, y_test)}")

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("Model saved to model.pkl")
