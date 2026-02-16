import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from xgboost import XGBClassifier

path = 'diabetes.csv'
diabetes_data = pd.read_csv(path)
filtered_data = diabetes_data[(diabetes_data['Glucose'] != 0) & (diabetes_data['BMI'] != 0) & (diabetes_data['BloodPressure'] != 0)]
X_full = filtered_data.loc[:, filtered_data.columns != 'Outcome'].to_numpy()
y = filtered_data['Outcome'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.25, shuffle=True, random_state=42)

# Create a Gradient Boosting model
GB_model = XGBClassifier()

# Fit the model on the training data
GB_model.fit(X_train,y_train)

# Compute accuracy
y_pred_gb_train = GB_model.predict(X_train)
y_pred_gb_test = GB_model.predict(X_test)
accuracy_train = accuracy_score(y_train, y_pred_gb_train)
accuracy_test = accuracy_score(y_test, y_pred_gb_test)
print("Training Accuracy of Gradient Boosting:", accuracy_train)
print("Testing Accuracy of Gradient Boosting:", accuracy_test)