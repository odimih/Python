import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

path = 'diabetes.csv'
diabetes_data = pd.read_csv(path)
filtered_data = diabetes_data[(diabetes_data['Glucose'] != 0) & (diabetes_data['BMI'] != 0) & (diabetes_data['BloodPressure'] != 0)]
X_full = filtered_data.loc[:, filtered_data.columns != 'Outcome'].to_numpy()
y = filtered_data['Outcome'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.25, shuffle=True, random_state=42)

d = 10 # depth of the tree
training_scores = [] #initialize an empty list
testing_scores = [] #initialize an empty list
depth_values = range(15) # create a range object representing integers 0 to 14
for depth in depth_values: #this loop runs 15 times
    dt = tree.DecisionTreeClassifier(max_depth=depth+1) #creates trees of depth 1 to 15
    dt.fit(X_train,y_train)
    train_score = dt.score(X_train,y_train) #calculates the accuracy of the model on the given dataset
    test_score = dt.score(X_test, y_test)
    print(f"Depth: {depth+1}, Train Accuracy: {train_score}, Test Accuracy: {test_score}")
    training_scores.append(train_score) #add train score data to the training_scores list at each iteration
    testing_scores.append(test_score) #add test score data to the testing_scores list at each iteration

# %%
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(training_scores,label='training')
ax.plot(testing_scores,label='testing')
ax.set_xlabel('tree depth')  # Add an x-label to the axes.
ax.set_ylabel('accuracy')  # Add a y-label to the axes.
ax.set_title("Train vs Test")  # Add a title to the axes.
ax.legend();  # Add a legend.

# %%
