import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# τεχνητά δεδομένα: data generation mechanism

 # return N labeled points
def gen_labeled_points(N,p):
  # First we generate uniformly distributed points
  rng = default_rng(12) # another way to set the random seed
  # %%
  X = rng.uniform(low=0.0, high=10.0, size=[N,2]) #specifies lower and upper bound of the range as well as the shape of the output array
  y = np.ones(N) # by default set all labels to 1
  # %%
  # Now we label them
  for i in range(N):
    # generate a p, 1-p coin
    c = 1
    if (rng.uniform(0,1) > 1-p): c = -1.
    if (X[i,0] > 5): y[i] = y[i]*c
    if (X[i,0] < 5): y[i] = -1*y[i]*c
  y = 0.5*(y+1) # make y 0's and 1's
  y = y.astype(int) # return integers, not floats
  return X,y

# N σημεία, πιθανότητα αλλαγής χρώματος = p
N = 200
p = 0.0
X,y = gen_labeled_points(N,p)

# Create a scatter plot (διάγραμμα διασποράς)
plt.figure(figsize=(10, 6))
for i in range(len(X)):
    if y[i] == 1:
        plt.scatter(X[i, 0], X[i, 1], color='blue', label='y=1' if 'y=1' not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(X[i, 0], X[i, 1], color='red', label='y=0' if 'y=0' not in plt.gca().get_legend_handles_labels()[1] else "")

# Adding labels and title
plt.title('Scatter Plot of Synthetic Data')
plt.legend()
plt.show()