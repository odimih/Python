# We import some of the basic packages we need.
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.datasets import cifar10
from keras.datasets import cifar10
from sklearn import tree
from PIL import Image # powerful library for image processing tasks

# Let's write a simple function to compute the accuracy.

def compute_accuracy(a1, a2): 
    return float((np.ravel(a1) == np.ravel(a2)).mean())

# μεταφορτώνουμε τα δεδομένα
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Έχουμε την δυνατότητα να ονομάσουμε εμείς τις 10 κατηγορίες
labels = ['αεροπλάνο', 'αυτοκίνητο', 'πτηνό', 'γάτα', 'ελάφι', 'σκύλος', 'βάτραχος', 'άλογο', 'πλοίο', 'φορτηγό']

# Let's check the size
print('There are', X_train.shape[0], 'training points.')
print('There are', X_test.shape[0], 'testing points.')
# How many times does each label appear?
for i in range(10): # a loop running from 0 to 9 (10 times in total)
  #ylist = y_train.tolist()
  num = sum(y_train==i) #calculates the sum of all the elements of y_train that are 0, 1 to 9 and stores it to variable num
  print('The label ', labels[i], 'appears ', num, 'times in the training data')#prints the number of times each label appears in y_train

print('These are the labels', y_train)
print('The feature vectors have shape', X_train[0].shape)
print('This is what the feature vector actually looks like',X_train[0])

# NumPy's ravel() function flattens into a 1-d array
print('These are the first 1000 feature values of the 16th training point:',np.ravel(X_train[16])[0:1000])
print('\n\n These are the first 1000 feature values of the 904th training point:',np.ravel(X_train[904])[0:1000])

img = X_train[16] #assigns the ?th image from X_train to the variable img
plt.figure(figsize=(6, 3)) #creates a new figure of 6 inches width and 3 inches height
plt.imshow(img) #displays the image
plt.axis('off')  # Optional: turn off axes for a cleaner look
plt.show()
img = X_train[904]
plt.figure(figsize=(6, 3))
plt.imshow(img)
plt.axis('off')
plt.show()

# Με τον ίδιο τρόπο μπορούμε να δούμε πως μοιάζουν και άλλες εικόνες
fig, axes = plt.subplots(ncols=5, nrows=3, figsize=(17, 8)) #creates a new figure (fig) and a set of subplots (axes) arranged in a grid
#of 5 columns, 3 rows of 17 inches wide and 8 inches high images
#axes is an array of objects where each element corresponds to a subplot.
index = 0
for i in range(3):
    for j in range(5):
        axes[i,j].set_title(labels[y_train[index][0]])
        axes[i,j].imshow(X_train[index])
        axes[i,j].get_xaxis().set_visible(False)
        axes[i,j].get_yaxis().set_visible(False)
        index += 1
plt.show()

# We reformat X to have the required shape that our algorithms need.
X_tr = X_train.reshape((X_train.shape[0], -1))
print('This is now the shape of X:',X_tr.shape)
print('The reshaped X has ' + repr(X_tr.shape[0]) + ' rows, one for each sample, and ' + repr(X_tr.shape[1]) + ' columns, one for each feature.')#repr is used to obtain a string representation of an object

# There are 50,000 data points. We pick a subset of them to speed up the training.
# The loader already shuffled, so it's safe to just pick the first ones.
N_tr = 5000
train_samples = N_tr
X_str = X_tr[:N_tr,:] # create a subset X_tr with N_tr samples.
y_str = y_train[:N_tr] # create  a subset target variable y_train with N_tr labels corresponding to the samples selected in X_str.
# check size
X_str.shape

# example of how it works
print(compute_accuracy([1,0,1,1,1,0,0,1,0,1],[0,1,1,1,1,0,0,1,1,0]))

# Ορίζουμε και εκπαιδεύουμε ένα δέντρο απόφασης: decision tree.
# Η μόνη παράμετρος που ορίζουμε είναι το βάθος: max_depth
decision_tree = tree.DecisionTreeClassifier(max_depth=1)
# Το εκπαιδεύουμε:
decision_tree.fit(X_str, y_str)

# Έτσι μοιάζει το δέντρο μας:
tree.plot_tree(decision_tree, impurity = False)
# Υπολογίζουμε την ακρίβεια στα δεδομένα εκπαίδευσης, τα training data

# Πρώτα χρησιμοποιούμε τον αλγόριθμό μας και σημειώνουμε τις εκτιμήσεις
# στα δεδομένα εκπαιδευσης.
# Μετα συγκρίνουμε τις προβλέψεις με τα πραγματικά labels.

yhat_train = decision_tree.predict(X_str)

print("Train score: %.4f" % compute_accuracy(y_str, yhat_train))

# Εναλακτικά μπορούμε να χρησιμοποιήσουμετ το model.score module.
train_score = decision_tree.score(X_str,y_str)
print("Train score: %.4f" % train_score)

# random guessing -- we are expecting something close to 0.1
import random

def generate_random_array(n):
    # Use a list comprehension to generate an array of n random integers between 0 and 9
    return [random.randint(0, 9) for _ in range(n)]

random_guess = generate_random_array(len(y_str))#the length of y_str is 5000
print("This is the accuracy of a random guess: %.4f" % compute_accuracy(y_str, random_guess))#prints accuracy to 4 decimal places
