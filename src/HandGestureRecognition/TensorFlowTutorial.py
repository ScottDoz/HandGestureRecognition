# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:13:23 2024

@author: scott

Tensor Flow Tutorial
--------------------

https://towardsdatascience.com/from-scikit-learn-to-tensorflow-part-1-9ee0b96d4c85

"""

# Import data loading and classification libraries
from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import tensorflow as tf
# Suppress verbose output from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

#%% Load Iris data
iris = datasets.load_iris()

# Convert to dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target


# Load features and classes
X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, 
                                                                    iris.target, 
                                                                    test_size=0.6, 
                                                                    random_state=42)


#%% ------------------------------------------
# Scikit Learn Implementation
# ------------------------------------------
# Use support vector classification
classifier_sk = svm.SVC()
# Use the train data to train this classifier
classifier_sk.fit(X_train, y_train)
# Use the trained model to predict on the test data
predictions = classifier_sk.predict(X_test)
score = metrics.accuracy_score(y_test, predictions)
print('Sklearn Accuracy: {0:f}'.format(score))


#%% Generate TFlite model
# https://www.geeksforgeeks.org/how-to-create-custom-model-for-android-using-tensorflow/
# https://medium.com/@nutanbhogendrasharma/tensorflow-deep-learning-model-with-iris-dataset-8ec344c49f91


# specifying the columns values into x and y variable
# iloc range based selecting 0 to 4 (4) values
X = iris.data
y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# Encode target labels with value between 0 and n_classes-1.
le = LabelEncoder()
y = le.fit_transform(y) # performing fit and transform data on y
y = to_categorical(y) # Convert to categorical

# Repeat for test and train sets
y_test = le.fit_transform(y_test); y_test = to_categorical(y_test);
y_train = le.fit_transform(y_train); y_train = to_categorical(y_train);


# Create model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
 
model = Sequential()
 
# input layer
# passing number neurons =64
# relu activation
# shape of neuron 4
model.add(Dense(64, activation='relu', input_shape=[4]))
 
# processing layer
# adding another denser layer of size 64
model.add(Dense(64))

# creating 3 output neuron
model.add(Dense(3, activation='softmax'))

# compiling model
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['acc'])


# training the model for fixed number of iterations (epoches)
model.fit(X_train, y_train, epochs=200)

from tensorflow import lite
converter = lite.TFLiteConverter.from_keras_model(model)

tfmodel = converter.convert()

open('iris.tflite', 'wb').write(tfmodel)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)


# Predict classes
y_pred = model.predict(X_test) # Probabilities
predicted = np.argmax(y_pred,axis=1)
# Get the actual vs predicted
actual = np.argmax(y_test,axis=1)
