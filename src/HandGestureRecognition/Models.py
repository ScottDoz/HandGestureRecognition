# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:12:58 2024

@author: scott

Models
------

Methods to load in and use classifier models

"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

from itertools import chain

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

import tensorflow as tf
from tensorflow import lite
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
# Suppress verbose output from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Module imports
from TrainModels import load_training_data

#%% tflite models

def load_model(model_name):
    
    # Tensorflow lite models
    if model_name in ['ASL0_9','ASL0_9.tflite']:
        # Load TFLite model and allocate tensors.
        tflite_file = Path.cwd()/'tflite'/'ASL0_9.tflite'
    elif model_name in ['Hagrid_ASL0_9','Hagrid_ASL0_9.tflite']:
        tflite_file = Path.cwd()/'tflite'/'Hagrid_ASL0_9.tflite'
    elif model_name in ['Hagrid_ASL0_9_v2','Hagrid_ASL0_9_v2.tflite']:
        # New versions
        tflite_file = Path.cwd()/'tflite'/'Hagrid_ASL0_9_v2.tflite'
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()
    
    
    return interpreter

def interpreter_predict(interpreter,x,labels=None):
    
    # Run a prediction from some input data
    
    
    # Get expected input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get labels
    if labels is None:
        labels = ['class{}'.format(i) for i in range(output_details[0]['shape'][1])]
    
    # TODO: Check input
    # if x.shape != tuple(input_shape):
    #     # Reshape
    
    # Format input
    # Expect input x as 1d array. Reshape to 1xn array
    x = np.array([x], dtype=np.float32)
    
    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], x)
    
    interpreter.invoke()
    
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    proba = interpreter.get_tensor(output_details[0]['index'])
    
    # Find argmax
    ind = np.argmax(proba)
    # Return class
    c = labels[ind]
    
    return c

def interpreter_predict_proba(interpreter,x):
    
    # Run a prediction from some input data
    
    # Get expected input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # TODO: Check input
    # if x.shape != tuple(input_shape):
    #     # Reshape
    
    # Format input
    # Expect input x as 1d array. Reshape to 1xn array
    x = np.array([x], dtype=np.float32)
    
    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], x)
    
    interpreter.invoke()
    
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    proba = interpreter.get_tensor(output_details[0]['index'])
    
    return proba

if __name__ == "__main__":
    
    

    # Test model on input data
    data_dir = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data') # Locations of all csv data
    
    # Hagrid ASL
    model_name = "Hagrid_ASL0_9" # Done
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'call', 'dislike', 'fist', 'like',
                 'mute', 'ok', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three2',
                 'two_up', 'two_up_inverted']
    
    # # Load training data
    # X_train,y_train = load_training_data(data_dir, model_name)
    
    # Load model
    interpreter = load_model(model_name)
    
    
    # Test cases (extracted from Hagrid training dataset)
    # X_train,y_train = load_training_data(data_dir, model_name)
    y0 = '0' # y0 = y_train[0] x0 = X_train[0,:];
    x0 = np.array([2.7135179 , 1.8724382 , 1.71462464, 1.62152599, 1.87660098,2.8905463 , 2.82735358, 2.83187897, 2.72111987, 0.21672117, 0.25754137, 0.40795112, 0.60001853, 0.93113643, 0.93018787, 1.04037834, 1.20954839])
    y1 = '1' # y1 = y_train[50] x1 = X_train[50,:];
    x1 = np.array([2.51641861, 3.01611291, 1.05689402, 0.94450264, 0.98894786, 2.9096163 , 2.50619087, 2.45559456, 2.41417377, 0.91463871, 0.57261869, 0.75784808, 0.95614866, 0.87011805, 0.83128634, 0.86581931, 1.04935726])
    y2 = '2' # y2 = y_train[100] x2 = X_train[100,:];
    x2 = np.array([2.5767001 , 3.03035254, 3.01480251, 1.12012675, 1.42461343, 2.71175149, 2.88785956, 2.35413022, 2.34984716, 1.22574485, 1.12111069, 0.47032298, 0.56052171, 0.7542891 , 0.59724987, 0.51736793, 0.6279337 ])

    # Run 
    interpreter_predict(interpreter,x0,labels=classes)
    
