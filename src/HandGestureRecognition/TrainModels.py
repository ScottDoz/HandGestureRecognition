# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:35:40 2024

@author: scott

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from ImageProcessing import *


#%% Train models

def train_model(data_dir, model_name):
    
    # Adapted from: https://towardsdatascience.com/from-scikit-learn-to-tensorflow-part-1-9ee0b96d4c85
    
    # Create output directory
    outdir = data_dir.parents[0]/'tfModels'/model_name
    outdir.mkdir(parents=True, exist_ok=True)
    
    
    # Load data
    df = read_training_data(data_dir,model_name)
    
    # Define features of model
    if model_name.lower() in ['asl0_9','asl1_9','hagrid_asl0_9','hagrid_asl1_9']:
        # Original models all trained on same set of features
        features = ['phi0','phi1','phi2','phi3','phi4',
                    'theta1','theta2','theta3','theta4',
                    'TT1n','TT2n','TT3n','TT4n',
                    'TB1n','TB2n','TB3n','TB4n',
                    ]
    elif model_name.lower() == 'hagrid_asl0_9_v2':
        # Select individual features for Hagrid ASL dataset
        features = ['phi0','phi1','phi2','phi3','phi4',  # Knuckle angles
                    'theta1','theta2','theta3','theta4', # Finger elevations 
                    'TT1n','TT2n','TT3n','TT4n',         # Thumb to finger tip
                    'TB1n','TB2n','TB3n','TB4n',         # Thumb to base of fingers
                    'D12n','D23n','D34n', # Distances between fingers
                    'alpha',              # Palm orientation angle
                    ]
        
    
    # Extract training data
    # X_train, X_test, y_train, y_test
    X_train = df[features][df.set == 'train'].to_numpy()
    y_train = df['label'][df.set == 'train'].to_numpy()
    
    # Extract test data
    X_test = df[features][df.set == 'test'].to_numpy()
    y_test = df['label'][df.set == 'test'].to_numpy()
    
    # Save test data
    dft = df[features+['label']][df.set == 'test']
    outname = model_name + '_testdata.csv'
    dft.to_csv(str(outdir/outname),index=False)
    # del dft, df # Free up some memory
    del dft
    
    # ------------------------------------------
    # Scikit Learn Implementation
    # ------------------------------------------
    # Use support vector classification
    classifier_sk = svm.SVC()
    # classifier_sk = GaussianNB()
    # Use the train data to train this classifier
    classifier_sk.fit(X_train, y_train)
    # Use the trained model to predict on the test data
    predictions = classifier_sk.predict(X_test)
    score_svm = metrics.accuracy_score(y_test, predictions)
    cm_svm = confusion_matrix(y_test, predictions) # Confusion matrix
    
    
    # Gaussian Naive Bayes
    # Use support vector classification
    # classifier_sk = svm.SVC()
    classifier_sk = GaussianNB()
    # Use the train data to train this classifier
    classifier_sk.fit(X_train, y_train)
    # Use the trained model to predict on the test data
    predictions = classifier_sk.predict(X_test)
    score_g = metrics.accuracy_score(y_test, predictions)
    cm_g = confusion_matrix(y_test, predictions) # Confusion matrix
    
    # ------------------------------------------
    # Tensorflow Implementation
    # ------------------------------------------
    
    # Encode target labels with value between 0 and n_classes-1.
    le = LabelEncoder()
    y_train = le.fit_transform(y_train); y_train = to_categorical(y_train);
    y_test_cat = le.fit_transform(y_test); y_test_cat = to_categorical(y_test_cat);
    classes = le.classes_ # Get class order

    
    
    # extract the input and output shapes
    input_shape = X_train.shape[1] # Number of features
    output_shape = y_train.shape[1] # Number of classes
    
    # Create Model
    # Use 64 neurors
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_shape, activation='softmax'))
    
    # compiling model
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['acc'])
    
    # training the model for fixed number of iterations (epoches)
    model.fit(X_train, y_train, epochs=200)
    
    # Convert model to tensorflow lite
    converter = lite.TFLiteConverter.from_keras_model(model)
    tfmodel = converter.convert()
    outfile = model_name + ".tflite"
    open( str(outdir/outfile), 'wb').write(tfmodel)
    print("\n"+str(outfile) + " model saved to " + str(outdir/outfile))
    
    # Test model
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    y_pred = model.predict_classes(X_test) # Returns indicies
    predictions = np.array([classes[i] for i in y_pred])
    cm_tf = confusion_matrix(y_test, predictions) # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_tf,display_labels=classes)
    disp.plot()
    plt.show()
    
    
    # -------------------------------------------
    #    Gaussian Naive Bayes
    # -------------------------------------------
    
    # Define prior function (fraction of occurances)
    def prior(x):
        return x.count()/num_samples
    
    # Get number of features, classes
    # Group by labels, and compute mean, variance.
    num_samples = len(df[df.set=='train'])
    dfstats = df[df.set=='train'].groupby('label').agg(["mean", "var", "std", "count",prior])
    # Swap level headings
    dfstats.columns = dfstats.columns.swaplevel(0,1)
    # Extract means, variances, priors
    means = dfstats['mean'][features]
    variances = dfstats['var'][features]
    stds = dfstats['std'][features]
    # priors = dfstats['prior']['x0'] # Priors. Same for each feature
    priors = np.ones(len(means))/len(means)
    
    print('Parameters for Gaussian Naive Bayes classifier')
    print('\nMeans')
    print( np.array2string(means.to_numpy(),separator=',').replace('[','{').replace(']','}') )
    print('\nVariances')
    print( np.array2string(variances.to_numpy(),separator=',').replace('[','{').replace(']','}') )
    
    
    print('\nPriors')
    print( np.array2string(priors,separator=',').replace('[','{').replace(']','}') )
    
    
    
    # Print results
    print('\n\nTFLITE Results')
    print('--------------')
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    
    print('\n\nSKlearn Results')
    print('---------------')
    print('Sklearn SVM Accuracy: {0:f}'.format(score_svm))
    print('Sklearn Gauss Accuracy: {0:f}'.format(score_g))
    
    
    
    # Print results to txt
    outfile = model_name + '_results.txt'
    f = open(str(outdir/outfile), "a")
    f.write("Model: "+ model_name)
    f.write("\nClasses: "+ str(classes))
    f.write("\nFeatures: "+ str(features))
    
    f.write("\n\nTFLITE Results")
    f.write('\nTest loss:' + str(loss))
    f.write('\nTest accuracy:' + str(accuracy))
    
    f.write("\n\nSKlearn Results")
    f.write('\nTest accuracy SVM:' + str(score_svm))
    f.write('\nTest accuracy SVM:' + str(score_svm))
    
    
    f.write("\n\nSKlearn Gaussian Results")
    
    
    f.write("\n\nGaussian Parameters")
    f.write('\nMeans')
    f.write( np.array2string(means.to_numpy(),separator=',').replace('[','{').replace(']','}') )
    f.write('\nVariances')
    f.write( np.array2string(variances.to_numpy(),separator=',').replace('[','{').replace(']','}') )
    f.write('\nPriors')
    f.write( np.array2string(priors,separator=',').replace('[','{').replace(']','}') )
    
    f.close()
    
    return means

def load_training_data(data_dir, model_name):
    
    # Load data
    df = read_training_data(data_dir,model_name)
    
    # Define features of model
    features = ['phi0','phi1','phi2','phi3','phi4',
                'theta1','theta2','theta3','theta4',
                'TT1n','TT2n','TT3n','TT4n',
                'TB1n','TB2n','TB3n','TB4n',
                ]
    
    # Extract training data
    # X_train, X_test, y_train, y_test
    X_train = df[features][df.set == 'train'].to_numpy()
    y_train = df['label'][df.set == 'train'].to_numpy()
    
    # Extract test data
    X_test = df[features][df.set == 'test'].to_numpy()
    y_test = df['label'][df.set == 'test'].to_numpy()
    
    
    return X_train, y_train

#%% Feature Selection

def feature_selection(data_dir, model_name):
    
    # Adapted from: https://towardsdatascience.com/from-scikit-learn-to-tensorflow-part-1-9ee0b96d4c85
    
    # Create output directory
    outdir = data_dir.parents[0]/'tfModels'/model_name
    outdir.mkdir(parents=True, exist_ok=True)
    
    
    # Load data
    df = read_training_data(data_dir,model_name)
    
    # Define features of model
    features = ['phi0','phi1','phi2','phi3','phi4',
                'theta1','theta2','theta3','theta4',
                'TT1n','TT2n','TT3n','TT4n',
                'TB1n','TB2n','TB3n','TB4n',
                ]
    features = list(chain.from_iterable(('{}_xn'.format(i), '{}_yn'.format(i), '{}_zn'.format(i)) for i in range(21)))
    


#%% Main function

if __name__ == "__main__":
    
    data_dir = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data') # Locations of all csv data
    # model_name = "ASL0_9" # Done
    # model_name = "ASL1_9" # Done
    # model_name = "Hagrid_ASL0_9" # Done
    # model_name = "Hagrid_ASL1_9"
    
    # New versions (including palm orientation angle)
    model_name = "Hagrid_ASL0_9_v2" #
    
    # df = read_training_data(data_dir,model_name)
    means = train_model(data_dir, model_name)
    