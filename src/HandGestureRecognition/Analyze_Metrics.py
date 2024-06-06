# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:48:34 2024

@author: scott

Metrics Module
--------------

Process hand landmark data and compute additional metrics to better distinguish
between gestures.

"""

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

import os
from pathlib import Path
from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

import pdb

from ImageProcessing import *



#%% Analyse raw data

def pca_analysis(data_dir):
    
    # Feature selection methods
    # https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
    
    # Load in data asl
    df = read_dataset(data_dir,'27CSL')
    
    # Select list of features
    # features = list(chain.from_iterable(('{}_x'.format(i), '{}_y'.format(i), '{}_z'.format(i)) for i in range(21)))
    features = list(chain.from_iterable(('{}_xn'.format(i), '{}_yn'.format(i), '{}_zn'.format(i)) for i in range(21)))
    
    # Extract data
    Xfull = df[features].to_numpy() # Extract feature data
        
    # Principal components
    # Run PCA on all numeric orbital parameters
    n_components = len(features)
    pca = PCA(n_components)
    pca.fit(Xfull)
    PC = pca.transform(Xfull)
    
    # Variance explained by each PC
    pca.explained_variance_ratio_
    
    # Format feature importance into a dataframe
    labels = ['PC'+str(i+1) for i in range(n_components)]    
    dffeatimp = pd.DataFrame(pca.components_.T,columns=labels)
    dffeatimp.insert(0,'Feature',features)
    dffeatimp.set_index('Feature',inplace=True)
        
    # # Heatmap of feature importance
    # import seaborn as sns
    # sns.heatmap(dffeatimp.abs(), annot=True)
    
    return df

def plot_metric_histograms_matplotlib(data_dir):
    
    # Load in data
    df = read_dataset(data_dir,'27CSL')
    
    # metrics = ['TT1','TT2','TT3','TT4']
    # metrics = ['TT1n','TT2n','TT3n','TT4n']
    metrics = ['theta1','theta2','theta3','theta4']
    
    classes = ['0','1','2','3','4','5','6','7','8','9']
    anno_y_spacing = 5
    anno_y_offset = 25
    
    # Thumb to finger distances
    fig, ax = plt.subplots(len(metrics),1,figsize=(12, 10))
    fig.suptitle("Thumb to Finger Distances")
    fig.tight_layout(pad=5.0)
    
    for j,m in enumerate(metrics):
        # Finger 1
        ax[j].set_ylabel(m)
        ax[j].set_ylim([0,100]) 
        for i,c in enumerate(classes):
            mu = np.mean(df[m][df.label==c])
            std = np.std(df[m][df.label==c])
            ax[j].hist(df[m][df.label==c],label=c,alpha=0.2,bins=50)
            ax[j].annotate('', xy=(mu-3*std, -anno_y_offset  -anno_y_spacing*i),xytext=(mu+3*std, -anno_y_offset  -.09 -anno_y_spacing*i),                     #draws an arrow from one set of coordinates to the other
                arrowprops=dict(arrowstyle='<->',facecolor='red'),   #sets style of arrow and colour
                annotation_clip=False)
    
    # # Finger 2
    # ax[1,0].set_ylabel('TT2')
    # ax[1,0].set_ylim([0,100]) 
    # for i,c in enumerate(classes):
    #     mu = np.mean(df['TT2'][df.label==c])
    #     std = np.std(df['TT2'][df.label==c])
    #     ax[1,0].hist(df['TT2'][df.label==c],label=c,alpha=0.2,bins=50)
    #     ax[1,0].annotate('', xy=(mu-3*std, -anno_y_offset  -anno_y_spacing*i),xytext=(mu+3*std, -anno_y_offset  -.09 -anno_y_spacing*i),                     #draws an arrow from one set of coordinates to the other
    #         arrowprops=dict(arrowstyle='<->',facecolor='red'),   #sets style of arrow and colour
    #         annotation_clip=False)
    # # Finger 3
    # ax[2,0].set_ylabel('TT3')
    # ax[2,0].set_ylim([0,100]) 
    # for i, c in enumerate(classes):
    #     mu = np.mean(df['TT3'][df.label==c])
    #     std = np.std(df['TT3'][df.label==c])
    #     ax[2,0].hist(df['TT3'][df.label==c],label=c,alpha=0.2,bins=50)
    #     ax[2,0].annotate('', xy=(mu-3*std, -anno_y_offset  -anno_y_spacing*i),xytext=(mu+3*std, -anno_y_offset  -.09 -anno_y_spacing*i),                     #draws an arrow from one set of coordinates to the other
    #         arrowprops=dict(arrowstyle='<->',facecolor='red'),   #sets style of arrow and colour
    #         annotation_clip=False)
    # # Finger 4
    # ax[3,0].set_ylabel('TT4')
    # ax[3,0].set_ylim([0,100]) 
    # for i,c in enumerate(classes):
    #     mu = np.mean(df['TT4'][df.label==c])
    #     std = np.std(df['TT4'][df.label==c])
    #     ax[3,0].hist(df['TT4'][df.label==c],label=c,alpha=0.2,bins=50)
    #     ax[3,0].annotate('', xy=(mu-3*std, -anno_y_offset  -anno_y_spacing*i),xytext=(mu+3*std, -anno_y_offset  -.09 -anno_y_spacing*i),                     #draws an arrow from one set of coordinates to the other
    #         arrowprops=dict(arrowstyle='<->',facecolor='red'),   #sets style of arrow and colour
    #         annotation_clip=False)
    
    # # 2nd Column: normalized
    # # Finger 1
    # ax[0,1].set_ylabel('TT1n')
    # ax[0,1].set_ylim([0,100]) 
    # for i,c in enumerate(classes):
    #     mu = np.mean(df['TT1n'][df.label==c])
    #     std = np.std(df['TT1n'][df.label==c])
    #     ax[0,1].hist(df['TT1n'][df.label==c],label=c,alpha=0.2,bins=50)
    #     ax[0,1].annotate('', xy=(mu-3*std, -anno_y_offset  -anno_y_spacing*i),xytext=(mu+3*std, -anno_y_offset  -.09 -anno_y_spacing*i),                     #draws an arrow from one set of coordinates to the other
    #         arrowprops=dict(arrowstyle='<->',facecolor='red'),   #sets style of arrow and colour
    #         annotation_clip=False)
    # # Finger 2
    # ax[1,1].set_ylabel('TT2n') 
    # ax[1,1].set_ylim([0,100]) 
    # for i,c in enumerate(classes):
    #     mu = np.mean(df['TT2n'][df.label==c])
    #     std = np.std(df['TT2n'][df.label==c])
    #     ax[1,1].hist(df['TT2n'][df.label==c],label=c,alpha=0.2,bins=50)
    #     ax[1,1].annotate('', xy=(mu-3*std, -anno_y_offset  -anno_y_spacing*i),xytext=(mu+3*std, -anno_y_offset  -.09 -anno_y_spacing*i),                     #draws an arrow from one set of coordinates to the other
    #         arrowprops=dict(arrowstyle='<->',facecolor='red'),   #sets style of arrow and colour
    #         annotation_clip=False)
    # # Finger 3
    # ax[2,1].set_ylabel('TT3n')
    # ax[2,1].set_ylim([0,100]) 
    # for i,c in enumerate(classes):
    #     mu = np.mean(df['TT3n'][df.label==c])
    #     std = np.std(df['TT3n'][df.label==c])
    #     ax[2,1].hist(df['TT3n'][df.label==c],label=c,alpha=0.2,bins=100)
    #     ax[2,1].annotate('', xy=(mu-3*std, -anno_y_offset  -anno_y_spacing*i),xytext=(mu+3*std, -anno_y_offset  -.09 -anno_y_spacing*i),                     #draws an arrow from one set of coordinates to the other
    #         arrowprops=dict(arrowstyle='<->',facecolor='red'),   #sets style of arrow and colour
    #         annotation_clip=False)
    # # Finger 3
    # ax[3,1].set_ylabel('TT4n') 
    # ax[3,1].set_ylim([0,100]) 
    # for i,c in enumerate(classes):
    #     mu = np.mean(df['TT4n'][df.label==c])
    #     std = np.std(df['TT4n'][df.label==c])
    #     ax[3,1].hist(df['TT4n'][df.label==c],label=c,alpha=0.2,bins=50)
    #     ax[3,1].annotate('', xy=(mu-3*std, -anno_y_offset  -anno_y_spacing*i),xytext=(mu+3*std, -anno_y_offset  -.09 -anno_y_spacing*i),                     #draws an arrow from one set of coordinates to the other
    #         arrowprops=dict(arrowstyle='<->',facecolor='red'),   #sets style of arrow and colour
    #         annotation_clip=False)
        
    plt.show()
        
    return

def feature_selection(data_dir,model_name):
    
    # Feature selection
    # https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
    
    # 1. Univariate feature selection
    # - consider each feature individually
    # - does not account for linear dependencies between features
    # Numerical Input, Categorical Output: 
    # - ANOVA correlation coefficient (linear). f_classif()
    # - Kendall’s rank coefficient (nonlinear). kendalltau
    
    
    
    # Classes
    if model_name.lower() == 'asl0_9':
        classes = ['0','1','2','3','4','5','6','7','8','9']
    elif model_name.lower() == 'asl1_9':
        classes = ['1','2','3','4','5','6','7','8','9']
    elif model_name.lower() == 'hagrid_asl0_9':
        classes = ['0','1','2','3','4','5','6','7','8','9','fist','like','dislike','stop','stop_inverted','two_up','two_up_inverted','call','rock','peace_inverted','three2','mute','ok']
    elif model_name.lower() == 'hagrid_asl1_9':
        classes = ['1','2','3','4','5','6','7','8','9','fist','like','dislike','stop','stop_inverted','two_up','two_up_inverted','call','rock','peace_inverted','three2','mute','ok']
    
    
    features = ['phi0','phi1','phi2','phi3','phi4',
                'theta1','theta2','theta3','theta4',
                'TT1n','TT2n','TT3n','TT4n',
                'TB1n','TB2n','TB3n','TB4n',
                'D12n','D23n','D34n',
                'alpha',
                ]
    
    # # Select features
    # # features = list(chain.from_iterable(('{}_x'.format(i), '{}_y'.format(i), '{}_z'.format(i)) for i in range(21)))
    # features = list(chain.from_iterable(('{}_xn'.format(i), '{}_yn'.format(i), '{}_zn'.format(i)) for i in range(21)))
    
    
    # Load data
    df = read_training_data(data_dir,model_name)
    
    # Extract label and feature data
    y_train = df['label'][df.set == 'train'].to_numpy()
    X_train = df[features][df.set == 'train'].to_numpy()
    
    
    # ANOVA feature selection for numeric input and categorical output
    from sklearn.datasets import make_classification
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    
    # define feature selection
    # See: https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-py
    k = len(features)
    # k=5
    selector = SelectKBest(f_classif, k=k)
    # apply feature selection
    # X_selected = selector.fit_transform(X_train, y_train)
    selector.fit(X_train, y_train)
    
    # Scores
    scores = selector.scores_
    freq_series = pd.Series(data=scores)
    
    # Get indicies of selected features
    selected_ind = selector.get_support(indices=True)
    selected_features = [features[i] for i in selected_ind]
    
    # Plot
    fig, ax = plt.subplots(1,1,figsize=(18, 10))
    # ax.bar(np.arange(X_train.shape[-1]) - 0.05, scores, width=0.2)
    freq_series.plot(kind='bar')
    ax.set_title("Feature univariate selection")
    # ax.set_xlabel("Feature variable")
    plt.xlabel("Feature variable", labelpad=70,fontsize=16);
    plt.ylabel("Univariate score", fontsize=16);
    ax.set_xticklabels(features)
    draw_brace(ax, (0,4),250.0, 'Knuckle angles')
    draw_brace(ax, (5,8),250.0, 'Finger elevations')
    draw_brace(ax, (9,12),250.0, 'Thumb to fingertip')
    draw_brace(ax, (13,16),250.0, 'Thumb to base of finger')
    draw_brace(ax, (17,19),250.0, 'Fingertip to fingertip')
    plt.tight_layout()
    plt.show()
    
    
    return selector

def draw_brace(ax, xspan, yy, text):
    """Draws an annotated brace outside the axes."""
    # https://stackoverflow.com/questions/18386210/annotating-ranges-of-data
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, -y, color='black', lw=1, clip_on=False)

    ax.text((xmax+xmin)/2., -yy-.10*yspan, text, ha='center', va='bottom')



#%% Main script

if __name__ == "__main__":
    
    # Define image directories
    img_dir_asl = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data\asl_dataset')
    img_dir_27CSL = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data\27 Class Sign Language Dataset')
    img_dir_SLFN = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data\Sign Language for Numbers')
    img_dir_hagrid = Path(r'E:\Files\Zero Robotics\HandGuestures\mmpretrain\data\hagrid\hagrid_dataset_512')
    
    # Read in dataset
    data_dir = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data') # Locations of all csv data
    dfasl = read_dataset(data_dir,'ASL')
    # df27CSL = read_dataset(data_dir,'27CSL')
    # dfSLFN = read_dataset(data_dir,'SLFN')
    # dfhagrid = read_dataset(data_dir,'hagrid')
    
    # df = read_training_data(data_dir,'Hagrid_ASL0_9')
    
    # # Compute palm centered data
    # dfn = compute_palm_centered_landmarks(dfasl)
    
    # Inspect images
    # inspect_images(img_dir_asl,dfasl, classes=['5'])     # ASL
    # inspect_images(img_dir_27CSL,df27CSL, classes=['5']) # 27CSL
    # inspect_images(img_dir_SLFN,dfSLFN, classes=['5'])   # SLFN
    
    # inspect_images(img_dir_hagrid,dfhagrid, classes=['palm']) # Hagrid
    
    # plot_single_hand_from_df(img_dir_asl,dfasl, classes=['0'], ind=0)
    # plot_single_hand_from_df(img_dir_27CSL,df27CSL, classes=['8'], ind=10)
    # plot_single_hand_from_df(img_dir_27CSL,df27CSL, ind=3548) # Test case: 4_537.jpg '4' at non-direct angle
    
    # Run pca on hagrid dataset
    # pca_analysis(data_dir)

    # # Plot histograms
    # plot_metric_histograms_matplotlib(data_dir)
    
    # # Feature selection
    # selector = feature_selection(data_dir,'ASL0_9')
    selector = feature_selection(data_dir,'Hagrid_ASL0_9')
  
    
    
