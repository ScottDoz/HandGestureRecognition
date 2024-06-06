"""
Image Processing
----------------

This module processes images and extracts hand landmarks to save to file.

"""

import cv2
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
from itertools import chain
from natsort import natsort_keygen

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# from matplotlib_venn import venn3, venn3_circles

from PIL import Image

try:
    import networkx as nx
except:
    pass

try:
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
except:
    pass

import pdb

# from Visualization import *

#%% Read Datasets

def read_dataset(data_dir,name,metrics=True,prune=True):
    
    # Individual datasets
    if name.lower() == 'asl':
        # ASL dataset
        df = pd.read_csv(str(data_dir/'hand_landmarks_asl_dataset.csv'))
    elif name.lower() == '27csl':
        # 27 Class Sign Language dataset
        df = pd.read_csv(str(data_dir/'hand_landmarks_27_class_sign_language_dataset.csv'))
        # FIXME: remove image 4_223.jpg (incorrect label)
    elif name.lower() == 'slfn':
        # Sign Language for Numbers dataset
        df = pd.read_csv(str(data_dir/'hand_landmarks_sign_language_for_numbers.csv'))
    elif name.lower() == 'hagrid':
        # Hagrid dataset
        df = pd.read_csv(str(data_dir/'hand_landmarks_hagrid.csv'))
        
    # Data prunning
    if prune:
        if name.lower() in ['asl','27csl','slfn']:
            # Images in these datasets all contain single right hand
            # - Remove all images with mutiple hands (num_hands > 1)
            # - Remove all images predicting left hand (handeness = 'left')
            df = df[df.num_hands == 1]
            df = df[df.handedness == 'Right']
        
        elif name.lower() == 'hagrid':
            # Hagrid dataset contains mixture of hands.
            
            # # Limit to hagrid annotated data
            # # TODO: complete hagrid_merge_annotation to filter by bboxes
            # df = hagrid_merge_annotation(data_dir)
            
            # Split off the image name as an image id
            df['image_id'] = df.filename.str.split("\\",expand=True)[2]
            df['image_id'] = df.image_id.str.split(".",expand=True)[0]
            
            # Merge in annotation data
            dfj = load_hagrid_annotations(data_dir)
            dfj.rename(columns = {"num_hands":"num_hands_annotated"}, inplace=True)
            dfj.sort_values(by='image_id',inplace=True)
            
            # Merge data
            df = pd.merge(df,dfj, how='left',on='image_id')
            
            # Remove annotated data with mutliple hands (for now)
            df = df[df.num_hands_annotated == 1]
            
            # Remove data with multiple detected hands
            df = df[df.num_hands == 1]
        
       
    # Convert labels to strings
    df['label'] = df['label'].astype(str)  
    df.label = df.label.str.strip()
    
    # Strip spaces from labels
    df.columns = [l.strip() for l in list(df.columns)]
    
    # Sort
    df = df.sort_values(by="filename",key=natsort_keygen()).reset_index(drop=True)
    
    # Add normalized points + metrics
    if metrics:
        # Palm-centered landmarks
        df = compute_palm_centered_landmarks(df)
        # Metrics
        df = compute_metrics(df)
    
    
    # TODO: Specific combinations of datasets for training models
    # # Group dataframe by filename and  count hands
    # dfg = df.groupby(['filename'])['handedness'].count()
    
    # # Split dataframe into images containing
    # # 1) Single hand
    # # 2) Multiple hands
    # mask = df.filename.duplicated(keep=False)
    # df1 = df[~mask] # Single hand
    # df2 = df[mask]  # Multiple hands
    
    # df1 = df1.head() # Limit rows for testing
    
    return df

def read_training_data(data_dir,model_name,metrics=True,prune=True):
    '''
    Prepare train/test/validation data for different models.
    Use 70/20/10 split.
    
    Each model has different sets of classes and different combinations of 
    datasets (ASL, 27CSL, Hagrid)
    Models:
        ASL0_9
        ASL1_9
        Hagrid_ASL0_9
        Hagrid_ASL1_9
    

    Parameters
    ----------
    data_dir : TYPE
        DESCRIPTION.
    model_name : TYPE
        DESCRIPTION.
    metrics : TYPE, optional
        DESCRIPTION. The default is True.
    prune : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    # Individual models
    
    # Classes: ASL digits 0 to 9
    # Datasets: ASL, 27CSL
    
    if model_name.lower() in ['asl0_9','asl0_9_v2']:
        classes = ['0','1','2','3','4','5','6','7','8','9']
    elif model_name.lower() in ['asl1_9','asl1_9_v2']:
        classes = ['1','2','3','4','5','6','7','8','9']
    elif model_name.lower() in ['hagrid_asl0_9','hagrid_asl0_9_v2']:
        classes = ['0','1','2','3','4','5','6','7','8','9','fist','like','dislike','stop','stop_inverted','two_up','two_up_inverted','call','rock','peace_inverted','three2','mute','ok']
    elif model_name.lower() in ['hagrid_asl1_9','hagrid_asl1_9_v2']:
        classes = ['1','2','3','4','5','6','7','8','9','fist','like','dislike','stop','stop_inverted','two_up','two_up_inverted','call','rock','peace_inverted','three2','mute','ok']
    
    # Load ASL
    dfasl = read_dataset(data_dir,'ASL')     # 2134 images
    dfasl = dfasl[dfasl.label.isin(classes)].reset_index(drop=True) # Limit classes
    # Training/testing/validation 70%/20%/10%
    dfasl['set'] = '' # Create new colum to specify train/test/val
    train, testval = train_test_split(dfasl, test_size=0.3,random_state=1) # Split training data (70% / 30%)
    test, val = train_test_split(testval, test_size=0.333,random_state=1) # Split (66%/33%)
    # train: 418, test: 120, val: 60. Total: 598
    dfasl['set'].loc[train.index] = 'train'
    dfasl['set'].loc[test.index] = 'test'
    dfasl['set'].loc[val.index] = 'val'
    
    # Load 27CSL
    df27CSL = read_dataset(data_dir,'27CSL') # 15698 images
    df27CSL = df27CSL[df27CSL.label.isin(classes)].reset_index(drop=True) # Limit classes
    # Training/testing/validation 70%/20%/10%
    df27CSL['set'] = '' # Create new colum to specify train/test/val
    train, testval = train_test_split(df27CSL, test_size=0.3,random_state=1) # Split training data (70% / 30%)
    test, val = train_test_split(testval, test_size=0.333,random_state=1) # Split (66%/33%)
    # train: 418, test: 120, val: 60. Total: 598
    df27CSL['set'].loc[train.index] = 'train'
    df27CSL['set'].loc[test.index] = 'test'
    df27CSL['set'].loc[val.index] = 'val'
    
    # Combine dataframes
    df = pd.concat([dfasl,df27CSL]).reset_index(drop=True)
    
    if model_name.lower() in ['hagrid_asl0_9','hagrid_asl0_9_v2','hagrid_asl1_9','hagrid_asl1_9_v2']:
        
        # Hagrid dataset
        dfhagrid = read_dataset(data_dir, 'Hagrid')
        dfhagrid = dfhagrid[dfhagrid.label.isin(classes)].reset_index(drop=True) # Limit classes
        # Remove extra columns from hagrid dataset
        dfhagrid = dfhagrid[dfasl.columns]
        
        # Combine dataframes
        df = pd.concat([df,dfhagrid]).reset_index(drop=True)
    
    return df
    
    
    


#%% File listing

def list_images(base_dir,classes=None):
    '''
    List all files and labels in a labeled dataset.
    Dataset is organized with subfolders containing images of individual classes.

    Parameters
    ----------
    base_dir : Path
        Base directory of the dataset containing sub-folders with class data.
    classes : List, optional
        List of classes. The default is None. If none, will be generatd from
        list of subfolders.

    Returns
    -------
    files : List
        List of full paths of all images.
    labels : List
        List of labels matching the images.

    '''

    # ASL Numbers dataset
    # base_dir = Path(r'F:\Files\Zero Robotics\HandGuestures\mediapipe\Training Data\asl_dataset')
    
    
    if classes is None:
        # List classes from names
        p = base_dir.glob('**/*')
        classes = [x.name for x in p if x.is_dir()]
    
    # Loop through image sets of each integer
    files = []
    labels = []
    for i in classes:
        dir_ = base_dir/str(i) # Imageset directory
        p = dir_.glob('**/*')
        files_i = [x for x in p if x.is_file()] # Filenames
        labels_i = [str(i)]*len(files_i)             # Labels (strings)
        labels += labels_i
        files += files_i

    return files, labels

#%% 27 Class Dataset

def process_27_Class_Dataset(base_dir):
    '''
    The 27 Class Sign Language Dataset contains images saved in raw .npy files.
    This function reads in the data from X.npy and labels from Y.npy.
    Each image is saved as individual jpg files in sub-folders of each class. 

    Data:
    Download raw data from: https://www.kaggle.com/datasets/ardamavi/27-class-sign-language-dataset?resource=download
    Extract zip file and rename to 27 Class Dataset

    Usage:
    Define directory of dataset
    >> base_dir = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data\27 Class Sign Language Dataset')
    >> process_27_Class_Dataset(base_dir)

    '''

    # Read in Y data (22801x1 array)
    Y = np.load(str(base_dir/'Y.npy'))
    # Read in X data (2801x128x128x3 array)
    X = np.load(str(base_dir/'X.npy'))


    # # FIXME: Save sample of X, Y for testing
    # X1 = X[:5,:,:,:]
    # Y1 = Y[:5]
    # np.save('X1.npy',X1)
    # np.save('Y1.npy',Y1)

    # Loop though images and extract
    prev_label = "" # Initialize previous label
    counter = 0
    for i in tqdm(range(len(Y))):
        # Extract image and label
        Xi = X[i,:,:,:] # Image
        labeli = str(Y[i][0])   # Label
        if labeli != prev_label:
            # Next set of labels. Reset counter
            counter = 0
            prev_label = labeli # Update

        # Create filename
        outdir = base_dir/labeli
        filename = labeli + "_" + str(counter) + ".jpg"
        fullfilename = outdir/filename

        # Create directory if doesnt exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Save image
        img = Image.fromarray(np.uint8(Xi*255), 'RGB') # Binary data to pixels
        img.save(fullfilename)


        # Increment counter
        counter += 1
        

    return

#%% Process images

def extract_hand_landmarks(base_dir,classes=None,out_dir=None,flip=False):
    '''
    Extract and save hand landmark data from dataset. 
    Each row contians the x,y,z, coordiantes of the 21 hand landmarks, along with the name of the image and the label.
    If multiple hands are detected in an image, they will both be saved.

    Example usage:
    Process the asl_dataset
    >> base_dir = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data\asl_dataset')
    >> extract_hand_landmarks(base_dir)
    A file named hand_landmarks.csv will be generated. Sample data:

    filename	 label	 handedness	0_x	 0_y	 0_z 	1_x	 1_y	 1_z 	2_x	 2_y	 2_z ...
    asl_dataset\0\hand1_0_bot_seg_1_cropped.jpeg	0	 Right	0.562653	0.634078	-9.2E-05	0.391319	0.524487	0.097748	0.32939	0.393489	0.104387
    asl_dataset\0\hand1_0_bot_seg_2_cropped.jpeg	0	 Right	0.548968	0.63699	5.1E-05	0.387985	0.54779	0.059822	0.31749	0.405543	0.051151
    asl_dataset\0\hand1_0_bot_seg_3_cropped.jpeg	0	 Right	0.541317	0.64597	-2.9E-05	0.3871	0.572246	0.010989	0.303467	0.414425	0.008751
    asl_dataset\0\hand1_0_bot_seg_4_cropped.jpeg	0	 Right	0.613558	0.660806	-0.00015	0.424134	0.58702	0.103826	0.331559	0.475439	0.100435
    asl_dataset\0\hand1_0_bot_seg_5_cropped.jpeg	0	 Right	0.564115	0.684686	-7.5E-05	0.413257	0.60465	0.070392	0.336177	0.452162	0.086764
    asl_dataset\0\hand1_0_dif_seg_1_cropped.jpeg	0	 Right	0.52341	0.714871	-0.000202	0.405447	0.601177	0.123745	0.373818	0.443053	0.105641

    Best pracitce to rename the hand_landmarks.csv file to specify dataset.

    Parameters
    ----------
    base_dir : Path
        Base directory of the dataset containing sub-folders with class data.
    classes : List, optional
        List of classes. The default is None. If none, will be generatd from
        list of subfolders.
    
    '''

    # List files and labels
    files,labels = list_images(base_dir,classes=classes)
    
    if classes is not None:
        print("Subset of classes: " + str(classes) ,flush=True)
    
    # Create output file
    if out_dir is None:
        # Save data into the base_dir
        out_dir = base_dir
    
    # Create csv file to write to
    f = open(str(out_dir/'hand_landmarks.csv'),'w+')
    # Write header
    ptlabels = ['{}_x, {}_y, {}_z'.format(i,i,i) for i in range(21)]
    ptlabels = ",".join(['{}_x,{}_y,{}_z'.format(i,i,i) for i in range(21)])
    f.write('filename,label,num_hands,handedness,score,'+ptlabels+'\n')

    # Create empty array for training data
    X = np.zeros((len(labels),17))

    # if show:
    #     plotter,actors,actorsb = initialize_3dplot() # Instantiate 3d plot

    # Initialize hands
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        
        # Loop through images
        for i,file in enumerate(tqdm(files)):
            
            # Read file
            frame = cv2.imread(str(file))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR 2 RGB
            if flip:
                # Flip image horzontally. Useful for webcam images.
                image = cv2.flip(image, 1) # Flip on horizontal. 
            image.flags.writeable = False # Set flag
            
            # Detections
            results = hands.process(image)
            
            # Set flag to true
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGB 2 BGR
            height, width, _ = image.shape # Get image size
                    
            # Detections
            # print(results)
            
            # Rendering results
            if results.multi_hand_landmarks:
                # Get number of hands in image
                num_hands = len(results.multi_hand_landmarks)

                if num_hands == 1:
                    # Single hand detected

                    # Determine if "Left" or "Right" hand
                    hand = results.multi_hand_landmarks[0]
                    handedness = results.multi_handedness[0].classification[0].label
                    score = results.multi_handedness[0].classification[0].score
                    
                    if flip==False:
                        # Handedness is labeled as if image is taken from webcam.
                        # Need to swap handedness if not flipping image.
                        if handedness == 'Right':
                            handedness = 'Left'
                        elif handedness == 'Left':
                            handedness = 'Right'
                    
                    # Get position values of key landmarks
                    # See here for abreviations: https://blog.missiondata.com/lab-notes-mediapipe-with-python-for-gesture-recognition/
                    
                    # Extract all points as 21x3 array
                    points = np.vstack([(float(hand.landmark[i].x),float(hand.landmark[i].y),float(hand.landmark[i].z)) for i in range(21)])  

                    # Prepare row entry and write to csv
                    # Filename, label, handedness, points
                    s = ",".join(np.char.mod('%f', points.flatten())) # Point coords as string
                    # filename,label,num_hands,handedness,score
                    txt = str(Path(*Path(file).parts[-3:])) + ',' + str(labels[i]) + ',' + str(num_hands) + ',' + handedness  + ',' + str(score) + ',' + s + '\n'
                    f.write(txt) #Give your csv text here.


                # Check for two of same hand
                if num_hands>1:
                    if results.multi_handedness[0].classification[0].label == results.multi_handedness[1].classification[0].label:
                        # Two hands detected, both the same. 
                        # Select result with highest confidence.
                        flag = 'two_same_hands'
                        ind = np.argmax(np.array([results.multi_handedness[0].classification[0].score,results.multi_handedness[1].classification[0].score]))
                        
                        # Determine if "Left" or "Right" hand
                        hand = results.multi_hand_landmarks[ind]
                        handedness = results.multi_handedness[ind].classification[0].label
                        score = results.multi_handedness[ind].classification[0].score
                        
                        if flip==False:
                            # Handedness is labeled as if image is taken from webcam.
                            # Need to swap handedness if not flipping image.
                            if handedness == 'Right':
                                handedness = 'Left'
                            elif handedness == 'Left':
                                handedness = 'Right'
                        
                        # Get position values of key landmarks
                        # See here forp abreviations: https://blog.missiondata.com/lab-notes-mediapipe-with-python-for-gesture-recognition/
                        
                        # Extract all points as 21x3 array
                        points = np.vstack([(float(hand.landmark[i].x),float(hand.landmark[i].y),float(hand.landmark[i].z)) for i in range(21)])  
                        
                        # Prepare row entry and write to csv
                        # Filename, label, handedness, points
                        s = ",".join(np.char.mod('%f', points.flatten())) # Point coords as string
                        txt = str(Path(*Path(file).parts[-3:])) + ',' + str(labels[i]) + ',' + str(num_hands) + ',' + handedness  + ',' + str(score) + ',' + s + '\n'
                        f.write(txt) #Give your csv text here.

                    else:
                        # Right and Left hands. Store both separatly

                        # Loop though hands
                        for num, hand in enumerate(results.multi_hand_landmarks):

                            # Determine if "Left" or "Right" hand
                            handedness = results.multi_handedness[num].classification[0].label
                            score = results.multi_handedness[num].classification[0].score
                            
                            if flip==False:
                                # Handedness is labeled as if image is taken from webcam.
                                # Need to swap handedness if not flipping image.
                                if handedness == 'Right':
                                    handedness = 'Left'
                                elif handedness == 'Left':
                                    handedness = 'Right'
                            
                            # Get position values of key landmarks
                            # See here forp abreviations: https://blog.missiondata.com/lab-notes-mediapipe-with-python-for-gesture-recognition/
                            
                            # Extract all points as 21x3 array
                            points = np.vstack([(float(hand.landmark[i].x),float(hand.landmark[i].y),float(hand.landmark[i].z)) for i in range(21)])  

                            # Prepare row entry and write to csv
                            # Filename, label, handedness, points
                            s = ",".join(np.char.mod('%f', points.flatten())) # Point coords as string
                            txt = str(Path(*Path(file).parts[-3:])) + ',' + str(labels[i]) + ',' + str(num_hands) + ',' + handedness  + ',' + str(score) + ',' + s + '\n'
                            f.write(txt) #Give your csv text here.


        # End loop
        f.close()


    # Save training data
    #df = pd.DataFrame(X,columns=["x{}".format(i) for i in range(X.shape[1])])
    #df['y'] = labels
    #dir_ = Path(os.path.abspath(__file__)).parent # Directory of this python script
    #df.to_csv(str(dir_/'training.csv'),index=False) # Save to csv

    
    return

def combine_results(base_dir,files):
    '''
    Combine the results of multiple hand_landmarks.csv files.
    
    For large datasets such as Hagrid, it is convenient to run the extract_hand_landmarks
    method in chunks for subsets of classes, while renaming the output files.
    This function combines a list of hand_landmarks.csv files in the base directory
    into a single csv file.
    
    Example: Combine the output files from the Hagrid database.
    >> base_dir = Path(r'E:\Files\Zero Robotics\HandGuestures\mmpretrain\data\hagrid\hagrid_dataset_512')
    >> files = ['hand_landmarks_call.csv','hand_landmarks_dislike.csv','hand_landmarks_fist.csv',
                'hand_landmarks_four.csv','hand_landmarks_mute.csv','hand_landmarks_ok.csv',
                'hand_landmarks_one.csv','hand_landmarks_palm.csv','hand_landmarks_peace.csv',
                'hand_landmarks_peace_inverted.csv','hand_landmarks_rock.csv','hand_landmarks_stop.csv',
                'hand_landmarks_stop_inverted.csv','hand_landmarks_three.csv','hand_landmarks_three2.csv',
                'hand_landmarks_two_up.csv','hand_landmarks_two_up_inverted.csv',
                ]
    >> combine_results(base_dir,files)
    
    
    Parameters
    ----------
    base_dir : Path
        Base directory of the dataset containing sub-folders with class data.
    files : list
        List of individual hand_landmarks.csv files to combine.

    Returns
    -------
    None.

    '''
    
    # Loop through files
    for i,f in tqdm(enumerate(files)):
        
        # Read in this file
        dfi = pd.read_csv(str(base_dir/f))
        
        if i==0:
            # Initialize on first instance
            df = dfi
        else:
            # Append
            df = pd.concat([df,dfi])
            
    # Save file
    df = df.reset_index(drop=True)
    df.to_csv(str(base_dir/'hand_landmarks.csv'),index=False)
    
    return df

def process_hagrid_annotation_data(base_dir):
    '''
    Process hagrid annotation data and store in a csv.
    Additional annotiaton data are stored in individual json files.
    Save this data as a single csv. This can be merged into the hagrid
    handlandmarks data to filter results.
    
    Additional data includes:
        bounding box of featured hand
        test/train/validation label

    Parameters
    ----------
    base_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    # # Read in data without processing
    # df = read_dataset(base_dir,'hagrid',metrics=False)
    # # Split off the image name as an image id
    # df['image_id'] = df.filename.str.split("\\",expand=True)[2]
    # df['image_id'] = df.image_id.str.split(".",expand=True)[0]
    
    # List of classes
    classes = ['fist','like','dislike','stop','stop_inverted',
               'two_up','two_up_inverted',
               'call','rock','peace_inverted','three2','mute','ok',
               'one','peace','three','four','palm']
    # classes = ['palm']
    
    
    dfj = pd.DataFrame() # Initialize dataframe of annotated data
    for c in tqdm(classes):
        
        # s0 = set(df['image_id'][df.label == c])
        
        # Load json data (tests)
        fname = c + ".json"
        json_file = base_dir/'annotations'/'test'/fname
        with open(str(json_file)) as data_file:    
            data = json.load(data_file) 
        dfj1 = pd.DataFrame.from_dict(data, orient='index')
        dfj1['image_id'] = dfj1.index
        dfj1 = dfj1.reset_index(drop=True)
        dfj1 = dfj1[['image_id','labels','bboxes','user_id']] # Reorder columns
        dfj1['set'] = 'test'
        s_test = set(dfj1.image_id) # Set of images
        dfj = pd.concat([dfj,dfj1]) # Add to dataframe
        
        # Load json data (train)
        json_file = base_dir/'annotations'/'train'/fname
        with open(str(json_file)) as data_file:    
            data = json.load(data_file) 
        dfj2 = pd.DataFrame.from_dict(data, orient='index')
        dfj2['image_id'] = dfj2.index
        dfj2 = dfj2.reset_index(drop=True)
        dfj2 = dfj2[['image_id','labels','bboxes','user_id']] # Reorder columns
        dfj2['set'] = 'train'
        s_train = set(dfj2.image_id) # Set of images
        dfj = pd.concat([dfj,dfj2]) # Add to dataframe
        
        # Load json data (val)
        json_file = base_dir/'annotations'/'val'/fname
        with open(str(json_file)) as data_file:    
            data = json.load(data_file)
        dfj3 = pd.DataFrame.from_dict(data, orient='index')
        dfj3['image_id'] = dfj3.index
        dfj3 = dfj3.reset_index(drop=True)
        dfj3 = dfj3[['image_id','labels','bboxes','user_id']] # Reorder columns
        dfj3['set'] = 'val'
        s_val = set(dfj3.image_id) # Set of images
        dfj = pd.concat([dfj,dfj3]) # Add to dataframe
        
    # Count number of hands
    dfj['num_hands'] = dfj.labels.apply(len)
    
    # Save dataframe
    dfj.to_csv(str(base_dir/'hagrid_annotations.csv'),sep = ';',index=False)
    
    # # TODO: Process bbox to extract limiting pixel coords
    # bboxes = dfj['bboxes'].astype(str) # Convert to string
    # dfj['bboxes'].astype(str).str[2:-2].str.split(",",expand=True)

    
    return

def load_hagrid_annotations(data_dir):
    
    df = pd.read_csv(str(data_dir/'hagrid_annotations.csv'),sep=';')
    
    return df

#%% New
    
#%% Transformations

def compute_palm_centered_landmarks(df):
    '''
    Transform coordinates of hand landmarks from real-world coords to a 
    palm-centered reference frame.
    
    This version of the function works on all images in the test/training data
    at once. Similar to function "" that works on a single hand.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    normalize_to = 'palm_bisector' # Normalize bisector of palm
    # normalize_to = 'wrist_to_index' # Normalize wrist to index vector
    
    # Extract points to 2d array
    cols = list(chain.from_iterable(('{}_x'.format(i), '{}_y'.format(i), '{}_z'.format(i)) for i in range(21)))
    x_cols = ["{}_x".format(i) for i in range(21)]
    y_cols = ["{}_y".format(i) for i in range(21)]
    z_cols = ["{}_z".format(i) for i in range(21)]
    landmarks = df[cols].to_numpy()
    
    # Extract key points of the palm
    p1 = df[['0_x','0_y','0_z']].to_numpy()    # Wrist
    p2 = df[['5_x','5_y','5_z']].to_numpy()    # Index finger MCP (Metacarpophalangeal Joint)
    p3 = df[['17_x','17_y','17_z']].to_numpy() # Pinky finger MCP (Metacarpophalangeal Joint)
    
    # Define unit vectors and their length
    r1mag = np.linalg.norm(p2-p1,axis=-1) # Wrist to Index MCP p1 to p2
    r2mag = np.linalg.norm(p3-p1,axis=-1) # Wrist to Pinky MCP p1 to p3
    r3mag = np.linalg.norm(p3-p2,axis=-1) # Index MCP to Pinky MCP p2 to p3
    r1u = (p2-p1)/r1mag[:, np.newaxis] # Wrist to Index MCP unit vector
    r2u = (p3-p1)/r2mag[:, np.newaxis] # Wrist to Pinky MCO unit vector
    
    # Length of palm (bisector r1u,r2u)
    l2 = (r1mag*r2mag/(r1mag+r2mag)**2)*( (r1mag+r2mag)**2 - r3mag**2) # Length of bisector angle
    l = np.sqrt(l2)
    if normalize_to == 'palm_bisector':
        palm_length = l # Define this as the palm length
    elif normalize_to == 'wrist_to_index':
        palm_length = r1mag # FIXME: changed for testing
        
    # Find left hands
    left_flag = df.handedness=='Left' # Boolean flag 
    left_ind = df.index[df.handedness=='Left'].tolist()
    
    # Define Basis vectors for palm plan
    # Note: difference in basis in left and right hand
    #
    # b2 vector along y direction (ambidexterous)
    if normalize_to == 'palm_bisector':
        b2 = (r1u + r2u)/np.linalg.norm(r1u+r2u,axis=-1)[:, np.newaxis] # Bisector of r1, r2 ****
    elif normalize_to == 'wrist_to_index':
        b2 = r1u #  along r1 vector for now
    # b3 direction normal to palm. Dependent on handedness
    b3 = np.cross(r1u,r2u); # Initialize for right hand 
    b3l = np.cross(r2u,r1u); b3[left_ind,:] = b3l[left_ind,:] # Replace for left hand
    b3 = b3/np.linalg.norm(b3,axis=-1)[:, np.newaxis] # Normalize
    
    # b1 completes the set
    b1 = np.cross(b2,b3); b1 = b1/np.linalg.norm(b1,axis=-1)[:, np.newaxis] # Complete the pair
    
    # Confirm three basis vectors are orthogonal
    assert np.around(max(abs(np.einsum('ij,ij->i',b1,b2))),decimals=12) == 0. # dot(b1,b2) ~= 0
    assert np.around(max(abs(np.einsum('ij,ij->i',b1,b3))),decimals=12) == 0. # dot(b1,b3) ~= 0
    assert np.around(max(abs(np.einsum('ij,ij->i',b2,b3))),decimals=12) == 0. # dot(b2,b3) ~= 0
    
    # Perform Transformation via Einstein notation.

    # Define Transformation matrix --------------------------
    C = np.dstack([b1,b2,b3])
    C = np.transpose(C, (2, 1, 0)) # Swap order
    # Confirm one of the hands C[:,:,0] 
    # Each row of C should match the vectors b1[0,:], b2[0,:], b3[0,:]
    # C1 = C[:,:,0] # First hand for testing
    
    
    # Define sets of points
    # X = 3x21xN array of points in real-workd coords
    x = df[x_cols].to_numpy(); #x = x - x[:,0][:, np.newaxis]; # Center at x0
    y = df[y_cols].to_numpy(); #y = y - y[:,0][:, np.newaxis]; # Center at x0
    z = df[z_cols].to_numpy(); #z = z - z[:,0][:, np.newaxis]; # Center at x0
    X = np.dstack([x,y,z])
    X = np.transpose(X, (2, 1, 0)) # Swap order
    # X1nos = X[:,:,0] # Extract first hand for testing (no-offset)
    # X1 = X1nos - np.tile(X[:,0,0],(21,1)).T # With offset
    
    # Offset to center at wrist
    x = x - x[:,0][:, np.newaxis]; # Center at x0
    y = y - y[:,0][:, np.newaxis];
    z = z - z[:,0][:, np.newaxis];
    X = np.dstack([x,y,z])
    X = np.transpose(X, (2, 1, 0)) # Swap order
    # X = np.transpose(X, (0, 2, 1))
    
    # Loop over each element to create true output for comparison.
    # Can use to validate einsum method.
    V = np.zeros(X.shape)*np.nan
    for i in tqdm(range(len(df))):
        Vi = np.matmul(C[:,:,i],X[:,:,i])/palm_length[i]
        V[:,:,i] = Vi
        
    
    # TODO: Matrix multiplication via einsum
    # Alternative to loop method
    # V = np.einsum("iik,ijk->ijk",C,X)
    # V = np.einsum("ij...,jk...->jk...",C,X)
    # V = np.einsum("nij,njk->njk",C,X)
    # Scale points by palm length
    #V = np.divide(V,palm_length)
    
    # Check results on first hand
    # V1 = V[:,:,0]/palm_length[0]
    # V1c = np.matmul(C1,X1)/palm_length[0]

    
    # Format dataframe
    # V is 3x21xN dataset
    # Extract all the x,y,z points separately
    xp = V[0,:,:].T 
    yp = V[1,:,:].T
    zp = V[2,:,:].T
    
    
    # Insert data in dataframe as columns 0_xn, 1_xn, etc.
    x_cols = ["{}_xn".format(i) for i in range(21)]
    y_cols = ["{}_yn".format(i) for i in range(21)]
    z_cols = ["{}_zn".format(i) for i in range(21)]
    df[x_cols] = xp
    df[y_cols] = yp
    df[z_cols] = zp
    
    # Compute orientation
    
    # # Re-order columns for output
    # cols = df.columns[:-126].to_list() # First columns #['filename', 'label', 'num_hands', 'handedness', 'score']
    # cols += list(chain.from_iterable(('{}_x'.format(i), '{}_y'.format(i), '{}_z'.format(i)) for i in range(21)))
    # cols += list(chain.from_iterable(('{}_xn'.format(i), '{}_yn'.format(i), '{}_zn'.format(i)) for i in range(21)))
    # df = df[cols]
    
    
    
    # Compute palm orientation ----------------------------
    # FIXME: Move to compute_metrics
    
    # Get projection of palm normal on xz plane
    b3proj = b3.copy(); b3proj[:,1] = 0. # Projection of palm normal on xz plane
    b3proj = b3proj/np.linalg.norm(b3proj,axis=-1)[:, np.newaxis] # Normalize
    
    # Compute angle from palm normal projection to -k vector [0,0,-1]
    # TODO: Look into orientation of z axis. For now, use -k since it works
    kvec = np.tile(np.array([0,0,-1]),(len(b3proj),1)) # Array of [0,0,-1]
    cosine_angle = np.einsum('ij,ij->i',b3proj,kvec) #/ (np.linalg.norm(ba,axis=-1) * np.linalg.norm(bc,axis=-1))
    angle = np.arccos(cosine_angle)
    # Add to dataframe
    df['alpha'] = np.nan
    df['alpha'] = angle
    
    # Check values
    # zp should be close to zero for hand points 0, 5, 17
    # V[2,0,:] ~= 0
    # V[2,5,:] ~= 0
    # V[2,17,:] ~= 0
    assert np.around(max(abs(V[2,0,:])),decimals=12) == 0. # Wrist
    assert np.around(max(abs(V[2,5,:])),decimals=12) == 0. # 1st finger
    assert np.around(max(abs(V[2,17,:])),decimals=12) == 0. # Wrist
    
    return df


def compute_metrics(df):
    
    # Thumb to finger tips
    df['TT1'] = np.sqrt( (df['8_x'] - df['4_x'])**2 + (df['8_y'] - df['4_y'])**2 + (df['8_z'] - df['4_z'])**2 )
    df['TT2'] = np.sqrt( (df['12_x'] - df['4_x'])**2 + (df['12_y'] - df['4_y'])**2 + (df['12_z'] - df['4_z'])**2 )
    df['TT3'] = np.sqrt( (df['16_x'] - df['4_x'])**2 + (df['16_y'] - df['4_y'])**2 + (df['16_z'] - df['4_z'])**2 )
    df['TT4'] = np.sqrt( (df['20_x'] - df['4_x'])**2 + (df['20_y'] - df['4_y'])**2 + (df['20_z'] - df['4_z'])**2 )
    # Thumb to finger tips (normalized)
    df['TT1n'] = np.sqrt( (df['8_xn'] - df['4_xn'])**2 + (df['8_yn'] - df['4_yn'])**2 + (df['8_zn'] - df['4_zn'])**2 )
    df['TT2n'] = np.sqrt( (df['12_xn'] - df['4_xn'])**2 + (df['12_yn'] - df['4_yn'])**2 + (df['12_zn'] - df['4_zn'])**2 )
    df['TT3n'] = np.sqrt( (df['16_xn'] - df['4_xn'])**2 + (df['16_yn'] - df['4_yn'])**2 + (df['16_zn'] - df['4_zn'])**2 )
    df['TT4n'] = np.sqrt( (df['20_xn'] - df['4_xn'])**2 + (df['20_yn'] - df['4_yn'])**2 + (df['20_zn'] - df['4_zn'])**2 )
    
    # Thumb to base of fingers
    df['TB1'] = np.sqrt( (df['5_x'] - df['4_x'])**2 + (df['5_y'] - df['4_y'])**2 + (df['5_z'] - df['4_z'])**2 )
    df['TB2'] = np.sqrt( (df['9_x'] - df['4_x'])**2 + (df['9_y'] - df['4_y'])**2 + (df['9_z'] - df['4_z'])**2 )
    df['TB3'] = np.sqrt( (df['13_x'] - df['4_x'])**2 + (df['13_y'] - df['4_y'])**2 + (df['13_z'] - df['4_z'])**2 )
    df['TB4'] = np.sqrt( (df['17_x'] - df['4_x'])**2 + (df['17_y'] - df['4_y'])**2 + (df['17_z'] - df['4_z'])**2 )
    # Thumb to base of fingers (normalized)
    df['TB1n'] = np.sqrt( (df['5_xn'] - df['4_xn'])**2 + (df['5_yn'] - df['4_yn'])**2 + (df['5_zn'] - df['4_zn'])**2 )
    df['TB2n'] = np.sqrt( (df['9_xn'] - df['4_xn'])**2 + (df['9_yn'] - df['4_yn'])**2 + (df['9_zn'] - df['4_zn'])**2 )
    df['TB3n'] = np.sqrt( (df['13_xn'] - df['4_xn'])**2 + (df['13_yn'] - df['4_yn'])**2 + (df['13_zn'] - df['4_zn'])**2 )
    df['TB4n'] = np.sqrt( (df['17_xn'] - df['4_xn'])**2 + (df['17_yn'] - df['4_yn'])**2 + (df['17_zn'] - df['4_zn'])**2 )
    
    # Distances between fingers (normalized)
    df['D12n'] = np.sqrt( (df['8_xn'] - df['12_xn'])**2 + (df['8_yn'] - df['12_yn'])**2 + (df['8_zn'] - df['12_zn'])**2 )
    df['D23n'] = np.sqrt( (df['12_xn'] - df['16_xn'])**2 + (df['12_yn'] - df['16_yn'])**2 + (df['12_zn'] - df['16_zn'])**2 )
    df['D34n'] = np.sqrt( (df['16_xn'] - df['20_xn'])**2 + (df['16_yn'] - df['20_yn'])**2 + (df['16_zn'] - df['20_zn'])**2 )
    
    # Knuckle joint angles
    # Use einsum to compute dot product
    # Thumb
    a = df[['2_x','2_y','2_z']].to_numpy()
    b = df[['3_x','3_y','3_z']].to_numpy()
    c = df[['4_x','4_y','4_z']].to_numpy()
    df['phi0'] = joint_angle_vectorized(a,b,c)
    # 1st finger
    a = df[['5_x','5_y','5_z']].to_numpy()
    b = df[['6_x','6_y','6_z']].to_numpy()
    c = df[['7_x','7_y','7_z']].to_numpy()
    df['phi1'] = joint_angle_vectorized(a,b,c)
    # 2nd finger
    a = df[[ '9_x', '9_y', '9_z']].to_numpy()
    b = df[['10_x','10_y','10_z']].to_numpy()
    c = df[['11_x','11_y','11_z']].to_numpy()
    df['phi2'] = joint_angle_vectorized(a,b,c)
    # 3rd finger
    a = df[['13_x','13_y','13_z']].to_numpy()
    b = df[['14_x','14_y','14_z']].to_numpy()
    c = df[['15_x','15_y','15_z']].to_numpy()
    df['phi3'] = joint_angle_vectorized(a,b,c)
    # 4th finger
    a = df[['17_x','17_y','17_z']].to_numpy()
    b = df[['18_x','18_y','18_z']].to_numpy()
    c = df[['19_x','19_y','19_z']].to_numpy()
    df['phi4'] = joint_angle_vectorized(a,b,c)
    
    
    # Finger Elevations above palm
    # TODO: Correct for negative angles (hyperflexed fingers)
    # 1st finger
    u = df[['6_xn','6_yn','6_zn']].to_numpy() - df[['5_xn','5_yn','5_zn']].to_numpy(); 
    u = u/np.linalg.norm(u,axis=-1)[:, np.newaxis]
    elev1 = np.arctan2( u[:,2], np.sqrt( u[:,0]**2 + u[:,1]**2 ) )
    df['theta1'] = np.pi - elev1 # Suplement. Angle from wrist
    # 2nd finger
    u = df[['10_xn','10_yn','10_zn']].to_numpy() - df[['9_xn','9_yn','9_zn']].to_numpy(); 
    u = u/np.linalg.norm(u,axis=-1)[:, np.newaxis]
    elev2 = np.arctan2( u[:,2], np.sqrt( u[:,0]**2 + u[:,1]**2 ) ) 
    df['theta2'] = np.pi - elev2 # Suplement. Angle from wrist
    # 3rd finger
    u = df[['14_xn','14_yn','14_zn']].to_numpy() - df[['13_xn','13_yn','13_zn']].to_numpy(); 
    u = u/np.linalg.norm(u,axis=-1)[:, np.newaxis]
    elev3 = np.arctan2( u[:,2], np.sqrt( u[:,0]**2 + u[:,1]**2 ) ) 
    df['theta3'] = np.pi - elev3 # Suplement. Angle from wrist
    # 4th finger
    u = df[['18_xn','18_yn','18_zn']].to_numpy() - df[['17_xn','17_yn','17_zn']].to_numpy(); 
    u = u/np.linalg.norm(u,axis=-1)[:, np.newaxis]
    elev3 = np.arctan2( u[:,2], np.sqrt( u[:,0]**2 + u[:,1]**2 ) ) 
    df['theta4'] = np.pi - elev3 # Suplement. Angle from wrist
    
    # Palm orientation
    # TODO: Move code from above to here
    
    
    return df

def joint_angle_vectorized(a,b,c):
    # Vectorized version to work on multiple hands
    # a = 3xn vector
    
    # Relvative vectors
    ba = a - b
    bc = c - b
    
    cosine_angle = np.einsum('ij,ij->i',ba,bc) / (np.linalg.norm(ba,axis=-1) * np.linalg.norm(bc,axis=-1))
    angle = np.arccos(cosine_angle)
    
    return angle


#%% Geometric transformations (old)

def compute_finger_metrics(points,POINTS):

    # Compute the elevation of each digit (1-4 excluding thumb) at the nuckle
    # Elevation angle is angle above/below the palm plane (suplement of polar angle)
    # elev = np.atan2(y,x)
    # In this case y = z, and x = sqrt(x^2 + y^2)
    # 1st digit
    u = POINTS[6,:]-POINTS[5,:] # Vector along digit
    elev1 = np.arctan2(u[2],np.sqrt( u[0]**2 + u[1]**2 )) # Index finger (from index knuckle pt 6 relative to pt 5)
    if u[2]<0: elev1*=-1 # Correct for negative angles
    # 2nd digit
    u = POINTS[10,:]-POINTS[9,:] # Vector along digit
    elev2 = np.arctan2(u[2],np.sqrt( u[0]**2 + u[1]**2 )) # Index finger (from index knuckle pt 6 relative to pt 5)
    if u[2]<0: elev2*=-1 # Correct for negative angles
    # 3rd digit
    u = POINTS[14,:]-POINTS[13,:] # Vector along digit
    elev3 = np.arctan2(u[2],np.sqrt( u[0]**2 + u[1]**2 )) # Index finger (from index knuckle pt 6 relative to pt 5)
    if u[2]<0: elev3*=-1 # Correct for negative angles
    # 4st digit
    u = POINTS[18,:]-POINTS[17,:] # Vector along digit
    elev4 = np.arctan2(u[2],np.sqrt( u[0]**2 + u[1]**2 )) # Index finger (from index knuckle pt 6 relative to pt 5)
    if u[2]<0: elev4*=-1 # Correct for negative angles
    print('\n\nCompute finger metrics\n----------------')
    print('\nElevation at knuckle of each digit (1-4 excluding thumb)')
    print('elev1 = {} // (rad)'.format(elev1))
    print('elev2 = {} // (rad)'.format(elev2))
    print('elev3 = {} // (rad)'.format(elev3))
    print('elev4 = {} // (rad)'.format(elev4))
    
    # Convert to joint angles (supplementary angle) = 180 - elev
    theta1 = np.pi - elev1
    theta2 = np.pi - elev2
    theta3 = np.pi - elev3
    theta4 = np.pi - elev4
    print('\nConvert to joint angles = 180 - elev')
    print('theta1 = {} // (rad)'.format(theta1))
    print('theta2 = {} // (rad)'.format(theta2))
    print('theta3 = {} // (rad)'.format(theta3))
    print('theta4 = {} // (rad)'.format(theta4))

    # New approach:
    # Compute the polar angle from the palm normal vector N to the digit
    # c = np.dot(u,v)/norm(u)/norm(v) # -> cosine of the angle
    # angle = arccos(clip(c, -1, 1)) # if you really want the angle
    #u = POINTS[6,:]-POINTS[5,:]; u = u/np.linalg.norm(u) # Unit vector
    #v = np.array([0,0,1])
    #phi1 = np.arccos(np.clip(np.dot(u,v), -1, 1)) # if you really want the angle
    

    # Middle joint angles
    phi0 = calculate_joint_angle(POINTS[2,:],POINTS[3,:],POINTS[4,:]) # Thumb
    phi1 = calculate_joint_angle(POINTS[5,:],POINTS[6,:],POINTS[7,:]) # 1st digit at knuckle
    phi2 = calculate_joint_angle(POINTS[9,:],POINTS[10,:],POINTS[11,:]) # 2st digit at knuckle
    phi3 = calculate_joint_angle(POINTS[13,:],POINTS[14,:],POINTS[15,:]) # 3rd digit at knuckle
    phi4 = calculate_joint_angle(POINTS[17,:],POINTS[18,:],POINTS[19,:]) # 3rd digit at knuckle
    #print(np.rad2deg(phi1))
    print('\nMiddle joint angles')
    print('phi0 = {} // (rad)'.format(phi0))
    print('phi1 = {} // (rad)'.format(phi1))
    print('phi2 = {} // (rad)'.format(phi2))
    print('phi3 = {} // (rad)'.format(phi3))
    print('phi4 = {} // (rad)'.format(phi4))

    # Thumb Distances
    # Compute distance from tip of thumb to tips of fingers and to base of fingers
    dtip1 = np.linalg.norm(POINTS[8,:] - POINTS[4,:]) # Tip finger 1
    dtip2 = np.linalg.norm(POINTS[12,:] - POINTS[4,:]) # Tip finger 2
    dtip3 = np.linalg.norm(POINTS[16,:] - POINTS[4,:]) # Tip finger 3
    dtip4 = np.linalg.norm(POINTS[20,:] - POINTS[4,:]) # Tip finger 4
    # Distances from tip of thumb to base of fingers
    dbase1 = np.linalg.norm(POINTS[5,:] - POINTS[4,:])  # Base finger 1
    dbase2 = np.linalg.norm(POINTS[9,:] - POINTS[4,:]) # Base finger 2
    dbase3 = np.linalg.norm(POINTS[13,:] - POINTS[4,:]) # Base finger 3
    dbase4 = np.linalg.norm(POINTS[17,:] - POINTS[4,:]) # Base finger 4
    print('\nThumb to tip distnaces (digits 1-4)')
    print('dtip1 = {} '.format(dtip1))
    print('dtip2 = {} '.format(dtip2))
    print('dtip3 = {} '.format(dtip3))
    print('dtip4 = {} '.format(dtip4))
    print('\nThumb to base distnaces (digits 1-4)')
    print('dbase1 = {} '.format(dbase1))
    print('dbase2 = {} '.format(dbase2))
    print('dbase3 = {} '.format(dbase3))
    print('dbase4 = {} '.format(dbase4))


    # Collect parameters for return
    params = [phi0,phi1,phi2,phi3,phi4,theta1,theta2,theta3,theta4,dtip1,dtip2,dtip3,dtip4,dbase1,dbase2,dbase3,dbase4]

    return params

def calculate_joint_angle(a,b,c):
    ''' 
    Compute the angle between any three points a,b,c 
    '''
    # Relvative vectors
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
        
    return angle

#%% Graph representation
    
def generate_hand_graph(xp,yp):
    
    # Create node positons dictionary
    
    pos = {i: (xp[i],yp[i]) for i in range(21)}
    
    G = nx.Graph()
    G.add_nodes_from(pos)
    G.add_edges_from([(0 ,1) , (1 ,2) , (2 ,3) , (3 ,4) ]) # Thumb
    G.add_edges_from([(0 ,5) , (5 ,9) , (9 ,13) , (13 ,17), (17,0) ]) # Palm
    G.add_edges_from([(5 ,6) , (6 ,7) , (7 ,8) ]) # 1st digit
    G.add_edges_from([(9 ,10) , (10 ,11) , (11 ,12) ]) # 2nd digit
    G.add_edges_from([(13 ,14) , (14 ,15) , (15 ,16) ]) # 3rd digit
    G.add_edges_from([(17 ,18) , (18 ,19) , (19 ,20) ]) # 4th digit
    
    
    return G, pos

#%% Inspect Images

def inspect_images(img_dir, df, classes=None, flip=False):
    '''
    Render images with predicted hand landmarks overlayed.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing lists of images and hand landmarks
    classes : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''

    
    if classes is not None:
        # Limit classes
        df = df[df.label.isin(classes)]

    
    # Limit dataframes to single image
    # df1 = df[df.num_hands == 1]   
    # df1 = df1.head() # Limit rows for testing
    
    # Get list of relevant columns ['0_x','0_y','0_z',...]
    # cols = list(chain.from_iterable(('{}_x'.format(i), '{}_y'.format(i), '{}_z'.format(i)) for i in range(21)))
    xcols = ['{}_x'.format(i) for i in range(21)]
    ycols = ['{}_y'.format(i) for i in range(21)]
    zcols = ['{}_z'.format(i) for i in range(21)]
    xncols = ['{}_xn'.format(i) for i in range(21)]
    yncols = ['{}_yn'.format(i) for i in range(21)]
    zncols = ['{}_zn'.format(i) for i in range(21)]
    
    
    # fig, ax = plt.subplots(figsize=(12, 12))
    fig, ax = plt.subplots(1,2,figsize=(18, 10))
    
    # Loop through single dataframe
    counter = 0
    for index, row in df.iterrows():
        counter += 1
        
        # Extract data
        x = row[xcols].to_numpy().astype(float)
        y = row[ycols].to_numpy().astype(float)
        z = row[zcols].to_numpy().astype(float)
        xn = row[xncols].to_numpy().astype(float)
        yn = row[yncols].to_numpy().astype(float)
        zn = row[zncols].to_numpy().astype(float)
        imagefile = img_dir.parent/row.filename
        short_filename = Path(*imagefile.parts[-3:]) # Shorten filename e.g. 27 Class Sign Language Dataset/5/5_0.jpg
        
        # Format into 21x3 array
        points = np.stack([x,y,z]).T
        
        # Read in image
        frame = cv2.imread(str(imagefile))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR 2 RGB
        if flip:
            image = cv2.flip(image, 1) # Flip on horizontal. Do not flip image
        image.flags.writeable = False # Set flag
        
        
        # Convert to pixel coords
        # Note: Need to flip coords left/right since trained on image that was flipped.
        image_height, image_width, _ = image.shape # Get image size
        xp = x*image_width
        # if flip:
        #     xp = image_width - x*image_width
        #     xn *= -1
        yp = y*image_height
        
        # Define hand graph
        G, pos = generate_hand_graph(xp,yp)
        G1, pos1 = generate_hand_graph(xn,yn) # Palm-centered frame
        
        # Plot image
        
        
        
        # ax[0].imshow(image)
        # ax[0].plot(xp,yp,'*r') # Landmarks
        # nx.draw_networkx(G, pos=pos)
        
        # Ax1: Palm-cen
        ax[0].clear()
        ax[0].title.set_text('Image Frame')
        ax[0].imshow(image)
        ax[0].plot(xp,yp,'*r') # Landmarks
        nx.draw_networkx(G, pos=pos, ax=ax[0])
        # Ax2: Palm-centered frame
        ax[1].clear()
        ax[1].title.set_text('Palm-Centered Frame')
        ax[1].plot(xn,yn,'*r') # Landmarks
        nx.draw_networkx(G1, pos=pos1, ax=ax[1])
        ax[1].set_aspect('equal', 'box')
        ax[1].set_xlim([-2,2])
        ax[1].set_ylim([-0.5,3])
        ax[1].grid()
        title_str = str(counter) + " of " + str(len(df)) + "\nFilename: " + str(short_filename) + "\nHandedness: " + row['handedness']
        fig.suptitle(title_str)
        plt.show()
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

        
    return points

def plot_single_hand_from_df(img_dir, df, classes=None, ind=0):
    '''
    Render images with predicted hand landmarks overlayed.

    Parameters
    ----------
    base_dir : TYPE
        DESCRIPTION.
    hand_landmarks_file : TYPE
        DESCRIPTION.
    classes : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    
    if classes is not None:
        # Limit classes
        df = df[df.label.isin(classes)]
    
    # Get list of relevant columns ['0_x','0_y','0_z',...]
    # cols = list(chain.from_iterable(('{}_x'.format(i), '{}_y'.format(i), '{}_z'.format(i)) for i in range(21)))
    xcols = ['{}_x'.format(i) for i in range(21)]
    ycols = ['{}_y'.format(i) for i in range(21)]
    zcols = ['{}_z'.format(i) for i in range(21)]
    xncols = ['{}_xn'.format(i) for i in range(21)]
    yncols = ['{}_yn'.format(i) for i in range(21)]
    zncols = ['{}_zn'.format(i) for i in range(21)]
    
    
    # Extract row
    row = df.iloc[ind]
        
    # Extract Data
    x = row[xcols].to_numpy().astype(float)
    y = row[ycols].to_numpy().astype(float)
    z = row[zcols].to_numpy().astype(float)
    xn = row[xncols].to_numpy().astype(float)
    yn = row[yncols].to_numpy().astype(float)
    zn = row[zncols].to_numpy().astype(float)
    imagefile = img_dir.parent/row.filename
    
    # Format into 21x3 array
    points = np.stack([x,y,z]).T
    
    # Read in image
    frame = cv2.imread(str(imagefile))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR 2 RGB
    # image = cv2.flip(image, 1) # Flip on horizontal
    image.flags.writeable = False # Set flag
    
    # Convert to pixel coords
    image_height, image_width, _ = image.shape # Get image size
    xp = x*image_width
    yp = y*image_height
    
    # Define hand graph
    G, pos = generate_hand_graph(xp,yp)
    G1, pos1 = generate_hand_graph(xn,yn) # Palm-centered frame
    
    # Plot image
    fig, ax = plt.subplots(1,2,figsize=(18, 10))
    # Ax1: Real world
    ax[0].title.set_text('Image Frame')
    ax[0].imshow(image)
    ax[0].plot(xp,yp,'*r') # Landmarks
    nx.draw_networkx(G, pos=pos, ax=ax[0])
    # Ax2: Palm-centered
    ax[1].title.set_text('Palm-Centered Frame')
    ax[1].plot(xn,yn,'*r') # Landmarks
    nx.draw_networkx(G1, pos=pos1, ax=ax[1])
    ax[1].set_aspect('equal', 'box')
    ax[1].set_xlim([-2,2])
    ax[1].set_ylim([-0.5,3])
    ax[1].grid()
    plt.show()

        
    return points
