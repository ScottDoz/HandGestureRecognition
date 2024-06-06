# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:12:20 2024

@author: scott

Run Image Processing
--------------------

This is a control script to use functions from the ImageProcessing module
to process the different hand gesture datasets.

Datasets
ASL
27SLN
SLfN
Hagrid *

Note, that the Hagrid dataset takes multiple hours to process. It is convenient
to run it in batches. Each of the 18 gestures takes around 25 mins to run.

"""

from ImageProcessing import *

if __name__ == "__main__":
    
    # Define base directories of where the image data are stored
    img_dir_asl = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data\asl_dataset')
    img_dir_27CSL = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data\27 Class Sign Language Dataset')
    img_dir_SLFN = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data\Sign Language for Numbers')
    img_dir_hagrid = Path(r'E:\Files\Zero Robotics\HandGuestures\mmpretrain\data\hagrid\hagrid_dataset_512')
    
    # # Process ASL (Done)
    # extract_hand_landmarks(img_dir_asl)
    
    # # Process 27 Class Sign Language (Done)
    # extract_hand_landmarks(img_dir_27CSL)
    
    # # Process Sign Language For Numbers (Done)
    # extract_hand_landmarks(img_dir_SLFN)
    
    # # Process Hagrid annotation data (Done)
    # hagrid_annotation_data(img_dir_hagrid)
    
    # # Process Hagrid Dataset (done)
    # base_dir = Path(r'E:\Files\Zero Robotics\HandGuestures\mmpretrain\data\hagrid\hagrid_dataset_512')
    # # # # Step 1: Process individual files
    # # # # extract_hand_landmarks(img_dir_hagrid,classes=['two_up_inverted'])
    # # Step 2: Combine into single csv
    # files = ['hand_landmarks_fist.csv','hand_landmarks_like.csv','hand_landmarks_dislike.csv',
    #           'hand_landmarks_stop.csv','hand_landmarks_stop_inverted.csv',
    #           'hand_landmarks_two_up.csv','hand_landmarks_two_up_inverted.csv',
    #           'hand_landmarks_call.csv','hand_landmarks_rock.csv','hand_landmarks_peace_inverted.csv',
    #           'hand_landmarks_three2.csv','hand_landmarks_mute.csv','hand_landmarks_ok.csv',
    #           'hand_landmarks_one.csv','hand_landmarks_peace.csv','hand_landmarks_three.csv',
    #           'hand_landmarks_four.csv','hand_landmarks_palm.csv']
    # combine_results(base_dir,files)
    # process_hagrid_annotation_data(img_dir_hagrid)
    
    # # # Step 3: Add additional annotation data
    # # df = hagrid_merge_annotation(img_dir_hagrid)
    # # df = pd.read_csv(str(img_dir_hagrid/'hagrid_annotations.csv'),delimiter=';')
    
    