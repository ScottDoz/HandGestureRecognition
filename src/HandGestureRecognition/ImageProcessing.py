"""
Image Processing
----------------

This module processes images and extracts hand landmarks to save to file.

"""

import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
from itertools import chain
import networkx as nx
import matplotlib.pyplot as plt

from PIL import Image

try:
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
except:
    pass

import pdb

# from Visualization import *

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

def extract_hand_landmarks(base_dir,classes=None,out_dir=None):
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
    
    # Create output file
    if out_dir is None:
        # Save data into the base_dir
        out_dir = base_dir
    
    # Create csv file to write to
    f = open(str(out_dir/'hand_landmarks.csv'),'w+')
    # Write header
    ptlabels = ['{}_x, {}_y, {}_z'.format(i,i,i) for i in range(21)]
    ptlabels = ",".join(['{}_x, {}_y, {}_z '.format(i,i,i) for i in range(21)])
    f.write('filename, label, handedness,'+ptlabels+'\n')

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
            image = cv2.flip(image, 1) # Flip on horizontal
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
                
                    # Get position values of key landmarks
                    # See here for abreviations: https://blog.missiondata.com/lab-notes-mediapipe-with-python-for-gesture-recognition/
                    
                    # Extract all points as 21x3 array
                    points = np.vstack([(float(hand.landmark[i].x),float(hand.landmark[i].y),float(hand.landmark[i].z)) for i in range(21)])  

                    # Prepare row entry and write to csv
                    # Filename, label, handedness, points
                    s = ",".join(np.char.mod('%f', points.flatten())) # Point coords as string
                    txt = str(Path(*Path(file).parts[-3:])) + ', ' + str(labels[i]) + ', ' + handedness  + ', ' + s + '\n'
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
                    
                        # Get position values of key landmarks
                        # See here forp abreviations: https://blog.missiondata.com/lab-notes-mediapipe-with-python-for-gesture-recognition/
                        
                        # Extract all points as 21x3 array
                        points = np.vstack([(float(hand.landmark[i].x),float(hand.landmark[i].y),float(hand.landmark[i].z)) for i in range(21)])  

                        # Prepare row entry and write to csv
                        # Filename, label, handedness, points
                        s = ",".join(np.char.mod('%f', points.flatten())) # Point coords as string
                        txt = str(Path(*Path(file).parts[-3:])) + ', ' + str(labels[i]) + ', ' + handedness  + ', ' + s + '\n'
                        f.write(txt) #Give your csv text here.

                    else:
                        # Right and Left hands. Store both separatly

                        # Loop though hands
                        for num, hand in enumerate(results.multi_hand_landmarks):

                            # Determine if "Left" or "Right" hand
                            handedness = results.multi_handedness[num].classification[0].label
                        
                            # Get position values of key landmarks
                            # See here forp abreviations: https://blog.missiondata.com/lab-notes-mediapipe-with-python-for-gesture-recognition/
                            
                            # Extract all points as 21x3 array
                            points = np.vstack([(float(hand.landmark[i].x),float(hand.landmark[i].y),float(hand.landmark[i].z)) for i in range(21)])  

                            # Prepare row entry and write to csv
                            # Filename, label, handedness, points
                            s = ",".join(np.char.mod('%f', points.flatten())) # Point coords as string
                            txt = str(Path(*Path(file).parts[-3:])) + ', ' + str(labels[i]) + ', ' + handedness  + ', ' + s + '\n'
                            f.write(txt) #Give your csv text here.


        # End loop
        f.close()


    # Save training data
    #df = pd.DataFrame(X,columns=["x{}".format(i) for i in range(X.shape[1])])
    #df['y'] = labels
    #dir_ = Path(os.path.abspath(__file__)).parent # Directory of this python script
    #df.to_csv(str(dir_/'training.csv'),index=False) # Save to csv

    
    return

#%% Geometric transformations

def transform_to_palm_centered_frame(points):

    # Extract key points of the palm
    p1 = points[0,:]  # Wrist
    p2 = points[5,:]  # Index finger MCP (Metacarpophalangeal Joint)
    p3 = points[17,:] # Pinky finger MCP (Metacarpophalangeal Joint)
    # Print
    print('\n--------------------------------------------------------')
    print('\nTransformation function data\n--------------------------')
    print('points (3D world coordinates)')
    print(str(points).replace('[','{').replace(']','}'))
    print('\nKey palm points')
    print('p1 = ' + str(p1).replace('[','{').replace(']','}'))
    print('p2 = ' + str(p2).replace('[','{').replace(']','}'))
    print('p3 = ' + str(p3).replace('[','{').replace(']','}'))

    # Define unit vectors and their length
    r1mag = np.linalg.norm(p2-p1)
    r2mag = np.linalg.norm(p3-p1)
    r3mag = np.linalg.norm(p3-p2)
    r1u = (p2 - p1)/r1mag # Wrist to Index unit vector
    r2u = (p3 - p1)/r2mag # Wrist to Pinky unit vector
    # Print
    print('\nVectors along palm')
    print('r1mag = {}'.format(r1mag))
    print('r2mag = {}'.format(r2mag))
    print('r3mag = {}'.format(r3mag))
    print('r1u = ' + str(r1u).replace('[','{').replace(']','}'))
    print('r2u = ' + str(r2u).replace('[','{').replace(']','}'))

    # Length of bisector
    l2 = (r1mag*r2mag/(r1mag+r2mag)**2)*( (r1mag+r2mag)**2 - r3mag**2) # Length of bisector angle
    l = np.sqrt(l2)
    palm_length = l # Define this as the palm length
    #print(palm_length)
    print('\nPalm lenth')
    print('l2 = {}'.format(l2))
    print('l = palm_length = {}'.format(l))

    # Define Basis vectors for palm plan
    b2 = (r1u + r2u)/np.linalg.norm(r1u+r2u) # Bisector of r1, r2
    b3 = np.cross(r2u,r1u); b3 = b3/np.linalg.norm(b3) # Normal to palm
    b1 = np.cross(b2,b3); b1 = b1/np.linalg.norm(b1) # Complete the pair
    print('\nBasis vectors for palm reference plane')
    print('Note: out of order. b2,b3,b1')
    print('b2 = ' + str(b2).replace('[','{').replace(']','}') + ' // Bisector of r1,r2')
    print('b3 = ' + str(b3).replace('[','{').replace(']','}') + ' // Normal to palm')
    print('b1 = ' + str(b1).replace('[','{').replace(']','}') + ' // Completes the orthogonal set')

    # Apply coordinate transform

    # Transform positions in image space to basis
    POINTS = points.copy()
    POINTS = POINTS - np.tile(POINTS[0,:],(len(points),1)) # Translate to center at wrist
    print('\npoints after translating to recenter at wrist')
    print(str(POINTS).replace('[','{').replace(']','}'))

    # Change of basis into {b1,b2,b3}
    A = np.column_stack((b1,b2,b3)) # Transformation matrix [b1,b2,b3]
    #POINTS = A.dot(points.T).T # Apply transformation matrix X = A*x
    # [X1  ...   Xn]   [a11 a12 a13]   [x1  ...   xn]
    # [Y1  ...   Yn] = [a21 a22 a23] * [y1  ...   yn] 
    # [Z1  ....  Zn]   [a31 a32 a33]   [z1  ....  zn]
    # (Transformed POINTS)   (Matrix)   *     (Origianl points)
    POINTS = np.matmul(A.T,POINTS.T).T
    print('\nTransformation matrix')
    print('Apply transformation matrix X = A*x')
    print('Note: need to take transpose of points to keep in the required shape for multiplication')
    print('[X1  ...   Xn]   [a11 a12 a13]   [x1  ...   xn]')
    print('[Y1  ...   Yn] = [a21 a22 a23] * [y1  ...   yn]')
    print('[Z1  ....  Zn]   [a31 a32 a33]   [z1  ....  zn]')
    print('\nA:')
    print(str(A).replace('[','{').replace(']','}'))
    print('\nPOINTS after matrix transform')
    print(str(POINTS).replace('[','{').replace(']','}'))

    # Scale points by length of the palm
    POINTS /= palm_length
    print('\nPOINTS after scaling by palm_length')
    print(str(POINTS).replace('[','{').replace(']','}'))



    # Check that points 0,5,17 are all alighned with the plane
    # print(POINTS[0,2])
    #print(POINTS[5,2])
    #print(POINTS[17,2])
    # Note: close enough for now (within 0.01). Try to improve precision later

    return POINTS, p1,p2,p3, b1, b2, b3, palm_length

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

def inspect_images(base_dir, hand_landmarks_file, classes=None):
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

    # Load hand_landmarks data
    df = pd.read_csv(hand_landmarks_file)
    # Strip spaces from labels
    df.columns = [l.strip() for l in list(df.columns)]
    
    # Convert labels to strings
    df['label'] = df['label'].astype(str)  
    df.label = df.label.str.strip()
    
    if classes is None:
        # List classes from names
        p = base_dir.glob('**/*')
        classes = [x.name for x in p if x.is_dir()]
    
    # Limit classes
    df = df[df.label.isin(classes)]
    
    
    # Group dataframe by filename and  count hands
    dfg = df.groupby(['filename'])['handedness'].count()
    
    # Split dataframe into images containing
    # 1) Single hand
    # 2) Multiple hands
    mask = df.filename.duplicated(keep=False)
    df1 = df[~mask] # Single hand
    df2 = df[mask]  # Multiple hands
    
    # df1 = df1.head() # Limit rows for testing
    
    # Get list of relevant columns ['0_x','0_y','0_z',...]
    # cols = list(chain.from_iterable(('{}_x'.format(i), '{}_y'.format(i), '{}_z'.format(i)) for i in range(21)))
    xcols = ['{}_x'.format(i) for i in range(21)]
    ycols = ['{}_y'.format(i) for i in range(21)]
    zcols = ['{}_z'.format(i) for i in range(21)]
    
    
    fig, ax = plt.subplots(figsize=(12, 12))
    # Loop through single dataframe
    for index, row in df1.iterrows():
        
        # Extract data
        x = row[xcols].to_numpy().astype(float)
        y = row[ycols].to_numpy().astype(float)
        z = row[zcols].to_numpy().astype(float)
        imagefile = base_dir.parent/row.filename
        
        # Format into 21x3 array
        points = np.stack([x,y,z]).T
        
        # Read in image
        frame = cv2.imread(str(imagefile))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR 2 RGB
        image = cv2.flip(image, 1) # Flip on horizontal
        image.flags.writeable = False # Set flag
        
        
        # Convert to pixel coords
        image_height, image_width, _ = image.shape # Get image size
        xp = x*image_width
        yp = y*image_height
        
        # Define hand graph
        G, pos = generate_hand_graph(xp,yp)
        
        # Plot image
        
        fig.clf(True) 
        plt.imshow(image)
        plt.plot(xp,yp,'*r') # Landmarks
        lines = nx.draw_networkx(G, pos=pos)
        
        plt.show()
        
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        
    return points


#%% Main function

if __name__ == "__main__":
    
    # # ASL Dataset
    # base_dir = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data\asl_dataset')
    # hand_landmarks_file = r'E:\Files\Zero Robotics\HandGuestures\Training Data\asl_dataset\hand_landmarks_asl_dataset.csv'

    # # 27 Class Sign Language
    # base_dir = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data\27 Class Sign Language Dataset')
    # hand_landmarks_file = r'E:\Files\Zero Robotics\HandGuestures\Training Data\27 Class Sign Language Dataset\hand_landmarks_27_class_sign_language_dataset.csv'
    
    # Sign Language For Numbers
    base_dir = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data\Sign Language for Numbers')
    hand_landmarks_file = r'E:\Files\Zero Robotics\HandGuestures\Training Data\Sign Language for Numbers\hand_landmarks_sign_language_for_numbers.csv'
    
    
    points = inspect_images(base_dir, hand_landmarks_file, classes=None)
