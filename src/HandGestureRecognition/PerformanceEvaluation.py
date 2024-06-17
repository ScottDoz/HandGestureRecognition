import os
import pdb
import cv2
from matplotlib import pyplot as plt
from natsort import natsort_keygen
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm
from pathlib import Path
from enum import Enum
import Models as md
import ImageProcessing  as ip


# def extract_hand_landmarks(image_path, label, output_path=".", flip=False):
#     """
#     Extract and save hand landmark data from a single image. 
#     Each row contains the x, y, z coordinates of the 21 hand landmarks, along with the name of the image and the label.
    
#     Parameters
#     ----------
#     image_path : str
#         Path to the image file.
#     label : str
#         Label of the image.
#     flip : bool, optional
#         Whether to flip the image horizontally. Default is False.
#     """
#     mp_hands = mp.solutions.hands

#     # Create output file if it does not exist
#     output_file = Path(output_path+'/hand_landmarks.csv')
#     if not output_file.exists():
#         with open(output_file, 'w+') as f:
#             ptlabels = ','.join(['{}_x,{}_y,{}_z'.format(i, i, i) for i in range(21)])
#             f.write('filename,label,num_hands,handedness,score,' + ptlabels + '\n')

#     with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
#         frame = cv2.imread(str(image_path))
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
#         if flip:
#             image = cv2.flip(image, 1)  # Flip horizontally
#         image.flags.writeable = False
#         results = hands.process(image)
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB to BGR

#         if results.multi_hand_landmarks:
#             num_hands = len(results.multi_hand_landmarks)
#             for num, hand in enumerate(results.multi_hand_landmarks):
#                 handedness = results.multi_handedness[num].classification[0].label
#                 score = results.multi_handedness[num].classification[0].score
#                 if not flip:
#                     handedness = 'Left' if handedness == 'Right' else 'Right'
#                 points = np.vstack([(hand.landmark[i].x, hand.landmark[i].y, hand.landmark[i].z) for i in range(21)])
#                 s = ','.join(np.char.mod('%f', points.flatten()))
#                 txt = f"{Path(*Path(image_path).parts[-3:])},{label},{num_hands},{handedness},{score},{s}\n"
#                 with open(output_file, 'a') as f:
#                     f.write(txt)

# def traverse_folders(root_dir):
#     """
#     Traverse folders and subfolders, processing pictures accordingly.
#     """
#     for label_name in os.listdir(root_dir):
#         label_path = os.path.join(root_dir, label_name)
#         if os.path.isdir(label_path):
#             for image_name in os.listdir(label_path):
#                 image_path = os.path.join(label_path, image_name)
#                 if os.path.isfile(image_path):
#                     extract_hand_landmarks(image_path, label_name)


def process_test_data(root_dir, ):
         ip.extract_hand_landmarks(root_dir, out_dir=root_dir)

def compute_palm_centered_landmarks(df):
    print("Starting computation of palm-centered landmarks...")
    
    normalize_to = 'palm_bisector'
    
    cols = list(md.chain.from_iterable(('{}_x'.format(i), '{}_y'.format(i), '{}_z'.format(i)) for i in range(21)))
    x_cols = ["{}_x".format(i) for i in range(21)]
    y_cols = ["{}_y".format(i) for i in range(21)]
    z_cols = ["{}_z".format(i) for i in range(21)]
    landmarks = df[cols].to_numpy()
    
    p1 = df[['0_x', '0_y', '0_z']].to_numpy()
    p2 = df[['5_x', '5_y', '5_z']].to_numpy()
    p3 = df[['17_x', '17_y', '17_z']].to_numpy()
    
    r1mag = np.linalg.norm(p2 - p1, axis=-1)
    r2mag = np.linalg.norm(p3 - p1, axis=-1)
    r3mag = np.linalg.norm(p3 - p2, axis=-1)
    r1u = (p2 - p1) / r1mag[:, np.newaxis]
    r2u = (p3 - p1) / r2mag[:, np.newaxis]
    
    l2 = (r1mag * r2mag / (r1mag + r2mag)**2) * ((r1mag + r2mag)**2 - r3mag**2)
    l = np.sqrt(l2)
    if normalize_to == 'palm_bisector':
        palm_length = l
    elif normalize_to == 'wrist_to_index':
        palm_length = r1mag
    
    left_flag = df.handedness == 'Left'
    left_ind = df.index[df.handedness == 'Left'].tolist()
    
    if normalize_to == 'palm_bisector':
        b2 = (r1u + r2u) / np.linalg.norm(r1u + r2u, axis=-1)[:, np.newaxis]
    elif normalize_to == 'wrist_to_index':
        b2 = r1u
    b3 = np.cross(r1u, r2u)
    b3l = np.cross(r2u, r1u)
    b3[left_ind, :] = b3l[left_ind, :]
    b3 = b3 / np.linalg.norm(b3, axis=-1)[:, np.newaxis]
    
    b1 = np.cross(b2, b3)
    b1 = b1 / np.linalg.norm(b1, axis=-1)[:, np.newaxis]
    
    assert np.around(max(abs(np.einsum('ij,ij->i', b1, b2))), decimals=12) == 0.
    assert np.around(max(abs(np.einsum('ij,ij->i', b1, b3))), decimals=12) == 0.
    assert np.around(max(abs(np.einsum('ij,ij->i', b2, b3))), decimals=12) == 0.
    
    C = np.dstack([b1, b2, b3])
    C = np.transpose(C, (2, 1, 0))
    
    x = df[x_cols].to_numpy()
    y = df[y_cols].to_numpy()
    z = df[z_cols].to_numpy()
    X = np.dstack([x, y, z])
    X = np.transpose(X, (2, 1, 0))
    
    x = x - x[:, 0][:, np.newaxis]
    y = y - y[:, 0][:, np.newaxis]
    z = z - z[:, 0][:, np.newaxis]
    X = np.dstack([x, y, z])
    X = np.transpose(X, (2, 1, 0))
    
    V = np.zeros(X.shape) * np.nan
    for i in range(len(df)):
        Vi = np.matmul(C[:, :, i], X[:, :, i]) / palm_length[i]
        V[:, :, i] = Vi
        if i % 100 == 0:
            print(f"Processed {i+1}/{len(df)} samples")
    
    xp = V[0, :, :].T
    yp = V[1, :, :].T
    zp = V[2, :, :].T
    
    x_cols = ["{}_xn".format(i) for i in range(21)]
    y_cols = ["{}_yn".format(i) for i in range(21)]
    z_cols = ["{}_zn".format(i) for i in range(21)]
    df[x_cols] = xp
    df[y_cols] = yp
    df[z_cols] = zp
    
    b3proj = b3.copy()
    b3proj[:, 1] = 0.
    b3proj = b3proj / np.linalg.norm(b3proj, axis=-1)[:, np.newaxis]
    
    kvec = np.tile(np.array([0, 0, -1]), (len(b3proj), 1))
    cosine_angle = np.einsum('ij,ij->i', b3proj, kvec)
    angle = np.arccos(cosine_angle)
    df['alpha'] = angle
    
    assert np.around(max(abs(V[2, 0, :])), decimals=12) == 0.
    assert np.around(max(abs(V[2, 5, :])), decimals=12) == 0.
    assert np.around(max(abs(V[2, 17, :])), decimals=12) == 0.
    
    print("Finished computation.")
    return df
def read_iss_test_data(base_dir,metrics=True):
    
    df = pd.read_csv(str(base_dir+'/hand_landmarks.csv'))

    # Convert labels to strings and strip whitespace
    df['label'] = df['label'].astype(str)
    df['label'] = df['label'].str.strip()

    # print(df.columns)
    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]
    print("-------------------------")
    print(len(df))
    # pdb.set_trace()
    # Sort the DataFrame by the 'filename' column using natural sort order and reset the index
    df = df.sort_values(by="filename", key=natsort_keygen()).reset_index(drop=True)
    print(len(df))
    print("-------------------------")
    # Print the first few rows of the DataFrame to verify the preprocessing
    print(df.head())
    
    # Add normalized points + metrics
    if metrics:
        # Palm-centered landmarks
        df = compute_palm_centered_landmarks(df)
        # Metrics
        df = ip.compute_metrics(df)
    print("metrics completed")
    return df

def get_features(model_name):
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
    return features

def get_classes(model_name):
    if model_name.lower() in ['asl0_9']:
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    else:
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'call', 'dislike', 'fist', 'like',
                    'mute', 'ok', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three2',
                    'two_up', 'two_up_inverted']
    return classes

def gen_confusion_matrix(data_dir, model_name):
     
    print("initialized")
    # Test model on input data
    try:
        features = get_features(model_name=model_name)
        classes = get_classes(model_name=model_name)
        df= read_iss_test_data(data_dir)
        X_test = df[features].to_numpy()
        y_test = df["label"].to_numpy()
        interpreter = md.load_model(model_name)
    except Exception as e:
        print(e)
    # output = []
    y_pred = {}
    # print(X_test)
    try:
        for i in range(len(X_test)):
            # y0 = y_test[0] 
            x0 = X_test[i,:]
            interpreter_value =  md.interpreter_predict(interpreter,x0,labels=classes)
            y_pred[i] = interpreter_value
    except Exception as e:
        print(e)   
    print(y_pred)
    # Calculate confusion matrix
    y_true = y_test
    y_pred_values = np.array(list(y_pred.values()))
    return confusion_matrix(y_true, y_pred_values, labels=classes)


def get_conf_matrix_tables(directory, name_prefix):
    models = ["ASL0_9", 'Hagrid_ASL0_9', 'Hagrid_ASL0_9_v2']
    for model_name in models:
        filename = str(name_prefix+"_"+model_name)
        conf_matrix = gen_confusion_matrix(data_dir=directory, model_name=model_name)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=get_classes(model_name=model_name))
        # plt.title(filename)  # Customize the title as needed
        # disp.plot()
        # plt.savefig(Path(directory + '/'+filename+'.png'))

        fig, ax = plt.subplots(figsize=(14, 14))  # Adjusted to be large enough for 25x25 matrix

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=get_classes(model_name=model_name))
        disp.plot(ax=ax, cmap='viridis')  # You can change the colormap if desired

        # Customize the font sizes for labels and title, and rotate x-axis labels
        plt.xticks(fontsize=12, rotation=45, ha='right')  # Rotate x-axis labels by 45 degrees
        plt.yticks(fontsize=12)
        plt.xlabel('Predicted Label', fontsize=16)
        plt.ylabel('True Label', fontsize=16)
        plt.title('Confusion Matrix for ' + model_name, fontsize=18)  # Add and customize title

        # Save the plot as an image file
        # plt.savefig('confusion_matrix.png', bbox_inches='tight')  
        plt.savefig(Path(directory + '/'+filename+'.png'))

def make_directory_if_not_exists(directory_path):
    """
    Helper function to create a directory if it doesn't exist already.
    
    Args:
    - directory_path (str): The path of the directory to be created.
    
    Returns:
    - bool: True if the directory was created or already exists, False otherwise.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
            return True
        except OSError as e:
            print(f"Error creating directory '{directory_path}': {e}")
            return False
    else:
        print(f"Directory '{directory_path}' already exists.")
        return True


class CombinationType(Enum):
    """
    Define the tests here. Add any combination you want here and refer to it in the main() function.
    Please note the following info which is venn diagram related
    ALL > Difference
    INTERSECTION > Intersection
    UNION > Union
    """
    AIRLOCK = { 
        "demo1_set1_airlock_1m_right" : r'D:/HandGestures/Test Data/2024-03-20 Tech Demo 1/actual data/mit.astrobee.hand_signals_demo1_delayed/set1 _airlock_1m_right',
        "demo1_set3_airlock__1m_left" : r'D:/HandGestures/Test Data/2024-03-20 Tech Demo 1/actual data/mit.astrobee.hand_signals_demo1_delayed/set3_airlock__1m_left',
        "demo2_set1_airlock__1m_right" :  r'D:/HandGestures/Test Data/2024-05-01 Tech Demo 2/set1_airlock__1m_right',
        "demo3_set3_airlock_1m_right" : r'D:/HandGestures/Test Data/2024-05-21 Tech Demo 3/set3_airlock_1m_right'
        }
    AFT =  { 
        "demo2_set2_aft_0_5m_right" : r'D:/HandGestures/Test Data/2024-05-01 Tech Demo 2/set2_aft_0.5m_right',
        "demo3_set1_aft_0_5m_right" : r'D:/HandGestures/Test Data/2024-05-21 Tech Demo 3/set1_aft_0.5m_right',
        "demo3_set2_aft_0_7m_right" : r'D:/HandGestures/Test Data/2024-05-21 Tech Demo 3/set2_aft_0.7m_right',
        }
    DIFFERENCE = {}
    INTERSECTION ={}
    UNION = {}
    CUSTOM = {}

    @staticmethod
    def combine(customCombination):
        """
        customCombination should have the CombinationType objects
        """
        difference = {}
        intersection = {}
        union = {}

        combinationObjects = CombinationType if len(customCombination)<=0 else customCombination
        for item in combinationObjects:
            if item.name != "DIFFERENCE" or item.name != "INTERSECTIOn" or item.name != "UNION":
                for key, value in item.value.items():
                    union[key] = value
                    if key not in difference:
                        difference[key] = value
                    else:
                        intersection[key] = value
                        pass
        return {
            "difference": difference,
            "intersection": intersection,
            "union": union
        }

    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


def optimizeCombination(customCombination):
    combinationValues = CombinationType.combine(customCombination)
    # Update ALL value after the class definition
    CombinationType.DIFFERENCE.value.update(combinationValues["difference"])
    CombinationType.INTERSECTION.value.update(combinationValues["intersection"])
    CombinationType.UNION.value.update(combinationValues["union"])

class CombinedOutputFolder(Enum):
    """
    Defines the folder outputs

    """
    COMPLETE_TEST = "Combined Test"
    AIR_LOCK_TEST = "Combined Test AirLock"
    AFT_TEST = "Combined Test Aft"


def CombinationFileCompiler(combinationType=CombinationType.DIFFERENCE, combinedOutputFolderName=CombinedOutputFolder.COMPLETE_TEST, customCombinationTypeCollection = []):
    """
    Combines the tests to be completed based on the background. 
    combinationType > If you want a set based output (venn diagram) of the customCOmbinationTypeCollection. 
    combinedOutputFolderName > just a name combined output folder
    customCombinationTypeCollection > collection of CombinationTypes for example; customCombinationTypeCollection = [CombinationType.AIRLOCK, CombinationType.AFT]
    NOTE: if customCombinationTypeCollection is not provided, combination type will provide the set combination of all CombinationTypes in the CombinationType
    """
    optimizeCombination(customCombinationTypeCollection)
    iss_demo = combinationType.value
    combined_csv_dir = r'D:/HandGestures/Test Data/' + combinedOutputFolderName.value
    combined_csv_file = r'D:/HandGestures/Test Data/'+ combinedOutputFolderName.value + r'/hand_landmarks.csv'

    return {
        "iss_demo": iss_demo,
        "combined_csv_dir": combined_csv_dir,
        "combined_csv_file": combined_csv_file
    }

def performanceEvaluator(combinationType, combinedOutputFolder, customCombinationTypeCollection):
    combinationCompiler = CombinationFileCompiler(combinationType, combinedOutputFolder, customCombinationTypeCollection)
    iss_demo = combinationCompiler['iss_demo']
    combined_csv_dir = combinationCompiler['combined_csv_dir']
    combined_csv_file = combinationCompiler['combined_csv_file']
    
    csv_hand_model_files = []
    
    make_directory_if_not_exists(combined_csv_dir)

    for demo_name, directory in iss_demo.items():
        ip.extract_hand_landmarks(Path(directory), out_dir=Path(directory))
        csv_hand_model_files.append(os.path.join(directory, "hand_landmarks.csv"))
        get_conf_matrix_tables(directory=directory, name_prefix = demo_name)

    
    combined_df = pd.concat([pd.read_csv(file) for file in csv_hand_model_files], ignore_index=True)
    combined_df.to_csv(combined_csv_file, index=False)
    
    get_conf_matrix_tables(directory=combined_csv_dir, name_prefix="combined_demos" )
def main():
    # NOTE: Change values here if needed
    # name of the output combination folder
    # CombinationType if not all are needed
    # for example if you need airlock only 
    # combinationType = CombinationType.DIFFERENCE # refer to CombinationType, but basically DIFFERENCE > ALL, INTERSECTION > INTERSECTION, UNION > UNION
    combinedOutputFolder = CombinedOutputFolder.AIR_LOCK_TEST # name of the output combination folder
    customCombinationTypeCollection = [ CombinationType.AIRLOCK ] 

    performanceEvaluator(combinedOutputFolder = combinedOutputFolder, customCombinationTypeCollection = customCombinationTypeCollection)


main()
