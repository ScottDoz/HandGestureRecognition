# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:17:38 2024

@author: scott
"""

# import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import uuid
import os
import pdb
import pyvista as pv
# from pyvistaqt import BackgroundPlotter
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands

from ImageProcessing import *


#%% Plotting functions

def initialize_3dplot():

    # Create plotter and instatiate scene objects
    plotter = BackgroundPlotter(shape=(1, 2))
    # Point cloud
    point_cloud = pv.PolyData(np.zeros((21,3))); plotter.add_mesh(point_cloud, color='maroon', point_size=10.0, render_points_as_spheres=True)
    # Basis vectors
    arrow_b1 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(arrow_b1, color="red", line_width=3)
    arrow_b2 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(arrow_b2, color="red", line_width=3)
    arrow_b3 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(arrow_b3, color="red", line_width=3)
    # Palm lines
    line_r1 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_r1, color="green", line_width=3)
    line_r2 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_r2, color="green", line_width=3)
    line_r3 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_r3, color="green", line_width=3)
    # Thumb
    line_f00 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f00, color="red", line_width=3)
    line_f01 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f01, color="red", line_width=3)
    line_f02 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f02, color="red", line_width=3)
    line_f03 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f03, color="red", line_width=3)
    # Finger 1
    line_f11 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f11, color="red", line_width=3)
    line_f12 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f12, color="red", line_width=3)
    line_f13 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f13, color="red", line_width=3)
    # Finger 2
    line_f21 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f21, color="red", line_width=3)
    line_f22 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f22, color="red", line_width=3)
    line_f23 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f23, color="red", line_width=3)
    # Finger 3
    line_f31 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f31, color="red", line_width=3)
    line_f32 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f32, color="red", line_width=3)
    line_f33 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f33, color="red", line_width=3)
    # Finger 4
    line_f41 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f41, color="red", line_width=3)
    line_f42 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f42, color="red", line_width=3)
    line_f43 = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f43, color="red", line_width=3)

    title1 = plotter.add_text('Real World', position='upper_left', color='red',shadow=False, font_size=20)

    plotter.add_axes()
    plotter.show_grid()
    #plotter.camera_position = 'xy'
    plotter.view_vector((0,0,-1), viewup=(0,1,0)) # View along z axis. y axis up
    plotter.set_focus((0.5,0.5,0))
    plotter.set_position([0.5, 0.5, 1.0])
    plotter.camera_position = [(0., 0.5, 3.2),(0., 0.5, 0.0),(0.0, 1.0, 0.0)]

    # Subplot B
    plotter.subplot(0,1)
    # Point cloud
    point_cloud_b = pv.PolyData(np.zeros((21,3)))
    plotter.add_mesh(point_cloud_b, color='maroon', point_size=10.0, render_points_as_spheres=True)
    # Palm lines
    line_r1b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_r1b, color="green", line_width=3)
    line_r2b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_r2b, color="green", line_width=3)
    line_r3b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_r3b, color="green", line_width=3)
    # Thumb
    line_f00b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f00b, color="red", line_width=3)
    line_f01b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f01b, color="red", line_width=3)
    line_f02b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f02b, color="red", line_width=3)
    line_f03b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f03b, color="red", line_width=3)
    # Finger 1
    line_f11b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f11b, color="red", line_width=3)
    line_f12b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f12b, color="red", line_width=3)
    line_f13b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f13b, color="red", line_width=3)
    # Finger 2
    line_f21b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f21b, color="red", line_width=3)
    line_f22b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f22b, color="red", line_width=3)
    line_f23b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f23b, color="red", line_width=3)
    # Finger 3
    line_f31b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f31b, color="red", line_width=3)
    line_f32b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f32b, color="red", line_width=3)
    line_f33b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f33b, color="red", line_width=3)
    # Finger 4
    line_f41b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f41b, color="red", line_width=3)
    line_f42b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f42b, color="red", line_width=3)
    line_f43b = pv.Line(pointa=(0.0, 0.0, 0.0), pointb=(0.0, 0.0, 0.0)); plotter.add_mesh(line_f43b, color="red", line_width=3)

    title2 = plotter.add_text('Palm-Centered', position='upper_left', color='red',shadow=False, font_size=20)

    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position = [(0., 0.8, 8.2),(0., 0.8, 0.0),(0.0, 1.0, 0.0)]

    # List of actors to return
    actors = [point_cloud,arrow_b1,arrow_b2,arrow_b3,
              line_r1,line_r2,line_r3, # Palm lines
              line_f00,line_f01,line_f02,line_f03, # Thumb
              line_f11,line_f12,line_f13, # Finger 1
              line_f21,line_f22,line_f23, # Finger 2
              line_f31,line_f32,line_f33, # Finger 3
              line_f41,line_f42,line_f43, # Finger 4
             ]
    
    actorsb = [point_cloud_b,
              line_r1b,line_r2b,line_r3b, # Palm lines
              line_f00b,line_f01b,line_f02b,line_f03b, # Thumb
              line_f11b,line_f12b,line_f13b, # Finger 1
              line_f21b,line_f22b,line_f23b, # Finger 2
              line_f31b,line_f32b,line_f33b, # Finger 3
              line_f41b,line_f42b,line_f43b, # Finger 4
             ]


    return plotter, actors, actorsb


def update_3dplot(plotter,actors,actorsb,points,POINTS,b1,b2,b3):

    # Extract actor items
    [point_cloud,arrow_b1,arrow_b2,arrow_b3,
    line_r1,line_r2,line_r3, # Palm lines
    line_f00,line_f01,line_f02,line_f03, # Thumb
    line_f11,line_f12,line_f13, # Finger 1
    line_f21,line_f22,line_f23, # Finger 2
    line_f31,line_f32,line_f33, # Finger 3
    line_f41,line_f42,line_f43, # Finger 4
    ] = actors
    # Extract actor items for plot b
    [point_cloud_b,
    line_r1b,line_r2b,line_r3b, # Palm lines
    line_f00b,line_f01b,line_f02b,line_f03b, # Thumb
    line_f11b,line_f12b,line_f13b, # Finger 1
    line_f21b,line_f22b,line_f23b, # Finger 2
    line_f31b,line_f32b,line_f33b, # Finger 3
    line_f41b,line_f42b,line_f43b, # Finger 4
    ] = actorsb

    # Select left plot
    plotter.subplot(0,0)
    point_cloud.overwrite(pv.PolyData(points)) # Hand landmarks
    # Palm
    line_r1.overwrite(pv.Line(pointa=points[0,:], pointb=points[5,:])) # Line r1
    line_r2.overwrite(pv.Line(pointa=points[0,:], pointb=points[17,:])) # Line r2
    line_r3.overwrite(pv.Line(pointa=points[5,:], pointb=points[17,:])) # Line r3
    # Thumb
    line_f00.overwrite(pv.Line(pointa=points[0,:], pointb=points[1,:])) # Wrist to base of thumb
    line_f01.overwrite(pv.Line(pointa=points[1,:], pointb=points[2,:])) # Inner
    line_f02.overwrite(pv.Line(pointa=points[2,:], pointb=points[3,:])) # Middle
    line_f03.overwrite(pv.Line(pointa=points[3,:], pointb=points[4,:])) # Outer
    # Finger 1
    line_f11.overwrite(pv.Line(pointa=points[5,:], pointb=points[6,:])) # Inner
    line_f12.overwrite(pv.Line(pointa=points[6,:], pointb=points[7,:])) # Middle
    line_f13.overwrite(pv.Line(pointa=points[7,:], pointb=points[8,:])) # Outer
    # Finger 2
    line_f21.overwrite(pv.Line(pointa=points[9,:], pointb=points[10,:])) # Inner
    line_f22.overwrite(pv.Line(pointa=points[10,:], pointb=points[11,:])) # Middle
    line_f23.overwrite(pv.Line(pointa=points[11,:], pointb=points[12,:])) # Outer
    # Finger 3
    line_f31.overwrite(pv.Line(pointa=points[13,:], pointb=points[14,:])) # Inner
    line_f32.overwrite(pv.Line(pointa=points[14,:], pointb=points[15,:])) # Middle
    line_f33.overwrite(pv.Line(pointa=points[15,:], pointb=points[16,:])) # Outer
    # Finger 4
    line_f41.overwrite(pv.Line(pointa=points[17,:], pointb=points[18,:])) # Inner
    line_f42.overwrite(pv.Line(pointa=points[18,:], pointb=points[19,:])) # Middle
    line_f43.overwrite(pv.Line(pointa=points[19,:], pointb=points[20,:])) # Outer
    
    # Basis vectors
    arrow_b1.overwrite(pv.Line(pointa=points[0,:], pointb=points[0,:]+0.15*b1)) # b1
    arrow_b2.overwrite(pv.Line(pointa=points[0,:], pointb=points[0,:]+0.15*b2)) # b2
    arrow_b3.overwrite(pv.Line(pointa=points[0,:], pointb=points[0,:]+0.15*b3)) # b3

    # 2nd Plot (transformed)
    plotter.subplot(0, 1)
    point_cloud_b.overwrite(pv.PolyData(POINTS)) # Hand landmarks
    # Palm
    line_r1b.overwrite(pv.Line(pointa=POINTS[0,:], pointb=POINTS[5,:])) # Line r1
    line_r2b.overwrite(pv.Line(pointa=POINTS[0,:], pointb=POINTS[17,:])) # Line r2
    line_r3b.overwrite(pv.Line(pointa=POINTS[5,:], pointb=POINTS[17,:])) # Line r3
    # Thumb
    line_f00b.overwrite(pv.Line(pointa=POINTS[0,:], pointb=POINTS[1,:])) # Wrist to base of thumb
    line_f01b.overwrite(pv.Line(pointa=POINTS[1,:], pointb=POINTS[2,:])) # Inner
    line_f02b.overwrite(pv.Line(pointa=POINTS[2,:], pointb=POINTS[3,:])) # Middle
    line_f03b.overwrite(pv.Line(pointa=POINTS[3,:], pointb=POINTS[4,:])) # Outer
    # Finger 1
    line_f11b.overwrite(pv.Line(pointa=POINTS[5,:], pointb=POINTS[6,:])) # Inner
    line_f12b.overwrite(pv.Line(pointa=POINTS[6,:], pointb=POINTS[7,:])) # Middle
    line_f13b.overwrite(pv.Line(pointa=POINTS[7,:], pointb=POINTS[8,:])) # Outer
    # Finger 2
    line_f21b.overwrite(pv.Line(pointa=POINTS[9,:], pointb=POINTS[10,:])) # Inner
    line_f22b.overwrite(pv.Line(pointa=POINTS[10,:], pointb=POINTS[11,:])) # Middle
    line_f23b.overwrite(pv.Line(pointa=POINTS[11,:], pointb=POINTS[12,:])) # Outer
    # Finger 3
    line_f31b.overwrite(pv.Line(pointa=POINTS[13,:], pointb=POINTS[14,:])) # Inner
    line_f32b.overwrite(pv.Line(pointa=POINTS[14,:], pointb=POINTS[15,:])) # Middle
    line_f33b.overwrite(pv.Line(pointa=POINTS[15,:], pointb=POINTS[16,:])) # Outer
    # Finger 4
    line_f41b.overwrite(pv.Line(pointa=POINTS[17,:], pointb=POINTS[18,:])) # Inner
    line_f42b.overwrite(pv.Line(pointa=POINTS[18,:], pointb=POINTS[19,:])) # Middle
    line_f43b.overwrite(pv.Line(pointa=POINTS[19,:], pointb=POINTS[20,:])) # Outer
    
    plotter.render()


    # Update actors lists before returning
    actors = [point_cloud,arrow_b1,arrow_b2,arrow_b3,
              line_r1,line_r2,line_r3, # Palm lines
              line_f00,line_f01,line_f02,line_f03, # Thumb
              line_f11,line_f12,line_f13, # Finger 1
              line_f21,line_f22,line_f23, # Finger 2
              line_f31,line_f32,line_f33, # Finger 3
              line_f41,line_f42,line_f43, # Finger 4
             ]
    actorsb = [point_cloud_b,
              line_r1b,line_r2b,line_r3b, # Palm lines
              line_f00b,line_f01b,line_f02b,line_f03b, # Thumb
              line_f11b,line_f12b,line_f13b, # Finger 1
              line_f21b,line_f22b,line_f23b, # Finger 2
              line_f31b,line_f32b,line_f33b, # Finger 3
              line_f41b,line_f42b,line_f43b, # Finger 4
             ]

    return plotter, actors, actorsb


def update_image_plot(image,p1,p2,p3,b1,b2,b3,height,width,palm_length,pred_label):
    
    # Draw palm (r1,r2,r1-r2)
    cv2.line(image, 
            (int(p1[0]*width),int(p1[1]*height)), 
            (int(p2[0]*width),int(p2[1]*height)), 
            (0, 255, 0), 
            2)
    cv2.line(image, 
            (int(p1[0]*width),int(p1[1]*height)), 
            (int(p3[0]*width),int(p3[1]*height)), 
            (0, 255, 0), 
            2)
    cv2.line(image, 
            (int(p2[0]*width),int(p2[1]*height)), 
            (int(p3[0]*width),int(p3[1]*height)), 
            (0, 255, 0), 
            2)
    # Draw basis vectors
    cv2.line(image, 
            (int(p1[0]*width),int(p1[1]*height)), 
            (int((p1[0]+0.5*palm_length*b1[0])*width),int((p1[1]+0.5*palm_length*b1[1])*height)), 
            (0, 0, 255), 
            2)
    cv2.line(image, 
            (int(p1[0]*width),int(p1[1]*height)), 
            (int((p1[0]+0.5*palm_length*b2[0])*width),int((p1[1]+0.5*palm_length*b2[1])*height)), 
            (0, 0, 255), 
            2)
    cv2.line(image, 
            (int(p1[0]*width),int(p1[1]*height)), 
            (int((p1[0]+0.5*palm_length*b3[0])*width),int((p1[1]+0.5*palm_length*b3[1])*height)), 
            (0, 0, 255), 
            2)

    # Display predicted label
    if pred_label is not None:
        cv2.putText(image, str(pred_label), 
                            (int(0.1*width),int(0.1*height)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA
                                )

    return image






#%% Main function

if __name__ == "__main__":
    
    # # ASL Dataset
    # base_dir = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data\asl_dataset')
    # hand_landmarks_file = r'E:\Files\Zero Robotics\HandGuestures\Training Data\asl_dataset\hand_landmarks_asl_dataset.csv'
    
    # # 27 Class Sign Language
    # base_dir = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data\27 Class Sign Language Dataset')
    # hand_landmarks_file = r'E:\Files\Zero Robotics\HandGuestures\Training Data\27 Class Sign Language Dataset\hand_landmarks_27_class_sign_language_dataset.csv'
    
    # Read dataset
    data_dir = Path(r'E:\Files\Zero Robotics\HandGuestures\Training Data')
    img_dir_hagrid = Path(r'E:\Files\Zero Robotics\HandGuestures\mmpretrain\data\hagrid\hagrid_dataset_512')
    dfhagrid = read_training_data(data_dir,'Hagrid_ASL0_9')
    inspect_images(img_dir_hagrid,dfhagrid, classes=['palm']) # Hagrid