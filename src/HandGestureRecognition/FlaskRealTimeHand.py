# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:48:45 2025

@author: scott
"""

import cv2
import numpy as np
import mediapipe as mp
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from flask import Response, Flask
import threading
from collections import deque

# Module imports
from Metrics import *
from Models import *

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#%% Inputs

# Select Model
# model_name = "Hagrid_ASL0_9"
# model_name = "ASL0_9"
model_name = "Hagrid_ASL0_9_v2"


# Predict metrics
if model_name == 'Hagrid_ASL0_9':
    # Trained on 17 input metrics
    features = ['phi0','phi1','phi2','phi3','phi4','theta1','theta2','theta3','theta4','TT1n','TT2n','TT3n','TT4n','TB1n','TB2n','TB3n','TB4n']
    classes = classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'call', 'dislike', 'fist', 'like', 'mute', 'ok', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three2', 'two_up', 'two_up_inverted']
elif model_name == 'ASL0_9':
    # Trained on 17 input metrics
    features = ['phi0','phi1','phi2','phi3','phi4','theta1','theta2','theta3','theta4','TT1n','TT2n','TT3n','TT4n','TB1n','TB2n','TB3n','TB4n']
    classes = classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
elif model_name == 'Hagrid_ASL0_9_v2':
    # New version Trained on  input metrics
    classes = classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'call', 'dislike', 'fist', 'like', 'mute', 'ok', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three2', 'two_up', 'two_up_inverted']
    features = ['phi0','phi1','phi2','phi3','phi4',  # Knuckle angles
                'theta1','theta2','theta3','theta4', # Finger elevations 
                'TT1n','TT2n','TT3n','TT4n',         # Thumb to finger tip
                'TB1n','TB2n','TB3n','TB4n',         # Thumb to base of fingers
                'D12n','D23n','D34n', # Distances between fingers
                'alpha',              # Palm orientation angle
                ]

#%% Flask app
server = Flask(__name__)

# Dash app
app = dash.Dash(__name__, server=server)

# Video capture
cap = cv2.VideoCapture(0)

# Load tflite model
interpreter = load_model(model_name)
print("Input shape expected:", interpreter.get_input_details()[0]['shape'])
# print("Input data received:", x.shape)

# 



# Function to capture video frames
def generate_frames():
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                
                # Get video frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR 2 RGB
                image = cv2.flip(image, 1) # Flip on horizontal
                results = hands.process(image)
                
                # Get image size
                height, width, _ = image.shape
                
                # Hand landmarks
                if results.multi_hand_landmarks:
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                                )
                    
                    # Extract all points as 21x3 array
                    points = np.vstack([(float(hand.landmark[i].x),float(hand.landmark[i].y),float(hand.landmark[i].z)) for i in range(21)])

                    # Coordinate transform to Palm-Centered Frame
                    POINTS, p1,p2,p3, b1, b2, b3, palm_length = transform_hand_landmarks_to_palm_centered_frame(points)

                    # Finger Angle Metrics
                    finger_metrics = compute_finger_metrics(points,POINTS)
                    
                    
                    # # Predict gesture using model
                    # x = np.array(params)
                    # ypred = model.predict([x])
                    # pred_label = ypred[0]
                    # pred_label = 0
                    
                    
                    
                    # Extract metrics
                    x = [finger_metrics[f] for f in features]
                    x = np.array(x)
                    # Make prediction
                    pred_label = interpreter_predict(interpreter,x,labels=classes)
                    
                    # Display predicted label
                    if pred_label is not None:
                        cv2.putText(image, str(pred_label), 
                                            (int(0.1*width),int(0.1*height)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA
                                                )
                        



                    
                
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask route to stream video
@server.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#%% Dash layout
app.layout = html.Div([
    html.H1("Hand Gesture Recognition"),
    html.Img(src="/video_feed", style={"width": "40%", "border": "2px solid black"}),
])

# Run Flask and Dash
if __name__ == '__main__':
    

    
    app.run(debug=True)
