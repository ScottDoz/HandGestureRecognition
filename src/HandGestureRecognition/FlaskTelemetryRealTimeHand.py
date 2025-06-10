# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:46:23 2025

@author: scott
"""

import cv2
import numpy as np
import mediapipe as mp
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from flask import Response, Flask
import threading
import time
from collections import deque
import pdb

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

# Load tflite model
interpreter = load_model(model_name)
print("Input shape expected:", interpreter.get_input_details()[0]['shape'])

#%% Set up data

# Store x-values and timestamps for the moving graph
time_window = 10  # seconds
time_data = deque(maxlen=1000)
start_time = time.time()

x1_data = deque(maxlen=1000)

# Phi data
phi_metrics = ['phi0','phi1','phi2','phi3','phi4']
phi_data = {feature: deque(maxlen=1000) for feature in phi_metrics}
phi_indices = [i for i, val in enumerate(features) if val in set(phi_metrics)]


# Buffer to store time-series data
history_length = 100  # Number of frames to store
history = {feature: deque(maxlen=history_length) for feature in features}
time_buffer = deque(maxlen=history_length)


#%% Flask app
server = Flask(__name__)

# Dash app
app = dash.Dash(__name__, server=server)

# Video capture
cap = cv2.VideoCapture(0)

# Function to capture video frames
def generate_frames():
    global x1_data, phi_data, time_data, start_time

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        
        while True:
            success, frame = cap.read()
            if not success:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            image = cv2.flip(image, 1)  
            results = hands.process(image)

            # Get image size
            height, width, _ = image.shape

            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

                # Extract hand landmark points
                points = np.vstack([(float(hand.landmark[i].x), float(hand.landmark[i].y), float(hand.landmark[i].z)) for i in range(21)])

                # Transform to palm-centered frame
                POINTS, p1, p2, p3, b1, b2, b3, palm_length = transform_hand_landmarks_to_palm_centered_frame(points)

                # Compute finger metrics
                finger_metrics = compute_finger_metrics(points, POINTS) # Dict
                

                # Extract feature values
                x = np.array([finger_metrics[f] for f in features])

                # Predict gesture
                pred_label = interpreter_predict(interpreter, x, labels=classes)

                # Display prediction on video
                if pred_label is not None:
                    cv2.putText(image, str(pred_label), (int(0.05*width), int(0.1*height)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2, cv2.LINE_AA)

                # Store x values for the moving graph
                # x1_data.append(x[1])  # Example: Tracking the first feature
                for i,f in zip(phi_indices,phi_metrics):
                    phi_data[f].append(x[i])  # Append new values to each deque
                
                
            else:
                # Append NaN when no hand is detected
                # x1_data.append(np.nan)
                for f in phi_metrics:
                    phi_data[f].append(np.nan)  # Append new values to each deque
            
            # Store current time values for the moving graph
            current_time = time.time() - start_time
            time_data.append(current_time)

            # Convert frame for streaming
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

    html.Div([
        # Video on the left (keeps aspect ratio)
        html.Img(src="/video_feed", style={
            "width": "30%",  # Fixed width, height auto to maintain aspect ratio
            "height": "auto",
            "border": "2px solid black"
        }),

        # Right side with graph on top, image below
        html.Div([
            dcc.Graph(id='x-value-plot', style={"height": "350px"}),  # Line plot on top
            html.Img(src="/assets/Hand_Gestures_23.png", style={
                "width": "100%",  # Full width of right side
                "height": "auto",  # Auto height to maintain aspect ratio
                "border": "2px solid black",
                "object-fit": "contain"
            })  # Image below the graph
        ], style={
            "display": "flex",
            "flex-direction": "column",
            "width": "50%"  # Fixed width for the right side
        })

    ], style={"display": "flex", "align-items": "flex-start"}),  # Align items at the top

    # Interval Component to Update Graph Every 100 ms
    dcc.Interval(id='interval-component', interval=100, n_intervals=0)
])


# app.layout = html.Div([
#     html.H1("Hand Gesture Recognition"),
    
    
#     html.Div([
#         html.Img(src="/video_feed", style={"width": "40%", "height": "40%", "border": "2px solid black"}),
#         html.Img(src="/assets/Hand_Gestures_23.png", style={
#             "width": "auto",  # Auto width to maintain aspect ratio
#             "height": "300px",  # Set max height to fit nicely
#             "border": "2px solid black",
#             "object-fit": "contain"
#         })
#     ], style={"display": "flex", "justify-content": "space-around"}),

#     # Graph for X Values
#     dcc.Graph(id='x-value-plot', style={"height": "350px"}),

#     # Interval Component to Update Graph Every 100 ms
#     dcc.Interval(id='interval-component', interval=100, n_intervals=0)
# ])

#%% Callback for Live Updating the Graph
@app.callback(
    Output('x-value-plot', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    # Remove old data beyond 10s
    while len(time_data) > 1 and time_data[0] < (time_data[-1] - time_window):
        time_data.popleft()
        # x1_data.popleft()
        for feature in phi_metrics: phi_data[feature].popleft()

    # Create the graph
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=list(time_data), y=list(x1_data), mode='lines', legendgroup='phi', line=dict(color='blue'), name=features[0]))
    for f in phi_metrics:
        fig.add_trace(go.Scatter(x=list(time_data), y=list(phi_data[f]), mode='lines', name=f))
    
    
    # Set graph properties
    fig.update_layout(
        title="Hand Metrics",
        xaxis_title="Time (s)",
        yaxis_title="Knuckle angle (rad)",
        showlegend=True, uirevision=None,
        # xaxis=dict(range=[max(0, time_data[-1] - time_window), time_data[-1]] if time_data else [0, 10]),
        xaxis=dict(range=[time_data[-1] - time_window, time_data[-1]] if time_data else [-time_window, 0]),

        # yaxis=dict(range=[min(x1_data) - 0.1, max(x1_data) + 0.1] if x1_data else [-1, 1]),
        yaxis=dict(range=[0, 3.15] if x1_data else [0, 3.2]),
        # yaxis=dict(yrange=[0,3.15]),
        template="plotly_dark"
    )
    return fig

#%% Run Flask and Dash
if __name__ == '__main__':
    app.run(debug=True)
