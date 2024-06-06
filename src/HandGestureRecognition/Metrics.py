# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:29:09 2024

@author: scott

Metrics
-------

Define data transformations and metrics used to predict hand gestures.

"""

import numpy as np
import pdb


#%% Geometric transformations

def transform_hand_landmarks_to_palm_centered_frame(points):

    # Extract key points of the palm
    p1 = points[0,:]  # Wrist
    p2 = points[5,:]  # Index finger MCP (Metacarpophalangeal Joint)
    p3 = points[17,:] # Pinky finger MCP (Metacarpophalangeal Joint)


    # Define unit vectors and their length
    r1mag = np.linalg.norm(p2-p1)
    r2mag = np.linalg.norm(p3-p1)
    r3mag = np.linalg.norm(p3-p2)
    r1u = (p2 - p1)/r1mag # Wrist to Index unit vector
    r2u = (p3 - p1)/r2mag # Wrist to Pinky unit vector

    # Length of bisector
    l2 = (r1mag*r2mag/(r1mag+r2mag)**2)*( (r1mag+r2mag)**2 - r3mag**2) # Length of bisector angle
    l = np.sqrt(l2)
    palm_length = l # Define this as the palm length


    # Define Basis vectors for palm plane
    b2 = (r1u + r2u)/np.linalg.norm(r1u+r2u) # Bisector of r1, r2
    b3 = np.cross(r2u,r1u); b3 = b3/np.linalg.norm(b3) # Normal to palm
    b1 = np.cross(b2,b3); b1 = b1/np.linalg.norm(b1) # Complete the pair


    # Apply coordinate transform

    # Transform positions in image space to basis
    POINTS = points.copy()
    POINTS = POINTS - np.tile(POINTS[0,:],(len(points),1)) # Translate to center at wrist


    # Change of basis into {b1,b2,b3}
    A = np.column_stack((b1,b2,b3)) # Transformation matrix [b1,b2,b3]
    #POINTS = A.dot(points.T).T # Apply transformation matrix X = A*x
    # [X1  ...   Xn]   [a11 a12 a13]   [x1  ...   xn]
    # [Y1  ...   Yn] = [a21 a22 a23] * [y1  ...   yn] 
    # [Z1  ....  Zn]   [a31 a32 a33]   [z1  ....  zn]
    # (Transformed POINTS)   (Matrix)   *     (Origianl points)
    POINTS = np.matmul(A.T,POINTS.T).T


    # Scale points by length of the palm
    POINTS /= palm_length

    # Check that points 0,5,17 are all alighned with the plane
    # print(POINTS[0,2])
    #print(POINTS[5,2])
    #print(POINTS[17,2])
    # Note: close enough for now (within 0.01). Try to improve precision later

    return POINTS, p1,p2,p3, b1, b2, b3, palm_length

#%% Finger Metrics
def compute_finger_metrics(points,POINTS):
    '''
    Compute finger metrics for a single hand.
    points is a 21x3 array of the x,y,z coordinates of each hand landmark

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.
    POINTS : TYPE
        DESCRIPTION.

    Returns
    -------
    metrics : TYPE
        DESCRIPTION.

    '''
    
    # Compute basis functions again -------------
    
    # Extract key points of the palm
    p1 = points[0,:]  # Wrist
    p2 = points[5,:]  # Index finger MCP (Metacarpophalangeal Joint)
    p3 = points[17,:] # Pinky finger MCP (Metacarpophalangeal Joint)
    
    # Define unit vectors and their length
    r1mag = np.linalg.norm(p2-p1)
    r2mag = np.linalg.norm(p3-p1)
    r3mag = np.linalg.norm(p3-p2)
    r1u = (p2 - p1)/r1mag # Wrist to Index unit vector
    r2u = (p3 - p1)/r2mag # Wrist to Pinky unit vector

    # Length of bisector
    l2 = (r1mag*r2mag/(r1mag+r2mag)**2)*( (r1mag+r2mag)**2 - r3mag**2) # Length of bisector angle
    l = np.sqrt(l2)
    palm_length = l # Define this as the palm length


    # Define Basis vectors for palm plane
    b2 = (r1u + r2u)/np.linalg.norm(r1u+r2u) # Bisector of r1, r2
    b3 = np.cross(r2u,r1u); b3 = b3/np.linalg.norm(b3) # Normal to palm
    b1 = np.cross(b2,b3); b1 = b1/np.linalg.norm(b1) # Complete the pair
    
    
    # -----------------
    
    # Thumb to fingertip (normalized)
    # Compute distance from tip of thumb to tips of fingers and to base of fingers
    TT1n = np.linalg.norm(POINTS[8,:] - POINTS[4,:]) # Tip finger 1
    TT2n = np.linalg.norm(POINTS[12,:] - POINTS[4,:]) # Tip finger 2
    TT3n = np.linalg.norm(POINTS[16,:] - POINTS[4,:]) # Tip finger 3
    TT4n = np.linalg.norm(POINTS[20,:] - POINTS[4,:]) # Tip finger 4
    
    # Thumb to base of fingers (normalized)
    # Distances from tip of thumb to base of fingers
    TB1n = np.linalg.norm(POINTS[5,:] - POINTS[4,:])  # Base finger 1
    TB2n = np.linalg.norm(POINTS[9,:] - POINTS[4,:]) # Base finger 2
    TB3n = np.linalg.norm(POINTS[13,:] - POINTS[4,:]) # Base finger 3
    TB4n = np.linalg.norm(POINTS[17,:] - POINTS[4,:]) # Base finger 4
    
    # Distances between fingers (normalized)
    D12n = np.linalg.norm(POINTS[8,:] - POINTS[12,:])   # Fingers 1 and 2
    D23n = np.linalg.norm(POINTS[12,:] - POINTS[16,:])  # Fingers 2 and 3
    D34n = np.linalg.norm(POINTS[16,:] - POINTS[20,:])  # Fingers 3 and 4
    
    # Middle Knuckle joint angles
    # TODO: Change to use real-world coords
    phi0 = calculate_joint_angle(POINTS[2,:],POINTS[3,:],POINTS[4,:]) # Thumb
    phi1 = calculate_joint_angle(POINTS[5,:],POINTS[6,:],POINTS[7,:]) # 1st digit at knuckle
    phi2 = calculate_joint_angle(POINTS[9,:],POINTS[10,:],POINTS[11,:]) # 2st digit at knuckle
    phi3 = calculate_joint_angle(POINTS[13,:],POINTS[14,:],POINTS[15,:]) # 3rd digit at knuckle
    phi4 = calculate_joint_angle(POINTS[17,:],POINTS[18,:],POINTS[19,:]) # 3rd digit at knuckle
    
    
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
    # if u[2]<0: elev3*=-1 # Correct for negative angles
    # 4st digit
    u = POINTS[18,:]-POINTS[17,:] # Vector along digit
    elev4 = np.arctan2(u[2],np.sqrt( u[0]**2 + u[1]**2 )) # Index finger (from index knuckle pt 6 relative to pt 5)
    if u[2]<0: elev4*=-1 # Correct for negative angles
    # Convert to joint angles (supplementary angle) = 180 - elev
    theta1 = np.pi - elev1
    theta2 = np.pi - elev2
    theta3 = np.pi - elev3
    theta4 = np.pi - elev4

    # New approach:
    # Compute the polar angle from the palm normal vector N to the digit
    # c = np.dot(u,v)/norm(u)/norm(v) # -> cosine of the angle
    # angle = arccos(clip(c, -1, 1)) # if you really want the angle
    #u = POINTS[6,:]-POINTS[5,:]; u = u/np.linalg.norm(u) # Unit vector
    #v = np.array([0,0,1])
    #phi1 = np.arccos(np.clip(np.dot(u,v), -1, 1)) # if you really want the angle
    
    
    # Palm orientation angle (alpha)
    
    # Get projection of palm normal in xz plane
    b3proj = b3.copy(); b3proj[1] = 0;
    b3proj = b3proj/np.linalg.norm(b3proj,axis=-1)
    # Compute angle from palm normal projection to -k vector [0,0,-1]
    # TODO: Look into orientation of z axis. For now, use -k since it works
    cosine_angle = np.dot(b3proj,np.array([0,0,-1]))
    alpha = np.arccos(cosine_angle)
    # print("Palm orientation angle: {} deg".format(np.rad2deg(alpha)))


    # # Collect parameters for return
    # params = [phi0,phi1,phi2,phi3,phi4,theta1,theta2,theta3,theta4,TT1n,TT2n,TT3n,TT4n,TB1n,TB2n,TB3n,TB4n]
    
    # Collect metrics in dictionary
    metrics = {'TT1n':TT1n,'TT2n':TT2n,'TT3n':TT3n,'TT4n':TT4n, # Thumb to fingertips (normalized)
               'TB1n':TB1n,'TB2n':TB2n,'TB3n':TB3n,'TB4n':TB4n, # Thumb to base of fingers (normalized)
               'D12n':D12n,'D23n':D23n,'D34n':D34n,             # Fingertips between digits (normalized)
               'phi0':phi0,'phi1':phi1,'phi2':phi2,'phi3':phi3,'phi4':phi4,     # Middle knuckle angles
               'theta1':theta1,'theta2':theta2,'theta3':theta3,'theta4':theta4, # Finger elevations
               'alpha':alpha, # Palm orientation angle
               }
    
    return metrics

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

