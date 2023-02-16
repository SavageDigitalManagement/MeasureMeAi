import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
mp_drawing = mp.solutions.mediapipe.python.solutions.drawing_utils
mp_pose = mp.solutions.mediapipe.python.solutions.pose

angle_position = []
time_position = []
frame_rate = 144
frame_interval = 1/frame_rate
class MeasureMeAi:
    i = 0
    t = float(0)

    def __init__(self) -> None:
         pass

    def turnLeft():
        try:
            landmarks= results.pose_landmarks.landmark
            #print(landmarks)
            # Coords
            lShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            lHeel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            rShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            angle = calculate_angle(lHeel, lShoulder, rShoulder)
            angle_position.append(float(angle))
            print(angle_position)
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(lShoulder, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA
                                )
        except Exception as e:
            print(e)
            pass

    def turnRight():
        try:
            landmarks= results.pose_landmarks.landmark
            #print(landmarks)
            # Coords
            rShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            rHeel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            lShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            angle = calculate_angle(rHeel, lShoulder, rShoulder)
            
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(lShoulder, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA
                                )
        except Exception as e:
            print(e)
            pass

    
    def jete_left():
        try:
            landmarks= results.pose_landmarks.landmark
            #print(landmarks)
            # Coords
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            lHeel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            
            angle = calculate_angle(lHeel, left_hip, right_heel)
            
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA
                                )
        except Exception as e:
            print(e)
            pass

    
    def jete_right():
        try:
            landmarks= results.pose_landmarks.landmark
            #print(landmarks)
            # Coords
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            rHeel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            lHeel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            
            angle = calculate_angle(rHeel, right_hip, lHeel)
            
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA
                                )
        except Exception as e:
            print(e)
            pass

    def arabesque_left():
        try:
            landmarks= results.pose_landmarks.landmark
            #print(landmarks)
            # Coords
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            rHeel = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            lHeel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            
            angle = calculate_angle(rHeel, right_hip, lHeel)
            
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA
                                )
        except Exception as e:
            print(e)
            pass

    def arabesque_right():
        try:
            landmarks= results.pose_landmarks.landmark
            #print(landmarks)
            # Coords
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            lHeel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            
            angle = calculate_angle(right_shoulder, left_hip, lHeel)
            
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA
                                )
        except Exception as e:
            print(e)
            pass


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle
def counter():
    while True:
        MeasureMeAi.i += 1
        print(MeasureMeAi.i)
        time.sleep(1)
#Disply Starts Here
cap = cv2.VideoCapture(0)
timer = threading.Thread(target=counter)
timer.start()
while True:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence = 0.5) as pose:
        while cap.isOpened():
            ret,frame = cap.read()


            #Detect and render
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #Extract Landmarks
            try:
                MeasureMeAi.turnLeft()
                #MeasureMeAi.turnRight()
                #MeasureMeAi.jete_left()
                #MeasureMeAi.jete_right()
                #MeasureMeAi.arabesque_left()
                #MeasureMeAi.arabesque_right()
                #Next step map hotkeys to swap modes.
            except Exception as e:
                print(e)
                pass


            #Render landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)





            cv2.imshow("Window", image)
            print(MeasureMeAi.i)
            time.sleep(frame_interval)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()