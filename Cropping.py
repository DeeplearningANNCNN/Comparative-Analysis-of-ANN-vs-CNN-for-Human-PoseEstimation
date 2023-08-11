# Import required libraries

import cv2
import mediapipe as mp
import time
import numpy as np
import keyboard
import datetime

#Define MediaPipe Objects
mpDraw = mp.solutions.drawing_utils 
mpPose = mp.solutions.pose
pose = mpPose.Pose()


#Define VideoCapture object
cap = cv2.VideoCapture(0)


#Define pressing time and Key press object
pTime = 0
key = keyboard.read_key()



cx = [0 for x in range(33)]
cy = [0 for y in range(33)]

if key == 'z':
    time.sleep(5)
    start_time = datetime.datetime.now()
    i=503
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        image = np.zeros((480, 640, 3))

        if results.pose_landmarks:
            mpDraw.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(h, w, c)
                print(id, lm)
                cx[id], cy[id] = int(lm.x * w), int(lm.y * h)
                x_max = max(cx)
                x_min = min(cx)
                y_max = max(cy)
                y_min = min(cy)
                roi = image[y_min-10:y_max+10, x_min-10:x_max+10]
                #cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            path = f"D:\\Pycharm\Anomalous_detection_CNN\\data_four_poses\\test\\normal\\Image{i}.png"
            cv2.imwrite(path, roi)
            i = i + 1

        later_time = datetime.datetime.now()
        difference = later_time - start_time
        sec_diff = difference.total_seconds()
        sec_diff = round(sec_diff)
        #print(sec_diff)
        #print("\n")
        if sec_diff == 2:
            break

        cv2.imshow("Image", img)
        cv2.waitKey(80)