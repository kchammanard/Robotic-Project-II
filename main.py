import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
import math
import time
from roboflow import Roboflow

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

rf = Roboflow(api_key="xryMW9j32H1tQ20ehfWF")
project = rf.workspace().project("american-sign-language-letters-l980l")
model = project.version(1).model

def hands_feed():
    while True:
        ret,frame = cap.read()
        hands = detector.findHands(frame, draw=False)
        #hands, frame = detector.findHands(frame)
        if hands:
            hand = hands[0]
            x,y,w,h = hand['bbox']

            imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255  #uint8 is 0-255
            imgCrop = frame[y-offset:y+h+offset,x-offset:x+w+offset]

            aspectRatio = h/w

            if aspectRatio > 1:
                k = imgSize/h
                width = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop,(width,imgSize))
                wGap = math.ceil((imgSize-width)/2)
                imgWhite[:,wGap:width+wGap] = imgResize

            else:
                k = imgSize/w
                height = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop,(imgSize,height))
                hGap = math.ceil((imgSize-height)/2)
                imgWhite[hGap:height+hGap,:] = imgResize


            cv2.imshow("Cropped Image",imgCrop)
            cv2.imshow("Whites", imgWhite)

        cv2.imshow("frame",frame)
        if cv2.waitKey(1) == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__== "__main__":
    hands_feed()
