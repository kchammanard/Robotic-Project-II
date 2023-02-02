import serial
import time
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ser = serial.Serial("COM11", '9600', timeout=2)

Xposition = 90
Yposition = 90

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (300, 300))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_lower = np.array([161, 155, 84], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, red_lower, red_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    rows, cols, _ = frame.shape
    center_x = int(rows / 2)
    center_y = int(cols / 2)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "x = "+str(x)+"y = "+str(y)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255))
        #////////////////////////////////////////////////////////////////////
        medium_x = int((x + x+w)/2)
        medium_y = int((y + y+h)/2)
        #////////////////////////////////////////////////////////////////////
        cv2.line(frame, (medium_x,0),(medium_x,600),(0,255,0),2)
        text2 = "mediumX = " + str(medium_x)
        cv2.putText(frame, text2, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 50))
        #////////////////////////////////////////////////////////////////////
        cv2.line(frame, (0, medium_y), (600, medium_y), (0, 255, 0), 2)
        text3 = "mediumY = " + str(medium_y)
        cv2.putText(frame, text3, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 50))

        # ////////////////////////////////////////////////////////////////////
        if medium_x > center_x +40:
            Xposition += 2
            ser.write((str(Xposition) + 'a').encode('utf-8'))
           # time.sleep(0.03)
        if medium_x < center_x -40:
            Xposition -= 2
            ser.write((str(Xposition) + 'a').encode('utf-8'))
           # time.sleep(0.03)

        break


    cv2.imshow("frame",frame)
  #  cv2.imshow("mask",mask )
    key = cv2.waitKey(1)
    if key ==27:
        break
cap.release()
cv2.destroyAllWindows()






