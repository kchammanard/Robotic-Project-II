import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
import math
import time
from roboflow import Roboflow
import os
import string
from glob import glob
import pyttsx3
from ultralytics import YOLO

PATH_TO_MODEL = 'yolov8/training_results/sign_language/weights/best.pt'
model = YOLO(PATH_TO_MODEL)

cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300
classes = dict( (i, key) for i,key in enumerate(string.ascii_lowercase))

detector = HandDetector(maxHands=1)
sentence = []

def hands_feed():
    count = 0
    letter_count = 0
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

            if count > 5:
                count = 0
                letter_count += 1
                cv2.imwrite(f"imglib/image_{letter_count}.jpg",imgWhite)
            count += 1
            results = model.predict(source=imgWhite, conf=0.3, show=True)[0]
        
        else:
            count = 0
            
            #cv2.imshow("Whites", imgWhite)

        #cv2.imshow("frame",frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def predict_image():
    folder = "imglib/"
    images = glob(os.path.join(folder,"*jpg"))
    for image in images:
        img = cv2.imread(image)
        results = model.predict(source = img, conf = 0.5)[0]
        if results.boxes:
            for i,obj in enumerate(results.boxes):
                x1, y1, x2, y2, conf, cls = obj.data.cpu().detach().numpy()[0]
                letter = classes[int(cls)]
                sentence.append(letter)
        else:
            sentence.append(" ")
    print("".join(sentence))

def reset_images():
    folder = "imglib/"
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for file in files:
        os.remove(os.path.join(folder, file))

def text_to_speech():
    engine = pyttsx3.init()
    engine.say("".join(sentence))
    engine.runAndWait()

if __name__== "__main__":
    reset_images()
    hands_feed()
    predict_image()
    text_to_speech()
