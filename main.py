import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import time
import os
import string
from glob import glob
import pyttsx3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from HandTracking import HandTracking

model = load_model("test/asl_model2.h5",compile= False)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()])

cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300
classes = dict( (i, key) for i,key in enumerate(string.ascii_lowercase))
classes[26] = ' '
classes[27] = '.'

detector = HandDetector(maxHands=1)
sentence = []


def hands_feed():
    HT = HandTracking()

    count = 0
    color = (0, 0, 255)

    # capture from live web cam
    cap = cv2.VideoCapture(0)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    start = time.time()
    pred = "No predictions"
    prev_pred = None
    l = []
    while cap.isOpened():
        insert = False

        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # hand tracking
        hands_results = HT.track(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # init frame each loop
        HT.read_results(image, hands_results)

        if hands_results.multi_hand_landmarks:
            HT.draw_hand()
            HT.draw_hand_label()

            hand = hands_results.multi_hand_landmarks[0]
            l = []
            for i in range(21):
                l += HT.get_moy_coords(hand, i)

            # sl = ",".join(map(str, l))
            # print(sl)
        else: 
            pred = "No predictions"

        if len(l) == 0:
            pass
        else:
            a = [l]
            p = model.predict(a)    
            p_index = np.argmax(p, axis=1)
            pred = classes[int(str(p_index)[1:-1])]
        
        print("prev pred:", prev_pred)
        print("pred:",pred)

        if prev_pred == pred:
            count += 1
            if count >= 5:
                color = (0,255,0)
                count = 0
                insert = True
        else:
            count = 0
            color = (0,0,255)

        if pred != "No predictions":
            prev_pred = pred
        
        if insert:
            sentence.append(pred)

        print(insert)

        cv2.putText(image, "".join(sentence) + pred, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)

        cv2.imshow("image", image)

        key = cv2.waitKey(250)

        if key == ord("q"):
            cap.release()
            break

    cv2.destroyAllWindows()

def text_to_speech():
    engine = pyttsx3.init()
    engine.say("".join(sentence))
    engine.runAndWait()
    print("".join(sentence))

if __name__== "__main__":
    hands_feed()
    text_to_speech()

