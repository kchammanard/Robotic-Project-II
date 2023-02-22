import cv2
import time
import string
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

from socket_module.HandTracking import HandTracking
from HandTrackingModule import handDetector
from tts_module.tts_module import ttsModule

model = load_model("trained_models/asl_model2.h5",compile= False)
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
classes[28] = 'back'

detector = handDetector(maxHands=1)
sentence = []


def hands_feed():
    HT = HandTracking()

    count = 0
    color = (0, 0, 255)

    # capture from live web cam
    cap = cv2.VideoCapture(0)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    start = time.time()
    pred = ""
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

        l = []
        if hands_results.multi_hand_landmarks:
            HT.draw_hand()
            HT.draw_hand_label()

            hand = hands_results.multi_hand_landmarks[0]
            for i in range(21):
                l += HT.get_moy_coords(hand, i)

            # sl = ",".join(map(str, l))
            # print(sl)
        else: 
            pred = ""

        print(l)
        print(len(l))

        if len(l) == 0:
            pred = ""
        else:
            a = [l]
            p = model.predict(a)    
            p_index = np.argmax(p, axis=1)
            pred = classes[int(str(p_index)[1:-1])]
        
        print("prev pred:", prev_pred)
        print("pred:", pred)

        if prev_pred == pred and pred != "":
            count += 1
            if count >= 5:
                color = (0,255,0)
                count = 0
                insert = True
        else:
            count = 0
            prev_pred = None
            color = (0,0,255)

        if pred != "":
            prev_pred = pred
        
        if insert:
            if pred == ".":
                print("sentence captured")
                text_to_speech(text="sentence captured")
                break
            elif pred == "back" and len(sentence) != 0:
                text_to_speech("deleting previous character")
                sentence.pop()
            else:
                text_to_speech(text=f"appending {pred}")
                sentence.append(pred)

        print(insert)

        cv2.putText(image, "".join(sentence) + pred, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)

        cv2.imshow("image", image)

        key = cv2.waitKey(250)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def text_to_speech(text):
    engine = ttsModule(text=text)
    engine.change_property(rate=100, volume=0.8, gender="f")
    engine.speak()


if __name__== "__main__":
    hands_feed()
    text_to_speech("".join(sentence))
