import mediapipe as mp
import cv2
import numpy as np
import time
import math
from HandTracking import HandTracking

def main():
    # init model
    HT = HandTracking()

    save = False
    ind = None
    count = 0

    # capture from live web cam
    cap = cv2.VideoCapture(0)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    start = time.time()
    while cap.isOpened():
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

            sl = ",".join(map(str, l))
            print(sl)
            if save:
                with open("test.csv", "a") as h:
                    h.write(sl+f",{ind}\n")
                print(f"Saving {ind}")
                count += 1

        # get fps
        fps = 1 / (time.time() - start)
        start = time.time()
        cv2.putText(image, "fps: " + str(round(fps, 2)), (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow("image", image)

        key = cv2.waitKey(250)

        if key == ord("q") or count >= 50:
            cap.release()
            count = 0
        elif key == ord("s"):
            if save == False:
                ind = int(input("INDEX: "))
                print(f"Saving as {ind}")
            else:
                print("Stop saving..")
            save = not save

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
