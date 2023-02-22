import cv2
from glob import glob
import numpy as np
from socket_module.HandTracking import HandTracking
import socket

########################################################################################
HEADER = 64 #first message to server will always be of 64 bytes, and will tell us the size of the next message
PORT = 5050
# SERVER = socket.gethostbyname(socket.gethostname())
SERVER = "10.202.250.150"
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
ADDR = (SERVER,PORT)

client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client.connect(ADDR) 
########################################################################################
def send(msg):
    message = msg.encode(FORMAT) 
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT) #pad so its length 64
    send_length += b' ' * (HEADER-len(send_length))
    client.send(send_length)
    client.send(message)

def hands_feed():
    HT = HandTracking()
    # capture from live web cam
    cap = cv2.VideoCapture(0)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

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

        send(str(l))
        print(l)

        cv2.imshow("image", image)

        key = cv2.waitKey(250)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__== "__main__":
    hands_feed()
    send(DISCONNECT_MESSAGE)