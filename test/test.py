import socket
import threading
import cv2
import math
import time
import os
import string
from glob import glob
import pyttsx3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
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
classes[28] = 'back'

HEADER = 64 #first message to server will always be of 64 bytes, and will tell us the size of the next message
PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER,PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
#(type of socket,method)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#bind to addres
server.bind(ADDR)

def handle_client(conn,addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT) #put how many bytes from client
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            if msg == DISCONNECT_MESSAGE:
                connected = False
            
            print(f"[{addr}] {msg}")
            
            conn.send("Msg received".encode(FORMAT))
    conn.close()

def start():
    #listen for new connections
    server.listen()
    print(f"server is listening on {SERVER}")
    while True:
        conn,addr = server.accept() #will wait until new connection occurs
        thread = threading.Thread(target= handle_client,args= (conn,addr))
        thread.start()
        print("[ACTIVE CONNECTIONS]" , threading.active_count()-1) #start thread is running always so -1

print("server starting")
start()