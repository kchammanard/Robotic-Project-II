import cv2
import numpy as np
from adafruit_servokit import ServoKit 
# (do lesson 31 to config servo)

kit = ServoKit(channels=16) #servo driver has 16 channels

pan = 90 #x dir
tilt = 45  #y dir
kit.servo[0].angle = pan
kit.servo[1].angle = tilt

