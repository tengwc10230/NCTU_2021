from imagezmq.imagezmq import ImageHub
from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2


imagehub = imagezmq.ImageHub(open_port='tcp://0.0.0.0:6006')

lastActive = {}
lastActiveCheck = datetime.now()

while True:
    client_name, frame = imagehub.recv_image()
    cv2.imshow(rpi_name, image)
    imagehub.send_reply(b'OK')
    if rpiName not in lastActive.keys():
        print("[INFO] receiving data from {}...".format(client_name))
    
    lastActive[client_name] = datetime.now()

    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    cv2.imshow(frame)

