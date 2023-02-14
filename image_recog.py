import numpy as np
import pickle
import imutils
from config import config
from face_verification.face_verify import FaceRecognition
import cv2
import os


with open('test.pickle', 'rb') as f:
    data = pickle.load(f)

for filename in os.listdir(config.INPUT_PATH):
    print(f"Processing Input Face: {filename})")
    image = cv2.imread(os.path.join(config.INPUT_PATH, filename))
    image = imutils.resize(image, 650)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # After processing convert the Image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    fv = FaceRecognition(image=image, data=data)
    boxes, names, _ = fv.faceAuth()

    for ((top, right, bottom, left), name) in zip(boxes, names):
        x = top - 15 if top - 15 > 15 else top + 15
        if name=='Unknown':
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(image, name, (left, x), config.FONT, 0.5, (0, 0, 255), 2)
        else:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, name, (left, x), config.FONT, 0.5, (0, 255, 0), 2)

    
    cv2.imshow("Image", image)
    cv2.waitKey(0)
