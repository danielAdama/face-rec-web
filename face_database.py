import numpy as np
import pickle
import face_recognition
import cv2
import os
from config import config



def face_db(filePath):

    knownEncodings = []
    knownNames = []
    flag = False

    for name in os.listdir(filePath):
        for filename in os.listdir(os.path.join(filePath,name)):
            print(f"Processing {name}'s face!")
            image = cv2.imread(os.path.join(filePath,name,filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect the (x, y) co-ordinate of face(s)
            boxes = face_recognition.face_locations(image, model=config.MODEL) #cnn
            encodings = face_recognition.face_encodings(image, boxes)
            # Loop over the encodings and add their name(s) and face(s), respectively to the database
            for encoding in encodings:
                try:
                    with open('encodings.pickle', 'rb+') as f:
                        existingdata = pickle.load(f)
                    existingdata["encodings"].append(encoding)
                    existingdata["names"].append(name)
                except:
                    flag = True
                    knownEncodings.append(encoding)
                    knownNames.append(name)

    if flag:
        print('\nSuccessfully Serialized Face(s) into the Database'+"..."*3)
        data = {"encodings": knownEncodings, "names": knownNames}
        with open("encodings.pickle", "wb+") as f:
            f.write(pickle.dumps(data))
            f.close()
    else:
        print('\nSuccessfully Appended Face(s) into the Database'+"..."*3)
