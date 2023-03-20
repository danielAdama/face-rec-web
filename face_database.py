import numpy as np
import pickle
import face_recognition
from dotenv import load_dotenv
import cv2
import boto3
import imutils
import io
from PIL import Image
import os
from config import config


def face_db(image_bytes, bucket_name):

    path = []
    nameList = []
    fileList = []
    
    s3 = boto3.client(
    "s3",
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    )

    remoteObjs = s3.list_objects_v2(Bucket=bucket_name)
    for obj in remoteObjs['Contents']:
        path.append(obj['Key'])
        nameList.append(obj['Key'].split('/')[1])
        fileList.append(obj['Key'].split('/')[2])

        
    knownEncodings = []
    knownNames = []
    flag = False

    filenames = list(fileList)
    for name in list(set(nameList)):
        for filename in filenames:
            for file_path in path:
                if (file_path == os.path.join('known_face',name,filename)):
                    print(f"Processing {name}'s face!")
                    image = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
                    
                    # Convert RGB to BGR 
                    cv_image = image[:, :, ::-1].copy()
                    image = imutils.resize(cv_image, 440)
                    (height, width) = image.shape[:2]
                    image = cv2.resize(cv_image, (width, height))
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



