from flask import Flask, render_template, Response, request, make_response, jsonify, send_file
from werkzeug.utils import secure_filename
from utils import VideoCapture, gen_frames, frame_detection, s3_store_cv_image, delete_s3_folder_contents, verify_image
from config import config
import numpy as np
import logging
import argparse
from flask_cors import CORS
import imutils
import base64
import io
from PIL import Image
import os
import boto3
import threading
import queue
import time
import face_database
import pickle
import cv2
from face_verification.face_verify import FaceRecognition
from config import config
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
handle = "face-app"
logger = logging.getLogger(handle)
logger.setLevel(level=logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(level=logging.DEBUG)
logger.addHandler(consoleHandler)


s3 = boto3.client(
    "s3",
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
)

app = Flask(__name__)
CORS(app)

if os.path.exists(os.path.join(os.getcwd(),'encodings.pickle')):
    with open('encodings.pickle', 'rb+') as f:
        data = pickle.load(f)
    flag = True

if (flag):
    faceRec = FaceRecognition(data=data)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/check', methods = ['POST'])
def check():
    if request.method == 'POST':
        if 'image' not in request.files:
            return make_response(jsonify({
                "BaseResponse":{
                    "Status":False,
                    "Message": "Please make select an image(s)",
                    "Data": None
                },
                'Error': "You can only upload image"
            }), 
            config.HTTP_500_INTERNAL_SERVER_ERROR)

        file = request.files.getlist("image")[0]
        if file.mimetype not in ['image/jpg','image/jpeg', 'image/png']:
                return make_response(jsonify({
                    "BaseResponse":{
                        "Status":False,
                        "Message": "Please make sure image extensions is - jpg, jpeg, png",
                        "Data": None
                    },
                    "Error":"These are the allowed image extension - jpg, jpeg, png"
                }),
            config.HTTP_400_BAD_REQUEST)
        
        filename = secure_filename(file.filename)
        verified_face = verify_image(file)
        # Check if a train image is been passed and test to verify face
        remoteObjs = s3.list_objects_v2(Bucket=os.environ.get("AWS_BUCKET_NAME"))
        if 'Contents' in remoteObjs.keys():
            for obj in remoteObjs['Contents']:
                if (obj['Key'].split('/')[2] == filename):
                    nameCheck = obj['Key'].split('/')[1]
                    logger.debug(f"APPLICATION CHECK > You can not verify a trained image, the image {filename} been trained for {nameCheck}")
                    return make_response(jsonify({
                        "BaseResponse":{
                            "Status":False,
                            "Message":str('You can not verify a trained image, the image '+filename+' been trained for '+nameCheck.capitalize())
                        }
                    }), config.HTTP_400_BAD_REQUEST)
                
        path = f"detected_face/{filename}"                
        ret, buffer = cv2.imencode('.jpg', verified_face)
        byte = buffer.tobytes()
        s3_store_cv_image(path, byte, file)
        s3_object = s3.get_object(Bucket=os.environ.get("AWS_BUCKET_NAME"), Key=path)
        predicted_image_byte = s3_object['Body'].read()
        ## Clear predicted image from s3
        delete_s3_folder_contents(os.environ.get("AWS_BUCKET_NAME"), 'detected_face/')
        predicted_image = Image.open(io.BytesIO(predicted_image_byte))
        rawBytes = io.BytesIO()
        predicted_image.save(rawBytes, format="JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
    return make_response(jsonify({
                    "BaseResponse":{
                        "Status":True,
                        "Data":str(img_base64),
                        "Messsage":"Operation successfully"
                    }
                }), config.HTTP_200_OK)

@app.route('/live')
def live():
    return render_template("verify.html")

@app.route('/upload', methods = ['POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return make_response(jsonify({
                "BaseResponse":{
                    "Status":False,
                    "Message": "Please make select an image(s)",
                    "Data": None
                },
                'Error': "You can only upload image"
            }), 
            config.HTTP_500_INTERNAL_SERVER_ERROR)

        files = request.files.getlist("image")
        name = request.form["name"]
        try:
            for file in files:
                if file.mimetype not in ['image/jpg','image/jpeg', 'image/png']:
                    return make_response(jsonify({
                        "BaseResponse":{
                            "Status":False,
                            "Message": "Please make sure image extensions is - jpg, jpeg, png",
                            "Data": None
                        },
                        "Error":"These are the allowed image extension - jpg, jpeg, png"
                    }),
                config.HTTP_400_BAD_REQUEST)

                image_bytes = file.read()
                filename = secure_filename(file.filename)
                remoteObjs = s3.list_objects_v2(Bucket=os.environ.get("AWS_BUCKET_NAME"))
                try:
                    filePath = f"known_face/{name}/{filename}"
                    remoteObjs = s3.list_objects_v2(Bucket=os.environ.get("AWS_BUCKET_NAME"))
                    if 'Contents' in remoteObjs.keys():
                        for obj in remoteObjs['Contents']:
                            if (obj['Key'] == filePath):
                                logger.debug(f"APPLICATION UPLOAD > The image {filename} been trained for {name.split('_')[0]}")
                                return make_response(jsonify({
                                    "BaseResponse":{
                                        "Status":False,
                                        "Message":str('The image '+filename+' been trained for '+name.split('_')[0].capitalize())
                                    }
                                }), config.HTTP_400_BAD_REQUEST)
                    objName = filePath
                    s3_store_cv_image(objName, image_bytes, file)
                except Exception as ex:
                    logger.debug(f"APPLICATION ERROR while storing images in s3 bucket - {str(ex)}")
                    return make_response(jsonify({
                        "BaseResponse":{
                            "Status":False,
                            "Message": f"Error accessing image store",
                        }
                    }),
                config.HTTP_500_INTERNAL_SERVER_ERROR)

            try:
                face_database.face_db(image_bytes, os.environ.get("AWS_BUCKET_NAME"))
            except Exception as ex:
                logger.debug(f"APPLICATION ERROR while storing images in s3 bucket - {str(ex)}")
                return make_response(jsonify({
                    "BaseResponse":{
                        "Status":False,
                        "Message": f"Error accessing image store",
                    }
                }),
            config.HTTP_500_INTERNAL_SERVER_ERROR)
            return make_response(jsonify({
                        "BaseResponse":{
                            "Status":True,
                            "Messsage":"Operation successfully"
                        }
                    }), config.HTTP_200_OK)
        except Exception as ex:
            logger.debug(f"APPLICATION ERROR while recognizing face - {ex}")
            return make_response(jsonify({
                "BaseResponse":{
                            "Status":False,
                            "Message":str(ex)
                        },
                "Error":"Something went wrong",
            }), config.HTTP_500_INTERNAL_SERVER_ERROR)

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face Recognition Application")
    parser.add_argument(
        "-p", 
        "--port", 
        default=8080, 
        type=int, 
        help="port number"
    )
    args = parser.parse_args()
    app.run(
        host='0.0.0.0', 
        debug=True, 
        port=args.port
    )
