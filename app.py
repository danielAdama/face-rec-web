from flask import Flask, render_template, Response, request, make_response, jsonify
from werkzeug.utils import secure_filename
from config import config
import numpy as np
import logging
import face_database
import pickle
from face_verification.face_verify import FaceRecognition
import cv2
import re
import os
import imutils
import io
from PIL import Image
import time
from config import config



app = Flask(__name__)
app.config['noMatchNames'] = []
app.config['matchNames'] = []
app.config['filenames'] = []
app.config['images'] = []
handle = "my-app"
#cors(app)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.NOTSET, datefmt='%d-%b-%y %H:%M:%S')
_logger = logging.getLogger(handle)


if os.path.exists(os.path.join(os.getcwd(),'encodings.pickle')):
    with open('encodings.pickle', 'rb+') as f:
        app.config['data'] = pickle.load(f)

def video_stream():
    webcam = cv2.VideoCapture(0)
    time.sleep(0.02)
    try:
        if (webcam.isOpened() == False):
            print('\nUnable to read camera feed')

        while True:
            success, frame = webcam.read()
            if success == True:
                rgb = cv2.cvtColor(frame, config.COLOR)
                rgb = imutils.resize(frame, 640)
                (h, w) = frame.shape[:2]
                r = w / rgb.shape[1]
                
                fv = FaceRecognition(rgb, data=app.config['data'])
                boxes, names, accs = fv.faceAuth()

                for ((top, right, bottom, left), name) in zip(boxes, names):
                    top, right, bottom, left = (int(top*r)), (int(right*r)), (int(bottom*r)), (int(left*r))

                    x = top - 15 if top - 15 > 15 else top + 15
                    if name=='Unknown':
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(frame, (left, bottom + 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                        cv2.putText(frame, name, (left+30, bottom+20), config.FONT, 0.5, 
                        (255, 255, 255), 2)
                    else:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        for acc in accs:
                            # Status box
                            cv2.rectangle(frame, (left, bottom + 25), (right, bottom), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, f"{name} {acc*100:.2f}%", (left+30, bottom+20), config.FONT, 0.5, 
                            (255, 255, 255), 2)
                ret, buffer = cv2.imencode(".jpg", frame)
                yield(
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'+ buffer.tobytes() +
                    b'\r\n\r\n'
                    )

                key = cv2.waitKey(1)
                if key == 27:
                    break
            else:
                break
        webcam.release()
        cv2.destroyAllWindows()

    except Exception as ex:
            _logger.warning(f"APPLICATION ERROR while recognizing face in camera")
            return jsonify({
                "BaseResponse":{
                            "Status":False,
                            "Message":str(ex)
                        },
                "Error":"Something went wrong",
            }), config.HTTP_500_INTERNAL_SERVER_ERROR


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/live')
def live():
    return render_template("result.html")

@app.route('/upload', methods = ['POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return make_response(jsonify({
                "BaseResponse":{
                    "Status":False,
                    "Message": "Please make sure the javascript formData is tagged - image",
                    "Data": None
                },
                'Error': "You can only upload image"
            }), 
            config.HTTP_500_INTERNAL_SERVER_ERROR)

        files = request.files.getlist("image")
        name = request.form["name"]
        try:
            for file in files:
                fileExt = file.filename.split('.')[1].lower()
                if fileExt not in ['jpg','jpeg', 'png']:
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
                image = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
                # Convert RGB to BGR 
                cv_image = image[:, :, ::-1].copy()
                image = imutils.resize(cv_image, 440)
                (height, width) = image.shape[:2]
                image = cv2.resize(cv_image, (width, height))
                
                filename = secure_filename(file.filename)
                path = os.path.join(os.getcwd(),'known_face')
                if not os.path.exists(path):
                    os.mkdir(path)
                

                filePath = os.path.join(os.getcwd(),'known_face',name)
                if not os.path.exists(filePath):
                    os.mkdir(filePath)
                cv2.imwrite(os.path.join(filePath, filename), image)

            face_database.face_db(path)
            return make_response(jsonify({
                        "BaseResponse":{
                            "Status":True,
                            "Messsage":"Operation successfully"
                        }
                    }), config.HTTP_200_OK)
        except Exception as ex:
            _logger.warning(f"APPLICATION ERROR while recognizing face - {ex}")
            return make_response(jsonify({
                "BaseResponse":{
                            "Status":False,
                            "Message":str(ex)
                        },
                "Error":"Something went wrong",
            }), config.HTTP_500_INTERNAL_SERVER_ERROR)

    return render_template('result.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), 
    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)