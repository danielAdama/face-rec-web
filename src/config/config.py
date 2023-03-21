import os
import cv2

MODEL = 'hog'
DATAPATH = r'/home/chuby/Desktop/programming/computerVision/face_recognition/face-rec-web'
KNOWN_PATH = os.path.join(DATAPATH,'known_face')
INPUT_PATH = os.path.join(DATAPATH, 'input_face')
##VIDEO_PATH = r'/home/daniel/Desktop/programming/pythondatascience/datascience/computerVision/video_for_vision_experiment'
#VIDEO = os.path.join(DATAPATH,"video",'Jeff-Bezos2.mp4')
VIDEO = 0
COLOR = cv2.COLOR_BGR2RGB
FONT = cv2.FONT_HERSHEY_SIMPLEX

HTTP_400_BAD_REQUEST = 400
HTTP_200_OK = 200
HTTP_500_INTERNAL_SERVER_ERROR = 500
HTTP_404_NOT_FOUND = 404