import os
import cv2

MODEL = 'hog'
DATAPATH = r'C:\\Users\\DELL\\Desktop\\programming\\computerVision\\face_recognition\\'
KNOWN_PATH = os.path.join(DATAPATH,'known_face')
INPUT_PATH = os.path.join(DATAPATH, 'input_face')
##VIDEO_PATH = r'/home/daniel/Desktop/programming/pythondatascience/datascience/computerVision/video_for_vision_experiment'
#VIDEO = os.path.join(DATAPATH,"video",'Jeff-Bezos2.mp4')
VIDEO = 0
COLOR = cv2.COLOR_BGR2RGB
FONT = cv2.FONT_HERSHEY_SIMPLEX
