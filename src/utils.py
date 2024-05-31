class VideoCapture:
    """
    A class used to capture video frames from a source and store them in a queue.

    ...

    Attributes
    ----------
    cap : cv2.VideoCapture
        a VideoCapture object from the OpenCV library
    q : queue.Queue
        a queue to store the video frames

    Methods
    -------
    _reader():
        Read frames as soon as they are available, keeping only the most recent one.
    read():
        Get the most recent frame from the queue.
    """
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

def gen_frames():
    """
    A generator function that yields video frames one by one.

    The function creates a VideoCapture object to capture frames from a source,
    processes each frame using the frame_detection function, and yields the frame
    encoded as a JPEG image. If an error occurs during frame processing, the function
    logs the error and returns a server error response.

    Yields
    ------
    bytes
        A byte string representing a JPEG image of a video frame.
    """
    camera = VideoCapture(config.VIDEO)
    while True:
        time.sleep(.2)   # simulate time between events
        frame = camera.read()
        try:
            frame_detection(frame)
        except Exception as ex:
                logger.debug(f"APPLICATION ERROR while reading webcam - {str(ex)}")
                return make_response(jsonify({
                    "BaseResponse":{
                        "Status":False,
                        "Message": f"Error reading camera",
                    }
                }),
            config.HTTP_500_INTERNAL_SERVER_ERROR)
         
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def frame_detection(frame):
    rgb = cv2.cvtColor(frame, config.COLOR)
    rgb = imutils.resize(frame, 440)
    (h, w) = frame.shape[:2]
    r = w / rgb.shape[1]
    boxes, names, accs = faceRec.faceAuth(rgb)
    for ((top, right, bottom, left), name) in zip(boxes, names):
        top, right, bottom, left = (int(top*r)), (int(right*r)), (int(bottom*r)), (int(left*r))
        x = top - 15 if top - 15 > 15 else top + 15
        if name=='Unknown':
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)
            cv2.rectangle(frame, (left, bottom + 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left+6, bottom+20), config.FONT, 0.8, 
            (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)
            for acc in accs:
                # Status box
                cv2.rectangle(frame, (left, bottom + 30), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f"{name} {acc*100:.2f}%", (left+6, bottom+20), config.FONT, 0.8, 
                (255, 255, 255), 2)
    return frame

def s3_store_cv_image(filePath: any, bytes: any, file: str) -> None:
    s3.put_object(
        Bucket=os.environ.get("AWS_BUCKET_NAME"),
        Key=filePath,
        Body=bytes,
        ContentType= file.content_type
    )

def delete_s3_folder_contents(bucket_name, directory):
    # List all the objects in the prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=directory)
    objects = response.get('Contents', [])

    # Delete all the objects in the prefix
    if len(objects) > 0:
        keys = [{'Key': obj['Key']} for obj in objects]
        s3.delete_objects(Bucket=bucket_name, Delete={'Objects': keys})

def verify_image(image_file):
    image_bytes = image_file.read()
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    # Convert RGB to BGR 
    frame = img[:, :, ::-1].copy()
    return frame_detection(frame)