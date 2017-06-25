# getEmotion returns outputs, which is a list of information from last 300 frames.
# each element is a list containing time and probability of emotions of all the faces in the frame

from __future__ import print_function

import threading
import numpy as np
import cv2

# local modules
from video import create_capture
from common import clock, draw_str

from keras.models import load_model

import threading
import time

class EmotionDetectionThread(threading.Thread):
    def detect(self, img, cascade):
        rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:,2:] += rects[:,:2]
        return rects

    def draw_rects(self, img, rects, color):
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    def __init__(self):
        super(EmotionDetectionThread, self).__init__()
        self.model_path = 'emotion.hd5'
        self.cascade_fn = 'haarcascade_frontalface_alt.xml'
        self.lock = threading.Lock()
        self.cascade = cv2.CascadeClassifier(self.cascade_fn)
        self.video_src = 0
        self.interval=0.05
        self.outputs = []
        self.cam = create_capture(self.video_src)
        print('capture started\n')
        self.running=True

    def run(self):
        print('running thread\n')
        self.model = load_model(self.model_path)
        print('model loaded\n')
        while self.running:
            print('running thread\n')
            ret, img = self.cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            t = clock()
            rects = self.detect(gray, self.cascade)
            vis = img.copy()
            #self.draw_rects(vis, rects, (0, 255, 0))
            outputs=[t]
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                input = cv2.resize(roi,(48,48))
                output=self.model.predict_proba(np.reshape(input,(1,48,48,1)),verbose=0)
                outputs.append(output)
                vis_roi = vis[y1:y2, x1:x2]
                emotions = ('Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral')
            #    draw_str(vis_roi, (20, 20),  emotions[output[0].argmax()])
            with self.lock:
                if len(self.outputs)<300:
                    self.outputs.append(outputs)
                else:
                    self.outputs = self.outputs[1:] + outputs

            dt = clock() - t
            #draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
            #cv2.imshow('facedetect', vis)
            #time.sleep(self.interval)
    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.running=False

    def __del__(self):
        cv2.destroyAllWindows()

    def getEmotion(self):
        with self.lock:
            return self.outputs
