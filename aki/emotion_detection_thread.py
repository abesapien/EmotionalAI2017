from __future__ import print_function

import threading
import numpy as np
import cv2

# local modules
from video import create_capture
from common import clock, draw_str

from keras.models import load_model

import threading

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
        self.outputs = []

    def run(self):
        cam = create_capture(self.video_src)
        model = load_model(self.model_path)
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            t = clock()
            rects = self.detect(gray, self.cascade)
            vis = img.copy()
            self.draw_rects(vis, rects, (0, 255, 0))
            outputs=[]
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                input = cv2.resize(roi,(48,48))
                output=model.predict(np.reshape(input,(1,48,48,1)))
                outputs.append(output)
                vis_roi = vis[y1:y2, x1:x2]
                emotions = ('Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral')
                draw_str(vis_roi, (20, 20),  emotions[output[0].argmax()])
            with self.lock:
                if len(self.outputs)<10:
                    self.outputs.append(outputs)
                else:
                    self.outputs = self.outputs[1:] + outputs

            dt = clock() - t
            draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
            cv2.imshow('facedetect', vis)

    def __del__(self):
        cv2.destroyAllWindows()

    def getEmotion(self):
        with self.lock:
            return self.outputs
