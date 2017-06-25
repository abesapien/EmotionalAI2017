import emotion_detection_thread
import time
edt=emotion_detection_thread.EmotionDetectionThread()
with edt:
    while True:
        print edt.getEmotion()
        time.sleep(1)
