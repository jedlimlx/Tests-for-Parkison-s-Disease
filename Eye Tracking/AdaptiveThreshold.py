from skimage.measure import compare_ssim
import os
import cv2
import numpy as np
from time import sleep

def adaptiveThreshold():
    def similarity(imageA, imageB):
        score, diff = compare_ssim(imageA, imageB, full=True)
        return score

    class ImageSource:
        def __init__(self):
            self.capture = cv2.VideoCapture(0)

        def get_current_frame(self, gray=False):
            ret, frame = self.capture.read()
            frame = cv2.flip(frame, 1)
            if not gray:
                return frame
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        def release(self):
            self.capture.release()

    img_source = ImageSource()

    test_lst = []
    for i in os.listdir("threshold_test"):
        test_lst.append(f"threshold_test/{i}")

    roi_lst = []
    for i in range(2):
        eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')
        while True:
            img = img_source.get_current_frame()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
            if len(eyes) == 2:
                x, y, w, h = eyes[0]
                roi = gray[y:y+h, x:x+w]
                roi_lst.append(roi)
                break

    threshold = 10
    aver_score_lst = []
    while True:
        score_lst = []
        #aver_score_lst = []
        for i in range(2):
            _, roi = cv2.threshold(roi_lst[i], threshold, 255, cv2.THRESH_BINARY)
            roi = cv2.resize(roi, (240, 240))
            
            test = cv2.imread(test_lst[i])
            test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
            
            test = cv2.resize(test, (240, 240))

            score = similarity(roi, test)

            score_lst.append(score)
            
            #print("Score:", score)

        threshold += 1
        #print(f"===========THRESHOLD {threshold}==========")
        aver_score_lst.append(sum(score_lst) / len(score_lst))

        if threshold == 100:
            break

    _, roi = cv2.threshold(roi_lst[i], aver_score_lst.index(max(aver_score_lst)), 255, cv2.THRESH_BINARY)
    roi = cv2.resize(roi, (1000, 1000))
    cv2.imshow("roi", roi)
    for i in range(len(test_lst)):
        test = cv2.imread(test_lst[i])
        test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        test = cv2.resize(test, (1000, 1000))
        cv2.imshow(f"test{i}", test)
            
    return max(aver_score_lst), aver_score_lst.index(max(aver_score_lst))

print(adaptiveThreshold())
