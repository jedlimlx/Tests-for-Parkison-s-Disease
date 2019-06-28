import cv2
import time
import numpy as np
import math
import threading
import copy
from Classify import classify

cap = cv2.VideoCapture(0)

threshold = 60  
blurValue = 41
bgSubThreshold = 50
learningRate = 0
bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def get_hand(gray):
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    ret, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    
    return cv2.countNonZero(thresh)

def predict_palm(gray):
    kernel = np.ones((3, 3),np.uint8)
    palm = cv2.erode(gray, kernel,iterations = 5)
    palm = cv2.GaussianBlur(palm, (blurValue, blurValue), 0)
    ret, thresh_palm = cv2.threshold(palm, 25, 255, cv2.THRESH_BINARY)

    return cv2.countNonZero(thresh_palm)

def Test():
    print("Open your Palm Wide...")
    time.sleep(5)
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    img = removeBG(frame)
    img = img[0:250, 0:250]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

    cv2.imwrite("predict.jpg", blur)
    result = classify("predict.jpg")

    print("Close your Palm...")
    time.sleep(5)

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    img = removeBG(frame)
    img = img[0:250, 0:250]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    
    cv2.imwrite("predict.jpg", blur)
    
    result2 = classify("predict.jpg")
    
    if result == 'open':
        print("Feedback: Good Job! Your hand was fully opened!")
        
    elif result == 'nothing':
        print("Feedback: Please place your hand in the blue rectangle!")
        
    else:
        print("Feedback: Your hand was not fully opened.")

    if result2 == 'close':
        print("Feedback: Good Job! Your hand is fully closed!")
        
    elif result2 == 'nothing':
       print("Feedback: Please place your hand in the blue rectangle!")
       
    else:
        print("Feedback: Your hand was not fully closed.")

def distance(a, b):
    return (abs(a[0] - b[0]) ** 2 + abs(a[1] - b[1]) ** 2) ** 0.5

def drawing(thresh, img):
    thresh1 = copy.deepcopy(thresh)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i

        res = contours[ci]
        hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 1)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 1)

        hull = list(hull)

        for i in range(len(hull)):
            if i + 1 < len(hull):
                if distance(hull[i][0], hull[i + 1][0]) < 5:
                    hull[i] = np.average(np.array([hull[i][0], hull[i + 1][0]]), axis=1)
                    hull = hull[:i] + hull[i+2:-1]
                    i -= 1
                try:
                    if hull[i][0][1] > 200:
                        hull = hull[:i] + hull[i+1:-1]
                except Exception:
                    pass

        hull = np.array(hull)
        for i in hull:
            cv2.circle(drawing, tuple(i[0]), 2, (255, 255, 255), -1)
        
        return drawing

i = 0        
while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (250, 0), (0, 250), (255, 0, 0), 2)

    img = removeBG(frame)
    img = img[0:250, 0:250]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    
    ret, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    
    '''
    kernel = np.ones((3, 3),np.uint8)
    palm = cv2.erode(gray, kernel,iterations = 5)
    palm = cv2.GaussianBlur(palm, (blurValue, blurValue), 0)
    ret, thresh_palm = cv2.threshold(palm, 25, 255, cv2.THRESH_BINARY)
    '''
    
    draw = drawing(thresh, img)
    
    cv2.imshow('frame', cv2.resize(frame, (1600, 1200)))
    cv2.imshow('blur', cv2.resize(blur, (1000, 1000)))
    if draw is not None:
        cv2.imshow('drawing', cv2.resize(draw, (1000, 1000)))

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('w'):
        t = threading.Thread(target=Test)
        t.start()
