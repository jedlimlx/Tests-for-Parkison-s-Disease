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

i = 0        
while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (250, 0), (0, 250), (255, 0, 0), 2)

    img = removeBG(frame)
    img = img[0:250, 0:250]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    
    '''
    25, 255, cv2.THret, thresh = cv2.threshold(blur, RESH_BINARY)

    kernel = np.ones((3, 3),np.uint8)
    palm = cv2.erode(gray, kernel,iterations = 5)
    palm = cv2.GaussianBlur(palm, (blurValue, blurValue), 0)
    ret, thresh_palm = cv2.threshold(palm, 25, 255, cv2.THRESH_BINARY)
    '''

    cv2.imshow('frame', cv2.resize(frame, (1600, 1200)))
    cv2.imshow('blur', cv2.resize(blur, (1000, 1000)))
    #cv2.imshow('thresh_palm', cv2.resize(thresh_palm, (1000, 1000)))

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('w'):
        t = threading.Thread(target=Test)
        t.start()
