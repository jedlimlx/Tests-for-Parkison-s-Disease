# importing modules
import cv2
import numpy as np
from time import sleep
from RegistryEditor import *

# declare variables
global cv2image
cv2image = 0

# find difference
def difference(a, b):
    if len(a) == 2:
        return(abs(a[0][0] - b[0][0]) + abs(a[1][0] - b[1][0])) / 2
    else:
        return -1

# returns frames from camera
class ImageSource:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

    def get_current_frame(self, gray=False):
        while True:
            try:
                ret, frame = self.capture.read()
                break
            except Exception as e:
                print("Error", e)
                continue
        
        frame = cv2.flip(frame, 1) # flip the frame
        if not gray:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def release(self):
        self.capture.release()

# find irises
class CascadeClassifier:
    def __init__(self):
        self.frame_grey = 0
        self.eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml') # use haar cascade
       
    # preprocess the image and detect keypoints
    def preprocess(self, roi):
        # parameters for simple blob detector
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.maxArea = 500
        detector_params.minArea = 100
        detector = cv2.SimpleBlobDetector_create(detector_params)
        i = 0
        while True:
            threshold = int(get_reg("Threshold", REG_PATH)) - 10
            #print("Thresh:",threshold)
            _, roi = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY) # threshold
            roi = cv2.erode(roi, None, iterations=2)
            roi = cv2.dilate(roi, None, iterations=2)
            roi = cv2.medianBlur(roi, 5)
            keypoints = detector.detect(roi) # detect the pupil as a blob
            pts = cv2.KeyPoint_convert(keypoints) 
            if len(pts) > 0:
                return pts[0]
            elif i > 6:
                break
            else:
                i += 1
                continue
    
    # get cordinates of irises
    def get_irises_location(self, frame_gray):
        eyes = self.eye_cascade.detectMultiScale(frame_gray, 1.3, 5)
        irises = []
        
        for (ex, ey, ew, eh) in eyes:
            roi = frame_gray[ey:ey+eh, ex:ex+ew]
            pts = self.preprocess(roi) # preprocess the roi
            if pts is not None:
                iris_w = int(ex + pts[0])
                iris_h = int(ey + pts[1])
            else:
                iris_w = int(ex + ew / 2)
                iris_h = int(ey + eh / 2)
            
            irises.append([np.float32(iris_w), np.float32(iris_h)]) # append irises coordinates
            
        return np.array(irises)

# eye tracker          
class LucasKanadeTracker:
    def __init__(self, blink_threshold=9):
        # parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.blink_threshold = blink_threshold # determine it was not due to a blink

    def track(self, old_gray, gray, irises):
        lost_track = False
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, irises, None, **self.lk_params)
        if st[0][0] == 0 or st[1][0] == 0:  # lost track on eyes
            lost_track = True
            
        elif err[0][0] > self.blink_threshold or err[1][0] > self.blink_threshold:  # high error rate in eye tracking
            lost_track = True
            
        else:
            irises = p1
            #for w, h in p1:
            #    irises.append([w, h])
            #irises = numpy.array(irises)
        return irises, lost_track

# calibration to user's eye
class CalibrateV2:
    def __init__(self, eye_index, img_source):
        self.eye_cascade = cv2.CascadeClassifier("haar/haarcascade_eye_tree_eyeglasses.xml")
        #self.face_cascade  = cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml")
        self.img_source = img_source
        self.eye_index = eye_index
        self.x, self.y = 0, 0
        self.contours = []
        self.threshold = 30
       
    # get the region of interest (containing eye)
    def update_roi(self):
        self.img = self.img_source.get_current_frame()
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.roi = 0
        
        while True:
            self.img = self.img_source.get_current_frame()
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.eyes = self.eye_cascade.detectMultiScale(self.gray, 1.3, 5)
            if len(self.eyes) == 2 and abs(self.eyes[0][1] - self.eyes[1][1]) < 5: # both eyes should have similar x coor.
                c1 = self.eye_index == 0
                c2 = max(self.eyes[0][0], self.eyes[1][0]) == self.eyes[0][0]
                if c1 == c2:
                    self.x, self.y, self.w, self.h = self.eyes[0]
                else:
                    self.x, self.y, self.w, self.h = self.eyes[1]
                self.roi = self.gray[self.y:self.y+self.h, self.x:self.x+self.w]
                break
            else:
                continue

        return self.roi
            
    def process_roi(self, roi):
        _, self.roi_contours = cv2.threshold(roi, self.threshold, 255, cv2.THRESH_BINARY_INV) # perform threshold
        return self.roi_contours
       
    # update the contours of the eye
    def update_contours(self):
        self.roi_contours = self.process_roi(self.update_roi())
        self.contours, self.hierarchy = cv2.findContours(self.roi_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # process the contours to be displayed on an image
    def get_processed_contours(self):
        self.contours2 = self.contours
        for j in range(len(self.contours2)):
            for i in range(len(self.contours2[j])):
                self.contours2[j][i][0][0] += self.x
                self.contours2[j][i][0][1] += self.y
        return self.contours2
    
    # main function
    def main(self):
        self.update_contours()
        while True:
            if len(self.contours) != 0:
                self.l = 0
                self.s = 10**2
                for i in self.contours:
                    for j in i:
                        if self.l < j[0][0]:
                            self.l = j[0][0]
                        elif self.s > j[0][0]:
                            self.s = j[0][0]

                break
            else:
                self.update_contours()
                continue

        return self.l, self.s, (self.l + self.s) / 2, [self.x, self.y]
    
    # preprocess the roi and find key points
    def preprocess(self, roi):
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.maxArea = 500
        detector = cv2.SimpleBlobDetector_create(detector_params)
        
        _, roi = cv2.threshold(roi, 25, 255, cv2.THRESH_BINARY)
        
        roi = cv2.erode(roi, None, iterations=2)
        roi = cv2.dilate(roi, None, iterations=4)
        roi = cv2.medianBlur(roi, 5)
        
        keypoints = detector.detect(roi)
        roi = cv2.drawKeypoints(roi, keypoints, roi)
        return roi
    
    # visual data to aid debugging
    def visualise_data(self):
        self.update_contours()
        self.img = self.img_source.get_current_frame()

        self.visual_3 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        roi = self.update_roi()

        self.visual_4 = self.preprocess(roi)
        
        self._, self.visual_3 = cv2.threshold(self.visual_3, self.threshold, 255, cv2.THRESH_BINARY)
        
        self.visual_1 = self.img
        cv2.drawContours(self.visual_1, self.get_processed_contours(), -1, (0,255,0), 1) # drawing contours of the eye
        
        self.visual_2 = self.process_roi(roi)
        self.visual_2 = 255 - self.visual_2
        self.visual_2 = cv2.cvtColor(self.visual_2, cv2.COLOR_GRAY2BGR)

        return self.visual_1, self.visual_2, self.visual_3, self.visual_4

    
