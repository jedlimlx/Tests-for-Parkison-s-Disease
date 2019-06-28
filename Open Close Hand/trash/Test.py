from __future__ import division
import cv2
import time
import numpy as np
import copy
import math

threshold = 60  #  BINARY threshold
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

def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 3, [211, 84, 0], -1)
            print(defects)
            for i in defects[0]:
                cv2.circle(drawing, (i[2], i[3]), 3, [0, 255, 255], -1)
                pass
            return True, cnt, defects
    return False, 0, None

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


class LucasKanadeTracker:
    def __init__(self):
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3))

    def track(self, old_gray, gray, fingers):
        lost_track = False
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, fingers, None, **self.lk_params)
        for i in range(len(st)):
            if st[i][0] == 0:
                lost_track = True
                break
            print("Error:", err)
            '''
            if err[i][0] > 10:
                lost_track = True
                break
            '''
        else:
            fingers = p1

        return fingers, lost_track

class Fingers:
    def __init__(self):
        protoFile = "hand/pose_deploy.prototxt"
        weightsFile = "hand/pose_iter_102000.caffemodel"
        self.nPoints = 22
        POSE_PAIRS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
        self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    def GetFingers2(self, frame, show = False):
        frameCopy = np.copy(frame)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        aspect_ratio = frameWidth/frameHeight

        threshold = 0.1

        t = time.time()
        # input image dimensions for the network
        inHeight = 368
        inWidth = int(((aspect_ratio*inHeight)*8)//8)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

        self.net.setInput(inpBlob)

        output = self.net.forward()

        # Empty list to store the detected keypoints
        points = []

        for i in range(self.nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > threshold :
                cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
            else :
                points.append([0, 0])

        num = [4, 8, 12, 16, 20]

        fingers = []

        for i in num:
            fingers.append(np.float32(points[i]))
            
        return fingers

    def GetFingers(self, frame):
        img = removeBG(frame)
        img = img[0:250, 0:250]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        kernel = np.ones((5,5),np.uint8)
        blur_process = cv2.erode(gray,kernel,iterations = 1)
        blur_process = cv2.GaussianBlur(blur_process, (blurValue, blurValue), 0)
        ret, blur_mask = cv2.threshold(blur_process, 25, 255, cv2.THRESH_BINARY_INV)
            
        blur = cv2.bitwise_and(gray, gray, mask=blur_mask)
        blur_mask = cv2.GaussianBlur(blur, (blurValue, blurValue), 0)
        ret, blur_mask = cv2.threshold(blur_mask, 27, 255, cv2.THRESH_BINARY_INV)
            
        blur = cv2.bitwise_and(gray, gray, mask=blur_mask)

        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.maxArea = 1000
        detector_params.minArea = 30
        detector = cv2.SimpleBlobDetector_create(detector_params)

        keypoints = detector.detect(blur_mask)
        blur_mask = cv2.drawKeypoints(blur_mask, keypoints, blur_mask)
        pts = cv2.KeyPoint_convert(keypoints)
        
        return pts, gray

class FingerDetector:
    def __init__(self, image_source, classifier, tracker):
        self.tracker = tracker
        self.classifier = classifier
        self.image_source = image_source
        self.fingers = []

    def distance(self, a, b):
        return (abs(a[0] - b[0]) ** 2 + abs(a[1] - b[1]) ** 2) ** 0.5

    def drawing(self, thresh, img):
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
            #isFinishCal, cnt, defects = calculateFingers(res, drawing)


            hull = list(hull)

            '''
            for j in range(0):
                for i in range(len(hull)):
                    if i + 1 < len(hull):
                        if self.distance(hull[i][0], hull[i + 1][0]) < 5:
                            hull[i] = np.average(np.array([hull[i][0], hull[i + 1][0]]), axis=1)
                            hull = hull[:i] + hull[i+2:-1]
                            i -= 1
                    try:
                        if hull[i][0][1] > 200:
                           hull = hull[:i] + hull[i+1:-1]
                    except Exception:
                        pass
            '''

            for i in range(len(hull)):
                hull[i][0][0] = np.float32(hull[i][0][0])
                hull[i][0][1] = np.float32(hull[i][0][1])

            hull = np.array(hull)
            print("HULL:", hull.flatten())

            
            for i in hull:
                cv2.circle(drawing, tuple(i[0]), 2, (255, 255, 255), -1)

            return drawing

    def run(self):
        k = cv2.waitKey(30) & 0xff
        i = 0
        while k != 27:
            frame = self.image_source.get_current_frame()
            _, blur = self.classifier.GetFingers(frame)

            self.fingers = np.array(self.fingers)
            
            drawing = self.drawing(thresh, img)
            
            if len(self.fingers) == 5:
                print("Track")
                track_result = self.tracker.track(old_blur, blur, self.fingers)
                self.fingers, lost_track = track_result
                if lost_track:
                    print("Lost Track")
                    self.fingers, blur = self.classifier.GetFingers(frame)
                    pass

            else:
                self.fingers, blur = self.classifier.GetFingers(frame)

            show_image_with_data(frame, drawing, self.fingers)
            k = cv2.waitKey(30) & 0xff
            old_blur = blur.copy()

        self.image_source.release()
        cv2.destroyAllWindows()

def show_image_with_data(frame, blur, fingers):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for w, h in fingers:
        cv2.circle(frame, (w, h), 5, (0, 255, 255), -1)

    cv2.rectangle(frame, (250, 0), (0, 250), (255, 0, 0), 2)
        
    cv2.imshow('Finger Detector', cv2.resize(frame, (1600, 1200)))

    cv2.imshow('Blur', cv2.resize(blur, (1200, 1200)))

cap = cv2.VideoCapture(0)
'''
while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (250, 0), (0, 250), (255, 0, 0), 2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''    
eyeris_detector = FingerDetector(image_source=ImageSource(), classifier=Fingers(),
                                 tracker=LucasKanadeTracker())
eyeris_detector.run()
