import cv2
from os.path import join
def CalibrateV2(eye_index, img_source):
    eye_cascade = cv2.CascadeClassifier(join('haar', 'haarcascade_eye.xml'))
    
    while True:
        img = img_source.get_current_frame()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        if len(eyes) == 2:
            x, y, w, h = eyes[eye_index]
            roi = gray[y:y+h, x:x+w]
            break
        
    threshold = 35
        
    _, gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    #gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    
    _, roi_contours = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY_INV)

    #roi_contours = gray[y:y+h, x:x+w]

    contours, hierarchy = cv2.findContours(roi_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    roi_contours = 255 - roi_contours

    roi_contours = cv2.cvtColor(roi_contours, cv2.COLOR_GRAY2BGR)

    cv2.imwrite("threshold_test/thresh_5.png", roi_contours)
    #cv2.imwrite("threshold_test/gray_4.png", roi)

    cv2.drawContours(roi_contours, contours, -1, (0,255,0), 1)

    l = 0
    s = 10**10
    for i in contours:
        for j in i:
            #print(j)
            if l < j[0][0]:
                l = j[0][0]
            elif s > j[0][0]:
                s = j[0][0]
    
    for j in range(len(contours)):
        for i in range(len(contours[j])):
            contours[j][i][0][0] += x
            contours[j][i][0][1] += y

    #print(contours)
    
    cv2.drawContours(img, contours, -1, (0,255,0), 1)

    cv2.circle(img, (int((l + s) / 2 + x), int(y + 30)), 2, (0, 0, 255), -1)

    cv2.circle(img, (int(l + x), int(y + 30)), 2, (255, 0, 255), -1)
    cv2.circle(img, (int(s + x), int(y + 30)), 2, (255, 0, 255), -1)

    #cv2.circle(img, ((l + s) // 2 + int(x), int(y) + 22), 2, (0, 255, 0), 2)

    pts = preprocess(roi)
    #preprocess(gray)

    cv2.circle(img, (int(x + pts[0]), int(y + pts[1])), 2, (255, 0, 0), -1)

    cv2.namedWindow('roi_contours',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('roi_contours', 600,600)
    cv2.imshow('roi_contours', roi_contours)
    cv2.imshow('gray', gray)
    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 3200,2400)
    cv2.imshow('img', img)

    return l, s, (l + s) / 2

class ImageSource:
    """
    Returns frames from camera
    """
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

    def get_current_frame(self, gray=False):
        ret, frame = self.capture.read()
        frame = cv2.flip(frame, 1)
        #cv2.imshow('frame', frame)
        if not gray:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def release(self):
        self.capture.release()

def preprocess(roi):
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 500
    detector_params.minArea = 50
    detector = cv2.SimpleBlobDetector_create(detector_params)

    while True:
        threshold = 30
        _, roi = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
        
        roi = cv2.erode(roi, None, iterations=2)
        roi = cv2.dilate(roi, None, iterations=4)
        roi = cv2.medianBlur(roi, 5)

        keypoints = detector.detect(roi)
        roi = cv2.drawKeypoints(roi, keypoints, roi)

        cv2.namedWindow('roi_processed',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('roi_processed', 600,600)
        cv2.imshow('roi_processed', roi)

        pts = cv2.KeyPoint_convert(keypoints)

        if len(pts) == 0:
            continue
        else:
            return pts[0]

img_source = ImageSource()
print(CalibrateV2(0, img_source))
