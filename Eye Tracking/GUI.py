"""
Problems:
AI is inaccurate.
"""

# import modules
import threading
import pickle
import PIL
import pandas as pd
import pyglet
import sys, os
from PIL import Image, ImageTk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from EyeTracking import *
from time import time
from Classify import classify

# declare variables
global cv2image
cv2image = []

global d
d = {}

global break_bool
break_bool = False

global file
file = "./"

global Quit
Quit = False

global recurse
recurse = 0

global recurse2
recurse2 = 0

global space
space = False

global ind
ind = 0

# reset all data collected
def reset():
    global d
    d["Time"] = []
    d["Right Iris X"] = []
    d["Right Iris Y"] = []
    d["Left Iris X"] = []
    d["Left Iris Y"] = []
    d["Iris Distance"] = []

reset()

# all the text
text_lst = ["Calibrating Left Iris ... Hold Still\n",
        "Calibrating Right Iris ... Hold Still\n",
        "Calibrating Normal Iris ... Hold Still\n",
        "Calibrating Complete!\n",
        "Move your Iris to Your Right and Look at the Golden Eye\n",
        "(Press <SPACE> when you are Ready)\n",
        "You pressed <SPACE>. Hold still... Processing Data...\n",
        "Move your Iris to Your Left and Look at the Golden Eye\n",
        "Good!\n",
        "Bad!\n",
        "Feedback: Your eye did not go far enough left or right.\n",
        "Feedback: Good Job! Your eye went far enough right and left.\n",
        "The Image of your Eye has been saved! You can rest your eye now.\n"]

# collect data
def UpdateData():
    global d
    global break_bool
    global file
    global Quit
    i = 0
    while True:
        iris = eyeris_detector.irises
        #print(len(iris))
        if len(iris) == 2:
            d["Time"] = np.append(d["Time"], [round(i,2)])
            d["Right Iris X"] = np.append(d["Right Iris X"], [iris[0][0]])
            d["Right Iris Y"] = np.append(d["Right Iris Y"], [iris[0][1]])
            d["Left Iris X"] = np.append(d["Left Iris X"], [iris[1][0]])
            d["Left Iris Y"] = np.append(d["Left Iris Y"], [iris[1][1]])
            d["Iris Distance"] = np.append(d["Iris Distance"], [eyeris_detector.distance()])
            
            sleep(0.1)
            i += 0.1
        if break_bool:
            text.insert(END, "DOWNLOAD CSV FILE STARTING...\n")
            pd.DataFrame.from_dict(d).to_csv(f"{file}/IrisDATA.csv")
            text.insert(END, "DOWNLOAD CSV FILE SUCCESFUL\n")
            break_bool = False
            reset()
        if Quit:
            break
            
# update images and save in real time
def UpdateImages():
    global break_bool_2
    global file
    global Quit
    sleep(0.1)
    while True:
        try:
            if cRealVar.get():
                images = CalRight.visualise_data()
                if len(cv2image) != 0:
                    v4 = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
                    i = 0
                    for j in images:
                        i += 1
                        if eval(f"c{i}Var.get()"):
                            cv2.imwrite(f"{file}/images/visual_{i}_{d['Time'][-1]}.png", j)
                    if c5Var.get():
                        cv2.imwrite(f"{file}/images/visual_{len(images) + 1}_{d['Time'][-1]}.png", v4)
                else:
                    v4 = 0
        except MemoryError: # prevent memory error
            sleep(1)
            print("Image Memory Exceeded")
        if Quit:
            break

# start download of data into the csv file
def downloadData():
    global break_bool
    break_bool = True

# detect the irises
class EyerisDetector:
    def __init__(self, image_source, classifier, tracker):
        self.tracker = tracker
        self.classifier = classifier
        self.image_source = image_source
        self.irises = []
        self.frame = 0

    # find distance between two irises sqrt((x1 - x2) ^ 2 + (y1 - y2) ^ 2)
    def distance(self):
        if len(self.irises) < 2:
            return -1
        else:
            return (abs(self.irises[0][0] - self.irises[1][0]) ** 2 + abs(self.irises[0][1] - self.irises[1][1]) ** 2) ** 0.5
    
    # start detection
    def run(self):
        global Quit
        i = 0
        while True:
            start_time = time()
            self.frame = self.image_source.get_current_frame()
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) # grayscale image
            try:            
                if len(self.irises) >= 2: # if there are more than 2 irises
                    track_result = self.tracker.track(old_gray, gray, self.irises) # use tracker
                    self.irises, lost_track = track_result
                    
                    if lost_track: # lost track
                        self.irises = self.classifier.get_irises_location(gray)
                
                else:
                    self.irises = self.classifier.get_irises_location(gray)
                    
            except Exception as e: # unexplained errors occasionally occur
                print(e)
                continue
            
            old_gray = gray.copy()

            FPS = round(1 / (time() - start_time), 2)
            if not i % 10:
                frameRateLabel.config(text=f"{FPS} FPS") # FPS Counter
                i = i % 10
            i += 1
            
            if Quit:
                break

        self.image_source.release()

# display webcam feed + processing
def show_image_with_data():
    try:
        global cv2image
        global recurse
        global ind
            
        frame = eyeris_detector.frame
        
        '''
        l1, s1, n1, lst1 = CalLeft.main()
        l2, s2, n2, lst2 = CalRight.main()

        cv2.circle(frame, (int(n1 + lst1[0]), int(lst1[1] + 25)), 4, (0, 0, 255), -1)
        cv2.circle(frame, (int(n2 + lst2[0]), int(lst2[1] + 25)), 4, (0, 0, 255), -1)

        cv2.circle(frame, (int(l1 + lst1[0]), int(lst1[1] + 25)), 4, (255, 0, 255), -1)
        cv2.circle(frame, (int(s1 + lst1[0]), int(lst2[1] + 25)), 4, (255, 0, 255), -1)

        cv2.circle(frame, (int(l2 + lst2[0]), int(lst1[1] + 25)), 4, (255, 0, 255), -1)
        cv2.circle(frame, (int(s2 + lst2[0]), int(lst2[1] + 25)), 4, (255, 0, 255), -1)
        '''
        eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml') # load haar cascade
        eyes = eye_cascade.detectMultiScale(frame, 1.3, 5) # detect eyes with haar cascade
        for (x,y,w,h) in eyes:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) # draw rectangle
            roi = frame[y:y+w, x:x+h] # get region of interest
            cv2.imwrite(fr"C:\Users\Jed Lim\Documents\roi_{ind}.jpg", cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)) # grayscale images

        #for w, h in irises:
        #    cv2.circle(frame, (w, h), 2, (0, 255, 0), -1)
                
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # convert image to RBGA for display (OpenCV uses BGR)
        
        img = PIL.Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.configure(image=imgtk)
        
        recurse += 1
        ind += 1
        sleep(0.01)
        
        if recurse < 400 and not Quit: # recursion may cause memory issues
            show_image_with_data()
        else:
            recurse = 0
            return None
            
    except Exception as e: # recursion may cause memory issues
        print(e)
        return None
    
# run the test
def TestV2():
    global space
    sleep(3)
    allowance = 3
    iris_detect = eyeris_detector
    max_score = 2 * repSlider.get()
    score = 0

    for i in range(repSlider.get()):
        text.insert(END, text_lst[4] + str(round(d["Time"][-1], 1)) + "\n")
        if cSpaceVar.get():
             text.insert(END, text_lst[5])

        LookImage2.config(image = imgLook)
        LookImage.config(image = transImage)
             
        soundRight = pyglet.media.load('Right Iris.wav', streaming=True) # play audio
        soundRight.play()

        while True: # check if user pressed space
            sleep(0.1)
            if space:
                if cSpaceVar.get():
                    text.insert(END, text_lst[6])
                else:
                    sleep(3)
                space = False
                break

        eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml') # load haar cascade
        eyes = eye_cascade.detectMultiScale(eyeris_detector.frame, 1.3, 5) # detect eyes with haar cascade
        while len(eyes) < 2:
            eyes = eye_cascade.detectMultiScale(cv2.cvtColor(eyeris_detector.frame, cv2.COLOR_BGR2GRAY), 1.3, 5) # detect eyes with haar cascade
            
        for i in range(2):
            (x,y,w,h) = eyes[i]
            roi = eyeris_detector.frame[y:y+w, x:x+h] # get region of interest
            if i == 1:
                cv2.imwrite("left.jpg", cv2.cvtColor(cv2.resize(roi, (400, 400)), cv2.COLOR_BGR2GRAY)) # write the images
            else:
                cv2.imwrite("right.jpg", cv2.cvtColor(cv2.resize(roi, (400, 400)), cv2.COLOR_BGR2GRAY))
        
        text.insert(END, text_lst[12])

        soundDone = pyglet.media.load('ImageTaken.wav', streaming=True) # play audio
        soundDone.play()

        # calling the AI 
        a = classify("left.jpg") 
        if a != "normal":
            b = classify("right.jpg")

        if a != "normal" and b != "normal":
            text.insert(END, text_lst[8])
            score += 1
        else:
            text.insert(END, text_lst[9])
        
        LookImage2.config(image = transImage)
        LookImage.config(image = imgLook)

        text.insert(END, text_lst[7] + str(round(d["Time"][-1], 1)) + "\n")
        if cSpaceVar.get():
             text.insert(END, text_lst[5])

        soundLeft = pyglet.media.load('Left Iris.wav', streaming=False) # play audio
        soundLeft.play()

        while True: # detect if user pressed space
            sleep(0.1)
            if space:
                if cSpaceVar.get():
                    text.insert(END, text_lst[6])
                else:
                    sleep(3)
                space = False
                break

        left_roi = CalLeft.update_roi()
        right_roi = CalRight.update_roi()

        cv2.imwrite("left.jpg", (cv2.resize(left_roi, (400, 400))))
        cv2.imwrite("right.jpg", (cv2.resize(right_roi, (400, 400))))

        text.insert(END, text_lst[12])

        soundDone = pyglet.media.load('ImageTaken.wav', streaming=True) # play audio
        soundDone.play()
        
        a = classify("left.jpg")
        if a != "normal":
            b = classify("right.jpg")
            
        if a != "normal" and b != "normal":
            text.insert(END, text_lst[8])
            score += 1
        else:
            text.insert(END, text_lst[9])

    # display score
    text.insert(END, f"Test Complete! You got {score} out of {max_score}\nTime: " + str(round(d["Time"][-1], 1)) + "\n")
    
    # display feedback
    if (score / max_score) * 1000 < 650:
        text.insert(END, text_lst[10])
    else:
        text.insert(END, text_lst[11])

    sound = pyglet.media.load('Test Complete.wav', streaming=False) # play audio
    sound.play()

    LookImage2.config(image = transImage)
    LookImage.config(image = transImage)

# auto scroll for text widget
def scroll():
    global Quit
    while True:
        try:
            text.see("end") 
        except Exception:
            break
        
        if Quit:
            break

# save settings to registry
def SaveSettings():
    set_reg("Reps", str(repSlider.get()), REG_PATH)
    set_reg("ImgCheckbox", str([cRealVar.get(), c1Var.get(), c2Var.get(), c3Var.get(), c4Var.get(), c5Var.get()]), REG_PATH)
    set_reg("DataDir", file, REG_PATH)
    set_reg("Space", str(cSpaceVar.get()), REG_PATH)
    UpdateSettings()

# update settings according to registry values
def UpdateSettings():
    global file
    x = get_reg("Reps", REG_PATH) # get values from registry
    if x is not None:
        repSlider.set(int(x))

    x = get_reg("ImgCheckbox", REG_PATH) # get values from registry
    if x is not None:
        x = eval(x)
        #print(x)
        cRealVar.set(x[0])
        c1Var.set(x[1])
        c2Var.set(x[2])
        c3Var.set(x[3])
        c4Var.set(x[4])
        c5Var.set(x[5])

    x = get_reg("DataDir", REG_PATH) # get values from registry
    if x is not None:
        dirLabel.config(text=f"Data Storage Location: {x}")
        file = x

    x = get_reg("Space", REG_PATH) # get values from registry
    if x is not None: 
        if x == "True":
            cSpaceVar.set(True)
        else:
            cSpaceVar.set(False)
            try:
                t8.start()
            except Exception:
                pass
# start the test   
def StartTest():
    try:
        tabControl.select(0)
        t2.start()
        t4.start()
        t5.start()
    except Exception as e:
        print(e)

# restart the test
def ResetTest():
    text.delete(1.0, END)
    tabControl.select(0)
    t2 = threading.Thread(target=TestV2)
    t2.start()

# get the directory to save data
def getDataDir():
    global file
    file = filedialog.askdirectory()
    dirLabel.config(text = f"Data Storage Location: {file}")

# quit the application
def Quit2():
    global Quit
    Quit = True
    root.destroy()
    python = sys.executable
    os.execl(python, python, * sys.argv)

# solution to recursion memory issues
def fixRecurse():
    sleep(1)
    while True:
        if Quit:
            break
        
        a = show_image_with_data()
            
        if a is None:
            a = 0
            continue

# detect press of space
def callback(event):
    global space
    #print("Prev Space", space)
    space = True
    #print("Now Space", space)

# check if user wants to use space
def dospace():
    while True:
        sleep(0.1)
        callback(0)
        if Quit or cSpaceVar.get():
            space = False
            break
    
# initialise irises detector
eyeris_detector = EyerisDetector(image_source=ImageSource(), classifier=CascadeClassifier(),
                                     tracker=LucasKanadeTracker())

# calibration 
CalLeft = CalibrateV2(0, eyeris_detector.image_source)
CalRight = CalibrateV2(1, eyeris_detector.image_source)
width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# initialise window
root = Tk()
root.state("zoomed")
root.title("Parkison's Test")
root.bind('<space>', callback)
root.iconbitmap('Logo.ico')

# initialise tabs
tabControl = ttk.Notebook()
tabTracker = ttk.Frame()
tabSettings = ttk.Frame()
tabControls = ttk.Frame()

# add tabs to window
tabControl.add(tabTracker, text = "Eye Tracking Test")
tabControl.add(tabControls, text = "Control")
tabControl.add(tabSettings, text = "Settings")

tabControl.pack(expand=1, fill="both")

# display images to look at
imgFrame = Frame(tabTracker)
imgFrame.pack()

imgLook = cv2.imread("Look.png")
imgLook = cv2.cvtColor(imgLook, cv2.COLOR_BGR2RGBA)
imgLook[np.all(imgLook == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0] # black part transparent
imgLook = cv2.resize(imgLook, (800, 400)) # resize the image

imgLook = PIL.Image.fromarray(imgLook)
imgLook = ImageTk.PhotoImage(image=imgLook)

transImage = PhotoImage(file = "Transparent.gif")

LookImage = Label(imgFrame, image=transImage)
LookImage.grid(row=0, column=0, padx=300)

LookImage2 = Label(imgFrame, image=transImage)
LookImage2.grid(row=0, column=2, padx=300)

# widget to display webcam feed
lmain = Label(imgFrame, width=800, height=600)
lmain.grid(row=0, column=1)

# display FPS
frameRateLabel = Label(tabTracker, text="FPS", font=("Arial", 30))
frameRateLabel.pack(pady=20)

# text widgte for displaying output
text = Text(tabTracker)
text.pack()

# button to download data
csvButton = Button(tabTracker, text="Download Data (.csv)", command=downloadData).pack()

# settings
testSettings = ttk.Labelframe(tabSettings, text="Eye Test Settings")
testSettings.pack(pady = 10)

# label + slider for number of repetitions of the test
repLabel = Label(testSettings, text="Number of Repetitions")
repLabel.pack(pady = 10)

repSlider = Scale(testSettings, from_ = 1, to_ = 5, orient=HORIZONTAL, length=500)
repSlider.pack(pady = 30, padx = 30)

# checkboxes for data downloads
visualCheckboxes = ttk.Labelframe(tabSettings, text="Data Download Settings")
visualCheckboxes.pack(pady=20)

cRealVar = BooleanVar()
c1Var = BooleanVar()
c2Var = BooleanVar()
c3Var = BooleanVar()
c4Var = BooleanVar()
c5Var = BooleanVar()

cReal = Checkbutton(visualCheckboxes, text="Real Time Image Download", var=cRealVar)
cReal.grid(row=0, column=2, pady=10)
c1 = Checkbutton(visualCheckboxes, text="Eye Contours", var=c1Var)
c1.grid(row=1, column=1, pady=10)
c2 = Checkbutton(visualCheckboxes, text="Threshold Eye", var=c2Var)
c2.grid(row=1, column=3, pady=10)
c3 = Checkbutton(visualCheckboxes, text="Threshold Image", var=c3Var)
c3.grid(row=2, column=0, pady=10, padx=10)
c4 = Checkbutton(visualCheckboxes, text="Processed Eye", var=c4Var)
c4.grid(row=2, column=2, pady=10)
c5 = Checkbutton(visualCheckboxes, text="Webcam", var=c5Var)
c5.grid(row=2, column=4, pady=10, padx=10)

# choose the directory to save data
dirLabel = Label(visualCheckboxes, text="Data Storage Location:")
dirLabel.grid(row=3, column=2, pady=30)

dirButton = Button(visualCheckboxes, text="Click to Choose", command=getDataDir)
dirButton.grid(row=4, column=2, pady=10)

# check if user wants to se space bar
cSpaceVar = BooleanVar()
cSpace = Checkbutton(tabSettings, text="Use Space Bar", var=cSpaceVar)
cSpace.pack(pady=10)

# controls -->  Quit, Start and Reset
settingsButton = Button(tabSettings, text="Save Settings", command=SaveSettings)
settingsButton.pack(pady = 10)

startButton = Button(tabControls, text="Start", font=("Arial", 30), command=StartTest)
startButton.pack(pady=20)

resetButton = Button(tabControls, text="Reset", font=("Arial", 30), command=ResetTest)
resetButton.pack(pady=20)

quitButton = Button(tabControls, text="Quit", font=("Arial", 30), command=Quit2)
quitButton.pack(pady=20)

UpdateSettings()

# start all the threads
t = threading.Thread(target=scroll)
t.start()

t2 = threading.Thread(target=TestV2)

t3 = threading.Thread(target=eyeris_detector.run)
t3.start()

t4 = threading.Thread(target=UpdateData)

t5 = threading.Thread(target=UpdateImages)

t6 = threading.Thread(target=fixRecurse)
t6.start()

if not cSpaceVar.get():
    t8 = threading.Thread(target=dospace)
    t8.start()

LookImage.image = 0
LookImage2.image = 0

root.mainloop()
