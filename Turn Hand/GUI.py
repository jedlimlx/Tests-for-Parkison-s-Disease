# import modules
import cv2
import time
import numpy as np
import threading
import PIL
import os
import tensorflow as tf
from multiprocessing import Process
from Classify import classify
from PIL import Image, ImageTk
from tkinter import *
from collections import Counter 

# declare variables
cap = cv2.VideoCapture(0)

blurValue = 41
bgSubThreshold = 50
learningRate = 0

global bgModel
bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

global blur
global i
i = 0

global space
space = False

global ans
ans = []

text_display = ["Rotate your Hand Anticlockwise.\nPosition your Fingers as if you are twisting a light bulb.\n",
                "Press <SPACE> when you start.\n",
                "When you are done, press <SPACE> again.\n",
                "Test is Complete! Calculating Results...\n",
                "Feedback: Good Job! Your hand is positioned properly.\n",
                "Feedback: Your hand is not positioned properly.\n",
                "Feedback: Your hand is not in the blue rectangle.\n"]

instruct = """Welcome to Parkison's Hand Test 2!
To Begin or Restart the test, press <SPACE>!
Do not move the laptop camera throughout the duration of this program.
If you have accidentally moved the laptop camera during the program, press <B>.
For instructions, press <I>.\n"""

font = "Verdana"

image_size = 250
max_reps = 10

# find most frequent element in a list
def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0]

# remove background with opencv background remover
def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate) # OpenCV bgmodel

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1) # erode the image
    res = cv2.bitwise_and(frame, frame, mask=fgmask) # use eroded image as a mask
    return res

# calling ai
def callback(i):
    global ans
    try:
        print(f"Classifying (predict/predict_{i}.jpg)")
        start_time = time.time()
        ans.append(classify(f"predict/predict_{i}.jpg"))
        print("Time:", time.time() - start_time) # print time taken
    except Exception as e:
        print(e)
    #text.insert(END, f"Calculating... {i + 1} / {j} Complete\n")

# run the test
def Test():
    # run callback in threads to speed up process
    def callback2():
        start_time = time.time()
        i = 0
        while True:
            p = threading.Thread(target=callback, args=(i,))
            p.start()
            time.sleep(sample_frequency + 0.01)
            i += 1
            if not space and i > j:
                break
            
    global ans
    global blur
    global space
    ans = []

    # frequency images are taken to be classified by AI
    sample_frequency = 1.25
    
    # display text
    text.insert(END, text_display[0])

    for i in range(s.get()):
        ans = []

        # display text
        text.insert(END, text_display[1], "emphasize")
        text.insert(END, text_display[2], "emphasize")
        
        l.config(text=text_display[1][:-2], fg="black")

        # save images for prediction
        for i in os.listdir("predict"):
            os.remove("predict/" + i)

        # wait unitl <space> key is pressed
        while True:
            time.sleep(0.1)
            if space:
                break

        # display text
        l.config(text=text_display[2][:-2], fg="black")

        # collecting the images
        j = 0
        while space:
            cv2.imwrite(f"predict/predict_{j}.jpg", blur)
            j += 1
            if j == 1:
                t = threading.Thread(target=callback2)
                t.start()
            time.sleep(sample_frequency)

        # display text
        text.insert(END, text_display[3])

        l.config(text=text_display[3][:-2], fg="black")
        
        # wait until AI is done classifying
        start_time = time.time()
        while len(ans) < j:
            time.sleep(0.5)
            print(len(ans), j)
            continue
        
        print("Time Taken:", time.time() - start_time)

        # display feedback
        if most_frequent(ans) == "rotate": # most frequent class used
            text.insert(END, text_display[4], "good")
            l.config(text=text_display[4][:-2], fg="green")
            
        elif most_frequent(ans) == "hand":
            text.insert(END, text_display[5], "bad")
            l.config(text=text_display[5][:-2], fg="red")
        else:
            text.insert(END, text_display[6], "bad")
            l.config(text=text_display[6][:-2], fg="red")

# run test  
def run_test(event):
    t = threading.Thread(target=Test)
    t.start()
'''
def save_image(event):
    global blur
    global i
    cv2.imwrite(f"test_{i}.jpg", blur)
    i += 1
    if not i % 100:
        text.insert(END, "Done! " + str(i))
'''

# reset background model
def reset_bg(event):
    global bgModel
    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

# auto scroll text widget
def see_bottom():
    while True:
        try:
            text.see("end")
        except Exception:
            break

# display instructions   
def instructions(event):
    text.insert(END, instruct, "emphasize")

# detect space bar press
def space_toggle(event):
    global space
    space = not space

# initalise window    
root = Tk()
root.state('zoomed')

# add key binds
root.bind('<t>', run_test)
root.bind('<b>', reset_bg)
root.bind('<i>', instructions)
root.bind('<space>', space_toggle)
root.title("Parkison's Hand Test 2")
root.focus_force()

# widget to display webcam feed
lmain = Label(root)
lmain.pack(pady=20)

# display the webcam feed
def show_frame():
    global blur
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (image_size, 0), (0, image_size), (255, 0, 0), 2)
    img = removeBG(frame)
    img = img[0:image_size, 0:image_size]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale image
    
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0) # apply Gaussian Blur

    #ret, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
    
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # convert to RGBA (OpenCV uses BGR)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()

# display important information
l = Label(text="IMPORTANT", font=(font, 30))
l.pack(pady=10)

# slider to control number of repetitions of the test
s = Scale(label="Number of Repetitions", from_ = 1, to_ = max_reps, orient=HORIZONTAL, length=400)
s.pack(pady=10)

# text widget to display output
text = Text(height=10, font=(font, 9), width=125)
text.pack(pady=20)
text.tag_configure("good", font=(font, 10, "bold"), foreground = "green")
text.tag_configure("bad", font=(font, 10, "bold"), foreground = "red")
text.tag_configure("emphasize", font=(font, 10, "bold"), foreground = "black")

# load images 
img_1 = ImageTk.PhotoImage(PIL.Image.open("Rotate_1.jpg"))
img_2 = ImageTk.PhotoImage(PIL.Image.open("Rotate_2.jpg"))
img_3 = ImageTk.PhotoImage(PIL.Image.open("Rotate_3.jpg"))
img_4 = ImageTk.PhotoImage(PIL.Image.open("Rotate_4.jpg"))
img_5 = ImageTk.PhotoImage(PIL.Image.open("Rotate_5.jpg"))

# widget to display images
pics = Frame()
pics.pack()

# displaying images
for i in range(5):
    exec(f"label{i} = Label(pics, image=img_{i+1})")
    exec(f"label{i}.grid(row=0, column=i)")

# displaying instructions
text.insert(END, instruct, "emphasize")

# always auto scroll in thread
t = threading.Thread(target=see_bottom)
t.start()

root.mainloop()
