# import modules
import cv2
import time
import numpy as np
import threading
import PIL
from PIL import Image, ImageTk
from tkinter import *
from Classify import *

# define variables
cap = cv2.VideoCapture(0)

blurValue = 41
bgSubThreshold = 50
learningRate = 0

global bgModel
bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

global blur
global i
i = 0

image_size = 250

delay = 3

# important text
text_display = ["Open your Palm Wide...\n", "Calculating...\n",
                "Feedback: Good Job! Your hand was fully opened!\n",
                "Feedback: Good Job! Your hand is fully closed!\n",
                "Feedback: Please place your hand in the blue rectangle!\n",
                "Feedback: Your hand was not fully opened.\n",
                "Feedback: Your hand is not fully closed.\n",
                "Close your Palm...\n",
                "Face your Palm perpendicular to the Camera...\n"]

label_display = ["Open your Palm Wide...", "Calculating...",
                "Feedback: Good Job! Your hand was fully opened!",
                "Feedback: Good Job! Your hand is fully closed!",
                "Feedback: Please place your hand in the blue rectangle!",
                "Feedback: Your hand was not fully opened.",
                "Feedback: Your hand is not fully closed.",
                 "Close your Palm...",
                 "Face your Palm perpendicular to the Camera..."]

instruct = """Welcome to Parkison's Hand Test 1!
To Begin or Restart the test, press <SPACE>!
Do not remove the laptop camera throughout the duration of this program.
If you have accidentally moved the laptop camera during the program, press <B>.
For instructions, press <I>.\n"""

font = "Verdana"

# remove background with opencv background subtractor
def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# run test on hand
def Test():
    global blur
    
    '''
    # display instructions
    text.insert(END, text_display[0], "emphasize")
    l.config(text=label_display[0], fg="black")
    time.sleep(1)

    # display hand image
    img = ImageTk.PhotoImage(PIL.Image.open("Hand Open.jpg"))
    pic.config(image=img)
    pic.image = img

    # wait 5 seconds
    for i in range(delay , 0, -1):
        text.insert(END, str(i) + "\n")
        l.config(text=str(i), fg="black")
        time.sleep(1)

    # display more text
    text.insert(END, text_display[1])
    l.config(text=label_display[1], fg="black")

    # call the ai
    cv2.imwrite("predict.jpg", blur)
    label_lines, top_k = classify("predict.jpg")
    result = label_lines[top_k[0]]
    '''
    
    # display instructions
    text.insert(END, text_display[8], "emphasize")
    l.config(text=label_display[8], fg="black")
    time.sleep(1)
    
    # display hand image
    img = ImageTk.PhotoImage(PIL.Image.open("Pen Hand.jpg"))
    pic.config(image=img)
    pic.image = img

    # wait 5 seconds
    for i in range(delay , 0, -1):
        text.insert(END, str(i) + "\n")
        l.config(text=str(i), fg="black")
        time.sleep(1)

    # display more text
    text.insert(END, text_display[1])
    l.config(text=label_display[1], fg="black")

    # call the ai
    cv2.imwrite("predict2.jpg", blur)
    label_lines, top_k = classify("predict2.jpg")
    result_2 = "half"
    for i in top_k:
        if "side" in label_lines[i]:
            result_2 = label_lines[i]
            break

    # display feedback
    if result_2 == 'open side':
        text.insert(END, text_display[2], "good")
        l.config(text=label_display[2], fg="green")
        
    elif result_2 == "nothing":
        text.insert(END, text_display[4], "bad")
        l.config(text=label_display[4], fg="red")
        
    else:
        text.insert(END, text_display[5], "bad")
        l.config(text=label_display[5], fg="red")

        '''
        img = cv2.addWeighted(cv2.imread("predict2.jpg"), 0.5, cv2.imread("Pen Hand.jpg"), 0.5, 0) # blending both images
        img = PIL.Image.fromarray(img)
        pic.config(image=img)
        pic.image = img
        '''
        
    time.sleep(5)

    # display text
    text.insert(END, text_display[7], "emphasize")
    l.config(text=label_display[7], fg="black")
    time.sleep(1)

    # display image
    img = ImageTk.PhotoImage(PIL.Image.open("Hand Close.jpg"))
    pic.config(image=img)
    pic.image = img

    # wait 5 seconds
    for i in range(delay, 0, -1):
        text.insert(END, str(i) + "\n")
        l.config(text=str(i), fg="black")
        time.sleep(1)

    # display more text         
    text.insert(END, text_display[1])
    l.config(text=label_display[1], fg="black")

    # call the ai
    cv2.imwrite("predict3.jpg", blur)

    label_lines, top_k = classify("predict3.jpg")
    result = label_lines[top_k[0]]

    # display feedback
    if result == 'close':
        text.insert(END, text_display[3], "good")
        l.config(text=label_display[3], fg="green")
        
    elif result == 'nothing':
        text.insert(END, text_display[4], "bad")
        l.config(text=label_display[4], fg="red")
       
    else:
        text.insert(END, text_display[6], "bad")
        l.config(text=label_display[6], fg="red")

    # remove the image
    pic.image = None

# run the test   
def run_test(event):
    t = threading.Thread(target=Test)
    t.start()
def save_image(event):
    global blur
    global i
    cv2.imwrite(f"open_side_{i}.jpg", blur)
    i += 1
    if i == 200:
        text.insert(END, "Done1")

# reset the background model
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

# initialise tkinter window    
root = Tk()
root.state('zoomed')
root.bind('<space>', run_test)
root.bind('<b>', reset_bg)
root.bind('<i>', instructions)
root.bind('<e>', save_image)
root.title("Parkison's Hand Test 1")

# widget to display webcam feed
lmain = Label(root)
lmain.pack()

# display webcam feed
def show_frame():
    global blur
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (image_size, 0), (0, image_size), (255, 0, 0), 2) # draw rectangle to place hand in
    img = removeBG(frame) # remove the background
    img = img[0:image_size, 0:image_size]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale images
    
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0) # perform gaussian blur
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()

# display important messages
l = Label(text="IMPORTANT", font=(font, 30))
l.pack(pady=10)

# text widget to display output
text = Text(font=(font, 9), width=100, height=15)
text.pack(pady=20)
text.tag_configure("good", font=(font, 10, "bold"), foreground = "green")
text.tag_configure("bad", font=(font, 10, "bold"), foreground = "red")
text.tag_configure("emphasize", font=(font, 10, "bold"), foreground = "black")

# picture widget
pic = Label()
pic.pack()

# display instructions
text.insert(END, instruct, "emphasize")

# run auto scroll in a thread
t = threading.Thread(target=see_bottom)
t.start()

# start window
root.mainloop()
