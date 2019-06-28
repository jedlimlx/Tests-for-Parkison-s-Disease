# Tests-for-Parkison-s-Disease
Parkison's Eye Test
-------------------

Operating System:
Windows 10 --> Uses Registry to Store Settings
MUST INSTALL AVBIN64 --> https://avbin.github.io/AVbin/Download.html
IF NOT --> NO SOUND!!!

To Run the Test:
Press the <START> button under the controls to start.
Press the <RESET> button under the controls to restart the test.
Follow the instructions by the voice and look to the right and left.

Settings:
You can adjust number of repetitions, where to output data and whether or not to use the space bar to capture your iris.

Troubleshooting:
If you press <SPACE> repeatedly and no response, click on the grey area of the window and try again.

Algorithm:
Use Haar Cascade from OpenCV to extract the region of interest containing the eye.
Perform Transfer Learning with Google Inception V3 Model for Image Classification.
Model Weights are stored in a frozen .pb file.
tkinter is used for creating GUI.

Parkison's Hand Test 1
----------------------

Operating System:
Probably Any?

To Run the Test:
Press the <SPACE> button.
Place your hand in the blue box.
Open your hand and close your and as the program says.

Other Instructions:
If you moved the laptop camera, press \<B\>.
For instructions, press \<I\>.

Troubleshooting:
If you press <SPACE> repeatedly and no response, click on the grey area of the window and try again.

Algorithm:
Use OpenCV background subtractor to remove the background so that only the hand remains.
Perform Transfer Learning with Google Inception V3 Model for Image Classification.
Model Weights are stored in a frozen .pb file.
tkinter is used for creating GUI.

Parkison's Hand Test 2
----------------------

Operating System:
Also Probably Any??

To Run the Test:
Press the <T> button
Place your hand in the blue box
Press <SPACE> to start recording
Follow the images and do a twisting light bulb action.
Press <SPACE> again after you are done.

Settings:
You can change number of repetitions.

Other Instructions:
If you moved the laptop camera, press \<B\>.
For instructions, press \<I\>.
When performing the action, move your ahnd slowly.

Troubleshooting:
If you press <SPACE> repeatedly and no response, click on the grey area of the window and try again.

Algorithm:
Use OpenCV background subtractor to remove the background so that only the hand remains.
Mutiple shots of the hand are taken during the motion.
Perform Transfer Learning with Google Inception V3 Model for Image Classification.
Model Weights are stored in a frozen .pb file.
tkinter is used for creating GUI.
AI is used to classify images and most frequent result will become the overall result

AI
----------
All trained models are in AI.zip
Place them in logs.
Code to train AI taken from here:
https://github.com/xuetsing/image-classification-tensorflow
