import sys
#import FindMattes as fm ### only if you have a big gpu
import findMattesCPU as fm
import cv2
import faceDetector
from imutils import face_utils
import numpy as np
import imutils
import dlib

#TODO we take a video in, for each frame of it we output a matte at the same size

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture('test.mp4')
count = 0
while cap.isOpened(): # as long as the video is open

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    ret,frame = cap.read() # we take the next frame

    cv2.imwrite("frame%d.jpg" % count, frame) # we write it on the disk
    #### voodoo
        # load the input image and convert it to grayscale
    image = cv2.imread("frame%d.jpg" % count)
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # detect faces in the grayscale image
    rects = detector(gray, 1)
	
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    
            # loop over the subset of facial landmarks, drawing the
            # specific face part
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
	
    # extract the ROI of the face region as a separate image
    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
    roi = image[y:y + h, x:x + w]
    roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
 
	# visualize all facial landmarks with a transparent overlay
    #output = face_utils.visualize_facial_landmarks(image, shape) # we move these two line so we draw the face OVER the matte
    #cv2.imwrite(str(frame), output)
    ####

    fm.createMatte("frame%d.jpg" % count, "matte_frame%d.jpg" % count, 720) # 720 is video res. Change it at need

    output = face_utils.visualize_facial_landmarks(cv2.imread("matte_frame%d.jpg" % count), shape) # we move these two line so we draw the face OVER the matte
    cv2.imwrite("matte_frame%d.jpg" % count, output)

    count = count + 1

cap.release()
cv2.destroyAllWindows()  # destroy all the opened windows