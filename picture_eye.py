#-*- coding: UTF-8 -*-
# write by feng
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

img = cv2.imread('./xiao.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
rects = detector(gray,0)

for rect in rects:
    shape = predictor(gray,rect)
    shape = face_utils.shape_to_np(shape)
    for i,(x,y) in enumerate(shape):
        cv2.circle(img,(x,y),2,(255,0,0),-1)
        cv2.putText(img,"{}".format(i),(x+3,y+3),
        cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),2)
cv2.imshow("Frame",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
