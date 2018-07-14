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


#save date with queue
def save_date(x):
    Queue.append(x)
    if len(Queue) > Maxsize:
        Queue.pop(0)

#draw sign
def draw_sign(img):
    pre_x = 100
    pre_y = 100
    cv2.line(img,(0,ARXES),(Frame_width,ARXES),(0,0,255),2)
    for i,d in enumerate(Queue):
        x = int((i+1)*Per_width)
        if i == 0:
            y = int(ARXES)
            cv2.circle(img,(x,y),4,(0,255,0),-1)
        else:
            y = int(ARXES + ( d - EYE_OPEN)*Per_hight)
            cv2.circle(img,(x,y),4,(0,255,0),-1)
            cv2.line(img,(pre_x,pre_y),(x,y),(255,0,0),2)
        #print('x={},y={},i={},d={},len={}'.format(x,y,i,d,len(Queue)))
        pre_x = x
        pre_y = y

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
#input argument 
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
    help="path to input video file")
args = vars(ap.parse_args())


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")

#input from camera 
#vs = VideoStream(src=0).start()

#input from file 
vs = cv2.VideoCapture('feng.avi')
time.sleep(1.0)

#get video attribute
print('the fps is {}'.format(vs.get(5)))
Frame_hight = int(vs.get(4))
Frame_width = int(vs.get(3))
ARXES = int(Frame_hight * (4.0/5))
print("Frame_width {},Frame_hight {}".format(Frame_width,Frame_hight))

Queue = []
Maxsize = 60
#Frame_width = 600
Per_width = (Frame_width) / Maxsize
Per_hight = 1000

EYE_OPEN = 0.30
EYE_AR_THRESH = 0.28
EYE_AR_CONSEC_FRAMES = 15
COUNTER = 0
TOTAL = 0
#save video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('out.avi',fourcc,20.0,(Frame_width,Frame_hight))

while True:
    #if not vs.isOpened():
    #    break
    ret,frame = vs.read()
    #frame = vs.read()

    #frame = imutils.resize(frame, width=Frame_width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        (x, y ,w , h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

        save_date(ear)
        draw_sign(frame)

        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for (x,y) in shape:
            cv2.circle(frame,(x,y),2,(255,0,0),-1)

    if out.isOpened():
        out.write(frame)
    else:
        print("writer closs")
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
out.release()
cv2.destroyAllWindows()
vs.release()
