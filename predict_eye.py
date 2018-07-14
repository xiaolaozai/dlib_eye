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
from sklearn import svm
from sklearn.externals import joblib

#save date with queue
def save_date(data):
    Queue.append(data)
    if len(Queue) > Maxsize:
        Queue.pop(0)

VECTOR_SIZE = 9
def queue_in(queue,data):
    queue.append(data)
    if len(queue) > VECTOR_SIZE:
        queue.pop(0)
    #print(queue)
    return queue
def turning_head(three_node):
    Left = dist.euclidean(three_node[0],three_node[1])
    Righ = dist.euclidean(three_node[1],three_node[2])
    value = Left / Righ
    return value

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
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1],mouth[7])
    B = dist.euclidean(mouth[2],mouth[6])
    C = dist.euclidean(mouth[3],mouth[5])
    D = dist.euclidean(mouth[0],mouth[4])

    mouth_dis = (A+B+C)/ (3.0 * D)
    return mouth_dis

#input argument 
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="",
    help="path to input video file")
args = vars(ap.parse_args())


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = (42,48)    #eye dex
(rStart, rEnd) = (36,42)
(mstart, mend) = (60,68)
# loading train mkdel

clf = joblib.load("./train/ear_svm.m")

print("[INFO] starting video stream thread...")

vs = cv2.VideoCapture(0)
time.sleep(1.0) #wait camera open

#get video attribute
print('the fps is {}'.format(vs.get(5)))
Frame_hight = 800 #int(vs.get(4))
Frame_width = 1024#int(vs.get(3))
ARXES = int(Frame_hight * (6.0/7))
print("Frame_width {},Frame_hight {}".format(Frame_width,Frame_hight))

Maxsize = 60
Per_width = (Frame_width) / Maxsize
Per_hight = 1000

EYE_OPEN = 0.30
EYE_AR_THRESH = 0.28
EYE_AR_CONSEC_FRAMES = 5
COUNTER = 0
TOTAL = 0
Queue = []
ear_vector =[]

#save video
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('9_vector.avi',fourcc,20.0,(Frame_width,Frame_hight))

while True:
    if not vs.isOpened():
        break

    ret,frame = vs.read()
    frame = imutils.resize(frame, width=Frame_width,height=Frame_hight)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        three_node = [shape[1],shape[30],shape[15]]
        mouth  = shape[mstart:mend]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        mouth_dis = mouth_aspect_ratio(mouth)

        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        (x, y ,w , h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        for i,(x,y) in enumerate(shape):
            cv2.circle(frame,(x,y),2,(255,0,0),-1)
            #cv2.putText(frame,"{}".format(i),(x+3,y+3),
            #cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),2)
        # draw sign
        save_date(ear)
        draw_sign(frame)
        # turn head
        turn_value = turning_head(three_node)
        # predict 
        ear_vector = queue_in(ear_vector,ear)
        if len(ear_vector) >= VECTOR_SIZE:
            #print(ear_vector)
            input_vector = []
            input_vector.append(ear_vector)
            res = clf.predict(input_vector)
            if res == 1:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0

        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "turn: {:.2f}".format(turn_value), (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "mouth: {:.2f}".format(mouth_dis), (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,0), 2)

#    if out.isOpened():
#        out.write(frame)
#    else:
#        print("writer close")
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
#out.release()
cv2.destroyAllWindows()
vs.release()
