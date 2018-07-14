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
import os
import csv
import math

beishu = 1000
VECTOR_SIZE = 9
def queue_in(queue, eye, head, mouth):
    eye = math.ceil(eye*beishu)
    head = math.ceil(head*beishu)
    mouth = math.ceil(mouth*beishu)
    queue.append(eye)
    queue.append(head)
    queue.append(mouth)
    while len(queue) > VECTOR_SIZE:
        #print(len(queue))
        queue.pop(0)
    return queue

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def turning_head(three_node):
    Left = dist.euclidean(three_node[0],three_node[1])
    Righ = dist.euclidean(three_node[1],three_node[2])
    value = Left / Righ
    return value

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1],mouth[7])
    B = dist.euclidean(mouth[2],mouth[6])
    C = dist.euclidean(mouth[3],mouth[5])
    D = dist.euclidean(mouth[0],mouth[4])
    mouth_dis = (A+B+C)/ (3.0 * D)
    return mouth_dis

#input argument 
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default=True,
    help="path to input video file")
args = vars(ap.parse_args())

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = (42,48)    #eye dex
(rStart, rEnd) = (36,42)
(mstart, mend) = (60,68)
class_num = 3

print("[INFO] starting video stream thread...")

vs = cv2.VideoCapture(0)
time.sleep(1.0)

#get video attribute
print('the fps is {}'.format(vs.get(5)))
Frame_hight = #int(vs.get(4))
Frame_width = #int(vs.get(3))
print("Frame_width {},Frame_hight {}".format(Frame_width,Frame_hight))


if not os.path.exists('train'):
    os.mkdir('train')
train_csv = open('train/train.csv','wb')
csvwriter = csv.writer(train_csv, delimiter=',')

count=[0,0,0]
flag = -1
vector = []
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('collect.avi',fourcc,20.0,(Frame_width,Frame_hight))

while True:
    if not vs.isOpened():
        break
    ret,frame = vs.read()

    #wait input 
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break

    frame = imutils.resize(frame, width=Frame_width,height=Frame_hight)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        three_node = [shape[1],shape[30],shape[15]]
        Mouth = shape[mstart:mend]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        head = turning_head(three_node)
        mouth = mouth_aspect_ratio(Mouth)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        (x, y ,w , h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        for (x,y) in shape:
            cv2.circle(frame,(x,y),2,(255,0,0),-1)

        if flag != -1:
            vector = queue_in(vector, ear, head , mouth)
            #print vector

        if len(vector) == VECTOR_SIZE and flag != -1:
            temp = vector[:]
            temp.append(flag)
            count[flag] = count[flag]+1
            csvwriter.writerow(temp)

        cv2.putText(frame, "0:fatigue image: {}".format(count[0]), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(frame, "1:fatigue image: {}".format(count[1]), (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, "2:fatigue image: {}".format(count[2]), (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, "eye: {:.3f}".format(ear), (Frame_width-150,30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, "head: {:.3f}".format(head), (Frame_width-150,60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, "mouth: {:.3f}".format(mouth), ( Frame_width-150,90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imshow("Frame", frame)
    #if out.isOpened():
    #    out.write(frame)

train_csv.close()
cv2.destroyAllWindows()
vs.release()
