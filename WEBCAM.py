import cv2
import numpy as np
import dlib
from math import hypot
import playsound
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpt(p1,p2):
	return int((p1.x + p2.x)/2),int((p1.y + p1.y)/2)

def blinks(pt,landmarks):
	left = (landmarks.part(pt[0]).x,landmarks.part(pt[0]).y)
	right  = (landmarks.part(pt[3]).x,landmarks.part(pt[3]).y)
	center_top    = midpt(landmarks.part(pt[1]),landmarks.part(pt[2]))
	center_bottom = midpt(landmarks.part(pt[5]),landmarks.part(pt[4]))
	hor_line = cv2.line(frame,left,right,(0,255,0),2)
	ver_line  = cv2.line(frame,center_top,center_bottom,(0,255,0),2)
	horlen = hypot((left[0]-right[0]),(left[1]-right[1]))
	verlen = hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))
	r = horlen/verlen
	return r

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
    	landmarks = predictor(gray, face)
    	l = blinks([36,37,38,39,40,41],landmarks)
    	r = blinks([42,43,44,45,46,47],landmarks)
    	ratio = (l+r)/2
    	print(ratio)
    	if ratio > 5.1:
    		cv2.putText(frame,"blinking",(50,150),font,7,(255,0,0))
    		#playsound.playsound("Police siren ringtone.mp3")




    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cap.destroyAllWindows()