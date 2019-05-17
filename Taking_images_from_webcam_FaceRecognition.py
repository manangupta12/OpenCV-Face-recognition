# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 19:46:33 2019

@author: lappy
"""

import cv2
face_classifier=cv2.CascadeClassifier("G:\\image dataset\\haarcascade_frontalface_alt.xml")


def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return None
    for (x,y,w,h)in faces:
        cropped_face=img[y:y+h,x:x+w]
    return cropped_face


count=0
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if face_extractor(frame) is not None:
        count=count+1
        face_resize=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face_resize,cv2.COLOR_BGR2GRAY)
        
        output_path="G:\\image dataset\\FaceSamples\\"+str(count)+".jpg"
        cv2.imwrite(output_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("FACE",face)
        print("Capturing Face")
    else:
        print("Face Not Found")
        pass
    if cv2.waitKey(1)==27 or count==100:
        break
cap.release()
cv2.destroyAllWindows()
print("Dataset Taken")
