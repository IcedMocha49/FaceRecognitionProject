import cv2
import numpy as np

#call classifier 
face_cascade = cv2.CascadeClassifier('haar_face.xml')

#read image
img = cv2.imread('photos/bts.jpg')

# convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces in image
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#iterate over faces detected 
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)

#display output
cv2.imshow('Detected Faces', img)
cv2. waitKey(0)


