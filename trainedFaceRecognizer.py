import cv2
import numpy as np

#use the trained model to identify faces 

#call classifier 
face_cascade = cv2.CascadeClassifier('haar_face.xml')

peopleList =  ['Ariana Grande', 'Beyonce', 'Henry Cavill', 'Mr Bean']

#read the .yml file
#instantiate face reconizer
#using LBPH face recognition algorithm
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trained_faces.yml')

#set variable to image path 
img = cv2.imread(r'C:\Users\gabri\OneDrive\Desktop\FaceRecognitionProject\TrainedFaces\Mr Bean\1b8ea024fc084cb7ee54cf3f6f84f0aa.jpg')

#convert to grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#display image
cv2.imshow('Person', gray)

#detect faces in image
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#iterate over faces detected 
for (x,y,w,h) in faces:
    #grab faces region of interested ROI
    face_roi = gray[y:y+h, x:x+w]

    #predict the confidence value 
    name, confidence = face_recognizer.predict(face_roi)

    """  # to display confidence will need a larger dataset for each face
     print(f'Name: {name-+*
     } confidence = {confidence}') """

    #puts name on image
    cv2.putText(img, str(peopleList[name]), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), thickness=2)

    #draws rectangle over face
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=3)

#display image
cv2.imshow('Detected Face', img)
cv2.waitKey(0)