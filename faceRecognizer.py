import cv2
import os
import numpy as np

#call classifier 
face_cascade = cv2.CascadeClassifier('haar_face.xml')

#create list of people in folders
peopleList =  ['Ariana Grande', 'Beyonce', 'Henry Cavill', 'Mr Bean']

#creates variable that calls folder with the folders of positive and negative images 
DIRECTORY = r'C:\Users\gabri\OneDrive\Desktop\FaceRecognitionProject\TrainedFaces'

#refers to training lists 
features = []
names = []

#loop over every folder in folders
def create_training():
    for person in peopleList:
        #path to folders
        path = os.path.join(DIRECTORY, person)
        #sets name to person
        name = peopleList.index(person)

        #loops over every image in folders
        for img in os.listdir(path):
            imgPath = os.path.join(path, img)
            #read image 
            img_array = cv2.imread(imgPath)

            #convert to grayscale image
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            #detect faces in image
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            #iterate over faces detected 
            for (x,y,w,h) in faces:

                #grab faces region of interested ROI
                face_roi = gray[y:y+h, x:x+w]

                #append the roi to the features and names lists
                features.append(face_roi)

                #mapping index of list, numerical value
                names.append(name)

               # cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)
create_training()

#convert the features and names lists to numpy arrays
features = np.array(features, dtype= 'object')
names = np.array(names)

#use appended lists to train the recognizer 

#instantiate face reconizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#use the features and names list to train the recognizer
#parse in lists 
face_recognizer.train(features, names)
#path to .yml
face_recognizer.save('trained_faces.yml')
#save features and names list
np.save('features.npy', features)
np.save('names.npy', names)