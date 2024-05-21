import cv2
from twilio.rest import Client

#initializes twilio client
client = Client("YOUR_TWILIO_ACCOUNT_SID", "YOUR_TWILIO_AUTH_TOKEN")
twilio_phone_number = '+YOUR_TWILIO_PHONE_NUMBER'
recipient_phone_number = '+RECIPIENT_PHONE_NUMBER'

def send_sms_notification(message):
    
    client.messages.create(
        to=recipient_phone_number,
        from_=twilio_phone_number,
        body=message
    )

# Use the trained model to identify faces

# Call classifier
face_cascade = cv2.CascadeClassifier('haar_face.xml')

peopleList =  ['Ariana Grande', 'Beyonce', 'Henry Cavill', 'Mr Bean']

# Read the .yml file
# Instantiate face recognizer
# Using LBPH face recognition algorithm
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trained_faces.yml')

# Set variable to image path
img = cv2.imread(r'C:image_path_here')

# Convert to grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display image
cv2.imshow('Person', gray)

# Detect faces in image
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#iterate over faces detected
for (x,y,w,h) in faces:
    #grab faces region of interest ROI
    face_roi = gray[y:y+h, x:x+w]

    #predicts the confidence value
    name, confidence = face_recognizer.predict(face_roi)

    #checks if the confidence is high enough to consider it a known face
    if confidence < 100:
        #sends SMS notification with the name of the known face detected
        send_sms_notification(f"Known face detected: {peopleList[name]}")

         #puts name on image
        cv2.putText(img, str(peopleList[name]), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), thickness=2)

        #draws rectangle over face
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=3)
    else:
        #sends SMS notification for unknown face detected
        send_sms_notification("Unknown face detected!")
        unknown_face_detected = True

#display image
cv2.imshow('Detected Face', img)
cv2.waitKey(0)

"""using toast notifications with plyer lib

from plyer import notification

send toast notifications
def send_toast_notification(message):
    notification.notify(
        title='Unknown Face Detected',
        message='Warning!',
        app_name='Face Recognition Project',
        app_icon=None,
        timeout=0,
    )
#sends toast notification if an unknown face was detected
if unknown_face_detected:
    send_toast_notification("An unknown face was detected!")"""



