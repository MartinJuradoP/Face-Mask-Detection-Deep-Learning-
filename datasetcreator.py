import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Face identification pattern
cap = cv2.VideoCapture(0)  # Camera Access
tag = input('Enter tag m is mask and f is face:  ')
name= input('Put the first letter of your name:  ')
sampleNum = 0;  # Number of samples 

while (True):
    ret, img = cap.read()  # Frame capture
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts images to gray for easy analysis (Pixel matrix with only one channel)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Recognize faces with cascade parameters

    for (x, y, w, h) in faces:
        cv2.putText(img, 'Identifying', (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        sampleNum = sampleNum + 1
        cv2.imwrite("data/" + tag +name + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
        print("dataSet/" + tag + name + str(sampleNum) + ".jpg")
        

    cv2.imshow('frame', img)
    # wait for 100 miliseconds
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 420
    elif sampleNum > 420:
        break
cap.release()
cv2.destroyAllWindows()