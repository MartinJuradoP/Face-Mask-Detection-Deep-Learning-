import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Patron de identificación de rostro
cap = cv2.VideoCapture(0)  # Acceso a la camara de la computadora
tag = input('Enter tag m is mask and f is face')
name= input('Put the first letter of your name')
sampleNum = 0;  # Variable que representa el numero de capturas que hara para el aprendizaje

while (True):
    ret, img = cap.read()  # Captura los frames y es lo que se analiza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # combierte las imagenes en color gris para facilitar el analisis
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # reconoce los rotros dependiendo de los parametros de cascade

    for (x, y, w, h) in faces:
        cv2.putText(img, 'Identifying', (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        sampleNum = sampleNum + 1
        cv2.imwrite("example/" + tag +name + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
        #print("dataSet/" + tag + name + str(sampleNum) + ".jpg")
        #print(x + w, y + h)

    cv2.imshow('frame', img)
    # wait for 100 miliseconds
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum > 420:
        break
cap.release()
cv2.destroyAllWindows()