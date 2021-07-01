import numpy as np
import cv2
import tensorflow as tf

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
loaded_model = tf.keras.models.load_model('modelo')

cap = cv2.VideoCapture(0)

sampleNum=0
while(True):

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        imge=gray[y:y + h, x:x + w]
        res = cv2.resize(imge, dsize=(56,56), interpolation=cv2.INTER_CUBIC)
        data = np.asarray(res) / 255
        xt = data.reshape(1,56, 56, 1)
        predictions = loaded_model.predict(xt)
        predicted_label = np.argmax(predictions)
        print(predicted_label,type(predicted_label))
        if predicted_label < 1:
            cv2.putText(img, 'No Mask', (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.putText(img, 'Mask', (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)



    cv2.imshow('frame',img)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
        # break if the sample number is morethan 20

cap.release()
cv2.destroyAllWindows()