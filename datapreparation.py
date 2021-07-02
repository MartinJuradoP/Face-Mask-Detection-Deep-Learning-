from os import listdir
from matplotlib import image
import numpy as np
from PIL import Image 
import cv2
from time import sleep
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.applications import VGG16
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split


class_name=['face','mask'] # Classes 
i=0
x = [] # trainning Data as Matrix, pictures
y =  [] # class

# load all images from data directory one by one to create classification

for filename in listdir('data'):
    i=i+1

    r=filename[0]
    
    img = cv2.imread('data' + '/' + filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray, dsize=(56,56), interpolation=cv2.INTER_CUBIC) # Resizing the images to 56x56 pixels, to have the same dimensions


    data = np.asarray(res) / 255 # Converting the image into an arrangement form. It is divided into 255 to normalize the values.
    
    x.append(data) # Each image is added in x Vector format to create the dataset.

#Labeling the pictures in the y Vector to have the classification. 
    if r == 'f':
        y.append(int(0))
    else:
        y.append(int(1))


#It is necessary to put in array format
xnp=np.array(x)
ynp=np.array(y)



#The parameter to train the neural network
batch_size=64 
num_classes=2
epochs=5

#Split the  dataset vectors into test and training vectors to avoid overfitting and validate our model.
xt, xtest, yt, ytest = train_test_split(xnp, ynp, test_size=0.30)
xr=xt
xre=xtest
yre=ytest
xt=xt.reshape(xt.shape[0],56,56,1) #transpose the input vector x.
xtest=xtest.reshape(xtest.shape[0],56,56,1) #transpose the input vector x.
yt=tensorflow.keras.utils.to_categorical(yt,num_classes)#create class vector
ytest=tensorflow.keras.utils.to_categorical(ytest,num_classes)#create class vector

#Creation of the Neural Network Layers

Entradas=Input(shape=(56,56,1))
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block_conv1')(Entradas)
x=Dropout(0.25)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block_conv2')(x)
x = MaxPooling2D((2, 2), name='block_pool')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv1')(Entradas)
x=Dropout(0.25)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), name='block1_pool')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x=Dropout(0.25)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), name='block2_pool')(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x=Dropout(0.25)(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = MaxPooling2D((2, 2), name='block3_pool')(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x=Dropout(0.25)(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = MaxPooling2D((2, 2), name='block4_pool')(x)

x=Flatten()(x)
x=Dense(128,activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(num_classes,activation='softmax')(x)


#Model Creation
modelo = Model(inputs=Entradas, outputs=x)
#modelo.summary()

#Optimization method Adam or SGD
Adam = Adam(lr=0.0001,beta_1=0.9,beta_2=0.9) #SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

modelo.compile(loss=tensorflow.keras.losses.categorical_crossentropy,optimizer=Adam,metrics=['categorical_accuracy'])


#Print the efficiency of the model with its loss function and its accuracy
history=modelo.fit(xt,yt,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(xtest,ytest))

puntuacion=modelo.evaluate(xtest,ytest,verbose=1)
modelo.save('modelo')
print(puntuacion)

plt.figure(1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Precision de Modelo')
plt.ylabel('Precision')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')


plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perdidas del Modelo')
plt.ylabel('Perdidas')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')
plt.show()


predictions = modelo.predict(xtest)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(xre[i],cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = yre[i]
    if predicted_label == true_label:
        color ='green'
    else:
        color='red'
    plt.xlabel('{}({})'.format(class_name[predicted_label],class_name[true_label]),color=color)
plt.show()