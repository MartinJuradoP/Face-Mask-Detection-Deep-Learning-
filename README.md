# Face Mask identification (Deep Learning) 😷

This project has the target to cover three stages of a Machine Learning implementation. 
1) Data Collection. 
2) Data Training 
3) Data Predictive

Where it pretends to identify when the people it is using masks or not. (Two classes, Mask and Not Mask), this will be applied using a neural network with some usabilities from Keras and tensor flow like Conv2D, Dropout, MaxPooling2D, Flatten to create our neural network.   

## Start 🚀

The project consists in 3 main files. 

1) Data Collection / datasetcreator.py
2) Data Training / datapreparation.py
3) Data Predictive / datapredictive.py

These do not have a main file, because the purpose of this project is to understand the three stages (Collect, training, and predicve).

## datasetcreator.py

This code uses Open Cv to enable the camera and identify faces to extract them and create the dataset, also it is possible to label the images during this process, with f if it is face or m it is a face with Mask. 

For this step it is important to remember that we have two classes, with mask and without mask. Once you run the code, it will save 420 images. It is not possible to save the two classes at the same time with mask and without because the label is defined at the beginning, it is necessary to do separately.

## datapreparation.py

This program extracts all the images taken previously, in order to create the input vector x of our data set and the vector Y that contains its result or classification, this to train our neural network, this classification is done by the name of the image to be able to tag the images automatically with the help of the previous code.

Separation of the dataset is performed in test and training vectors.

The layers of neural networks are created. And it is optimized by models like Adam which is a replacement optimization algorithm for stochastic gradient descent for training deep learning models.

The performance of the model is verified through curves during and after its creation.

## datapredictive.py

Finally, with the trained model, we proceed to predict in real time when someone has a mask or not. Through a visual identifier with the help of the camera.



### Pre requirements 📋

-Python3.

Libraries:

    1) Data Collection: Numpy, OpenCv(cv2)
    2) Data Training: Numpy, OpenCv (cv2), tensorflow (GPU or CPU), matplotlib, PIL, sklearn, time.
    3) Data Predictive: Numpy, OpenCv (cv2), tensorflow.

