# Face Mask identification (Deep Learning) 😷

This project has the target to cover three stages of a Machine Learning implementation. 
1) Data Collection. 
2) Data Training 
3) Data Predictive

Where it pretends to identify when the people it is using masks or not. (Two classes, Mask and Not Mask), this will be applied using a neural network with some usabilities from Keras and tensor flow like Conv2D, Dropout, MaxPooling2D, Flatten to create our neural network.   

## Comenzando 🚀

Para levantar los ambientes, nos tenemos que posicionar en la carpeta raíz del proyecto, no importa si es Windows o Mac con los accedemos mediante la consola y corremos las siguientes líneas.

$ vagrant up nodo1 nodo2 nodo3

Vagrant de manera natural consume el archivo de configuración para crear los ambientes virtuales o levantarlos si estos ya fueron creados.

Existe otros dos comandos que nos ayudan para poder mantener esos ambientes virtuales.

$ vagrant destroy nodo1 nodo2 nodo3

Este comando nos ayuda a destruir o eliminar las maquinas virtuales creadas.

$ vagrant suspend nodo1 nodo2 nodo3

Con la sintaxis suspend, las maquinas o maquina virtual se queda en pausa justamente en el momento que aplicamos el comando.

Recuerden que los nodo1 nodo2 y nodo3 son la referencia a las maquinas virtuales que creamos con vagrantfile. 

$vagrant ssh nodo1

Este comando nos ayuda a acceder a las maquinas virtuales creadas.




### Pre-requisitos 📋

-Instalar Virtual Box, que es el encargado consumir el recurso de nuestra maquina y asignarlo a los ambientes virtuales que queremos crear.

-Instalar Vagrant, será la herramienta para crear y configurar los ambientes virtuales que se comunicaran con virtual Box con las características que necesitamos para este proyecto.

