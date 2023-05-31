import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
##################################

def cargarDatos(rutaOrigen,numeroCategorias,limite,ancho,alto):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0,6):
        for i in range(1, categoria):
            for idImagen in range(61,68):
                ruta=rutaOrigen+str(categoria)+"x"+str(i)+"/"+str(categoria)+"x"+str(i)+" ("+str(idImagen)+").jpg"
                print(ruta)
                imagen = cv2.imread(ruta)
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                imagen = cv2.resize(imagen, (ancho, alto))
                imagen = imagen.flatten()
                imagen = imagen / 255
                imagenesCargadas.append(imagen)
                probabilidades = np.zeros(numeroCategorias)
                probabilidades[categoria+i] = 1
                valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados
def cargarPrueba(rutaOrigen,numeroCategorias,ancho,alto):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0,6):
        for i in range(1, categoria):
            for idImagen in range(68,71):
                ruta=rutaOrigen+str(categoria)+"x"+str(i)+"/"+str(categoria)+"x"+str(i)+" ("+str(idImagen)+").jpg"
                print(ruta)
                imagen = cv2.imread(ruta)
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                imagen = cv2.resize(imagen, (ancho, alto))
                imagen = imagen.flatten()
                imagen = imagen / 255
                imagenesCargadas.append(imagen)
                probabilidades = np.zeros(numeroCategorias)
                probabilidades[categoria+i] = 1
                valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados
#################################
ancho=100
alto=100
pixeles=ancho*alto
#Imagen RGB -->3
numeroCanales=1
formaImagen=(ancho,alto,numeroCanales)
numeroCategorias=12

cantidaDatosEntrenamiento=[80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80]
cantidaDatosPruebas=[20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]

#Cargar las imágenes
imagenes, probabilidades=cargarDatos("dataset/",numeroCategorias,cantidaDatosEntrenamiento,ancho,alto)

model = Sequential()
# Capa de entrada
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Flatten())

# Capas ocultas
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))

# Capa de salida
model.add(Dense(numeroCategorias, activation="softmax"))

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Entrenamiento
model.fit(x=imagenes, y=probabilidades, epochs=30, batch_size=60)

# Prueba del modelo
imagenesPrueba, probabilidadesPrueba = cargarPrueba("dataset/", numeroCategorias, ancho, alto)
resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)
print("Accuracy=", resultados[1])

# Guardar modelo
ruta = "models/modeloA.h5"
model.save(ruta)

# Informe de estructura de la red
model.summary()