import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
##################################

def cargarDatos(rutaOrigen,numeroCategorias,limite,ancho,alto):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0,7):
        for i in range(1, categoria+1):
            for idImagen in range(1,71):
                ruta=rutaOrigen+str(categoria)+"x"+str(i)+"/"+str(categoria)+"x"+str(i)+" ("+str(idImagen)+").jpg"
                print(ruta)
                imagen = cv2.imread(ruta)
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                imagen = cv2.resize(imagen, (ancho, alto))
                imagen = imagen.flatten()
                imagen = imagen / 255
                imagenesCargadas.append(imagen)
                probabilidades = np.zeros(numeroCategorias)
                probabilidades[categoria+i-1] = 1
                valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados
def cargarPrueba(rutaOrigen,numeroCategorias,limite,ancho,alto):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0,7):
        for i in range(1, categoria+1):
            for idImagen in range(71,81):
                ruta=rutaOrigen+str(categoria)+"x"+str(i)+"/"+str(categoria)+"x"+str(i)+" ("+str(idImagen)+").jpg"
                print(ruta)
                imagen = cv2.imread(ruta)
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                imagen = cv2.resize(imagen, (ancho, alto))
                imagen = imagen.flatten()
                imagen = imagen / 255
                imagenesCargadas.append(imagen)
                probabilidades = np.zeros(numeroCategorias)
                probabilidades[categoria+i-1] = 1
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
model.add(Reshape(formaImagen))

# Capas ocultas
model.add(Conv2D(kernel_size=3, strides=1, filters=32, padding="same", activation="relu", name="capa_1"))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=3, strides=1, filters=64, padding="same", activation="relu", name="capa_2"))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=3, strides=1, filters=128, padding="same", activation="relu", name="capa_3"))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Flatten())
model.add(Dense(256, activation="relu"))

# Capa de salida
model.add(Dense(numeroCategorias, activation="softmax"))

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Entrenamiento
model.fit(x=imagenes, y=probabilidades, epochs=30, batch_size=60)

#Prueba del modelo
imagenesPrueba,probabilidadesPrueba=cargarPrueba("dataset/",numeroCategorias,cantidaDatosPruebas,ancho,alto)
resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)
print("Accuracy =", resultados[1])

predicciones = model.predict(imagenesPrueba)
predicciones_etiquetas = np.argmax(predicciones, axis=1)
etiquetas_verdaderas = np.argmax(probabilidadesPrueba, axis=1)

# Calcula la matriz de confusión
matriz_confusion = confusion_matrix(etiquetas_verdaderas, predicciones_etiquetas)
print("Matriz de confusión:")
print(matriz_confusion)
print('KNN Reports\n',classification_report(etiquetas_verdaderas, predicciones_etiquetas))

# Guardar el modelo
ruta = "models/modeloD.h5"
model.save(ruta)

# Informe de estructura de la red
model.summary()