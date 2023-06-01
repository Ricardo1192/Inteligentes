import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
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

# Capas convolucionales y de pooling
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Añadir más capas convolucionales y de pooling
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanamiento
model.add(Flatten())

# Capas completamente conectadas
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Regularización mediante dropout

model.add(Dense(numeroCategorias, activation='softmax'))

#Traducir de keras a tensorflow
model.compile(optimizer=tf.keras.optimizers.Adam(),loss="categorical_crossentropy", metrics=["accuracy"])
#Entrenamiento
model.fit(x=imagenes,y=probabilidades,epochs=35,batch_size=60)

# Prueba del modelo
imagenesPrueba, probabilidadesPrueba = cargarPrueba("dataset/", numeroCategorias, cantidaDatosPruebas, ancho, alto)
resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)

# Accuracy
accuracy = resultados[1]

# Predicciones
predicciones = model.predict(imagenesPrueba)
etiquetas_predichas = np.argmax(predicciones, axis=1)

# Etiquetas reales
etiquetas_reales = np.argmax(probabilidadesPrueba, axis=1)

# Precision, Recall y F1 Score


precision = precision_score(etiquetas_reales, etiquetas_predichas, average='weighted')
recall = recall_score(etiquetas_reales, etiquetas_predichas, average='weighted')
f1 = f1_score(etiquetas_reales, etiquetas_predichas, average='weighted')

# Loss (Pérdida)
loss = resultados[0]

# Épocas de entrenamiento
epocas = 80  # O el número de épocas que especificaste durante el entrenamiento

# Tiempo de respuesta
# Calcula el tiempo de respuesta aquí según tus necesidades

# Imprimir los resultados
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Loss:", loss)
print("Épocas de entrenamiento:", epocas)

predicciones = model.predict(imagenesPrueba)
predicciones_etiquetas = np.argmax(predicciones, axis=1)
etiquetas_verdaderas = np.argmax(probabilidadesPrueba, axis=1)
matriz_confusion = confusion_matrix(etiquetas_verdaderas, predicciones_etiquetas)
print("Matriz de confusión:")
print(matriz_confusion)
print('KNN Reports\n',classification_report(etiquetas_verdaderas, predicciones_etiquetas))

# Guardar modelo
ruta="models/modelo5.h5"
model.save(ruta)
# Informe de estructura de la red
model.summary()
