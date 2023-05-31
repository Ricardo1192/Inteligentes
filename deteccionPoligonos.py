import cv2
import numpy as np
from Prediccion import Prediccion

nameWindow="Calculadora"
def nothing(x):
    pass
def constructorVentana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min",nameWindow,0,255,nothing)
    cv2.createTrackbar("max", nameWindow, 100, 255, nothing)
    cv2.createTrackbar("kernel", nameWindow, 1, 100, nothing)
    cv2.createTrackbar("areaMin", nameWindow, 500, 10000, nothing)

def calcularAreas(figuras):
    areas=[]
    for figuraActual in figuras:
        areas.append(cv2.contourArea(figuraActual))
    return areas

def detectarFigura(imagenOriginal):
    imagenGris=cv2.cvtColor(imagenOriginal,cv2.COLOR_BGR2GRAY)
    min = cv2.getTrackbarPos("min", nameWindow)
    max = cv2.getTrackbarPos("max", nameWindow)
    bordes = cv2.Canny(imagenGris, min, max)
    
    tamañoKernel = cv2.getTrackbarPos("kernel", nameWindow)
    kernel = np.ones((tamañoKernel, tamañoKernel), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    
    figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = calcularAreas(figuras)
    i = 0
    areaMin = cv2.getTrackbarPos("areaMin", nameWindow)
    
    return imagenOriginal

video=cv2.VideoCapture(0)
constructorVentana()
while True:
    _,frame=video.read()
    detectarFigura(frame)
    cv2.imshow("Imagen",frame)


    k=cv2.waitKey(5) & 0xFF
    if k==112:
        clases=["1","2","3","4","5","6","7","8","9","10","11","12"]
        
        ancho=100
        alto=100

        miModeloCNN=Prediccion("models/modeloA.h5",ancho,alto)
        cv2.imwrite("img.jpg", frame)

        claseResultado=miModeloCNN.predecir(frame)
        print("La imagen cargada es ",clases[claseResultado])
    elif k==27:
        print("Saliendo")
        break
video.release()
cv2.destroyAllWindows()

