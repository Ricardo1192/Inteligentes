
import cv2
from Prediccion import Prediccion

clases=["1","2","3","4","5","6","7","8","9","10","11","12"]

ancho=100
alto=100

miModeloCNN=Prediccion("models/modeloA.h5",ancho,alto)
imagen=cv2.imread("dataset/3x2/3x2 (72).jpg")

claseResultado=miModeloCNN.predecir(imagen)
print("La imagen cargada es ",clases[claseResultado])

while True:
    cv2.imshow("imagen",imagen)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()