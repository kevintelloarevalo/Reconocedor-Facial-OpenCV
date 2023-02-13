import cv2 as cv
import os
import imutils
dataRuta='C:/Users/KEVIN/Music/Curso/ReconocedorFacial/Reconocedor/Data' #Ruta de la Data
listaData=os.listdir(dataRuta)
entrenamientoEigenFaceRecognizer=cv.face.EigenFaceRecognizer_create()
entrenamientoEigenFaceRecognizer.read('C:/Users/KEVIN/Music/Curso/ReconocedorFacial/Reconocedor/EntrenamientoRostros.xml')  #Leer el modelo entrenado -RUTA
ruidos=cv.CascadeClassifier('C:/Users/KEVIN/Music/Curso/ReconocedorFacial/entrenamientos opencv ruidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml') #El modelo ruido para identificar objetos y eliminar #RUTA
camara=cv.VideoCapture(0) #leer la camara puede variar
while True:
    respuesta,captura=camara.read()
    if respuesta==False:break
    captura=imutils.resize(captura,width=640)
    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY) #ESCALA DE GRISES
    idcaptura=grises.copy()
    cara=ruidos.detectMultiScale(grises,1.3,5)
    for(x,y,e1,e2) in cara:
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        rostrocapturado=cv.resize(rostrocapturado, (182,182),interpolation=cv.INTER_CUBIC)
        resultado=entrenamientoEigenFaceRecognizer.predict(rostrocapturado)
        cv.putText(captura, '{}'.format(resultado), (x,y-5), 1,1.3,(0,255,0),1,cv.LINE_AA) #Letras del resultado , COLORES
        if resultado[1]<9000:
            cv.putText(captura, '{}'.format(listaData[resultado[0]]), (x,y-20), 2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)
        else:
            cv.putText(captura,"No encontrado", (x,y-20), 2,0.7,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)

       
    cv.imshow("Resultados", captura) 
    if cv.waitKey(1)==ord('s'):   #con la s se cierra la ventana
        break
camara.release()
cv.destroyAllWindows()


