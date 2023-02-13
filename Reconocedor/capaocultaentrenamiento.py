import cv2 as cv  #Usaremos cv significa que ya no llamaremos por cv2 SINO "cv"
import os  
import numpy as np #Usaremos np
from time import time  #Libreria para obtener los tiempos de los procesos
dataRuta='C:/Users/KEVIN/Music/Curso/ReconocedorFacial/Reconocedor/Data'  #RUTA DONDE ESTA LAS IMAGENES CARPETAS
listaData=os.listdir(dataRuta)  #Aqui listamos las carpetas QUEDARIA ASI --> [kevin, -, -]
#print('data',listaData)
ids=[]  #ARREGLO
rostrosData=[]#ARREGLO
id=0 #Contador
tiempoInicial=time()  #Tiempo de inicio
for fila in listaData:
    rutacompleta=dataRuta+'/'+ fila #Ingresa a la carpeta 
    print('Iniciando lectura...')
    for archivo in os.listdir(rutacompleta): 
       
        print('Imagenes: ',fila +'/'+archivo)  #Muestra imagen tal 
    
        ids.append(id)
        rostrosData.append(cv.imread(rutacompleta+'/'+archivo,0))    #Lee todo
      

    id=id+1  #Aumenta el contador hasta el final ULTIMA IMAGEN
    tiempofinalLectura=time()  #Tiempo del proceso
    tiempoTotalLectura=tiempofinalLectura-tiempoInicial  #Tiempo que se Utilizo
    print('Tiempo total lectura: ',tiempoTotalLectura)  # Mostramos el tiempo

entrenamientoEigenFaceRecognizer=cv.face.EigenFaceRecognizer_create()  #Esta función EigenFaceRecognizer_create() --> Hace diferentes procesamientos, como gauss, eliminar ruido, grises, binarización.
print('Iniciando el entrenamiento...espere')  #Mostramos
entrenamientoEigenFaceRecognizer.train(rostrosData,np.array(ids))  #Entrenamiento 
TiempofinalEntrenamiento=time() #Tiempo del proceso
tiempoTotalEntrenamiento=TiempofinalEntrenamiento-tiempoTotalLectura
print('Tiempo entrenamiento total: ',tiempoTotalEntrenamiento) #Mostrar
entrenamientoEigenFaceRecognizer.write('EntrenamientoRostros.xml') #NOMBRE DEL ARCHIVO CREADO
print('Entrenamiento concluido')#Mostrar

#Despues de acabar el entrenamiento de la DATA LES QUEDARA UN ARCHIVO.XML 