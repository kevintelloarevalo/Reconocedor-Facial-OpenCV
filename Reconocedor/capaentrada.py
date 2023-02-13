import cv2 as cv
import os
import imutils #Manejo de directorios
modelo='Kevin' #Nombre de la carpeta  //nombre de la carpeta de las fotos
ruta1='C:/Users/KEVIN/Music/Curso/ReconocedorFacial/Reconocedor/Data'  #SE USA ESTO '/' CUANDO SE CONTANE O SE USA VARIAS VECES LA RUTA.
rutacompleta = ruta1 + '/'+ modelo   # DECIMOS DONDE ES LA RUTA Y EL NOMBRE DE LA CARPETA A CREAR
if not os.path.exists(rutacompleta):  #CREAR SISKE NO EXISTE
    os.makedirs(rutacompleta)



camara=cv.VideoCapture(0)  #Captura camara  // SI DESEAN UN VIDEO PONER ('ruta del video')
ruidos=cv.CascadeClassifier('C:/Users/KEVIN/Music/Curso/ReconocedorFacial/entrenamientos opencv ruidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml') #PARA QUE EL MODELO IGNORE LOS RUIDOS Y SOLO CAPTURE CARAS
id=0  #Contador
while True:
    respuesta,captura=camara.read() #Tomar capturas
    if respuesta==False:break  #Cerrar Ventana si sucede algun problema
    captura=imutils.resize(captura,width=640)  #La imagen captura a menos dimensión para no sobrecargar memoria

    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY) #Convertir a escala de grises
    idcaptura=captura.copy() #Aqui Se copia todo el proceso anterior.

    cara=ruidos.detectMultiScale(grises,1.3,5)  #Imagen captura, solo Caras

    for(x,y,e1,e2) in cara: 
        cv.rectangle(captura, (x,y), (x+e1,y+e2), (0,255,0),2)  #Creamos el rectangulo, 1.Imagen, x&y, 2.(x+e1, x mas el vertice superior) (y+e2, y mas el vertice inferior)3. Color del rectangulo, 4.Tamaño del borde
        rostrocapturado=idcaptura[y:y+e2,x:x+e1] 
        rostrocapturado=cv.resize(rostrocapturado, (182,182),interpolation=cv.INTER_CUBIC) #Aqui formamos un cuadrado 
        cv.imwrite(rutacompleta+'/imagen_{}.jpg'.format(id), rostrocapturado) #Capturamos imagen. Y el archivo tendra como nombre "imagen_id".jpg
        id=id+1 
    
    cv.imshow("Resultado rostro", captura)

    if id==200:  #SOLO CAPTURA HASTA 200 imagenes  // PUEDE VARIAR SEGUN CUANTAS IMAGENES DESEEN  
        break  #Se cierra
camara.release() #Apagamos
cv.destroyAllWindows() #Cerramos ventanas