import numpy as np
import cv2
import pickle

# aprendizagem openCV...
# testando git e github

# biblioteca haar cascade (está na pasta do opencv, modifique aí)
# abre o python e digita import cv2
# depois print(cv2.__file__)
# copia a pasta "data" para a pasta do arquivo que tu extraiu esse programa e muda o diretório abaixo
classificadorRosto = cv2.CascadeClassifier('/Users/rafaellaweiss/coding/face_detection/data/haarcascade_frontalface_alt2.xml')
reconhecimento = cv2.face.LBPHFaceRecognizer_create()

reconhecimento.read("trainner.yml")
etiquetas = {"nome_pessoa": 1}
with open("labels.pkl", "rb") as f:
    og_etiquetas = pickle.load(f)
    etiquetas = {v:k for k,v in og_etiquetas.items()}

# função para abrir a câmera
capturadeVideo = cv2.VideoCapture(0)

# loop de leitura do vídeo
while (True):
    ret, imagemCor = capturadeVideo.read() # retângulo do frame
    imagemCinza = cv2.cvtColor(imagemCor, cv2.COLOR_BGR2GRAY) # usando o filtro de conversão para imagem cinza
    rostos = classificadorRosto.detectMultiScale(imagemCinza, scaleFactor=1.5, minNeighbors=5) # define os parâmetros da detecção


    for (x, y, w, h) in rostos:
            #print(x,y,w,h) # printa no terminal as coordenadas de detecção
            roi_cinza = imagemCinza[y:y+h, x:x+w] # região de interesse da imagem cinza
            roi_cor = imagemCor[y:y+h, x:x+w] # região de interesse da imagem colorida

            id_, conf = reconhecimento.predict(roi_cinza)
            if conf >= 45: 
                print(id_)
                print(etiquetas[id_])
                fonte = cv2.FONT_HERSHEY_TRIPLEX
                nome = etiquetas[id_]
                cor = (0, 255, 0)
                borda = 1
                cv2.putText(imagemCor, nome, (x,y), fonte, 1, cor, borda, cv2.LINE_AA)

            img_item = "7.png" # cria um arquivo com a última imagem de detecção
            cv2.imwrite(img_item, roi_cor)

            # parâmetros do retângulo da detecção

            cor = (255, 0, 0) # azul
            borda = 2 # largura da borda
            largura = x + w # tamanho vai seguir a área de detecção
            altura = y + h

            cv2.rectangle(imagemCor, (x, y), (largura, altura), cor, borda) # parâmetros da função de retângulo

    cv2.imshow('imagem colorida', imagemCor) # mostrar o frame colorido
    cv2.imshow('imagem cinza', imagemCinza) # mostrar o frame cinza
    if cv2.waitKey(20) & 0xFF == ord('q'): # essa condicional é para apertar 'q' e sair do programa
        break

capturadeVideo.release() #
cv2.destroyAllWindows()