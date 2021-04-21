import numpy as np
import cv2 as cv
import time

# carregar o nome das classes
# with open('object_detection_classes_coco.txt', 'r') as f:
#    class_names = f.read().split('\n')

# cores de cada classe
# colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# carregando o modelo DNN
model = cv.dnn.readNetFromDarknet('mask-yolov3-tiny-prn.cfg', 'mask-yolov3-tiny-prn.weights')
model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# captura do vídeo
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      image = frame
      image_height, image_width, _ = image.shape
      # criar o blob da imagem
      blob = cv.dnn.blobFromImage(image=image, scalefactor=0.01, size=(300,300), mean=(104, 117, 123), swapBR=True)
      # início do cálculo do tempo do FPS
      start = time.time()
      model.setInput(blob)
      output = model.forward()
      # fim do tempo do cálculo após a detecção
      end = time.time()
      fps = 1 / (end-start)
      
      for detection in output[0, 0, :, :]:
        confidence = detection[2]

        if confidence > .4:
          class_id = detection[1]
          # class_name = class_names[int(class_id)-1]
          # color = colors[int(class_id)]
      
          # coordenadas para a box de detecção
          box_x = detection[3] * image_width
          box_y = detection[4] * image_height
          # altura e largura da box de detecção
          box_width = detection[5] * image_width
          box_height = detection[6] * image_height
          # desenhando o retângulo
          cv.rectangle(frame, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color='#008F39', thickness=2)
          # colocar o FPS no topo do frame
          # cv.putText(frame, class_name, (int(box_x), int(box_y - 5)), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
          cv.putText(frame, f'{fps:.2f} FPS', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('Imagem', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()