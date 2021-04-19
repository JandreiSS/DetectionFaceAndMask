import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

model = cv.dnn.readNetFromDarknet('####caminho para os pesos','###caminho para a configuração')

# carregar o nome das classes
with open('object_detection_classes_coco.txt', 'r') as f:
   class_names = f.read().split('\n')

# cores de cada classe
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

while(True):
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blob = cv.dnn.blobFromImage(image=gray, scalefactor=0.01, size=(244,244), mean=(104, 117, 123))
    
    model.setInput(blob)

    output = model.forward()

    for detection in output[0, 0, :, :]:
      confidence = detection[2]
      if confidence > .4:
        class_id = detection[1]
        color = colors[int(class_id)]

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()