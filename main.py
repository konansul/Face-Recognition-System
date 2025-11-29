import cv2
import torch
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from xyzservices import providers

yolo = YOLO('models/yolov12s-face.pt')

if torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

yolo.to(device)

face_recognizer_model = FaceAnalysis(name = 'buffalo_l', providers = ['CPUExecutionProvider'])
face_recognizer_model.prepare(ctx_id = 0)

database = { }

def add_person(person_name, image_path):
    face = face_recognizer_model.get(cv2.imread(image_path))
    database[person_name] = face[0].normed_embedding
    print('Added:', person_name)

add_person('Kanan', 'images/kanan.jpeg')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    yolo_detections = yolo.predict(frame, verbose = False)

    frame = yolo_detections[0].plot()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()