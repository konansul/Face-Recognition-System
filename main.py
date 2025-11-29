import cv2
import torch
import numpy as np
from ultralytics import YOLO

yolo = YOLO('models/yolov12s-face.pt')

if torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

yolo.to(device)

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