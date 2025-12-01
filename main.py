import cv2
import torch
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

THRESHOLD = 0.6

yolo = YOLO('models/yolov12s-face.pt')

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
yolo.to(device)

face_recognizer_model = FaceAnalysis(name = 'buffalo_l', providers = ['CPUExecutionProvider'])
face_recognizer_model.prepare(ctx_id = 0)

database = { }

def add_person(person_name, image_path):
    face = face_recognizer_model.get(cv2.imread(image_path))
    database[person_name] = face[0].normed_embedding
    print('Added:', person_name)

add_person('Kanan', 'images/kanan_sultanov.jpeg')
add_person('Garnacho', 'images/alejandro_garnacho.jpeg')
add_person('E.Fernandez', 'images/enzo_fernandez.jpeg')
add_person('Caicedo', 'images/moises_caicedo.jpg')
add_person('Cucurella', 'images/marc_cucurella.jpg')

def recognize_face(frame):
    yolo_results = yolo.predict(frame, verbose = False)
    insightface_results = face_recognizer_model.get(frame)

    final_name = 'UNKNOWN'
    final_score = 0.0
    final_box = (0, 0, 0, 0)

    for yolo_detections in yolo_results[0].boxes:
        x1, y1, x2, y2 = yolo_detections.xyxy[0].cpu().numpy().astype(int)

        matched = None

        for insightface_detections in insightface_results:
            fx1, fy1, fx2, fy2 = insightface_detections.bbox.astype(int)

            if fx1 < x2 and fx2 > x1 and fy1 < y2 and fy2 > y1:
                matched = insightface_detections
                break

        if matched is None:
                name = 'UNKNOWN'
                score = 0.0

        else:
            embedding = matched.normed_embedding

            best_name = 'UNKNOWN'
            best_similarity = -1

            for person_name, saved_embedding in database.items():
                similarity = np.dot(embedding, saved_embedding)

                if similarity >= THRESHOLD:
                    best_name = person_name
                    best_similarity = similarity

                if best_similarity < THRESHOLD:
                    best_name = 'UNKNOWN'

                final_name = best_name
                final_score = best_similarity
                final_box = (x1, y1, x2, y2)

    return final_name, final_score, *final_box

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    name, score, x1, y1, x2, y2 = recognize_face(frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(frame, f'{name} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()