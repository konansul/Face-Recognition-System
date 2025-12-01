import cv2
import torch
import queue
import threading
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

THRESHOLD = 0.6
IOU_THRESHOLD = 0.3

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

arc_queue = queue.Queue(maxsize = 1)
name_results = { }
next_id = 0

def assign_id_to_boxes(new_box, prev_box):

    assigned = { }
    global next_id
    used_previous = set()

    for box in new_box:
        x1, y1, x2, y2 = box

        best_iou = 0
        best_prev_id = None

        for prev_id, (prev_x1, prev_y1, prev_x2, prev_y2) in prev_box.items():
            if old_id in used_previous
                continue

            intersection_x1 = max(x1, prev_x1)
            intersection_y1 = max(y1, prev_y1)
            intersection_x2 = min(x2, prev_x2)
            intersection_y2 = min(y2, prev_y2)

            if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
                continue

            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
            area_new = (x2 - x1) * (y2 - y1)
            area_prev = (prev_x2 - prev_x1) * (prev_y2 - prev_y1)

            intersection_over_union = intersection_area / (area_new + area_prev - intersection_area)

            if intersection_over_union > best_iou:
                best_iou = intersection_over_union
                best_prev_id = prev_id

            if best_prev_id is not None and best_iou > IOU_THRESHOLD:
                assigned[best_prev_id] = box
                used_previous.add(prev_id)

            else:
                assigned[next_id] = box
                next_id = next_id + 1

    return assigned

def recognize_face(frame):
    yolo_results = yolo.predict(frame, verbose = False)
    insightface_results = face_recognizer_model.get(frame)

    results = [ ]

    for yolo_detections in yolo_results[0].boxes:
        x1, y1, x2, y2 = yolo_detections.xyxy[0].cpu().numpy().astype(int)
        matched = None

        for insightface_detections in insightface_results:
            fx1, fy1, fx2, fy2 = insightface_detections.bbox.astype(int)

            if fx1 < x2 and fx2 > x1 and fy1 < y2 and fy2 > y1:
                matched = insightface_detections
                break

        if matched is None:
                results.append(('UNKNOWN', 0.0, (x1, y1, x2, y2)))
                continue

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

        results.append((best_name, best_similarity, (x1, y1, x2, y2)))

    return results

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    faces = recognize_face(frame)

    for name, score, (x1, y1, x2, y2) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, f'{name} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()