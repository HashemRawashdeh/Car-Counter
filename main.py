from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

model = YOLO('yolov8n.pt')
class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench',
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
               'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
               'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
               'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
               'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
               'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

cap = cv2.VideoCapture('traffic.mov')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, size)
mask = cv2.imread('mask.png')
# Tracker:
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
id_list = set()
while True:
    success, img = cap.read()
    img_region = cv2.bitwise_and(img, mask)
    results = model(img_region, stream=True)
    detections = np.empty((0,5))
    for i in results:
        boxes = i.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
            # confidence
            conf = math.ceil((box.conf[0]*100)) / 100
            # class
            cls = int(box.cls[0])
            current_class = class_names[cls]
            if (current_class =='car') and conf>=0.8:
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                cvzone.putTextRect(img, f' {class_names[cls]} {conf}', (max(0, x1), max(20, y1)),
                                   scale=1, thickness=1, offset=3)
                current_array = np.array([x1, y1, int(x2), int(y2), conf])
                detections = np.vstack((detections, current_array))
    results_tracker = tracker.update(detections)
    for result in results_tracker:
        x1, y1, x2, y2, id = result
        id_list.add(id)
        x1, y1, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
    cvzone.putTextRect(img, ('count: ' + str(len(id_list))), (50,50))
    out.write(img)
    cv2.imshow('image', img)
    cv2.waitKey(1)