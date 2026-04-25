from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_plate(image_path):

    results = model(image_path)
    plates = []

    for r in results:
        boxes = r.boxes

        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()

            # COCO class 2 = car
            if cls == 2 and conf > 0.2:   # 🔥 reduced threshold
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                # take lower part (plate area)
                h = y2 - y1
                y1_new = int(y1 + h * 0.6)

                plates.append({
                    "bbox": [int(x1), y1_new, int(x2), int(y2)],
                    "score": conf
                })

    # 🔥 IMPORTANT: fallback if no detection
    if len(plates) == 0:
        img = cv2.imread(image_path)
        h, w, _ = img.shape

        plates.append({
            "bbox": [0, 0, w, h],
            "score": 1.0
        })

    return plates