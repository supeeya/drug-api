from ultralytics import YOLO

# โหลดโมเดลเพียงครั้งเดียว
model = YOLO("models/drugs_yolov8.pt")

def process_images(file_path):
    RESULT_FOLDER = "runs/detect/exp"
    results = model.predict(source=file_path, save=True, save_dir=RESULT_FOLDER)

    # ดึงข้อมูลผลลัพธ์
    detections = []
    for result in results:
        for box in result.boxes:  # ดึงข้อมูล bounding box
            cls = int(box.cls[0])  # คลาสของวัตถุ
            confidence = float(box.conf[0])  # ความมั่นใจ
            label = model.names[cls]  # ชื่อคลาส
            bbox = box.xyxy[0].tolist()  # ตำแหน่ง [x1, y1, x2, y2]
            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": bbox
            })

    return detections, RESULT_FOLDER
