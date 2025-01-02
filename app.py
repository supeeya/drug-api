from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from io import BytesIO
import base64
from PIL import Image
from ultralytics import YOLO
import uuid  # เพิ่มการใช้งาน uuid
import numpy as np


app = FastAPI()

# โหลดโมเดล YOLO
model = YOLO("models/drugs_yolov8.pt")

# class names ของยา
class_names = {
    0 : "Alaxan",
    1 : "Bactidol",
    2 : "Bioflu", 
    3 : "Biogesic", 
    4 : "DayZinc",
    5 :  "Decolgen",
    6 : "Fish Oil", 
    7 : "Kremil S", 
    8 : "Medicol",
    9 : "Neozep"
}

# คำแนะนำการใช้ยาและข้อควรระวัง
instructions = {
    "Alaxan": "ใช้เพื่อบรรเทาอาการปวด ห้ามใช้เกินขนาดที่กำหนด.",
    "Bactidol": "ใช้เพื่อบ้วนปาก ห้ามกลืนลงไป.",
    "Bioflu": "ใช้เพื่อบรรเทาอาการหวัด ปฏิบัติตามคำแนะนำของแพทย์.",
    "Biogesic": "บรรเทาไข้และปวด ใช้อย่างระมัดระวัง.",
    "DayZinc": "เสริมสร้างภูมิคุ้มกัน รับประทานพร้อมอาหาร.",
    "Decolgen": "แก้ไข้หวัดและน้ำมูกไหล ห้ามใช้ร่วมกับแอลกอฮอล์.",
    "Fish Oil": "เสริมโอเมก้า 3 รับประทานวันละ 1 เม็ด.",
    "Kremil S": "บรรเทาอาการกรดไหลย้อน เคี้ยวก่อนกลืน.",
    "Medicol": "บรรเทาปวดศีรษะ หลีกเลี่ยงการใช้เกินขนาด.",
    "Neozep": "บรรเทาหวัดและน้ำมูกไหล หลีกเลี่ยงการขับขี่หลังรับประทาน."
}

# สร้างตัวจัดการเทมเพลต
templates = Jinja2Templates(directory="templates")

# Middleware สำหรับเพิ่ม Request ID
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    # สร้าง Request ID ใหม่หรือใช้ค่าใน Header
    request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    
    # บันทึก Request ID ใน log หรือใช้ใน response
    response = await call_next(request)
    response.headers['X-Request-ID'] = request_id
    
    return response

def convert_image_to_base64(image: Image) -> str:
    """แปลงภาพเป็น Base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/") 
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        # เปิดไฟล์ภาพที่อัปโหลด
        img = Image.open(file.file)

        # ประมวลผลด้วยโมเดล YOLO
        results = model.predict(source=img)

        # ผลลัพธ์จะถูกเก็บใน numpy.ndarray
        result_image_array = results[0].plot()  # วาดผลลัพธ์ลงบนภาพ

        # แปลง numpy.ndarray เป็น PIL.Image
        result_image = Image.fromarray(result_image_array)

        # แปลงภาพผลลัพธ์และภาพที่อัปโหลดเป็น Base64
        uploaded_image_base64 = convert_image_to_base64(img)
        result_image_base64 = convert_image_to_base64(result_image)

        # ดึงข้อมูลการตรวจจับ
        detection = []
        instructions_list = []

        for box in results[0].boxes:
            label_index = int(box.cls[0].item())  # ดึง index ของ class
            detection_name = class_names[label_index]  # แปลง index เป็นชื่อยา
            detection.append(detection_name)
            # เพิ่มข้อมูลคำแนะนำ
            instructions_list.append(instructions.get(detection_name, "ไม่พบข้อมูลคำแนะนำ"))

        # ส่งข้อมูลไปยังเทมเพลต
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "uploaded_image_base64": uploaded_image_base64,
                "result_image_base64": result_image_base64,
                "detections": zip(detection, instructions_list),
            }
        )

    except Exception as e:
        # ถ้ามีข้อผิดพลาดเกิดขึ้น
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # เปิดไฟล์ภาพที่อัปโหลด
        img = Image.open(file.file)

        # ประมวลผลด้วยโมเดล YOLO
        results = model.predict(source=img)

        # ดึงข้อมูลการตรวจจับ
        detections = []
        for box in results[0].boxes:
            label_index = int(box.cls[0].item())  # ดึง index ของ class
            detection_name = class_names[label_index]  # แปลง index เป็นชื่อยา
            instructions_text = instructions.get(detection_name, "ไม่พบข้อมูลคำแนะนำ")
            detections.append({
                "name": detection_name,
                "instructions": instructions_text
            })

        # ส่งผลลัพธ์เป็น JSON
        return JSONResponse(content={"detections": detections})

    except Exception as e:
        # ถ้ามีข้อผิดพลาดเกิดขึ้น
        return JSONResponse(content={"error": str(e)}, status_code=500)

# เพิ่ม endpoint สำหรับรับภาพจากกล้องเป็น Base64
@app.post("/capture/")
async def capture_image(request: Request, image_base64: str = Form(...)):
    try:
        # แปลง Base64 เป็นภาพ
        image_data = base64.b64decode(image_base64.split(",")[1])
        img = Image.open(BytesIO(image_data))

        # ประมวลผลภาพ
        results = model.predict(source=img)

        # ดึงข้อมูลการตรวจจับ
        detections = []
        for box in results[0].boxes:
            label_index = int(box.cls[0].item())
            detection_name = class_names[label_index]
            instructions_text = instructions.get(detection_name, "ไม่พบข้อมูลคำแนะนำ")
            detections.append({
                "name": detection_name,
                "instructions": instructions_text
            })

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
