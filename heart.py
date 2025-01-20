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
model = YOLO("models/drugs.pt")

# class names ของยา
class_names = {
    0:  'Amitriptyline Hydrochloride Tablets 10 mg', 
    1:  'Amitriptyline Hydrochloride Tablets 10 mg', 
    2:  'Acetylsalicylic Acid (Aspirin) 81 mg', 
    3:  'Alfuzosin Hydrochloride 10 mg', 
    4:  'Allopurinol 100 mg (Allopurinol Sodium)', 
    5:  'Allopurinol 300 mg (Allopurinol Sodium)', 
    6:  'Amiloride Hydrochlorothiazide (HCTZ)', 
    7:  'Amiodarone Hydrochloride 200 mg', 
    8:  'Amlodipine Besylate 10 mg',
    9:  'Amlodipine Besylate 5 mg',
    10: 'Anapril 5 mg (Enalapril Maleate 5 mg)', 
    11: 'Acetylsalicylic Acid 325 mg', 
    12: 'Atenolol 100 mg',
    13: 'Atenolol 25 mg',
    14: 'Atenolol 50 mg',
    15: 'Atorvastatin 40 mg', 
    16: 'Betahistine 12 mg', 
    17: 'Bisoprolol Fumarate 5 mg',
    18: 'Para-co (Paracetamol + Codeine)', 
    19: 'Calciferol (Vitamin D2 หรือ Vitamin D3)',
    20: 'Canagliflozin 100 mg', 
    21: 'Canagliflozin 100 mg',
    22: 'Dicloxacillin Capsules 500 mg', 
    23: 'Captopril 25 mg', 
    24: 'Carbamazepine 200 mg',
    25: 'Carvedilol 25 mg',
    26: 'Carvedilol 6.25 mg',
    27: 'Carvedilol 6.5 mg', 
    28: 'Celebrex (Celecoxib)',
    29: 'Cetirizine 10 mg', 
    30: 'Chalktab 1.5 (Calcium Carbonate 1.5 g)', 
    31: 'Chalktab 835 (Calcium Carbonate 835 mg)',
    32: 'Colchicine 0.6 mg (Chicine)',
    33: 'Ciprofloxacin 250 mg',
    34: 'Clindamycin 300 mg',
    35: 'Clonazepam 2 mg',
    36: 'Clopidogrel 75 mg',
    37: 'Clorazepate Dipotassium 5 mg',
    38: 'Colchicine 0.6 mg',
    39: 'Dapagliflozin 10 mg',
    40: 'Dapagliflozin 10 mg ', 
    41: 'Digoxin 0.25 mg',
    42: 'Diltiazem 30 mg',
    43: 'Dimenhydrinate 50 mg', 
    44: 'Diltiazem 120 mg',
    45: 'Diltiazem 60 mg',
    46: 'Doxazosin 2 mg', 
    47: 'Empagliflozin 10 mg',
    48: 'Empagliflozin 10 mg',
    49: 'Enalapril Maleate 5 mg',
    50: 'Entresto 100 mg (ประกอบด้วย Sacubitril 24 mg และ Valsartan 26 mg)', 
    51: 'Entresto 200 mg',
    52: 'Standardised Senna Extract Equivalent 7.5 mg',
    53: 'Ethambutol 400 mg',
    54: 'Ferrous Fumarate 200 mg', 
    55: 'Fimasartan 60 mg', 
    56: 'Flecainide Acetate 100 mg',
    57: 'Fluoxetine Hydrochloride 20 mg', 
    58: 'Furosemide 40 mg', 
    59: 'Furosemide 500 mg',
    60: 'Gabapentin 100 mg', 
    61: 'Gabapentin 300 mg',
    62: 'Gemfibrozil 600 mg',
    63: 'Gemigliptin 50 mg', 
    64: 'Glibenclamide (Glyburide) 5 mg',
    65: 'Glimepiride 2 mg',
    66: 'Glipizide 5 mg', 
    67: 'Hyoscine-N-Butylbromide 10 mg',
    68: 'Hydralazine Hydrochloride 25 mg',
    69: 'Hydralazine 25 mg', 
    70: 'Hydralazine 50 mg', 
    71: 'Metformin Hydrochloride 500 mg',
    72: 'Hydrochlorothiazide 25 mg',
    73: 'Hydroxychloroquine 200 mg',
    74: 'Isosorbide Dinitrate 10 mg',
    75: 'Isosorbide Dinitrate 5 mg',
    76: 'Ivabradine 5 mg',
    77: 'Lercanidipine 20 mg ', 
    78: 'Levothyroxine 50 mcg', 
    79: 'Losartan Potassium 50 mg',
    80: 'Losartan Potassium 50 mg', 
    81: 'Luseogliflozin 5 mg',
    82: 'Manidipine Hydrochloride 20 mg', 
    83: 'Metformin 850 mg', 
    84: 'Metformin Hydrochloride 500 mg',
    85: 'Metoprolol 100 mg',
    86: 'Montelukast 10 mg',
    87: 'Naproxen 250 mg',
    88: 'Nifedipine 20 mg', 
    89: 'Nifedipine 5 mg',
    90: 'Nimodipine 30 mg', 
    91: 'Omeprazole Capsules 20 mg',
    92: 'Paracetamol 450 mg',
    93: 'Pioglitazone 30 mg', 
    94: 'Prasugrel 10 mg',
    95: 'Propranolol 10 mg',
    96: 'Propranolol 40 mg',
    97: 'Propranolol 10 mg',
    98: 'Pyrazinamide 500 mg',
    99: 'Ramipril 5 mg',
    100: 'Rifampicin 450 mg', 
    101: 'Sertraline 50 mg',
    102: 'Simvastatin 20 mg',
    103: 'Sitagliptin 100 mg',
    104: 'Spironolactone 100 mg',
    105: 'Spironolactone 25 mg',
    106: 'Tamoxifen Film-coated tablets 20 mg', 
    107: 'Tizanidine 2 mg',
    108: 'Tolvaptan 15 mg',
    109: 'Tramadol',
    110: 'Tramadol 50 mg ', 
    111: 'Trimetazidine 35 mg',
    112: 'Valsartan 160 mg', 
    113: 'Verapamil 240 mg', 
    114: 'Verapamil 40 mg',
    115: 'Vitamin B Complex',
    116: 'Warfarin 2 mg',
    117: 'Warfarin 3 mg',
    118: 'Warfarin 5 mg'
}

# คำแนะนำการใช้ยาและข้อควรระวัง
instructions = {
    "Amitriptyline Hydrochloride Tablets 10 mg": "รับประทานยาตามคำสั่งแพทย์ วันละ 1-3 ครั้ง หรือก่อนนอนในกรณีใช้รักษาอาการนอนไม่หลับ.",
    "Acetylsalicylic Acid (Aspirin) 81 mg": "ใช้ป้องกันลิ่มเลือดอุดตัน ลดความเสี่ยงโรคหัวใจขาดเลือดเฉียบพลัน หรือโรคหลอดเลือดสมอง ตามคำสั่งแพทย์ รับประทานวันละ 1 เม็ด พร้อมน้ำเปล่า ควรรับประทานหลังอาหารเพื่อลดการระคายเคืองกระเพาะอาหาร.",
    "Alfuzosin Hydrochloride 10 mg": "ใช้รักษาอาการปัสสาวะขัดหรือปัสสาวะไม่ออกในผู้ป่วยต่อมลูกหมากโต (Benign Prostatic Hyperplasia - BPH) รับประทานวันละ 1 ครั้ง หลังอาหารมื้อหลัก เพื่อลดความเสี่ยงต่ออาการเวียนศีรษะหรือหน้ามืด.",
    "Allopurinol 100 mg (Allopurinol Sodium)": "ใช้ลดระดับกรดยูริกในเลือดสำหรับผู้ป่วยโรคเกาต์หรือภาวะกรดยูริกสูง รับประทานวันละ 1 ครั้ง หลังอาหาร พร้อมน้ำเปล่าปริมาณมาก เพื่อป้องกันการเกิดนิ่วในไต.",
    "Allopurinol 300 mg (Allopurinol Sodium)": "ใช้ลดระดับกรดยูริกในเลือดสำหรับผู้ป่วยโรคเกาต์หรือภาวะกรดยูริกสูง รับประทานวันละ 1 ครั้ง หลังอาหาร พร้อมน้ำเปล่ามาก ๆ เพื่อลดความเสี่ยงการเกิดนิ่วในไต ขนาดยาขึ้นอยู่กับระดับกรดยูริกในเลือด ควรปฏิบัติตามคำแนะนำของแพทย์อย่างเคร่งครัด.",
    "Amiloride Hydrochlorothiazide (HCTZ)": "ใช้รักษาความดันโลหิตสูงและภาวะบวมน้ำ (edema) โดยช่วยขับปัสสาวะและรักษาสมดุลโพแทสเซียมในร่างกาย รับประทานวันละครั้งในตอนเช้า พร้อมอาหารหรือหลังอาหารเพื่อลดการระคายเคืองกระเพาะอาหาร.",
    "Amiodarone Hydrochloride 200 mg": "ใช้รักษาภาวะหัวใจเต้นผิดจังหวะ เช่น หัวใจเต้นเร็วผิดปกติหรือหัวใจเต้นพริ้ว (Atrial Fibrillation) รับประทานตามแพทย์สั่ง ปกติมักเริ่มต้นด้วยขนาดสูงและลดลงเมื่ออาการควบคุมได้ รับประทานพร้อมอาหารหรือตามคำแนะนำของแพทย์.",
    "Amlodipine Besylate 10 mg": "ใช้รักษาความดันโลหิตสูง (Hypertension) และโรคหลอดเลือดหัวใจ (Angina) รับประทานวันละครั้ง เวลาเดียวกันทุกวัน พร้อมหรือไม่พร้อมอาหารก็ได้.",
    "Amlodipine Besylate 5 mg": "ใช้รักษาความดันโลหิตสูง (Hypertension) และโรคหลอดเลือดหัวใจตีบ (Angina) รับประทานวันละครั้ง เวลาเดียวกันทุกวัน พร้อมหรือไม่พร้อมอาหารก็ได้.",
    "Anapril 5 mg (Enalapril Maleate 5 mg)": " ใช้รักษาความดันโลหิตสูง (Hypertension) และภาวะหัวใจล้มเหลว (Heart Failure) รับประทานวันละ 1 ครั้ง หรือตามคำแนะนำของแพทย์ สามารถรับประทานได้ทั้งพร้อมหรือไม่พร้อมอาหาร.",
    "Acetylsalicylic Acid 325 mg": "ใช้ในการบรรเทาอาการปวด เช่น ปวดศีรษะ, ปวดกล้ามเนื้อ, ปวดท้องจากประจำเดือน และอาการอักเสบต่างๆ ใช้ในผู้ที่มีความเสี่ยงในการเกิดโรคหัวใจและหลอดเลือด เพื่อป้องกันการเกิดลิ่มเลือด (แนะนำให้ใช้ตามคำแนะนำของแพทย์) รับประทานพร้อมอาหารหรือหลังอาหารเพื่อลดการระคายเคืองในกระเพาะอาหาร หากใช้เพื่อบรรเทาอาการปวด ให้รับประทานในขนาด 325 mg ทุก 4-6 ชั่วโมงตามต้องการ แต่ไม่เกิน 4,000 mg ต่อวัน.",
    "Atenolol 100 mg ": " ใช้รักษาความดันโลหิตสูง (Hypertension) และภาวะหลอดเลือดหัวใจตีบ (Angina) ใช้ในการรักษาภาวะหัวใจเต้นผิดจังหวะ (Arrhythmia) รับประทานวันละ 1 ครั้ง โดยควรรับประทานในเวลาเดียวกันทุกวัน สามารถรับประทานพร้อมหรือไม่พร้อมอาหาร.",
    "Atenolol 25 mg": "ใช้รักษาความดันโลหิตสูง (Hypertension) และภาวะหลอดเลือดหัวใจตีบ (Angina) ใช้ในการรักษาภาวะหัวใจเต้นผิดจังหวะ (Arrhythmia) รับประทานวันละ 1 ครั้ง โดยควรรับประทานในเวลาเดียวกันทุกวัน สามารถรับประทานพร้อมหรือไม่พร้อมอาหาร.",
    "Atenolol 50 mg": " ใช้รักษาความดันโลหิตสูง (Hypertension) และภาวะหลอดเลือดหัวใจตีบ (Angina) ใช้ในการรักษาภาวะหัวใจเต้นผิดจังหวะ (Arrhythmia) รับประทานวันละ 1 ครั้ง โดยควรรับประทานในเวลาเดียวกันทุกวัน สามารถรับประทานพร้อมหรือไม่พร้อมอาหาร.",
    "Atorvastatin 40 mg": "ใช้ในการลดระดับคอเลสเตอรอล (Cholesterol) ในเลือด โดยเฉพาะคอเลสเตอรอลชนิด LDL และไตรกลีเซอไรด์ (Triglycerides) ใช้ป้องกันโรคหลอดเลือดหัวใจ เช่น หัวใจขาดเลือด (Coronary Artery Disease) และเส้นเลือดตีบ รับประทานวันละ 1 ครั้ง โดยควรรับประทานในเวลาเดียวกันทุกวัน สามารถรับประทานพร้อมหรือไม่พร้อมอาหาร ปรับขนาดยาและติดตามระดับคอเลสเตอรอลเป็นระยะตามคำแนะนำของแพทย์.",
    "Betahistine 12 mg": "ใช้ในการรักษาอาการเวียนศีรษะจากโรคเมเนียร์ (Meniere's Disease) เช่น เวียนศีรษะ คลื่นไส้ และหูอื้อ รับประทานยา 1-2 ครั้งต่อวัน โดยปกติจะรับประทานหลังอาหารเพื่อป้องกันการระคายเคืองกระเพาะอาหาร ปรับขนาดยาและความถี่ในการใช้ยาตามคำแนะนำของแพทย์.",
    "Bisoprolol Fumarate 5 mg": "ใช้รักษาความดันโลหิตสูง (Hypertension) และภาวะหัวใจล้มเหลว (Heart Failure) ใช้ในการรักษาภาวะหัวใจเต้นผิดจังหวะ (Arrhythmia) รับประทานวันละ 1 ครั้ง โดยปกติจะรับประทานในเวลาเดียวกันทุกวัน สามารถรับประทานได้ทั้งพร้อมหรือไม่พร้อมอาหาร .",
    "Para-co (Paracetamol + Codeine)": "ใช้ในการบรรเทาอาการปวดที่มีความรุนแรงปานกลางถึงรุนแรง เช่น ปวดหลัง ปวดข้อ ปวดหัว ประกอบด้วยสองส่วนคือ Paracetamol (ช่วยบรรเทาปวด) และ Codeine (ยาบรรเทาปวดที่มีฤทธิ์สูง) รับประทานตามคำแนะนำของแพทย์ โดยทั่วไปจะรับประทาน 1-2 เม็ดทุก 4-6 ชั่วโมง ขึ้นอยู่กับอาการปวด แต่ไม่ควรใช้มากกว่า 8 เม็ดต่อวัน.",
    "Calciferol (Vitamin D2 หรือ Vitamin D3)": "ใช้ในการรักษาหรือป้องกันภาวะขาดวิตามินดี (Vitamin D deficiency) เช่น ในผู้ที่มีภาวะกระดูกพรุน (Osteoporosis), โรคกระดูกอ่อน (Rickets), และโรคกระดูกเสื่อม (Osteomalacia) ใช้เพื่อส่งเสริมการดูดซึมแคลเซียมในลำไส้และช่วยให้กระดูกและฟันแข็งแรง ปริมาณการใช้ขึ้นอยู่กับคำแนะนำของแพทย์ โดยมักจะใช้ในขนาด 400-1000 IU ต่อวัน แต่สามารถปรับได้ตามภาวะสุขภาพของผู้ป่วย.",
    "Canagliflozin 100 mg": "ใช้ในการรักษาโรคเบาหวานชนิดที่ 2 (Type 2 Diabetes Mellitus) โดยช่วยลดระดับน้ำตาลในเลือด ใช้ร่วมกับการควบคุมอาหารและออกกำลังกายเพื่อปรับปรุงระดับน้ำตาลในเลือด ใช้ในการลดความเสี่ยงของโรคหลอดเลือดหัวใจในผู้ป่วยโรคเบาหวานชนิดที่ 2 ที่มีความเสี่ยงสูง รับประทานยา 1 ครั้งต่อวัน โดยควรรับประทานในเวลาเดียวกันทุกวัน สามารถรับประทานพร้อมหรือไม่พร้อมอาหาร.",
    "Canagliflozin 100 mg": "ใช้ในการรักษาโรคเบาหวานชนิดที่ 2 (Type 2 Diabetes Mellitus) โดยช่วยลดระดับน้ำตาลในเลือด ใช้ร่วมกับการควบคุมอาหารและออกกำลังกายเพื่อปรับปรุงระดับน้ำตาลในเลือด ใช้ในการลดความเสี่ยงของโรคหลอดเลือดหัวใจในผู้ป่วยโรคเบาหวานชนิดที่ 2 ที่มีความเสี่ยงสูง รับประทานยา 1 ครั้งต่อวัน โดยควรรับประทานในเวลาเดียวกันทุกวัน สามารถรับประทานพร้อมหรือไม่พร้อมอาหาร.",
    "Dicloxacillin Capsules 500 mg": "ใช้ในการรักษาการติดเชื้อที่เกิดจากแบคทีเรียที่ไวต่อยา dicloxacillin เช่น การติดเชื้อที่ผิวหนัง ระบบทางเดินหายใจ และระบบทางเดินปัสสาวะ ใช้ในการรักษาโรคติดเชื้อที่เกิดจากแบคทีเรีย Staphylococcus aureus ที่ไม่ดื้อต่อเมธาซิลลิน รับประทานยาตามคำแนะนำของแพทย์ โดยทั่วไปจะรับประทาน 1-2 แคปซูลทุก 6 ชั่วโมง หรือขึ้นอยู่กับคำแนะนำของแพทย์ ควรรับประทานยานี้ขณะท้องว่าง หรืออย่างน้อย 1 ชั่วโมงก่อนอาหารหรือ 2 ชั่วโมงหลังอาหาร.",
    "Captopril 25 mg": " ใช้ในการรักษาความดันโลหิตสูง (Hypertension) ใช้ในการรักษาภาวะหัวใจล้มเหลว (Heart Failure) ใช้ในกรณีที่มีปัญหาเกี่ยวกับการทำงานของไตในผู้ป่วยที่มีเบาหวาน (Diabetic Nephropathy) ใช้ในการรักษาภาวะหลังจากมีอาการหัวใจวาย (Post-myocardial infarction) รับประทานยาตามคำแนะนำของแพทย์ โดยทั่วไปจะเริ่มต้นที่ 25 mg วันละ 1-2 ครั้ง (ขึ้นอยู่กับภาวะของผู้ป่วย).",
    "Carbamazepine 200 mg": "ใช้ในการรักษาภาวะชัก (Epilepsy) โดยเฉพาะในโรคลมชัก (Seizure) ชนิด Partial seizures, Tonic-clonic seizures ใช้ในการรักษาโรคอาการปวดเส้นประสาท (Trigeminal Neuralgia) และภาวะความผิดปกติทางจิต เช่น โรคอารมณ์สองขั้ว (Bipolar disorder) รับประทานยาตามคำแนะนำของแพทย์ โดยทั่วไปเริ่มต้นที่ 200 mg วันละ 1-2 ครั้ง แล้วค่อย ๆ ปรับขนาดยาในระยะเวลาต่อมา.",
    "Carvedilol 25 mg": "ใช้ในการรักษาความดันโลหิตสูง (Hypertension) ใช้ในการรักษาภาวะหัวใจล้มเหลว (Heart Failure) ใช้ในการรักษาภาวะหลังจากมีอาการหัวใจวาย (Post-myocardial infarction) เพื่อลดความเสี่ยงในการเกิดภาวะหัวใจหยุดทำงาน รับประทานยาตามคำแนะนำของแพทย์ โดยทั่วไปเริ่มต้นที่ 12.5 mg-25 mg วันละ 1-2 ครั้ง ขึ้นอยู่กับภาวะของผู้ป่วย.",
    "Carvedilol 6.25 mg": "ใช้ในการรักษาความดันโลหิตสูง (Hypertension) ใช้ในการรักษาภาวะหัวใจล้มเหลว (Heart Failure) ใช้ในการรักษาภาวะหลังจากมีอาการหัวใจวาย (Post-myocardial infarction) เพื่อลดความเสี่ยงในการเกิดภาวะหัวใจหยุดทำงาน รับประทานยาตามคำแนะนำของแพทย์ โดยทั่วไปเริ่มต้นที่ 6.25 mg วันละ 2 ครั้ง (ขึ้นอยู่กับภาวะของผู้ป่วย).",
    "Carvedilol 6.5 mg": "ใช้ในการรักษาความดันโลหิตสูง (Hypertension) ใช้ในการรักษาภาวะหัวใจล้มเหลว (Heart Failure) ใช้ในการรักษาภาวะหลังจากมีอาการหัวใจวาย (Post-myocardial infarction) เพื่อลดความเสี่ยงในการเกิดภาวะหัวใจหยุดทำงาน รับประทานยาตามคำแนะนำของแพทย์ โดยทั่วไปเริ่มต้นที่ 6.5 mg วันละ 2 ครั้ง ขึ้นอยู่กับคำแนะนำของแพทย์หรือภาวะของผู้ป่วย.",
    "Celebrex (Celecoxib)": "ใช้ในการรักษาอาการปวดและการอักเสบจากโรคข้ออักเสบ (Osteoarthritis, Rheumatoid arthritis) ใช้ในการรักษาอาการปวดเฉียบพลัน เช่น ปวดหลัง ปวดฟัน หรือปวดกล้ามเนื้อ ใช้ในการรักษาโรคเกาต์ (Gout) ในระยะที่มีการอักเสบ ใช้ในการรักษาภาวะที่เกี่ยวข้องกับการอักเสบของระบบกระดูกและข้อ เช่น โรคข้อเสื่อม (Ankylosing spondylitis).",
    "Cetirizine 10 mg": "ใช้ในการรักษาอาการแพ้ (Allergic rhinitis) เช่น น้ำมูกไหล จาม คันจมูก คันตา ใช้ในการรักษาผื่นลมพิษ (Urticaria) หรืออาการคันจากผิวหนังที่เกิดจากอาการแพ้ ใช้ในการบรรเทาอาการแพ้ในกรณีของภูมิแพ้ต่าง ๆ.",
    "Chalktab 1.5 (Calcium Carbonate 1.5 g)": " ใช้ในการเสริมแคลเซียมในผู้ที่มีระดับแคลเซียมในเลือดต่ำ หรือในผู้ที่ต้องการแคลเซียมเสริม เช่น ผู้ที่มีภาวะกระดูกพรุน (Osteoporosis) ใช้ในการรักษาโรคกระเพาะอาหาร โดยช่วยในการลดกรดในกระเพาะ (Antacid) ใช้ในการเสริมแคลเซียมในสตรีตั้งครรภ์หรือให้นมบุตรหากแคลเซียมในอาหารไม่เพียงพอ.",
    "Chalktab 835 (Calcium Carbonate 835 mg)": "ใช้ในการเสริมแคลเซียมในผู้ที่มีระดับแคลเซียมในเลือดต่ำ หรือในผู้ที่ต้องการแคลเซียมเสริม เช่น ผู้ที่มีภาวะกระดูกพรุน (Osteoporosis) ใช้ในการรักษาโรคกระเพาะอาหาร โดยช่วยในการลดกรดในกระเพาะ (Antacid) ใช้ในการเสริมแคลเซียมในสตรีตั้งครรภ์หรือให้นมบุตรหากแคลเซียมในอาหารไม่เพียงพอ.",
    "Colchicine 0.6 mg (Chicine)": "ใช้ในการรักษาโรคเกาต์ (Gout) โดยช่วยลดอาการอักเสบและปวดที่เกิดจากการสะสมของกรดยูริกในข้อต่าง ๆ ใช้ในการรักษาโรค Pericarditis (อาการอักเสบของเยื่อหุ้มหัวใจ) ใช้ในบางกรณีในการรักษาภาวะ Amyloidosis ที่เกี่ยวข้องกับการสะสมของโปรตีนผิดปกติในอวัยวะ.",
    "Ciprofloxacin 250 mg": "ใช้ในการรักษาการติดเชื้อที่เกิดจากแบคทีเรีย เช่น การติดเชื้อในทางเดินปัสสาวะ (Urinary tract infections, UTI), การติดเชื้อในทางเดินหายใจ, การติดเชื้อที่ผิวหนังและเนื้อเยื่ออ่อน ใช้ในการรักษาโรคที่เกิดจากแบคทีเรียที่ไวต่อยา เช่น โรคท้องเสียจากแบคทีเรีย, การติดเชื้อกระเพาะอาหาร ใช้ในการรักษาภาวะติดเชื้อในกระแสเลือด (Septicemia) และการติดเชื้อในกระดูกและข้อ.",
    "Clindamycin 300 mg": " ใช้ในการรักษาการติดเชื้อที่เกิดจากแบคทีเรีย เช่น การติดเชื้อที่ผิวหนังและเนื้อเยื่ออ่อน, การติดเชื้อในทางเดินหายใจ, การติดเชื้อในช่องท้อง และการติดเชื้อในกระดูก ใช้ในการรักษาภาวะการติดเชื้อจากแบคทีเรียที่ไม่สามารถรักษาด้วยยาปฏิชีวนะประเภทอื่น ๆ ใช้ในกรณีที่มีการติดเชื้อจากแบคทีเรียแกรมบวกที่ไวต่อยา.",
    "Clonazepam 2 mg": "ใช้ในการรักษาภาวะวิตกกังวล (Anxiety), โรคลมชัก (Seizures), และโรคแพนิค (Panic disorder) ใช้ในบางกรณีเพื่อการบรรเทาอาการของโรคกล้ามเนื้อเกร็ง (Muscle spasms) ใช้ในกรณีที่จำเป็นในการควบคุมอาการวิตกกังวลหรืออาการที่เกี่ยวข้องกับการนอนไม่หลับในบางผู้ป่วย.",
    "Clopidogrel 75 mg": "ใช้ในการป้องกันและรักษาภาวะเสี่ยงต่อการเกิดโรคหลอดเลือดหัวใจ เช่น โรคหัวใจขาดเลือด (Angina), โรคหลอดเลือดสมอง, และการเกิดลิ่มเลือดหลังจากการทำหัตถการ เช่น การผ่าตัดบายพาสหลอดเลือดหัวใจ ใช้ในการป้องกันการเกิดลิ่มเลือดในผู้ป่วยที่มีภาวะหลอดเลือดแดงตีบ (Atherosclerosis) ใช้ในผู้ป่วยที่ได้รับการผ่าตัดเพื่อป้องกันการเกิดลิ่มเลือดในหลอดเลือดหัวใจหรือในหลอดเลือดสมอง.",
    "Clorazepate Dipotassium 5 mg": "ใช้ในการรักษาภาวะวิตกกังวล (Anxiety), โรคลมชัก (Seizures), และอาการที่เกี่ยวข้องกับการถอนตัวจากแอลกอฮอล์ (Alcohol withdrawal) ใช้ในการบรรเทาอาการเครียดหรือวิตกกังวลในผู้ป่วยที่มีอาการรุนแรง ใช้ในบางกรณีเพื่อบรรเทาอาการของการเกร็งของกล้ามเนื้อ.",
    "Colchicine 0.6 mg": "ใช้ในการรักษาโรคเกาต์ (Gout) โดยการลดอาการอักเสบและบรรเทาอาการปวดที่เกิดจากการสะสมของกรดยูริคในข้อต่อ ใช้ในการป้องกันการเกิดโรคเกาต์ในผู้ป่วยที่มีอาการเป็นซ้ำ ใช้ในการรักษาภาวะโรคแพ้ภูมิตัวเองบางประเภท เช่น Familial Mediterranean Fever (FMF).",
    "Dapagliflozin 10 mg": "ใช้ในการรักษาผู้ป่วยโรคเบาหวานชนิดที่ 2 (Type 2 diabetes) โดยช่วยลดระดับน้ำตาลในเลือด ใช้ในการรักษาภาวะหัวใจล้มเหลว (Heart failure) โดยช่วยลดความเสี่ยงจากการเกิดภาวะแทรกซ้อน ใช้ในการรักษาผู้ป่วยที่มีโรคไตเรื้อรัง (Chronic kidney disease) เพื่อชะลอการเสื่อมของไต ใช้ในบางกรณีร่วมกับยาชนิดอื่น ๆ เพื่อควบคุมระดับน้ำตาลในเลือดให้ดีขึ้น.",
    "Dapagliflozin 10 mg ": "ใช้ในการรักษาผู้ป่วยโรคเบาหวานชนิดที่ 2 (Type 2 diabetes) โดยช่วยลดระดับน้ำตาลในเลือด ใช้ในการรักษาภาวะหัวใจล้มเหลว (Heart failure) โดยช่วยลดความเสี่ยงจากการเกิดภาวะแทรกซ้อน ใช้ในการรักษาผู้ป่วยที่มีโรคไตเรื้อรัง (Chronic kidney disease) เพื่อชะลอการเสื่อมของไต ใช้ในบางกรณีร่วมกับยาชนิดอื่น ๆ เพื่อควบคุมระดับน้ำตาลในเลือดให้ดีขึ้น.",
    "Digoxin 0.25 mg": "ใช้ในการรักษาภาวะหัวใจล้มเหลว (Heart failure) โดยช่วยเพิ่มประสิทธิภาพในการบีบตัวของหัวใจ ใช้ในการรักษาภาวะหัวใจเต้นผิดปกติ (Atrial fibrillation, Atrial flutter) โดยช่วยชะลอการเต้นของหัวใจ ช่วยลดความดันโลหิตในผู้ป่วยบางกลุ่ม.",
    "Diltiazem 30 mg": "ใช้ในการรักษาภาวะความดันโลหิตสูง (Hypertension) โดยการช่วยขยายหลอดเลือดและลดความต้านทานของหลอดเลือด ใช้ในการรักษาภาวะหัวใจเต้นผิดจังหวะ (Arrhythmia) เช่น atrial fibrillation และ atrial flutter โดยช่วยลดอัตราการเต้นของหัวใจ ใช้ในการรักษาภาวะกล้ามเนื้อหัวใจขาดเลือด (Angina) โดยการช่วยเพิ่มการไหลเวียนของเลือดไปยังกล้ามเนื้อหัวใจ ใช้ในผู้ป่วยโรคหลอดเลือดหัวใจตีบ (Coronary artery disease) เพื่อลดอาการเจ็บหน้าอก.",
    "Dimenhydrinate 50 mg": "ใช้ในการรักษาอาการเมารถ (Motion sickness) หรือคลื่นไส้อาเจียนที่เกิดจากการเดินทาง ใช้ในการรักษาอาการเวียนศีรษะ (Dizziness) หรืออาการที่เกี่ยวข้องกับการเคลื่อนไหว ใช้ในการรักษาอาการคลื่นไส้ อาเจียนจากการใช้ยา หรือการบำบัดภาวะปัญหาทางเดินอาหารบางชนิด.",
    "Diltiazem 120 mg": "ใช้ในการรักษาภาวะความดันโลหิตสูง (Hypertension) โดยการขยายหลอดเลือดและลดความต้านทานของหลอดเลือด ใช้ในการรักษาภาวะหัวใจเต้นผิดจังหวะ (Arrhythmia) เช่น atrial fibrillation หรือ atrial flutter โดยช่วยชะลออัตราการเต้นของหัวใจ ใช้ในการรักษาภาวะกล้ามเนื้อหัวใจขาดเลือด (Angina) โดยช่วยเพิ่มการไหลเวียนของเลือดไปยังกล้ามเนื้อหัวใจ ใช้ในผู้ป่วยโรคหลอดเลือดหัวใจตีบ (Coronary artery disease) เพื่อลดอาการเจ็บหน้าอก.",
    "Diltiazem 60 mg": "ใช้ในการรักษาภาวะความดันโลหิตสูง (Hypertension) โดยการขยายหลอดเลือดและลดความต้านทานของหลอดเลือด ใช้ในการรักษาภาวะหัวใจเต้นผิดจังหวะ (Arrhythmia) เช่น atrial fibrillation หรือ atrial flutter โดยช่วยชะลออัตราการเต้นของหัวใจ ใช้ในการรักษาภาวะกล้ามเนื้อหัวใจขาดเลือด (Angina) โดยการเพิ่มการไหลเวียนของเลือดไปยังกล้ามเนื้อหัวใจ ใช้ในผู้ป่วยโรคหลอดเลือดหัวใจตีบ (Coronary artery disease) เพื่อลดอาการเจ็บหน้าอก.",
    "Doxazosin 2 mg": "ใช้ในการรักษาภาวะความดันโลหิตสูง (Hypertension) โดยการช่วยขยายหลอดเลือดและลดความต้านทานของหลอดเลือด ใช้ในการรักษาผู้ที่มีอาการต่อมลูกหมากโต (Benign Prostatic Hyperplasia หรือ BPH) โดยการผ่อนคลายกล้ามเนื้อในต่อมลูกหมากและกระเพาะปัสสาวะ ใช้ในการลดความเสี่ยงของภาวะหัวใจล้มเหลวหรือหลอดเลือดหัวใจตีบในผู้ป่วยที่มีปัจจัยเสี่ยง.",
    "Empagliflozin 10 mg": " ใช้ในการรักษาภาวะเบาหวานชนิดที่ 2 (Type 2 Diabetes) เพื่อควบคุมระดับน้ำตาลในเลือด โดยการยับยั้งการดูดซึมน้ำตาลจากปัสสาวะกลับสู่กระแสเลือด ใช้ในการลดความเสี่ยงของภาวะหัวใจล้มเหลวในผู้ป่วยที่มีภาวะเบาหวานชนิดที่ 2 ใช้ในการลดความเสี่ยงของการเกิดภาวะโรคหลอดเลือดหัวใจในผู้ป่วยที่มีเบาหวาน.",
    "Empagliflozin 10 mg": " ใช้ในการรักษาภาวะเบาหวานชนิดที่ 2 (Type 2 Diabetes) เพื่อควบคุมระดับน้ำตาลในเลือด โดยการยับยั้งการดูดซึมน้ำตาลจากปัสสาวะกลับสู่กระแสเลือด ใช้ในการลดความเสี่ยงของภาวะหัวใจล้มเหลวในผู้ป่วยที่มีภาวะเบาหวานชนิดที่ 2 ใช้ในการลดความเสี่ยงของการเกิดภาวะโรคหลอดเลือดหัวใจในผู้ป่วยที่มีเบาหวาน.",
    "Enalapril Maleate 5 mg": "ใช้ในการรักษาภาวะความดันโลหิตสูง (Hypertension) โดยการยับยั้งเอนไซม์แองจิโอเทนซินคอนเวอร์ติ้งเอ็นไซม์ (ACE) ซึ่งช่วยขยายหลอดเลือดและลดความดันโลหิต ใช้ในการรักษาภาวะหัวใจล้มเหลว (Heart failure) โดยช่วยลดการทำงานหนักของหัวใจ ใช้ในการรักษาผู้ป่วยที่มีภาวะไตเสื่อมจากเบาหวาน (Diabetic nephropathy).",
    "Entresto 100 mg (ประกอบด้วย Sacubitril 24 mg และ Valsartan 26 mg)": "ใช้ในการรักษาภาวะหัวใจล้มเหลว (Heart Failure) โดยเฉพาะในผู้ป่วยที่มีภาวะหัวใจล้มเหลวชนิดที่มีการบีบตัวของหัวใจต่ำ (Heart Failure with reduced Ejection Fraction - HFrEF) ช่วยลดความเสี่ยงในการเข้าโรงพยาบาลจากภาวะหัวใจล้มเหลวและการเสียชีวิตจากภาวะหัวใจล้มเหลว Sacubitril (ส่วนประกอบหนึ่งของยา) ช่วยยับยั้งเอนไซม์ neprilysin ซึ่งมีบทบาทในการสลายสารที่ช่วยขยายหลอดเลือดและลดความดันโลหิต Valsartan (ส่วนประกอบอีกตัว) เป็นยาต้านตัวรับแองจิโอเทนซิน II ซึ่งช่วยลดความดันโลหิตและลดภาระการทำงานของหัวใจ.",
    "Entresto 200 mg": "ใช้ในการรักษาภาวะหัวใจล้มเหลว (Heart failure) โดยช่วยปรับปรุงการทำงานของหัวใจและลดอาการบวมที่เกิดจากการคั่งของน้ำ ใช้ในการลดการเข้าโรงพยาบาลและการเสียชีวิตในผู้ป่วยที่มีภาวะหัวใจล้มเหลวจากสาเหตุที่ไม่สามารถควบคุมได้ด้วยยาอื่น ๆ ใช้ในผู้ป่วยที่มีความเสี่ยงต่อโรคหัวใจและหลอดเลือด เช่น ผู้ที่มีความดันโลหิตสูงหรือมีปัญหาหัวใจชนิดอื่นๆ.",
    "Standardised Senna Extract Equivalent 7.5 mg": " ใช้ในการรักษาอาการท้องผูก (Constipation) โดยการกระตุ้นการเคลื่อนไหวของลำไส้ สามารถช่วยบรรเทาอาการท้องผูกในผู้ป่วยที่ต้องการการกระตุ้นลำไส้ในระยะสั้น.",
    "Ethambutol 400 mg": "ใช้ในการรักษาภาวะวัณโรค (Tuberculosis) โดยจะช่วยยับยั้งการเจริญเติบโตของแบคทีเรีย Mycobacterium tuberculosis ที่ทำให้เกิดวัณโรค ใช้ร่วมกับยาต้านวัณโรคอื่นๆ เพื่อรักษาให้ได้ผลดีที่สุดและป้องกันการเกิดเชื้อที่ต้านทานยา.",
    "Ferrous Fumarate 200 mg": "ใช้ในการรักษาภาวะขาดธาตุเหล็ก (Iron deficiency) หรือภาวะโลหิตจางจากการขาดธาตุเหล็ก (Iron deficiency anemia) ช่วยเพิ่มระดับธาตุเหล็กในร่างกาย ซึ่งจำเป็นต่อการผลิตฮีโมโกลบินในเลือด.",
    "Fimasartan 60 mg": "ใช้ในการรักษาความดันโลหิตสูง (Hypertension) เพื่อช่วยลดความเสี่ยงของโรคหัวใจและหลอดเลือด ช่วยในการควบคุมความดันโลหิตให้อยู่ในระดับปกติและลดความเสี่ยงในการเกิดโรคหลอดเลือดสมองหรือโรคหัวใจ.",
    "Flecainide Acetate 100 mg": "รับประทานตามคำสั่งแพทย์ โดยปกติมักรับประทานวันละ 1-2 ครั้ง รับประทานพร้อมหรือไม่พร้อมอาหารก็ได้.",
    "Fluoxetine Hydrochloride 20 mg": "รับประทานวันละ 1 ครั้ง โดยปกติตอนเช้า เพื่อหลีกเลี่ยงอาการนอนไม่หลับ ควรรับประทานเวลาเดียวกันทุกวัน และตามคำสั่งแพทย์.",
    "Furosemide 40 mg": "รับประทานตามคำสั่งแพทย์ วันละ 1-2 ครั้ง ควรรับประทานในช่วงเช้าหรือก่อนบ่าย เพื่อหลีกเลี่ยงการตื่นปัสสาวะตอนกลางคืน.",
    "Furosemide 500 mg": "รับประทานหรือให้ทางหลอดเลือดตามคำสั่งแพทย์ ควรปรับขนาดยาตามความรุนแรงของอาการและการตอบสนอง.",
    "Gabapentin 100 mg": "รับประทานตามคำสั่งแพทย์ โดยเริ่มจากขนาดต่ำแล้วเพิ่มขึ้นทีละน้อย ควรรับประทานเวลาเดียวกันทุกวัน และสามารถรับประทานพร้อมอาหารได้.",
    "Gabapentin 300 mg": "รับประทานตามคำแนะนำของแพทย์ โดยเริ่มจากขนาดต่ำแล้วเพิ่มทีละน้อย รับประทานพร้อมหรือไม่พร้อมอาหารก็ได้ ควรรับประทานเวลาเดียวกันทุกวันเพื่อประสิทธิภาพสูงสุด.",
    "Gemfibrozil 600 mg": "รับประทานวันละ 2 ครั้ง ก่อนอาหารเช้าและอาหารเย็นประมาณ 30 นาที รับประทานตามคำสั่งแพทย์อย่างเคร่งครัด.",
    "Gemigliptin 50 mg": "รับประทานวันละ 1 ครั้ง พร้อมหรือไม่พร้อมอาหารก็ได้ ควรรับประทานในเวลาเดียวกันทุกวัน.",
    "Glibenclamide (Glyburide) 5 mg": ".",
    "Glimepiride 2 mg": ".",
    "Glipizide 5 mg": ".",
    "Hyoscine-N-Butylbromide 10 mg": ".",
    "Hydralazine Hydrochloride 25 mg": ".",
    "Hydralazine 25 mg": ".",
    "Hydralazine 50 mg": ".",
    "Metformin Hydrochloride 500 mg": ".",
    "Hydrochlorothiazide 25 mg": ".",
    "Hydroxychloroquine 200 mg": ".",
    "Isosorbide Dinitrate 10 mg": ".",
    "Isosorbide Dinitrate 5 mg": ".",
    "Ivabradine 5 mg": ".",
    "Lercanidipine 20 mg": ".",
    "Levothyroxine 50 mcg": ".",
    "Losartan Potassium 50 mg": ".",
    "Losartan Potassium 50 mg": ".",
    "Luseogliflozin 5 mg": ".",
    "Manidipine Hydrochloride 20 mg": ".",
    "Metformin 850 mg": ".",
    "Metformin Hydrochloride 500 mg": ".",
    "Metoprolol 100 mg": ".",
    "Montelukast 10 mg": ".",
    "Naproxen 250 mg": ".",
    "Nifedipine 20 mg": ".",
    "Nifedipine 5 mg": ".",
    "Nimodipine 30 mg": ".",
    "Omeprazole Capsules 20 mg": ".",
    "Paracetamol 450 mg": ".",
    "Pioglitazone 30 mg": ".",
    "Prasugrel 10 mg": ".",
    "Propranolol 10 mg": ".",
    "Propranolol 40 mg": ".",
    "Propranolol 10 mg": ".",
    "Pyrazinamide 500 mg": ".",
    "Ramipril 5 mg": ".",
    "Rifampicin 450 mg": ".",
    "Sertraline 50 mg": ".",
    "Simvastatin 20 mg": ".",
    "Sitagliptin 100 mg": ".",
    "Spironolactone 100 mg": ".",
    "Spironolactone 25 mg": ".",
    "Tamoxifen Film-coated tablets 20 mg": ".",
    "Tizanidine 2 mg": ".",
    "Tolvaptan 15 mg": ".",
    "Tramadol": ".",
    "Tramadol 50 mg ": ".",
    "Trimetazidine 35 mg": ".",
    "Valsartan 160 mg": ".",
    "Verapamil 240 mg": ".",
    "Verapamil 40 mg": ".",
    "Vitamin B Complex": ".",
    "Warfarin 2 mg": ".",
    "Warfarin 3 mg": ".",
    "Warfarin 5 mg": "."
    
}

# สร้างตัวจัดการเทมเพลต
templates = Jinja2Templates(directory="templates")

def convert_image_to_base64(image: Image) -> str:
    """แปลงภาพเป็น Base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("heart.html", {"request": request})

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
