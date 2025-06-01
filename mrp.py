from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import pytesseract
import numpy as np
import re

app = FastAPI()

# Cleans OCR noise from a single line
def clean_line(line: str) -> str:
    return re.sub(r'[^A-Z0-9 ,./\-]', '', line.upper()).strip()

# Improved extraction using sliding window over cleaned lines
def extract_structured_data(text: str):
    entries = []
    lines = [clean_line(line) for line in text.split("\n") if clean_line(line)]

    i = 0
    while i < len(lines) - 2:
        window = lines[i:i+4]  # Look at up to 4 lines at a time
        customer = None
        address = None
        product = None

        for line in window:
            # Identify customer name
            if not customer and re.search(r'[A-Z]+, [A-Z]+', line):
                customer = line.title()

            # Identify address
            if not address and re.search(r'\d{1,5} [A-Z ]+ (RD|AVE|LANE|PLACE|DR|ROAD|CIR|WAY|HILL)', line):
                address = line.title()

            # Identify product/task type
            if not product and re.search(r'(TUESDAY|MAINT|OPENING|VACUUM)', line):
                product = line.title()

        if customer and product:
            entries.append({
                "customer": customer,
                "address": address or "",
             "product": product,
                "quantity": 1,
                "completed": "N"
            })
            i += 3  # Move to next likely entry
        else:
            i += 1  # Slide window down one line

    return entries

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        # === Image preprocessing for better OCR ===
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 10
        )
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # === OCR with pytesseract ===
        raw_text = pytesseract.image_to_string(dilated)

        print("\n==== RAW OCR TEXT ====\n")
        print(raw_text)
        print("\n======================\n")

        # === Structured Data Extraction ===
        structured_data = extract_structured_data(raw_text)

        return JSONResponse(content={
            "entries": structured_data,
            "raw_text": raw_text
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
