import cv2
import pytesseract
import re
import numpy as np
from scipy import ndimage
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import uuid

app = FastAPI()

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------------- YOUR ORIGINAL FUNCTIONS ----------------

def rotate(image, center=None, scale=1.0):
    angle = int(re.search(r'(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    rotated = ndimage.rotate(image, float(angle) * -1)
    return rotated


def preprocessing(image):
    w, h = image.shape[0], image.shape[1]
    if w < h:
        image = rotate(image)

    resized_image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    grey_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.medianBlur(grey_image, 3)

    thres_image = cv2.adaptiveThreshold(
        blur_image, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 13, 7
    )
    return thres_image, resized_image


def aadhar_mask_and_ocr(thres_image, resized_image):
    d = pytesseract.image_to_data(thres_image, output_type=pytesseract.Output.DICT)
    number_pattern = r"(?<!\d)\d{4}(?!\d)"
    n_boxes = len(d['text'])

    c = 0
    temp = []
    UID = []
    final_image = resized_image.copy()

    for i in range(n_boxes):
        if int(d['conf'][i]) > 20:
            if re.match(number_pattern, d['text'][i]):

                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]

                if c < 2:
                    cv2.rectangle(final_image, (x, y), (x + w, y + h), (255, 255, 255), -1)
                    temp.append(d['text'][i])
                    c += 1

                elif (c >= 2) and (d['text'][i] in temp):
                    cv2.rectangle(final_image, (x, y), (x + w, y + h), (255, 255, 255), -1)

                elif c == 2:
                    UID = temp + [d['text'][i]]
                    c += 1

    final_image = cv2.resize(final_image, None, fx=0.33, fy=0.33)
    return final_image, UID


# ---------------- API ENDPOINT ----------------

@app.post("/mask-aadhaar")
async def mask_aadhaar(file: UploadFile = File(...)):

    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse({"status": "error", "message": "Invalid image file"})

    thres_image, resized_image = preprocessing(image)
    masked_image, UID = aadhar_mask_and_ocr(thres_image, resized_image)

    if len(UID) < 3:
        return JSONResponse({
            "status": "error",
            "message": "Aadhaar not detected clearly",
            "detected_parts": UID
        })

    filename = f"masked_{uuid.uuid4()}.png"
    cv2.imwrite(filename, masked_image)

    return JSONResponse({
        "status": "success",
        "uid_parts": UID,
        "masked_image_file": filename
    })
