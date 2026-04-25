import easyocr
import cv2

reader = easyocr.Reader(['en'])

def extract_text(image, bbox):

    x1, y1, x2, y2 = bbox
    plate = image[y1:y2, x1:x2]

    if plate.size == 0:
        return ""

    # improve OCR
    plate = cv2.resize(plate, None, fx=2, fy=2)

    result = reader.readtext(plate)

    if len(result) > 0:
        return result[0][1]

    return ""