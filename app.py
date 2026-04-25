from flask import Flask, request, jsonify
import os
import cv2

from detector import detect_plate
from ocr import extract_text
from utils import clean_text

app = Flask(__name__)

BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "License Plate Recognition API Running"

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    filename = file.filename.replace(" ", "_")

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    image = cv2.imread(file_path)

    if image is None:
        return jsonify({"error": "image read error"}), 400

    # Step 1: detect car region (approx plate area)
    plates = detect_plate(file_path)

    results = []

    # Step 2: OCR
    for p in plates:
        text = extract_text(image, p["bbox"])
        text = clean_text(text)

        p["text"] = text
        results.append(p)

        # draw box
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, text, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, image)

    return jsonify({
        "count": len(results),
        "plates": results,
        "output_image": output_path
    })


if __name__ == "__main__":
    app.run(debug=True)