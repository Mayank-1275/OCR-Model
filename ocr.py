import cv2
import re
from paddleocr import PaddleOCR

# Initialize PaddleOCR (latest versions: no show_log argument)
ocr_reader = PaddleOCR(lang='en', use_textline_orientation=True)

# Character mappings to fix OCR mistakes
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def license_complies_format(text):
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
    return re.match(pattern, text) is not None

def format_license(text):
    license_plate_ = ''
    mapping = {
        0: dict_int_to_char, 1: dict_int_to_char,
        2: dict_char_to_int, 3: dict_char_to_int,
        4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char
    }
    for j in range(len(text)):
        if j in mapping and text[j] in mapping[j]:
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_

def read_license_plate(img):
    results = ocr_reader.predict(img)

    if results and 'rec_texts' in results[0]:
        rec_texts = results[0]['rec_texts']
        rec_scores = results[0]['rec_scores']

        best_text = None
        best_score = 0

        for text, score in zip(rec_texts, rec_scores):
            text = text.upper().replace(' ', '')

            if license_complies_format(text):
                return format_license(text), score

            if score > best_score:
                best_text, best_score = text, score

        return best_text, best_score

    return None, 0

# ------------------- Main -------------------
def for_ocr(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Image not found at:", img_path)
    else:
        plate_number, confidence = read_license_plate(img)
        if plate_number:
            print("✅ Vehicle Number:", plate_number)
            print("Confidence:", confidence)
        else:
            print("No license plate detected")
