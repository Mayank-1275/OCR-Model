from ultralytics import YOLO
import cv2
import os
import numpy as np
from ocr import for_ocr

# Step 1: Load YOLO model
model = YOLO("license_plate_detector.pt")

# Step 2: Input image
img_path = os.path.join("images", "1.jpeg")
img = cv2.imread(img_path)

# Step 3: Run inference
results = model(img)

# Step 4: Create output folders
os.makedirs("plates", exist_ok=True)             # raw cropped plates
os.makedirs("processed_plates", exist_ok=True)   # enhanced plates

# Base name of input image (without extension)
base_name = os.path.splitext(os.path.basename(img_path))[0]

for i, r in enumerate(results):
    boxes = r.boxes.xyxy.cpu().numpy()  
    for j, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])  
        plate_crop = img[y1:y2, x1:x2]

        # # ==== Tight cropping using contour detection ====
        # gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # if contours:
        #     c = max(contours, key=cv2.contourArea)
        #     x, y, w, h = cv2.boundingRect(c)
        #     plate_crop = plate_crop[y:y+h, x:x+w]

        # ==== Save raw cropped plate ====
        raw_path = f"plates/{base_name}_plate.jpg"
        cv2.imwrite(raw_path, plate_crop)
        print(f"Raw plate saved at: {raw_path}")

        for_ocr(raw_path)
        

    