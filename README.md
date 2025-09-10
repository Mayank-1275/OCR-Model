# Automatic Number Plate Detection and Extraction

This project detects vehicle license plates using a **YOLO model**
(`.pt` file) and extracts the detected plates for further **OCR (Optical
Character Recognition)** processing.

## 🚀 Features

-   License plate detection using YOLOv8.\
-   Crops and saves number plates from input images.\
-   OCR integration for text extraction.\
-   Organized folders for raw and processed plates.\
-   Easy to extend for real-time video input.

------------------------------------------------------------------------

## 📂 Project Structure

    project/
    │── main.py                  # Main script
    │── license_plate_detector.pt # YOLO model weights
    │── ocr.py                   # OCR helper function
    │── images/                  # Input images
    │── plates/                  # Cropped plates (output)
    │── processed_plates/        # Preprocessed plates (output)

------------------------------------------------------------------------

## 🛠️ Installation

1.  Clone the repository:

``` bash
git clone https://github.com/your-username/number-plate-detection.git
cd number-plate-detection
```

2.  Install required dependencies:

``` bash
pip install ultralytics opencv-python numpy
```

> ⚠️ If OCR uses `pytesseract`, install it too:

``` bash
pip install pytesseract
```

And download [Tesseract
OCR](https://github.com/UB-Mannheim/tesseract/wiki) if not already
installed.

------------------------------------------------------------------------

## ▶️ Usage

1.  Place your test images inside the `images/` folder.\
    Example: `images/1.jpeg`

2.  Run the script:

``` bash
python main.py
```

3.  Output:
    -   Cropped plates will be saved inside `plates/`.\
    -   Processed/enhanced plates (if applied) will be saved inside
        `processed_plates/`.\
    -   Extracted text (from OCR) will be printed in the terminal.

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Support for **real-time video streams**.\
-   Improved preprocessing for higher OCR accuracy.\
-   Web or mobile app integration.
