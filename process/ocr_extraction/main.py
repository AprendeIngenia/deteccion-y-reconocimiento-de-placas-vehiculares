import cv2
import os
import numpy as np
from process.ocr_extraction.ocr import OcrProcess
from typing import List, Tuple, Union


class TextExtraction:
    def __init__(self):
        self.ocr = OcrProcess()
        self.min_vertical_distance = 12

    def clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(l)
        updated_lab_img2 = cv2.merge((clahe_img, a, b))
        CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
        return CLAHE_img

    def exposure_level(self, hist: np.ndarray) -> str:
        hist = hist / np.sum(hist)
        percent_over = np.sum(hist[200:])
        percent_under = np.sum(hist[:50])
        if percent_over > 0.75:
            return "Overexposed"
        elif percent_under > 0.75:
            return "Underexposed"
        else:
            return "Properly exposed"

    def image_contrast(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        if self.exposure_level(hist) == "Overexposed" or "Underexposed":
            img = self.clahe(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = cv2.Laplacian(gray, cv2.CV_64F).var()  # type: ignore
        if contrast < 100:
            img = cv2.equalizeHist(gray)
        return img

    def same_line(self, yi1, yi2):
        return abs(yi1 - yi2) < self.min_vertical_distance

    def process_text_line(self, text_detected) -> str:
        full_text = ''
        lines_list = []
        for i, text in enumerate(text_detected):
            text_bbox, text_extracted, text_confidence = self.ocr.extractor_text_line(text)
            lines_list.append(text_bbox)
            if i > 0:
                if self.same_line(lines_list[i][1], lines_list[i - 1][1]):
                    full_text += ' '
                else:
                    full_text += '\n'
            full_text += text_extracted
        return full_text

    def text_extraction(self, plate_image_crop: np.ndarray) -> str:
        number_line_text, text_detected = self.ocr.text_detection(plate_image_crop)
        full_text = self.process_text_line(text_detected)
        return full_text
