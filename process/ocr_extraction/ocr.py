import numpy as np
import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
from typing import List, Tuple, Union, Any


class OcrProcess:
    def __init__(self):
        # tr_ocr
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        self.ocr_extractor = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed").to(
            torch.device("cuda"))
        # easyocr
        self.ocr_detector = easyocr.Reader(['es'], gpu=True)

        self.text_bbox: list = []
        self.text_extracted: str = ''
        self.text_confidence: float = 0.0

    def text_detection(self, text_image: np.ndarray):
        text_line_detected = self.ocr_detector.readtext(text_image)
        return len(text_line_detected), text_line_detected

    def extractor_text_line(self, text) -> Tuple[List[int], str, float]:
        bbox, self.text_extracted, self.text_confidence = text
        xi, yi, xf, yf = int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])
        self.text_bbox = [xi, yi, xf, yf]
        return self.text_bbox, self.text_extracted, self.text_confidence

    def image_to_text(self, img: np.ndarray):
        pixel_values = self.processor(img, return_tensors="pt").pixel_values.to(torch.device("cuda"))
        generated_ids = self.ocr_extractor.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
