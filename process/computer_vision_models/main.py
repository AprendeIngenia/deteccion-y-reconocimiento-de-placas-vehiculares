import cv2
import numpy as np
import math
import torch
from typing import List, Any, Tuple
from ultralytics import YOLO
from process.computer_vision_models.models.config import ConfigModels


class VehicleDetection:
    def __init__(self):
        self.models = ConfigModels()

        # detection
        self.detection_model = YOLO(self.models.vehicle_model)
        self.detection_classes = self.models.vehicle_classes
        self.color = self.models.vehicle_color

    def check_vehicle(self, vehicle_image: np.ndarray) -> Tuple[bool, Any, np.ndarray]:
        clean_image = vehicle_image.copy()
        detect = False
        results = self.detection_model(vehicle_image, stream=False, conf=0.60)
        for res in results:
            boxes = res.boxes
            for box in boxes:
                cls = int(box.cls[0])
                cls = self.detection_classes[cls]
                if cls in self.color:
                    detect = True
        if detect is False:
            return False, results, clean_image
        else:
            return True, results, clean_image

    def extract_detection_info(self, vehicle_image: np.ndarray, detect_info: Any) -> Tuple[list, str, float]:
        height, width, _ = vehicle_image.shape
        bbox: List = []
        cls: int = 0
        conf: float = 0.0
        vehicle_type: str = ''

        for res in detect_info:
            boxes = res.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                bbox = [x1, y1, x2, y2]

                cls = int(box.cls[0])
                vehicle_type = self.detection_classes[cls]
                conf = math.ceil(box.conf[0])
        return bbox, vehicle_type, conf

    def image_vehicle_crop(self, vehicle_image: np.ndarray, bbox: List[int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return vehicle_image[y1:y2, x1:x2]

    # draw
    def draw_vehicle_detection(self, vehicle_image: np.ndarray, bbox: List[int], vehicle_type: str, conf: float) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        color = self.color.get(vehicle_type)
        vehicle_info = f'vehicle type: {vehicle_type} {conf*100}%'
        cv2.putText(vehicle_image, vehicle_info, (x1+50, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.50, color, 1)
        cv2.rectangle(vehicle_image, (x1, y1), (x2, y2), color, 2)
        return vehicle_image


class PlateSegmentation:
    def __init__(self):
        self.models = ConfigModels()
        # segmentation
        self.segmentation_model = YOLO(self.models.plate_model)
        self.segmentation_classes = self.models.plate_classes

        self.best_mask = None

    def check_vehicle_plate(self, crop_vehicle_image: np.ndarray) -> Tuple[bool, Any]:
        segment = None
        results = self.segmentation_model(crop_vehicle_image, stream=False, conf=0.60)
        for res in results:
            segment = res.masks

        if segment is None:
            return False, results
        else:
            return True, results

    def extract_plate_info(self, crop_vehicle_image: np.ndarray, mask_info: Any) -> Tuple[list, list, float]:
        height, width, _ = crop_vehicle_image.shape
        max_confidence = 0
        best_segment = None
        best_pos = 0
        for segment in mask_info:
            for i, boxes in enumerate(segment.boxes):
                if boxes.conf[0] > max_confidence:
                    best_pos = i
                    best_segment = segment
                    max_confidence = boxes.conf[0]

        self.best_mask = best_segment.masks[best_pos]
        best_box = best_segment.boxes[best_pos]

        x1, y1, x2, y2 = best_box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        bbox = [x1, y1, x2, y2]

        return self.best_mask, bbox, max_confidence

    def image_plate_crop(self, crop_vehicle_image: np.ndarray, plate_bbox: List[int]) -> np.ndarray:
        h, w, _ = crop_vehicle_image.shape
        offset_x, offset_y = int(w * 0.025), int(h * 0.025)
        xi, yi, xf, yf = plate_bbox
        xi, yi, xf, yf = xi - offset_x, yi - offset_y, xf + offset_x, yf + offset_y
        return crop_vehicle_image[yi:yf, xi:xf]

    def mask_processing(self, crop_plate_image: np.ndarray, plate_mask: Any) -> np.ndarray:
        h, w, _ = crop_plate_image.shape
        m = torch.squeeze(plate_mask.data)
        composite = torch.stack((m, m, m), 2)
        composite = cv2.resize(composite.cpu().numpy(), (w, h))
        img = crop_plate_image.astype(np.uint8)
        tmp = img * composite.astype(np.uint8)
        return tmp

    def calculate_mask_area(self, plate_mask: Any) -> int:
        mask = torch.squeeze(plate_mask.data).cpu().numpy()
        area = np.sum(mask)
        return area

    def draw_plate_segmentation(self, vehicle_image: np.ndarray, plate_mask: Any, vehicle_bbox: List[int]) -> np.ndarray:
        mask = torch.squeeze(plate_mask.data).cpu().numpy() * 255
        mask = mask.astype(np.uint8)
        color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_INFERNO)
        color_mask_resized = cv2.resize(color_mask, (vehicle_bbox[2] - vehicle_bbox[0], vehicle_bbox[3] - vehicle_bbox[1]))
        blank_mask = np.zeros_like(vehicle_image)
        blank_mask[vehicle_bbox[1]:vehicle_bbox[3], vehicle_bbox[0]:vehicle_bbox[2]] = color_mask_resized
        overlay = cv2.addWeighted(vehicle_image, 1, blank_mask, 0.5, 0)
        return overlay




