import cv2
import numpy as np
import math
from typing import List, Any
from ultralytics import YOLO
from process.computer_vision_models.models.config import ConfigModels


class VehicleDetection:
    def __init__(self):
        self.models = ConfigModels()

        # detection
        self.detection_model = YOLO(self.models.vehicle_model)
        self.detection_classes = self.models.vehicle_classes
        self.color = self.models.vehicle_color

    def check_vehicle(self, vehicle_image: np.ndarray, mode: bool):
        detect = False
        results = self.detection_model(vehicle_image, stream=mode, conf=0.85)
        for res in results:
            boxes = res.boxes
            for box in boxes:
                detect = int(box.cls[0])
        if detect is False:
            return False, results
        else:
            return True, results

    def extract_detection_info(self, vehicle_image: np.ndarray, detect_info: Any):
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

    # draw
    def draw_vehicle_detection(self, vehicle_image: np.ndarray, bbox: List[int], vehicle_type: str, conf: float):
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

