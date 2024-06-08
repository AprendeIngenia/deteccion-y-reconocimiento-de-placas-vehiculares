import numpy as np
import cv2
from process.computer_vision_models.main import ComputerVisionModels


class PlateRecognition:
    def __init__(self):
        self.models = ComputerVisionModels()

    def process_static_image(self, image_path: str):
        # Step 1: Load the image
        plate_image = cv2.imread(image_path)
        return self.process_vehicular_plate(plate_image, stream_mode=False)

    def process_vehicular_plate(self, vehicle_image: np.ndarray, stream_mode: bool):
        # step 1: vehicle detection
        vehicle_image, bbox, vehicle_type, confidence = self.models.vehicle_detection(vehicle_image, mode=stream_mode, draw=False)
        return vehicle_image, bbox, vehicle_type, confidence

