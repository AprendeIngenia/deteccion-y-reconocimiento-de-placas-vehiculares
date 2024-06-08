import numpy as np
import cv2
from vehicle_detection.main import VehicleDetector


class PlateRecognition:
    def __init__(self):
        self.vehicles = VehicleDetector()
        pass

    def process_static_image(self, image_path: str):
        # Step 1: Load the image
        plate_image = cv2.imread(image_path)
        return self.process_vehicular_plate(plate_image)

    def process_vehicular_plate(self, image: np.ndarray):
        # step 1: vehicle detection
        vehicle_type = self.vehicles.detector()

