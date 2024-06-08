import numpy as np
import cv2
from process.computer_vision_models.main import VehicleDetection


class PlateRecognition:
    def __init__(self):
        self.model_detect = VehicleDetection()

    def process_static_image(self, image_path: str, draw: bool):
        # Step 1: Load the image
        plate_image = cv2.imread(image_path)
        return self.process_vehicular_plate(plate_image, stream_mode=False, draw=draw)

    def process_vehicular_plate(self, vehicle_image: np.ndarray, stream_mode: bool, draw: bool):
        # step 1: check vehicle
        check_vehicle, info_vehicle = self.model_detect.check_vehicle(vehicle_image, mode=stream_mode)
        if check_vehicle is False:
            return vehicle_image, 'no vehicle detected'
        # step 2: extract info
        vehicle_bbox, vehicle_type, vehicle_conf = self.model_detect.extract_detection_info(vehicle_image, info_vehicle)

        # step 3: draw (optional)
        if draw:
            vehicle_image = self.model_detect.draw_vehicle_detection(vehicle_image, vehicle_bbox, vehicle_type,
                                                                     vehicle_conf)
        return vehicle_image, 'vehicle detected'

