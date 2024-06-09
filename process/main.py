import numpy as np
import cv2
from process.computer_vision_models.main import (VehicleDetection, PlateSegmentation)


class PlateRecognition:
    def __init__(self):
        self.model_detect = VehicleDetection()
        self.model_segmentation = PlateSegmentation()

    def process_static_image(self, image_path: str, draw: bool):
        # Step 1: Load the image
        plate_image = cv2.imread(image_path)
        return self.process_vehicular_plate(plate_image, stream_mode=False, draw=draw)

    def process_vehicular_plate(self, vehicle_image: np.ndarray, stream_mode: bool, draw: bool):
        # step 1: check vehicle
        check_vehicle, info_vehicle, clean_image = self.model_detect.check_vehicle(vehicle_image, mode=stream_mode)

        if check_vehicle is False:
            return vehicle_image, 'no vehicle detected'

        # step 2: extract info
        vehicle_bbox, vehicle_type, vehicle_conf = self.model_detect.extract_detection_info(vehicle_image, info_vehicle)

        # step 3: draw (optional)
        if draw:
            vehicle_image = self.model_detect.draw_vehicle_detection(vehicle_image, vehicle_bbox, vehicle_type,
                                                                     vehicle_conf)
        # step 4: crop vehicle
        image_vehicle_crop = self.model_detect.image_vehicle_crop(vehicle_image, vehicle_bbox)

        # step 5: plate segmentation
        check_plate, info_plate = self.model_segmentation.check_vehicle_plate(image_vehicle_crop, mode=stream_mode)

        if check_plate is False:
            return vehicle_image, 'vehicle detected but no plate detected'

        # step 6: extract plate info
        plate_mask, plate_bbox, plate_conf = self.model_segmentation.extract_plate_info(image_vehicle_crop, info_plate)

        # step 7: process mask
        processed_mask_image = self.model_segmentation.mask_processing(image_vehicle_crop, plate_mask)

        # step 8: crop plate
        image_plate_crop = self.model_segmentation.image_plate_crop(processed_mask_image, plate_bbox)

        return image_plate_crop, 'vehicle detected and plate detected'

