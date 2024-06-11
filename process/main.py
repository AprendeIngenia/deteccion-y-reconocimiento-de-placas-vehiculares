import numpy as np
import cv2
from process.computer_vision_models.main import (VehicleDetection, PlateSegmentation)
from process.ocr_extraction.main import TextExtraction


class PlateRecognition:
    def __init__(self):
        self.model_detect = VehicleDetection()
        self.model_segmentation = PlateSegmentation()
        self.process_text_extraction = TextExtraction()
        self.license_plate = ''

    def process_static_image(self, image_path: str, draw: bool):
        # Step 1: Load the image
        plate_image = cv2.imread(image_path)
        return self.process_vehicular_plate(plate_image, dynamic_image=False, draw=draw)

    def process_vehicular_plate(self, vehicle_image: np.ndarray, dynamic_image: bool, draw: bool):
        # step 1: check vehicle
        check_vehicle, info_vehicle, clean_image = self.model_detect.check_vehicle(vehicle_image)

        if check_vehicle is False:
            return vehicle_image, self.license_plate, 'no vehicle detected'

        # step 2: extract info
        vehicle_bbox, vehicle_type, vehicle_conf = self.model_detect.extract_detection_info(vehicle_image, info_vehicle)

        # step 3: draw detect (optional)
        if draw:
            vehicle_image = self.model_detect.draw_vehicle_detection(vehicle_image, vehicle_bbox, vehicle_type,
                                                                     vehicle_conf)
        # step 4: crop vehicle
        image_vehicle_crop = self.model_detect.image_vehicle_crop(vehicle_image, vehicle_bbox)

        # step 5: plate segmentation
        check_plate, info_plate = self.model_segmentation.check_vehicle_plate(image_vehicle_crop)

        if check_plate is False:
            return vehicle_image, self.license_plate, 'vehicle detected but no plate detected'

        # step 6: extract plate info
        plate_mask, plate_bbox, plate_conf = self.model_segmentation.extract_plate_info(image_vehicle_crop, info_plate)

        # step 7: draw segmentation (optional)
        if draw:
            vehicle_image = self.model_segmentation.draw_plate_segmentation(vehicle_image, plate_mask, vehicle_bbox)

        if dynamic_image:
            # step 8:
            plate_area = self.model_segmentation.calculate_mask_area(plate_mask)

            if 6500 > plate_area > 6000:
                # step 9: process mask
                processed_mask_image = self.model_segmentation.mask_processing(image_vehicle_crop, plate_mask)

                if processed_mask_image is None or processed_mask_image.size == 0:
                    return vehicle_image, self.license_plate, 'error: processed mask image is empty'

                # step 10: crop plate
                image_plate_crop = self.model_segmentation.image_plate_crop(processed_mask_image, plate_bbox)

                if image_plate_crop is None or image_plate_crop.size == 0:
                    return vehicle_image, self.license_plate, 'error: image_plate_crop is empty'

                # step 11: contrast plate
                image_plate_contrasted = self.process_text_extraction.image_contrast(image_plate_crop)

                # step 12: text extraction
                self.license_plate = self.process_text_extraction.text_extraction(image_plate_contrasted)

                return vehicle_image, self.license_plate, f'vehicle detected and plate detected'
            else:
                return vehicle_image, self.license_plate, f'vehicle detected and plate detected but is small'
        else:
            # step 8: process mask
            processed_mask_image = self.model_segmentation.mask_processing(image_vehicle_crop, plate_mask)

            # step 9: crop plate
            image_plate_crop = self.model_segmentation.image_plate_crop(processed_mask_image, plate_bbox)

            # step 10: contrast plate
            image_plate_contrasted = self.process_text_extraction.image_contrast(image_plate_crop)

            # step 11: text extraction
            self.license_plate = self.process_text_extraction.text_extraction(image_plate_contrasted)

            return vehicle_image, self.license_plate, f'vehicle detected and plate detected'


