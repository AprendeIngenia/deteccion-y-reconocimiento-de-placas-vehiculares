from pydantic import BaseModel
from typing import List
from process.computer_vision_models.models.vehicle_detection import (vehicle_detect_model, vehicle_detect_classes, vehicle_color)
from process.computer_vision_models.models.plate_segmentation import (plate_segmentation_model, plate_segmentation_classes)


class ConfigModels(BaseModel):
    # vehicle detect
    vehicle_model: str = vehicle_detect_model
    vehicle_classes: List[str] = vehicle_detect_classes
    vehicle_color: dict = vehicle_color
    # plate segmentation
    plate_model: str = plate_segmentation_model
    plate_classes: List[str] = plate_segmentation_classes
