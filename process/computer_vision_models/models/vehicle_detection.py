from typing import List

vehicle_detect_model: str = 'process/computer_vision_models/models/vehicle_detection.pt'
vehicle_detect_classes: List[str] = ['bus', 'car', 'microbus', 'motorbike', 'pickup-van', 'truck']
vehicle_color: dict = {'bus': [255, 0, 0],
                       'car': [0, 255, 0],
                       'microbus': [0, 0, 255],
                       'motorbike': [255, 255, 0],
                       'pickup-van': [0, 255, 255],
                       'truck': [255, 0, 255]}
