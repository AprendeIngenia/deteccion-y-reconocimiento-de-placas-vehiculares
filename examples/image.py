import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from process.main import PlateRecognition


if __name__ == "__main__":
    processor = PlateRecognition()
    image_path = 'examples/image_example.jpeg'
    vehicle_image, license_plate, info = processor.process_static_image(image_path, draw=True)
    print(f'license plate: {license_plate} \ninfo: {info}')
    cv2.imshow('plate recognition', vehicle_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()