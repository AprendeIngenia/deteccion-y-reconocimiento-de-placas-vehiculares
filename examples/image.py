import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from process.main import PlateRecognition


if __name__ == "__main__":
    processor = PlateRecognition()
    image_path = 'examples/43.jpg'
    analysis, error = processor.process_static_image(image_path)
    if error:
        print(error)

    cv2.waitKey(0)
    cv2.destroyAllWindows()