import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from process.main import PlateRecognition

processor = PlateRecognition()
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

if __name__ == "__main__":
    while True:
        ret, frame = cap.read()
        vehicle_image, license_plate, info = processor.process_vehicular_plate(frame, True, True)
        print(f'license plate: {license_plate} \ninfo: {info}')
        cv2.imshow('result_process', vehicle_image)
        t = cv2.waitKey(5)
        if t == 27:
            break

    cap.release()
    cv2.destroyAllWindows()