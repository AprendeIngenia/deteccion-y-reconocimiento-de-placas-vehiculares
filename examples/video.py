import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from process.main import PlateRecognition

processor = PlateRecognition()
cap = cv2.VideoCapture('file.mp4')

if __name__ == "__main__":

    while True:
        ret, frame = cap.read()
        cv2.imshow('stream', frame)
        result_img, error = processor.process_vehicular_plate(frame)
        if error:
            print(error)
        else:
            cv2.imshow('result_process', result_img)
        t = cv2.waitKey(5)
        if t == 27:
            break

    cap.release()
    cv2.destroyAllWindows()