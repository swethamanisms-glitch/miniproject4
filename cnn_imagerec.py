import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

classifier = MobileNetV2(weights="imagenet")
detector = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mode = "yolo"

while camera.isOpened():
    ok, frame = camera.read()
    if not ok:
        break

    if mode == "yolo":
        frame = detector(frame, verbose=False)[0].plot()
    else:
        rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        img = preprocess_input(np.expand_dims(rgb.astype(float), axis=0))
        _, name, score = decode_predictions(classifier.predict(img, verbose=0), top=1)[0][0]
        cv2.putText(frame, f"{name} {score:.0%}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("m"):
        mode = "mobilenet"
    elif key == ord("y"):
        mode = "yolo"

camera.release()
cv2.destroyAllWindows()