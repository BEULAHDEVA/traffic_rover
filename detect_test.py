from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # nano model, downloads automatically
cap = cv2.VideoCapture("sample.mp4")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    results = model(frame, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck
    cv2.imshow("detections", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()