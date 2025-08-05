import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO(r'F:\year 3\sem 2\MVI\assinment\best.pt')  # Replace 'best.pt' with your trained model path

# Access the webcam
cap = cv2.VideoCapture(0)  # '0' is the default webcam

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model.predict(frame, conf=0.5)  # You can adjust confidence threshold

    # Annotate frame with the results
    annotated_frame = results[0].plot()

    # Extract results to identify objects
    for result in results:
        for box in result.boxes:
            cls = box.cls  # Get the class index
            label = model.names[int(cls)]  # Get the label name for the class

            # Draw the label on the frame if it's a vehicle type of interest
            if label in ['car', 'bus', 'motorcycle', 'truck']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates of bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow('Webcam Vehicle Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
