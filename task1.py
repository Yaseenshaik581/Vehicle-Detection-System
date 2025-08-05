import cv2
import numpy as np

# Step 1: Load the Video
video_path = r'C:\Users\SHAIK YASEEN\Desktop\Vehicle-detection-master\assets\video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Step 2: Process Video Frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Step 3: Preprocess Frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    edges = cv2.Canny(blurred_frame, 50, 150)  # Edge detection using Canny

    # Step 4: Find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Step 5: Filter by Size (to avoid small noise)
        area = cv2.contourArea(contour)
        if area < 500:  # Threshold to ignore small contours
            continue

        # Step 6: Draw Bounding Box Around Contour
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Step 7: Heuristic Vehicle Type Detection
        if area > 10000 and aspect_ratio > 1.2:
            vehicle_type = "Bus or Truck"
        elif 5000 < area <= 10000 and aspect_ratio > 1.2:
            vehicle_type = "Car"
        elif area < 5000 and aspect_ratio <= 1.2:
            vehicle_type = "Motorcycle"
        else:
            vehicle_type = "vehicle"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Vehicle Type: {vehicle_type}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Step 8: Display Frame
    cv2.imshow('Vehicle Detection', frame)

    # Step 9: Break on 'q' Key Press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture Object and Close All Windows
cap.release()
cv2.destroyAllWindows()
