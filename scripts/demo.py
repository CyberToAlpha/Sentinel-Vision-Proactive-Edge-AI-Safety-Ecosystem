from ultralytics import YOLO
import cv2

# Load the model directly (Ultralytics handles the HuggingFace link automatically)
model = YOLO('safety_model.pt')

# Open Webcam (Source 0)
cap = cv2.VideoCapture(0)

print("Starting Demo... Press 'q' to exit.")

# Loop through video
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read from webcam.")
        break

    # Run Inference
    # conf=0.5 (Standard threshold)
    # show=True (We just want to show it)
    results = model.predict(frame, conf=0.5, show=True)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
