from ultralytics import YOLO

model = YOLO('safety_model.pt')
print("Model Classes:", model.names)
