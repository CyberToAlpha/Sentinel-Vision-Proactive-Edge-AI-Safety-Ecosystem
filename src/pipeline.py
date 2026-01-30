from ultralytics import YOLO
import cv2
import numpy as np
import time
from .preprocessing import apply_clahe
from .logic import check_ppe_compliance, check_geofence
from .privacy import blur_face
from .logger import SafetyLogger

class SafetyPipeline:
    def __init__(self, ppe_model_path='safety_model.pt', person_model_path='yolov8n.pt'):
        print("Loading models...")
        self.ppe_model = YOLO(ppe_model_path)
        self.person_model = YOLO(person_model_path) # Standard YOLOv8n for people
        self.logger = SafetyLogger()
        self.last_log_time = 0 # Rate limiter for logging

        
        # Determine class IDs for PPE
        self.ppe_names = self.ppe_model.names
        self.hard_hat_id = None
        self.vest_id = None
        
        # Flexible matching
        for k, v in self.ppe_names.items():
            name = v.lower()
            if 'hard' in name or 'helmet' in name:
                self.hard_hat_id = k
            if 'vest' in name:
                self.vest_id = k
                
        print(f"PPE Classes Map: HardHat={self.hard_hat_id}, Vest={self.vest_id}")
        
        # Person class ID in COCO is 0
        self.person_id = 0 

    def process_frame(self, frame, zone_polygon=None, auto_blur=True):
        """
        Main pipeline processing.
        frame: BGR image
        zone_polygon: list of (x,y) or None
        auto_blur: bool
        """
        # 1. Preprocessing (CLAHE)
        # processed_frame = apply_clahe(frame) 
        # Note: CLAHE changes colors, might affect detection? 
        # User said: "CLAHE -> YOLOv8", so yes.
        processed_frame = apply_clahe(frame)
        
        # 2. Person Detection & Tracking
        # Track persons. persist=True for ID tracking
        person_results = self.person_model.track(processed_frame, persist=True, classes=[self.person_id], verbose=False)
        
        # 3. PPE Detection
        # Predict on same frame (or CLAHE frame)
        ppe_results = self.ppe_model.predict(processed_frame, verbose=False)
        
        # Parse PPE detections
        helmets = []
        vests = []
        
        for r in ppe_results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy()
                if cls_id == self.hard_hat_id:
                    helmets.append(xyxy)
                elif cls_id == self.vest_id:
                    vests.append(xyxy)
                    
        # 4. Logic & Association
        annotated_frame = frame.copy() # Draw on original frame (or processed one?) usually original.
        
        # Draw Zone
        if zone_polygon:
            pts = np.array(zone_polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [pts], True, (0, 0, 255), 2)
        
        if person_results and len(person_results) > 0:
             # Iterate over tracks
            for r in person_results:
                boxes = r.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    # Get Person ID and Box
                    track_id = int(box.id[0]) if box.id is not None else -1
                    px1, py1, px2, py2 = box.xyxy[0].cpu().numpy()
                    person_box = [px1, py1, px2, py2]
                    
                    # Logic: Check PPE
                    has_helmet, has_vest = check_ppe_compliance(person_box, helmets, vests)
                    
                    # Logic: Check Geofence
                    in_danger_zone = False
                    if zone_polygon:
                        centroid = ((px1 + px2)/2, (py1 + py2)/2)
                        in_danger_zone = check_geofence(centroid, zone_polygon)
                    
                    # Determine Status
                    is_safe = has_helmet and has_vest
                    status_color = (0, 255, 0) if is_safe else (0, 0, 255) # Green vs Red
                    
                    if in_danger_zone:
                        status_color = (0, 0, 255) # Red if in danger zone
                        
                    # Privacy Blur
                    # Logic: IF (Violation == True): Disable Blur. ELSE: Keep Blurred.
                    # Violation = Not Safe OR In Danger Zone? 
                    # User said: "IF (Violation == True): Disable Blur (Unmask for Evidence). ELSE: Keep Blurred."
                    violation = not is_safe or in_danger_zone
                    
                    if auto_blur and not violation:
                        annotated_frame = blur_face(annotated_frame, person_box)
                        
                    # Logging (Rate Limited to once per 2 seconds to avoid spam)
                    if violation and (time.time() - self.last_log_time > 2.0):
                        v_type = []
                        if not has_helmet: v_type.append("NoHelmet")
                        if not has_vest: v_type.append("NoVest")
                        if in_danger_zone: v_type.append("ZoneIntrusion")
                        
                        if v_type:
                            self.logger.log_violation(",".join(v_type), frame) # Log original frame or annotated? usually original frame is better for evidence, but we want context. Let's log annotated for now or maybe both? User said "Snapshot Path". Let's log annotated frame so we see what the AI saw.
                            # Actually, user wants "Snapshot". Usually raw evidence is preferred, but for hackathon, annotated is clearer.
                            # Let's stick to annotated for the "wow" factor.
                            self.logger.log_violation(",".join(v_type), annotated_frame)
                            self.last_log_time = time.time()
                    
                    # Annotate
                    label = f"ID:{track_id} "
                    if not has_helmet: label += "NoHelmet "
                    if not has_vest: label += "NoVest "
                    if in_danger_zone: label += "ZONE!"
                    if is_safe and not in_danger_zone: label += "SAFE"
                    
                    cv2.rectangle(annotated_frame, (int(px1), int(py1)), (int(px2), int(py2)), status_color, 2)
                    cv2.putText(annotated_frame, label, (int(px1), int(py1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        return annotated_frame
