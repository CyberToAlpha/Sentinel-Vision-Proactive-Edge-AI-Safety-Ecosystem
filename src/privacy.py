import cv2

def blur_face(frame, person_box):
    """
    Blurs the face region of the detected person.
    person_box: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = map(int, person_box)
    
    # Face ROI: Top 1/8th of the person bounding box
    height = y2 - y1
    face_height = int(height / 8)
    
    # Define face region
    # slightly padded width? No, just use full width of person box for simplicity or narrow it?
    # Usually heads are centered. Let's take full width but top 1/8th.
    
    roi_y2 = min(frame.shape[0], y1 + face_height)
    roi_y1 = max(0, y1)
    roi_x1 = max(0, x1)
    roi_x2 = min(frame.shape[1], x2)
    
    if roi_y2 > roi_y1 and roi_x2 > roi_x1:
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        # Strong blur
        # Kernel size must be odd
        ksize = 51
        roi = cv2.GaussianBlur(roi, (ksize, ksize), 30)
        frame[roi_y1:roi_y2, roi_x1:roi_x2] = roi
        
    return frame
