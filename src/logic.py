import cv2
import numpy as np
import shapely.geometry

def check_ppe_compliance(person_box, helmets, vests):
    """
    Checks if a person has a helmet and vest.
    person_box: [x1, y1, x2, y2]
    helmets: list of [x1, y1, x2, y2]
    vests: list of [x1, y1, x2, y2]
    
    Returns: (has_helmet, has_vest)
    """
    x1, y1, x2, y2 = person_box
    
    # Head ROI: Top 1/5th
    # This is where we expect the helmet to be
    height = y2 - y1
    head_height = int(height / 5)
    head_roi = [x1, y1, x2, y1 + head_height]
    
    has_helmet = False
    for h_box in helmets:
        # Check if center of helmet is within Head ROI
        hx_center = (h_box[0] + h_box[2]) / 2
        hy_center = (h_box[1] + h_box[3]) / 2
        
        if (head_roi[0] < hx_center < head_roi[2]) and (head_roi[1] < hy_center < head_roi[3]):
            has_helmet = True
            break
            
    has_vest = False
    for v_box in vests:
        # Check if center of vest is within Person Box (loosely)
        vx_center = (v_box[0] + v_box[2]) / 2
        vy_center = (v_box[1] + v_box[3]) / 2
        
        if (x1 < vx_center < x2) and (y1 < vy_center < y2):
            has_vest = True
            break
            
    return has_helmet, has_vest

def check_geofence(person_centroid, polygon_pts):
    """
    Checks if person_centroid is inside the polygon.
    person_centroid: (x, y)
    polygon_pts: list of (x, y) tuples
    """
    # Using OpenCV pointPolygonTest
    # polygon_pts needs to be numpy array of shape (N, 1, 2)
    pts = np.array(polygon_pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # measureDist=False returns +1 (inside), -1 (outside), 0 (on edge)
    dist = cv2.pointPolygonTest(pts, person_centroid, False)
    return dist >= 0
