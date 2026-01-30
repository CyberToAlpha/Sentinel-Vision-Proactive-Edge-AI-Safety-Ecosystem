import streamlit as st
import cv2
import numpy as np
from src.pipeline import SafetyPipeline
import tempfile

st.set_page_config(page_title="Sentinel Vision AI", layout="wide")

st.title("Sentinel Vision: Construction Safety AI")
st.sidebar.title("Settings")

# Settings
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
use_clahe = st.sidebar.checkbox("Use Night Vision (CLAHE)", True)
enable_blur = st.sidebar.checkbox("Enable Privacy Blur", True)

# Zone Configuration
st.sidebar.subheader("Danger Zone Configuration")
# Simple default zone for demo
zone_active = st.sidebar.checkbox("Active Danger Zone", False)

# Initialize Pipeline
@st.cache_resource
def get_pipeline():
    return SafetyPipeline()

pipeline = get_pipeline()

import json
import os

# Load Configuration
config_path = "config.json"
danger_zone = []
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        danger_zone = config.get("danger_zone", [])
else:
    # Default fallback
    danger_zone = [(300, 300), (600, 300), (600, 600), (300, 600)]

if zone_active and not danger_zone:
    st.warning("Zone active but no zone defined!")

# Input Source
st.sidebar.subheader("Input Source")
input_source = st.sidebar.radio("Source", ["Webcam", "Video File"])

if st.button("Start System"):
    stframe = st.empty()
    
    cap = None
    if input_source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        # File uploader
        # For simplicity in this demo loop, we might need to handle file upload separately
        # But here we just assume webcam for 'Start System' button or handle file
        pass
        
    if input_source == "Video File":
        uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
             tfile = tempfile.NamedTemporaryFile(delete=False) 
             tfile.write(uploaded_file.read())
             cap = cv2.VideoCapture(tfile.name)
    
    if cap is None or not cap.isOpened():
        st.error("Could not open video source.")
    else:
        stop_button = st.button("Stop")
        
        # Dashboard placeholders
        col1, col2 = st.columns([3, 1])
        
        with col1:
            stframe = st.empty()
            
        with col2:
            st.subheader("Real-time Alerts")
            alert_box = st.empty()
            
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run Pipeline
            zone = danger_zone if zone_active else None
            output_frame = pipeline.process_frame(frame, zone_polygon=zone, auto_blur=enable_blur)
            
            # Display
            output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            stframe.image(output_frame_rgb, channels="RGB", use_column_width=True)
            
            # Update Stats
            # In a real app we might want to poll the DB less frequently, but this is fine for demo
            recent = pipeline.logger.get_recent_violations(5)
            if recent:
                 msg = ""
                 for r in recent:
                     msg += f"⚠️ {r[1]} at {r[0]}\n\n"
                 alert_box.warning(msg)

        cap.release()

st.markdown("---")
st.header("Safety Analytics")
# Show charts
stats = pipeline.logger.get_stats()
if stats:
    st.bar_chart(stats)
    
    st.subheader("Recent Violations")
    recent_logs = pipeline.logger.get_recent_violations(10)
    for log in recent_logs:
        col_a, col_b = st.columns([1, 4])
        with col_a:
            if os.path.exists(log[2]):
                st.image(log[2], caption=log[1])
        with col_b:
            st.write(f"**Type:** {log[1]}")
            st.write(f"**Time:** {log[0]}")
            st.write("---")
else:
    st.info("No violations recorded yet.")
