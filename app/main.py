# app.py - Run: streamlit run app.py
import streamlit as st
from ultralytics import YOLO
import cv2

st.title("🚨 AI Surveillance Demo")

model = YOLO('yolov8n.pt')
uploaded_file = st.file_uploader("Upload CCTV", type=['mp4','avi'])

if uploaded_file:
    results = model(uploaded_file, conf=0.5, save=True)
    st.video(uploaded_file)
    st.success(f"✅ {len(results[0].boxes)} detections!")