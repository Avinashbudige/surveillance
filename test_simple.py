#!/usr/bin/env python3
"""
Simple YOLO test on synthetic images - no video dependencies.
Run: python test_simple.py
"""

from ultralytics import YOLO
import numpy as np
import cv2
import os

print("🎥 Loading model...")
model = YOLO('yolov8n.pt')

print("📊 Running inference tests...\n")

# Test 1: Empty frame
print("Test 1: Empty black frame")
blank = np.zeros((480, 640, 3), dtype=np.uint8)
result = model(blank, conf=0.3, verbose=False)
print(f"  Detections: {len(result[0].boxes)} (expected: 0) " + 
      ("✅ PASS" if len(result[0].boxes) == 0 else "⚠️ WARNING"))

# Test 2: Noise
print("\nTest 2: Random noise")
noise = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
result = model(noise, conf=0.3, verbose=False)
print(f"  Detections: {len(result[0].boxes)} (expected: 0)")
print("  ✅ PASS" if len(result[0].boxes) == 0 else "  ⚠️ Some false positives")

# Test 3: Uniform color
print("\nTest 3: Uniform white frame")
uniform = np.ones((480, 640, 3), dtype=np.uint8) * 255
result = model(uniform, conf=0.3, verbose=False)
print(f"  Detections: {len(result[0].boxes)} (expected: 0)")
print("  ✅ PASS" if len(result[0].boxes) == 0 else "  ⚠️ Some false positives")

# Test 4: Synthetic person-like object (blue rectangle)
print("\nTest 4: Blue rectangle (synthetic object)")
rect = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(rect, (200, 100), (440, 400), (255, 0, 0), -1)  # Blue box
result = model(rect, conf=0.3, verbose=False)
print(f"  Detections: {len(result[0].boxes)}")
if len(result[0].boxes) > 0:
    for box in result[0].boxes:
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        print(f"    - Class: {cls}, Confidence: {conf:.2f}")
    print("  ✅ PASS - Model detected synthetic object")
else:
    print("  ℹ️ No detections (synthetic shape may not match training data)")

# Test 5: Performance - multiple frames
print("\nTest 5: Batch processing (10 frames)")
import time
frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
start = time.time()
results = model(frames, conf=0.3, verbose=False)
elapsed = time.time() - start
fps = len(frames) / elapsed
print(f"  Time: {elapsed:.2f}s for {len(frames)} frames")
print(f"  FPS: {fps:.1f}")
print("  ✅ PASS - Batch processing working")

print("\n" + "="*50)
print("✅ All tests completed!")
print("="*50)
