#!/usr/bin/env python3
"""
Quick YOLO test on video files.
Run: python test_video.py
"""

from ultralytics import YOLO
import os
from pathlib import Path
import urllib.request
import subprocess

MODEL_PATH = 'yolov8n.pt'
DATA_DIR = 'data'
VIDEO_PATH = None

def download_video(url, output_path):
    """Download video using curl, wget, or urllib."""
    print(f"📥 Downloading: {url}")
    
    try:
        # Try curl first (faster, shows progress)
        result = subprocess.run(
            ['curl', '-L', url, '-o', output_path, '-# '],
            capture_output=True,
            timeout=300
        )
        if result.returncode == 0 and os.path.exists(output_path):
            return True
    except:
        pass
    
    try:
        # Try wget
        result = subprocess.run(
            ['wget', url, '-O', output_path],
            capture_output=True,
            timeout=300
        )
        if result.returncode == 0 and os.path.exists(output_path):
            return True
    except:
        pass
    
    try:
        # Fallback: urllib
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return False

def find_or_download_video():
    """Find valid video or download fresh one."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check existing videos
    candidates = [
        ('data/cctv_back.mp4', 'cctv_back'),
        ('data/cctv_person.mp4', 'cctv_person'),
        ('data/cctv.mp4', 'cctv'),
    ]
    
    print("🔍 Searching for video files...")
    for path, name in candidates:
        if os.path.exists(path):
            print(f"✓ Found: {path}")
            file_size = os.path.getsize(path) / (1024*1024)
            print(f"  Size: {file_size:.1f} MB")
            if file_size < 1:  # Suspiciously small
                print(f"  ⚠️ File too small (likely corrupted)")
                continue
            return path
    
    # No valid local video - download fresh
    print("\n📥 No valid video found. Downloading sample...\n")
    
    sample_path = os.path.join(DATA_DIR, 'sample.mp4')
    url = "https://commondatastorage.googleapis.com/gtv-videos-library/sample/BigBuckBunny.mp4"
    
    if download_video(url, sample_path):
        print(f"✅ Downloaded: {sample_path}")
        return sample_path
    else:
        return None

def create_fallback_test():
    """Create synthetic test image."""
    print("\n⚠️  Creating synthetic test frame instead...")
    try:
        import cv2
        import numpy as np
        
        # Create simple test frame with blue rectangle
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2.rectangle(test_img, (150, 150), (490, 330), (255, 0, 0), -1)
        
        test_path = os.path.join(DATA_DIR, 'test_frame.jpg')
        cv2.imwrite(test_path, test_img)
        print(f"✓ Test image created: {test_path}")
        return test_path
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return None

# Find or download video
VIDEO_PATH = find_or_download_video()

if VIDEO_PATH is None:
    VIDEO_PATH = create_fallback_test()

if VIDEO_PATH is None:
    print("\n❌ No video or test image available")
    exit(1)

print(f"\n🎥 Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

print(f"🎬 Processing: {VIDEO_PATH}")
try:
    results = model(
        VIDEO_PATH,
        conf=0.5,
        save=True,
        verbose=False
    )
    
    # Show results
    total_detections = 0
    for result in results:
        total_detections += len(result.boxes)
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Total detections: {total_detections}")
    print(f"📁 Results saved to: runs/detect/predict/")
    
    # Get output path
    output_dir = Path('runs/detect/predict')
    if output_dir.exists():
        videos = list(output_dir.glob('*.avi'))
        if videos:
            print(f"🎥 Output video: {videos[0]}")
        else:
            images = list(output_dir.glob('*.jpg'))
            if images:
                print(f"🖼️ Output image: {images[0]}")

except Exception as e:
    print(f"\n❌ Processing failed: {e}")
    print(f"\n💡 Tip: Download a fresh video manually:")
    print(f"  curl -L 'https://commondatastorage.googleapis.com/gtv-videos-library/sample/BigBuckBunny.mp4' -o data/sample.mp4")
