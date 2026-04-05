#!/usr/bin/env bash
# Download sample surveillance video for testing
# Usage: bash download_sample.sh

echo "📥 Downloading sample video..."
mkdir -p data

# Option 1: Free public video (Big Buck Bunny)
VIDEO_URL="https://commondatastorage.googleapis.com/gtv-videos-library/sample/BigBuckBunny.mp4"
OUTPUT="data/sample.mp4"

echo "URL: $VIDEO_URL"
echo "Output: $OUTPUT"

if command -v curl &> /dev/null; then
    curl -L "$VIDEO_URL" -o "$OUTPUT" && echo "✅ Downloaded: $OUTPUT"
elif command -v wget &> /dev/null; then
    wget "$VIDEO_URL" -O "$OUTPUT" && echo "✅ Downloaded: $OUTPUT"
else
    echo "❌ curl or wget not found"
fi

ls -lh "$OUTPUT"
