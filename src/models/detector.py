"""Inference wrapper for YOLO detection."""

from ultralytics import YOLO


class Detector:
    def __init__(self, weights_path: str) -> None:
        self.model = YOLO(weights_path)

    def predict(self, frame, conf: float = 0.3):
        return self.model(frame, conf=conf)
