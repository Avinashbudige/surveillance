# In test_video.py - Add MLflow
import mlflow
with mlflow.start_run():
    results = model('data/cctv_sample.mp4', conf=0.5)
    mlflow.log_metric("detections", len(results[0].boxes))
    mlflow.ultralytics.log_model(model, "yolo_surveillance")