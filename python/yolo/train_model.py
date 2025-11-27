from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.pt")

# model.to('cuda')

# Train the model
results = model.train(data="data.yaml", epochs=200, imgsz=640, batch=16, patience=50, name="yolo_icp_correct_label")

model.export(format="onnx")