from ultralytics import YOLO

model = YOLO('yolo11m.pt')

model.train(
    data='stroke_detection.yaml',
    epochs=100,
    batch=32,
    imgsz=640,
    device=0,
    workers=8,
    optimizer='SGD',
    augment=True,
    degrees=5,
    scale=0.05,
    shear=0.0, 
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    blur=0.1,
    save_period=10,
    patience=20,
    verbose=True
)
