from ultralytics import YOLO

model = YOLO('yolo11m-seg.pt')

results = model.train(
    data='../datasets/dataset.yaml',
    imgsz=512,
    epochs=30,
    batch=8,
    device=0,
    workers=4,
    name='inme-segmentation-augmented',
    patience=10,
    optimizer='SGD',
    verbose=True
)