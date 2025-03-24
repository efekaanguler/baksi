from ultralytics import YOLO
import torch
import time

torch.cuda.empty_cache()

model = YOLO('yolo11l.pt')

start_time = time.time()

model.train(
    data='/home/efekaan/Desktop/baksi/yolo/stroke_detection.yaml',
    epochs=100,
    batch=8,
    imgsz=640,
    device=0,
    workers=8,
    
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,
    
    iou=0.3,
    patience=0,
    save_period=5,
    verbose=True,
    conf=0.01,
    agnostic_nms=True,
    augment=True,
)

end_time = time.time()
total_time = end_time - start_time

print(f"\n===== EĞİTİM TAMAMLANDI =====")
print(f"Toplam Eğitim Süresi: {total_time:.2f} saniye")
