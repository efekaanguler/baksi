from ultralytics import YOLO
import torch
import time

torch.cuda.empty_cache()

model = YOLO('yolo11l-seg.pt')

start_time = time.time()

model.train(
    data='/home/efekaan/Desktop/baksi/yolo/stroke_segmentation.yaml',
    epochs=200,
    batch=8,
    imgsz=640,
    device=0,
    workers=8,
    
    iou=0.3,
    patience=0,
    save_period=5,
    verbose=True,
    conf=0.01,
    agnostic_nms=True,
    augment=True
)

end_time = time.time()
total_time = end_time - start_time

print(f"\n===== EĞİTİM TAMAMLANDI =====")
print(f"Toplam Eğitim Süresi: {total_time:.2f} saniye")