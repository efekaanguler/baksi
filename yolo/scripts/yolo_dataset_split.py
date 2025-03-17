import os
import random
import shutil
from glob import glob

def split_train_val(dataset_dir, val_ratio=0.2, seed=42):
    random.seed(seed)
    
    # Klasör yolları
    images_dir = os.path.join(dataset_dir, 'images', 'train')
    labels_dir = os.path.join(dataset_dir, 'labels', 'train')
    
    # Yeni val dizinleri
    images_val_dir = os.path.join(dataset_dir, 'images', 'val')
    labels_val_dir = os.path.join(dataset_dir, 'labels', 'val')
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    
    # Tüm görüntü dosyaları
    image_files = glob(os.path.join(images_dir, '*.png'))

    # Shuffle ve split
    random.shuffle(image_files)
    val_size = int(len(image_files) * val_ratio)
    val_files = image_files[:val_size]

    # Val dosyalarını taşı
    for img_path in val_files:
        filename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, filename.replace('.png', '.txt'))
        
        # Move images
        shutil.move(img_path, os.path.join(images_val_dir, filename))
        
        # Move labels
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(labels_val_dir, filename.replace('.png', '.txt')))
    
    print(f"Split completed. {val_size} images transferred to val set.")

split_train_val('../datasets')
