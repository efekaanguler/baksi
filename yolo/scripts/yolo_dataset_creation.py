import os
import cv2
import numpy as np
import pydicom
from glob import glob
from tqdm import tqdm

def prepare_yolo_segmentation_dataset(data_dirs, save_dir, class_id=0):

    image_save_dir = os.path.join(save_dir, 'images', 'train')
    label_save_dir = os.path.join(save_dir, 'labels', 'train')
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)
    
    for data_dir in data_dirs:
        dicom_dir = os.path.join(data_dir, 'DICOM')
        mask_dir = os.path.join(data_dir, 'MASKS')

        dicom_files = glob(os.path.join(dicom_dir, '*.dcm'))

        for dcm_path in tqdm(dicom_files, desc=f"Processing {data_dir}"):
            # Dosya adı
            filename = os.path.splitext(os.path.basename(dcm_path))[0]
            mask_path = os.path.join(mask_dir, f"{filename}.png")
            
            if not os.path.exists(mask_path):
                print(f"Mask not found for {filename}, skipping.")
                continue

            try:
                ds = pydicom.dcmread(dcm_path)
                img = ds.pixel_array

                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = img.astype(np.uint8)
                
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if img.shape != mask.shape:
                    print(f"Shape mismatch for {filename}, skipping. DICOM: {img.shape}, MASK: {mask.shape}")
                    continue

                image_out_path = os.path.join(image_save_dir, f"{filename}.png")
                cv2.imwrite(image_out_path, img)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                label_lines = []
                h, w = mask.shape

                for contour in contours:
                    if len(contour) < 3:
                        continue

                    points = []
                    for point in contour:
                        x, y = point[0]
                        norm_x = x / w
                        norm_y = y / h
                        points.extend([norm_x, norm_y])

                    line = f"{class_id} " + " ".join([f"{p:.6f}" for p in points])
                    label_lines.append(line)

                if not label_lines:
                    print(f"No contours found in mask for {filename}, skipping.")
                    continue
                
                label_out_path = os.path.join(label_save_dir, f"{filename}.txt")
                with open(label_out_path, 'w') as f:
                    f.write('\n'.join(label_lines))

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    print(f"Dataset preperation completed. Images: {image_save_dir}, Labels: {label_save_dir}")

prepare_yolo_segmentation_dataset(["../../data/test1", "../../data/test2"], "../datasets")
