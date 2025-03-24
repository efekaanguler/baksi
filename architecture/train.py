import os
import sys
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from model import StrokeDICOMDataset, BaksiDetection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(f"stroke_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger()

CONFIG = {
    'data_dir': '../datasets/classification_ds',
    'yolo_detection_model': '../yolo/best_det.pt',
    'yolo_segmentation_model': '../yolo/best_seg.pt',
    'output_dir': '../trained_models',
    'batch_size': 16,
    'num_workers': 4,
    'epochs': 5,
    'initial_lr': 3e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'pretrained': True,
    'freeze_backbone_epochs': 5,
    'image_size': (640, 640),
    'val_split': 0.15,
    'test_split': 0.15,
    'dropout': 0.3,
    'spatial_dropout': 0.1,
    'mixup_alpha': 0.2,
    'label_smoothing': 0.1,
    'lr_scheduler': 'cosine_warmup',
    'warmup_epochs': 1,
    'lr_scheduler_patience': 2,
    'lr_scheduler_factor': 0.5,
    'early_stopping_patience': 3,
    'early_stopping_min_delta': 0.001,
    'grad_clip': 1.0,
    'grad_accum_steps': 2,
    'use_amp': True,
    'eval_threshold': 0.5,
    'optimize_threshold': True,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'log_interval': 10,
}

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG['seed'])


class MedicalImageTransforms:
    def __init__(self, intensity='medium', image_size=(640, 640)):
        self.intensity = intensity
        self.image_size = image_size
        
    def get_train_transforms(self):
        if self.intensity == 'light':
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                self._random_gamma_correction,
                self._random_window_adjustment,
            ])
        elif self.intensity == 'medium':
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                self._random_gamma_correction,
                self._random_window_adjustment,
                self._random_noise,
                transforms.RandomApply([self._simulate_low_contrast], p=0.2),
            ])
        elif self.intensity == 'strong':
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                self._random_gamma_correction,
                self._random_window_adjustment,
                self._random_noise,
                transforms.RandomApply([self._simulate_low_contrast], p=0.3),
                transforms.RandomApply([self._simulate_motion_artifact], p=0.2),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])
        else:
            raise ValueError(f"Unsupported intensity level: {self.intensity}")
            
    def get_val_transforms(self):
        return transforms.Compose([])
    
    def _random_gamma_correction(self, img):
        if torch.rand(1).item() > 0.5:
            gamma = torch.distributions.uniform.Uniform(0.8, 1.2).sample().item()
            return transforms.functional.adjust_gamma(img, gamma)
        return img
        
    def _random_window_adjustment(self, img):
        if torch.rand(1).item() > 0.5:
            brightness = torch.distributions.uniform.Uniform(0.9, 1.1).sample().item()
            contrast = torch.distributions.uniform.Uniform(0.9, 1.1).sample().item()
            return transforms.functional.adjust_brightness(
                transforms.functional.adjust_contrast(img, contrast),
                brightness
            )
        return img
        
    def _random_noise(self, img):
        if torch.rand(1).item() > 0.5:
            noise_level = torch.distributions.uniform.Uniform(0.01, 0.05).sample().item()
            noise = torch.randn(img.size()) * noise_level
            noisy_img = img + noise
            return torch.clamp(noisy_img, 0, 1)
        return img
        
    def _simulate_low_contrast(self, img):
        alpha = torch.distributions.uniform.Uniform(0.5, 0.9).sample().item()
        mean = img.mean()
        return alpha * img + (1 - alpha) * mean
        
    def _simulate_motion_artifact(self, img):
        strength = torch.distributions.uniform.Uniform(2, 5).sample().item()
        angle = torch.distributions.uniform.Uniform(0, 180).sample().item()
        kernel = torch.zeros((int(strength), int(strength)))
        center = int(strength) // 2
        for i in range(int(strength)):
            kernel[center, i] = 1.0 / int(strength)
        
        blurred = transforms.functional.gaussian_blur(img, kernel_size=int(strength))
        blend_factor = torch.distributions.uniform.Uniform(0.1, 0.3).sample().item()
        return img * (1 - blend_factor) + blurred * blend_factor
    

        
def prepare_datasets(data_dir, image_size=(640, 640)):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    transforms = MedicalImageTransforms(intensity='medium', image_size=image_size)
    train_transform = transforms.get_train_transforms()
    val_transform = transforms.get_val_transforms()
    
    train_dataset = StrokeDICOMDataset(train_dir, img_size=image_size, transform=train_transform)
    val_dataset = StrokeDICOMDataset(val_dir, img_size=image_size, transform=val_transform)
    test_dataset = StrokeDICOMDataset(test_dir, img_size=image_size, transform=val_transform)
    
    logger.info(f"Dataset split complete - Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
                
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=8, num_workers=4):
    train_labels = [sample[1] for sample in train_dataset.samples]
    class_counts = {0: train_labels.count(0), 1: train_labels.count(1)}
    num_samples = len(train_labels)
    weights = [num_samples / (class_counts[label] * len(class_counts)) for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, grad_accum_steps, log_interval):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        inputs, targets = inputs.to(device), targets.to(device)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        with torch.amp.autocast(device_type='cuda', enabled=CONFIG['use_amp']):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * grad_accum_steps

        if batch_idx % log_interval == 0:

            torch.cuda.empty_cache()
    
    avg_loss = running_loss / len(train_loader.dataset)
    return avg_loss

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_predictions = []
    all_raw_preds = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            predictions = torch.sigmoid(outputs).cpu().numpy()
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions)
            all_raw_preds.extend(outputs.cpu().numpy())
    
    avg_loss = val_loss / len(val_loader.dataset)
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    threshold = CONFIG['eval_threshold']
    if CONFIG['optimize_threshold']:
        best_f1 = 0
        best_threshold = 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            temp_f1 = f1_score(all_targets, all_predictions > t, zero_division=1)
            if temp_f1 > best_f1:
                best_f1 = temp_f1
                best_threshold = t
        threshold = best_threshold
        logger.info(f"Optimized threshold: {threshold:.3f}")
    
    accuracy = accuracy_score(all_targets, all_predictions > threshold)
    precision = precision_score(all_targets, all_predictions > threshold, zero_division=1)
    recall = recall_score(all_targets, all_predictions > threshold, zero_division=1)
    f1 = f1_score(all_targets, all_predictions > threshold, zero_division=1)
    roc_auc = roc_auc_score(all_targets, all_predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'threshold': threshold
    }
    
    return avg_loss, metrics

def plot_training_curves(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Plot loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    
    # Plot accuracy
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    # Plot precision
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['train_precision'], label='Train Precision')
    plt.plot(epochs, history['val_precision'], label='Val Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Precision')
    
    # Plot recall
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['train_recall'], label='Train Recall')
    plt.plot(epochs, history['val_recall'], label='Val Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.title('Recall')
    
    # Plot F1 score
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['train_f1'], label='Train F1')
    plt.plot(epochs, history['val_f1'], label='Val F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score')
    
    # Plot ROC AUC
    plt.subplot(2, 3, 6)
    plt.plot(epochs, history['train_roc_auc'], label='Train ROC AUC')
    plt.plot(epochs, history['val_roc_auc'], label='Val ROC AUC')
    plt.xlabel('Epochs')
    plt.ylabel('ROC AUC')
    plt.legend()
    plt.title('ROC AUC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def plot_training_curves_with_focal_params(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 10), dpi=300)
    
    # Plot loss
    plt.subplot(3, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    
    # Plot accuracy
    plt.subplot(3, 3, 2)
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    # Plot precision
    plt.subplot(3, 3, 3)
    plt.plot(epochs, history['train_precision'], label='Train Precision')
    plt.plot(epochs, history['val_precision'], label='Val Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Precision')
    
    # Plot recall
    plt.subplot(3, 3, 4)
    plt.plot(epochs, history['train_recall'], label='Train Recall')
    plt.plot(epochs, history['val_recall'], label='Val Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.title('Recall')
    
    # Plot F1 score
    plt.subplot(3, 3, 5)
    plt.plot(epochs, history['train_f1'], label='Train F1')
    plt.plot(epochs, history['val_f1'], label='Val F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score')
    
    # Plot ROC AUC
    plt.subplot(3, 3, 6)
    plt.plot(epochs, history['train_roc_auc'], label='Train ROC AUC')
    plt.plot(epochs, history['val_roc_auc'], label='Val ROC AUC')
    plt.xlabel('Epochs')
    plt.ylabel('ROC AUC')
    plt.legend()
    plt.title('ROC AUC')
    
    # Plot Focal Loss Alpha
    plt.subplot(3, 3, 7)
    plt.plot(epochs, history['focal_alpha'], label='Alpha')
    plt.xlabel('Epochs')
    plt.ylabel('Alpha')
    plt.title('Focal Loss Alpha')
    
    # Plot Focal Loss Gamma
    plt.subplot(3, 3, 8)
    plt.plot(epochs, history['focal_gamma'], label='Gamma')
    plt.xlabel('Epochs')
    plt.ylabel('Gamma')
    plt.title('Focal Loss Gamma')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves_with_focal_params.png'))
    plt.close()
    
    plot_training_curves(history, output_dir)

def train():
    set_seed(CONFIG['seed'])
    
    train_dataset, val_dataset, test_dataset = prepare_datasets(CONFIG['data_dir'], CONFIG['image_size'])
    
    model = BaksiDetection(
        det_model_path=CONFIG['yolo_detection_model'],
        seg_model_path=CONFIG['yolo_segmentation_model'],
        pretrained=CONFIG['pretrained'],
        device=CONFIG['device']
    ).to(CONFIG['device'])
    
    criterion = nn.BCEWithLogitsLoss().to(CONFIG['device'])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['initial_lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr'])
    
    scaler = GradScaler(device="cuda", enabled=CONFIG['use_amp'])
    
    best_val_loss = float('inf')
    best_model_wts = None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'train_f1': [],
        'val_f1': [],
        'train_roc_auc': [],
        'val_roc_auc': []
    }
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        if epoch <= CONFIG['freeze_backbone_epochs']:
            model.freeze_backbone()
            logger.info("EfficientNet backbones are frozen")
            current_batch_size = 16
            current_grad_accum_steps = CONFIG['grad_accum_steps']
        else:
            model.unfreeze_backbone()
            logger.info("EfficientNet backbones are unfrozen and trainable")
            current_batch_size = 2
            current_grad_accum_steps = 8
        
        # Create dataloaders with current batch size
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset, 
            batch_size=current_batch_size, 
            num_workers=CONFIG['num_workers']
        )
        
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, 
            CONFIG['device'], epoch, current_grad_accum_steps, CONFIG['log_interval']
        )
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, CONFIG['device'])
        
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                    f"Batch Size: {current_batch_size}, Grad Accum Steps: {current_grad_accum_steps}, "
                    f"Val Accuracy: {val_metrics['accuracy']:.4f}, Val Precision: {val_metrics['precision']:.4f}, "
                    f"Val Recall: {val_metrics['recall']:.4f}, Val F1: {val_metrics['f1']:.4f}, "
                    f"Val ROC AUC: {val_metrics['roc_auc']:.4f}, "
                    f"Threshold: {val_metrics.get('threshold', CONFIG['eval_threshold']):.2f}")
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(val_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['train_precision'].append(val_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(val_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        history['train_f1'].append(val_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_roc_auc'].append(val_metrics['roc_auc'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], f'model_epoch_{epoch}.pth'))
        
        torch.cuda.empty_cache()
    
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], 'best_model.pth'))
    
    plot_training_curves(history, CONFIG['output_dir'])
    
    logger.info("Training complete.")
    return model

if __name__ == "__main__":
    model = train()