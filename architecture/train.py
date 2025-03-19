import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from model import (
    StrokeDICOMDataset, 
    BaksiDetection, 
    PreprocessorLayer,
    CombinedToEfficientNet,
    BinaryClassificationHead
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Baksi Stroke Detection Model')
    parser.add_argument('--data_dir', type=str, default='../datasets/classification_ds',
                        help='Directory with train/val/test data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='../saved_models', help='Directory to save models')
    parser.add_argument('--checkpoint_freq', type=int, default=2, help='Checkpoint frequency in epochs')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--det_model', type=str, default='../yolo/best_det.pt', help='Detection model path')
    parser.add_argument('--seg_model', type=str, default='../yolo/best_seg.pt', help='Segmentation model path')
    parser.add_argument('--scheduler', type=str, default='plateau', help='LR scheduler type: plateau or cosine')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs')
    parser.add_argument('--img_size', type=int, default=640, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    return parser.parse_args()

def get_stroke_dataloaders(data_root, batch_size=32, img_size=(640, 640), 
                          num_workers=4, transforms=None):
    """
    Create train, validation and test DataLoaders for stroke detection DICOM data
    
    Args:
        data_root: Root directory containing 'train', 'val', and 'test' folders,
                  each with 'class_0' and 'class_1' subfolders
        batch_size: Batch size (default 32)
        img_size: Target image size (height, width)
        num_workers: Number of workers for loading data
        transforms: Dictionary with optional transforms for each split
                   e.g., {'train': transform_train, 'val': transform_val, 'test': transform_test}
        
    Returns:
        Dictionary with train, val, test dataloaders
    """
    # Setup transforms
    if transforms is None:
        transforms = {
            'train': None,
            'val': None,
            'test': None
        }
    
    dataloaders = {}
    splits = ['train', 'val', 'test']
    
    # Create a dataloader for each split
    for split in splits:
        split_dir = os.path.join(data_root, split)
        
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist. Skipping {split} split.")
            continue
        
        # Create dataset
        dataset = StrokeDICOMDataset(
            root_dir=split_dir,
            img_size=img_size,
            transform=transforms.get(split)
        )
        
        # Print dataset statistics
        class0_count = sum(1 for _, label in dataset.samples if label == 0)
        class1_count = sum(1 for _, label in dataset.samples if label == 1)
        print(f"{split.capitalize()} dataset loaded with {len(dataset)} samples:")
        print(f"  - Class 0 (no stroke): {class0_count} samples")
        print(f"  - Class 1 (stroke): {class1_count} samples")
        
        # Create dataloader
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),  # Only shuffle training data
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders

def get_lr_scheduler(optimizer, args, train_loader_len):
    """Create learning rate scheduler with optional warmup"""
    if args.scheduler.lower() == 'plateau':
        # ReduceLROnPlateau: reduce LR when validation loss plateaus
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                     patience=5, verbose=True)
    elif args.scheduler.lower() == 'cosine':
        # CosineAnnealingLR: gradually decreases LR following a cosine curve
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        raise ValueError(f"Unknown scheduler type: {args.scheduler}")
    
    return scheduler

def warmup_learning_rate(optimizer, epoch, batch_idx, total_batches, warmup_epochs, base_lr):
    """Gradually warm up learning rate during early epochs"""
    if epoch >= warmup_epochs:
        return
    
    # Calculate current progress through warmup (0 to 1)
    progress = (epoch * total_batches + batch_idx) / (warmup_epochs * total_batches)
    
    # Adjust learning rate: start from 10% of base_lr, gradually increase to base_lr
    new_lr = base_lr * (0.1 + 0.9 * progress)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train_model(model, dataloaders, criterion, optimizer, args, device='cuda'):
    """
    Train the BaksiDetection model with visualization and dynamic learning rate
    
    Args:
        model: BaksiDetection model
        dataloaders: Dictionary containing 'train' and 'val' dataloaders
        criterion: Loss function
        optimizer: Optimizer for model parameters
        args: Training arguments
        device: Device to train on
        
    Returns:
        Dictionary containing trained model and training history
    """
    # Create save directory if it doesn't exist
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get train and validation loaders
    train_loader = dataloaders['train']
    val_loader = dataloaders.get('val')
    
    # Create learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, args, len(train_loader))
    
    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # For tracking best model
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_path = None
    no_improvement_count = 0
    
    # Training loop with progress bar for epochs
    epoch_pbar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # Track metrics for current epoch
        epoch_loss = 0
        epoch_acc = 0
        steps = 0
        
        # Set model to training mode
        model.train()
        
        # Progress bar for batches
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", 
                          leave=False, unit="batch")
        
        # Track current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Training loop for current epoch
        for batch_idx, batch in enumerate(batch_pbar):
            # Apply learning rate warmup if in warmup phase
            if args.warmup_epochs > 0:
                warmup_learning_rate(optimizer, epoch, batch_idx, len(train_loader), 
                                    args.warmup_epochs, args.lr)
            
            # Forward pass and calculate loss
            metrics = model.train_step(batch, criterion, optimizer)
            
            # Update metrics
            epoch_loss += metrics['loss']
            epoch_acc += metrics['accuracy']
            steps += 1
            
            # Update progress bar
            batch_pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}", 
                "acc": f"{metrics['accuracy']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Calculate epoch metrics
        avg_train_loss = epoch_loss / steps
        avg_train_acc = epoch_acc / steps
        
        # Store in history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        
        # Validation phase
        val_loss = 0
        current_val_acc = 0
        
        if val_loader:
            val_metrics = model.evaluate(val_loader, criterion)
            val_loss = val_metrics['loss']
            current_val_acc = val_metrics['accuracy']
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(current_val_acc)
            
            # Update main progress bar with validation metrics
            epoch_pbar.set_postfix({
                "train_loss": f"{avg_train_loss:.4f}",
                "train_acc": f"{avg_train_acc:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{current_val_acc:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Update learning rate based on validation loss
            if args.scheduler.lower() == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Check if this is the best model so far
            is_best = False
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_epoch = epoch + 1
                is_best = True
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not is_best:  # Don't count twice if both improved
                    is_best = True
            
            if is_best:
                # Save best model
                best_model_path = os.path.join(save_dir, f"model_best_{timestamp}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                    'val_accuracy': best_val_acc,
                    'val_loss': best_val_loss,
                    'history': history,
                }, best_model_path)
                print(f"\nNew best model saved with val_acc: {best_val_acc:.4f}, val_loss: {best_val_loss:.4f}")
                
                # Reset no improvement counter
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            # Early stopping check
            if args.early_stopping is not None and no_improvement_count >= args.early_stopping:
                print(f"\nEarly stopping triggered after {no_improvement_count} epochs without improvement")
                break
        else:
            # If no validation set, just update with training metrics
            epoch_pbar.set_postfix({
                "train_loss": f"{avg_train_loss:.4f}",
                "train_acc": f"{avg_train_acc:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Update learning rate scheduler if it's not ReduceLROnPlateau
            if args.scheduler.lower() != 'plateau':
                scheduler.step()
        
        # Save model checkpoint periodically
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            model_save_path = os.path.join(save_dir, f"model_epoch{epoch+1}_{timestamp}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'history': history,
            }, model_save_path)
            print(f"\nModel checkpoint saved to {model_save_path}")
    
    # Plot and save training curves
    plot_training_curves(history, save_dir, timestamp)
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"model_final_{timestamp}.pth")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'history': history,
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Print best model information
    if best_model_path:
        print(f"Best model was from epoch {best_epoch} with val_acc: {best_val_acc:.4f}, val_loss: {best_val_loss:.4f}")
        print(f"Best model saved to: {best_model_path}")
    
    return {
        'model': model,
        'history': history,
        'model_path': final_model_path,
        'best_model_path': best_model_path,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'plot_path': os.path.join(save_dir, f"training_curves_{timestamp}.png")
    }

def plot_training_curves(history, save_dir, timestamp=None):
    """
    Plot training and validation curves including learning rate
    
    Args:
        history: Dictionary containing training and validation metrics
        save_dir: Directory to save the plot
        timestamp: Optional timestamp for unique filename
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss subplot
    ax1.plot(epochs, history['train_loss'], 'b-', marker='o', markersize=3, label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', marker='s', markersize=3, label='Validation Loss')
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy subplot
    ax2.plot(epochs, history['train_acc'], 'b-', marker='o', markersize=3, label='Training Accuracy')
    if 'val_acc' in history and history['val_acc']:
        ax2.plot(epochs, history['val_acc'], 'r-', marker='s', markersize=3, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Learning rate subplot
    if 'learning_rates' in history and history['learning_rates']:
        ax3.plot(epochs, history['learning_rates'], 'g-', marker='o', markersize=3)
        ax3.set_title('Learning Rate', fontsize=14)
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')  # Log scale for better visualization
        ax3.grid(True, alpha=0.3)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(save_dir, f"training_curves_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {plot_path}")
    
    # Close figure to free memory
    plt.close(fig)

def main():
    """Main function to run the training"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = BaksiDetection(
        det_model_path=args.det_model,
        seg_model_path=args.seg_model,
        device=device
    )
    
    # Set up dataloaders
    print(f"Loading datasets from {args.data_dir}...")
    dataloaders = get_stroke_dataloaders(
        data_root=args.data_dir, 
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        num_workers=args.num_workers
    )
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    
    print("\n" + "="*50)
    print("Starting training with parameters:")
    print(f"- Learning rate: {args.lr}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Epochs: {args.epochs}")
    print(f"- LR Scheduler: {args.scheduler}")
    print(f"- Warmup epochs: {args.warmup_epochs}")
    print(f"- Early stopping patience: {args.early_stopping}")
    print("="*50 + "\n")
    
    # Train model
    results = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        args=args,
        device=device
    )
    
    print("\nTraining completed!")
    return results

if __name__ == "__main__":
    main()
