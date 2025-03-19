import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

# Import from model.py
from model import (
    StrokeDICOMDataset,
    BaksiDetection,
    PreprocessorLayer,
    CombinedToEfficientNet,
    BinaryClassificationHead,
    get_stroke_dataloaders
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test BaksiDetection model for stroke detection')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--test_data', type=str, default='../datasets/classification_ds', help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--output_dir', type=str, default='../test_results', help='Directory to save test results')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold (default: 0.5)')
    return parser.parse_args()

def load_model(model_path, device):
    """
    Load trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model to
        
    Returns:
        Loaded BaksiDetection model
    """
    print(f"Loading model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model (you need to know the original parameters)
    model = BaksiDetection(
        det_model_path='../yolo/best_det.pt',  # These should match your training settings
        seg_model_path='../yolo/best_seg.pt',
        device=device
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    return model, checkpoint

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()

def plot_roc_curve(y_true, y_scores, save_path=None):
    """
    Plot and save ROC curve
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        save_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()
    
    return roc_auc

def evaluate_model(model, test_loader, device, output_dir, threshold=0.5):
    """
    Evaluate model on test data
    
    Args:
        model: BaksiDetection model
        test_loader: DataLoader with test data
        device: Device for computation
        output_dir: Directory to save results
        threshold: Classification threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    criterion = nn.BCEWithLogitsLoss()
    
    # Lists to store predictions and ground truth
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    test_loss = 0.0
    
    # Set model to evaluation mode
    model.eval()
    
    # Create progress bar
    pbar = tqdm(test_loader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for inputs, labels in pbar:
            # Move to device
            labels = labels.to(device)
            
            # Model prediction
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels.float().unsqueeze(1))
            test_loss += loss.item()
            
            # Get predictions and probabilities
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities >= threshold).astype(int)
            
            # Store batch results
            all_predictions.extend(predictions.squeeze())
            all_probabilities.extend(probabilities.squeeze())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    avg_loss = test_loss/len(test_loader)
    
    # Calculate ROC AUC
    roc_path = os.path.join(output_dir, "roc_curve.png")
    roc_auc = plot_roc_curve(all_labels, all_probabilities, roc_path)
    
    # Print metrics
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    print(f"Loss:      {avg_loss:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("="*50)
    
    # Save metrics to file
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        
    # Create confusion matrix visualization
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(all_labels, all_predictions, cm_path)
    
    # Save predictions for further analysis
    predictions_path = os.path.join(output_dir, "predictions.csv")
    with open(predictions_path, "w") as f:
        f.write("true_label,predicted_label,probability\n")
        for true, pred, prob in zip(all_labels, all_predictions, all_probabilities):
            f.write(f"{true},{pred},{prob:.6f}\n")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "loss": avg_loss
    }

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create timestamp for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory with timestamp
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Save test configuration
    with open(os.path.join(output_dir, "test_config.txt"), "w") as f:
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Test data path: {args.test_data}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Classification threshold: {args.threshold}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Timestamp: {timestamp}\n")
    
    # Load model
    try:
        model, checkpoint = load_model(args.model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    try:
        test_loader = get_stroke_dataloaders(
            data_root=args.test_data,
            batch_size=args.batch_size
        ).get('test')
        
        if not test_loader:
            print("Error: No test data found!")
            return
            
        print(f"Test data loaded with {len(test_loader.dataset)} samples")
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Evaluate model
    print("\nStarting evaluation...")
    metrics = evaluate_model(model, test_loader, device, output_dir, args.threshold)
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Stroke Detection Test Summary\n")
        f.write("===========================\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test data: {args.test_data}\n\n")
        
        f.write("Metrics:\n")
        f.write(f"- Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"- Precision: {metrics['precision']:.4f}\n")
        f.write(f"- Recall:    {metrics['recall']:.4f}\n")
        f.write(f"- F1 Score:  {metrics['f1']:.4f}\n")
        f.write(f"- ROC AUC:   {metrics['roc_auc']:.4f}\n")
    
    print("\nTesting completed!")
    print(f"All results saved to {output_dir}")

if __name__ == "__main__":
    main()
