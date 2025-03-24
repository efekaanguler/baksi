import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, balanced_accuracy_score
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import gc
import traceback

from model import (
    StrokeDICOMDataset,
    BaksiDetection,
    PreprocessorLayer,
    YOLOPredictor
)

def parse_args():
    parser = argparse.ArgumentParser(description='DualPathwayModel Test for Stroke Detection')
    parser.add_argument('--model_path', type=str, default='../saved_models/model_best.pth', help='Path to the trained model checkpoint')
    parser.add_argument('--test_data', type=str, default='../datasets/classification_ds/test', help='Path to the test data directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--output_dir', type=str, default='../test_results', help='Directory to save test results')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold (default: 0.5)')
    parser.add_argument('--use_cpu_yolo', action='store_true', help='Run YOLO on CPU to save GPU memory')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for processing')
    return parser.parse_args()

def load_model(model_path, device, args):
    print(f"Loading model: {model_path}...")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"Checkpoint loaded successfully. Checkpoint keys: {list(checkpoint.keys())}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        det_model_path = '../yolo/best_det.pt'
        seg_model_path = '../yolo/best_seg.pt'
        
        if not os.path.exists(det_model_path):
            print(f"WARNING: Detection model not found: {det_model_path}")
        
        if not os.path.exists(seg_model_path):
            print(f"WARNING: Segmentation model not found: {seg_model_path}")
        
        yolo_device = 'cpu'
        print(f"YOLO models will run on {yolo_device}")
        
        print("Initializing model...")
        model = BaksiDetection(
            det_model_path=det_model_path,
            seg_model_path=seg_model_path,
            pretrained=True,
            device=yolo_device
        )
        
        print("Loading model weights...")
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("Model loaded using 'model_state_dict' key")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("Model loaded using 'state_dict' key")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print("Entire checkpoint used as model weights")
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model = model.to(device)
        
        if hasattr(model, 'feature_extractor') and hasattr(model.feature_extractor, 'yolo_stream'):
            if hasattr(model.feature_extractor.yolo_stream, 'predictor'):
                print("Checking YOLO predictor device...")
                model.feature_extractor.yolo_stream.predictor.device = yolo_device
                print(f"YOLO predictor device set to {yolo_device}")
        
        model.eval()
        
        print(f"Model loaded successfully and moved to {device}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def prepare_test_dataloader(test_data_path, batch_size, img_size):
    print(f"Preparing test data: {test_data_path}")
    
    try:
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data directory not found: {test_data_path}")
        
        test_dataset = StrokeDICOMDataset(
            root_dir=test_data_path,
            img_size=(img_size, img_size)
        )
        
        if len(test_dataset) == 0:
            print(f"WARNING: No samples found in test dataset: {test_data_path}")
            return None
        
        print(f"Test dataset loaded: {len(test_dataset)} samples")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return test_loader
        
    except Exception as e:
        print(f"Error preparing test dataloader: {e}")
        traceback.print_exc()
        return None

def evaluate_model(model, test_loader, device, output_dir, threshold=0.5):
    print("\nEvaluating model...")
    
    if model is None or test_loader is None:
        print("Model or test dataloader not available, evaluation cannot proceed.")
        return {}
    
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Test Evaluation")):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                
                if outputs.device != device:
                    outputs = outputs.to(device)
                
                y_true.extend(labels.detach().cpu().numpy())
                scores = torch.sigmoid(outputs).detach().cpu().numpy()
                y_scores.extend(scores)
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA out of memory for batch {batch_idx}. Switching to CPU.")
                    torch.cuda.empty_cache()
                    try:
                        images = images.cpu()
                        labels = labels.cpu()
                        model.to('cpu')
                        outputs = model(images)
                        y_true.extend(labels.numpy())
                        y_scores.extend(torch.sigmoid(outputs).cpu().numpy())
                        model.to(device)
                    except Exception as cpu_err:
                        print(f"Error on CPU: {cpu_err}")
                        traceback.print_exc()
                        continue
                else:
                    print(f"Error processing batch {batch_idx}: {e}")
                    traceback.print_exc()
                    continue
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                traceback.print_exc()
                continue
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    if len(y_true) == 0 or len(y_scores) == 0:
        print("No valid data for evaluation.")
        return {}
        
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        metrics['specificity'] = 0
    
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        metrics['roc_auc'] = auc(fpr, tpr)
    except Exception as e:
        print(f"Error calculating ROC: {e}")
        metrics['roc_auc'] = 0
    
    threshold_analysis = analyze_thresholds(y_true, y_scores)
    metrics['threshold_analysis'] = threshold_analysis
    
    try:
        plot_results(y_true, y_scores, metrics, output_dir)
    except Exception as e:
        print(f"Error visualizing results: {e}")
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    return metrics

def analyze_thresholds(y_true, y_scores):
    thresholds = np.linspace(0.1, 0.9, 9)
    results = {
        'thresholds': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'specificity': [],
        'f1': []
    }
    
    best_accuracy = {'threshold': 0.5, 'value': 0}
    best_f1 = {'threshold': 0.5, 'value': 0}
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            spec = 0
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if acc > best_accuracy['value']:
            best_accuracy = {'threshold': threshold, 'value': acc}
            
        if f1 > best_f1['value']:
            best_f1 = {'threshold': threshold, 'value': f1}
        
        results['thresholds'].append(threshold)
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['specificity'].append(spec)
        results['f1'].append(f1)
    
    idx = np.argmin(np.abs(np.array(results['recall']) - np.array(results['specificity'])))
    balanced_op = {
        'threshold': results['thresholds'][idx],
        'recall': results['recall'][idx],
        'specificity': results['specificity'][idx]
    }
    
    analysis = {
        'metrics': results,
        'best_accuracy': best_accuracy,
        'best_f1': best_f1,
        'balanced_op': balanced_op
    }
    
    return analysis

def plot_results(y_true, y_scores, metrics, output_dir):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, (y_scores >= 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    threshold_analysis = metrics['threshold_analysis']['metrics']
    plt.plot(threshold_analysis['thresholds'], threshold_analysis['accuracy'], label='Accuracy')
    plt.plot(threshold_analysis['thresholds'], threshold_analysis['precision'], label='Precision')
    plt.plot(threshold_analysis['thresholds'], threshold_analysis['recall'], label='Recall')
    plt.plot(threshold_analysis['thresholds'], threshold_analysis['specificity'], label='Specificity')
    plt.plot(threshold_analysis['thresholds'], threshold_analysis['f1'], label='F1 Score')
    
    best_acc = metrics['threshold_analysis']['best_accuracy']
    best_f1 = metrics['threshold_analysis']['best_f1']
    balanced_op = metrics['threshold_analysis']['balanced_op']
    
    plt.axvline(x=best_acc['threshold'], color='r', linestyle='--', alpha=0.5, 
                label=f'Best Accuracy Threshold: {best_acc["threshold"]:.2f}')
    plt.axvline(x=best_f1['threshold'], color='g', linestyle='--', alpha=0.5,
                label=f'Best F1 Threshold: {best_f1["threshold"]:.2f}')
    plt.axvline(x=balanced_op['threshold'], color='b', linestyle='--', alpha=0.5,
                label=f'Balanced Point Threshold: {balanced_op["threshold"]:.2f}')
    
    plt.xlabel('Threshold Value')
    plt.ylabel('Metric Value')
    plt.title('Threshold Analysis')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    args = parse_args()
    
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA. Available GPU(s): {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  Free CUDA memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        if args.gpu and not torch.cuda.is_available():
            print("GPU specified but CUDA not found. Using CPU.")
        else:
            print("Using CPU.")
    
    torch.cuda.empty_cache()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    with open(os.path.join(output_dir, "test_config.txt"), "w") as f:
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Test data path: {args.test_data}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Classification threshold: {args.threshold}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Use CPU YOLO: {args.use_cpu_yolo}\n")
        f.write(f"Image size: {args.img_size}\n")
        f.write(f"Timestamp: {timestamp}\n")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = load_model(args.model_path, device, args)
        if model is None:
            print("Model could not be loaded. Exiting test.")
            return
        
        test_loader = prepare_test_dataloader(
            test_data_path=args.test_data,
            batch_size=args.batch_size,
            img_size=args.img_size
        )
        
        if not test_loader:
            print("Test data could not be loaded. Exiting test.")
            return
        
        print("\nStarting evaluation...")
        metrics = evaluate_model(model, test_loader, device, output_dir, args.threshold)
        
        if not metrics:
            print("Metrics could not be calculated. Exiting test.")
            return
        
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("Stroke Detection Test Summary\n")
            f.write("===========================\n\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Test data: {args.test_data}\n\n")
            
            f.write("Metrics:\n")
            f.write(f"- Accuracy:          {metrics['accuracy']:.4f}\n")
            f.write(f"- Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
            f.write(f"- Precision:         {metrics['precision']:.4f}\n")
            f.write(f"- Recall:            {metrics['recall']:.4f}\n")
            f.write(f"- Specificity:       {metrics['specificity']:.4f}\n")
            f.write(f"- F1 Score:          {metrics['f1']:.4f}\n")
            f.write(f"- ROC AUC:           {metrics['roc_auc']:.4f}\n\n")
            
            f.write("Optimal Thresholds:\n")
            f.write(f"- For Accuracy:      {metrics['threshold_analysis']['best_accuracy']['threshold']:.4f}\n")
            f.write(f"- For F1 Score:      {metrics['threshold_analysis']['best_f1']['threshold']:.4f}\n")
            f.write(f"- Balanced Point:    {metrics['threshold_analysis']['balanced_op']['threshold']:.4f}\n")
        
        print("\nTesting completed!")
        print(f"All results saved to {output_dir}")
    
    except Exception as e:
        print(f"Unexpected error during test: {e}")
        traceback.print_exc()
        
        with open(os.path.join(output_dir, "error_log.txt"), "w") as f:
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write("Error Details:\n")
            traceback.print_exc(file=f)

if __name__ == "__main__":
    main()