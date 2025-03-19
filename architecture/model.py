import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from ultralytics import YOLO

class StrokeDICOMDataset(Dataset):
    """
    Dataset for loading DICOM files from class_0 and class_1 directories
    """
    def __init__(self, root_dir, img_size=(640, 640), transform=None):
        """
        Args:
            root_dir: Directory with 'class_0' and 'class_1' subdirectories
            img_size: Target image size (height, width)
            transform: Optional additional transforms to apply
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        
        # List all DICOM files and their corresponding labels
        self.samples = []
        
        # Class 0 files
        class0_dir = os.path.join(root_dir, 'class_0')
        if os.path.exists(class0_dir):
            class0_files = [os.path.join(class0_dir, f) for f in os.listdir(class0_dir) 
                           if f.endswith('.dcm')]
            self.samples.extend([(path, 0) for path in class0_files])
        
        # Class 1 files
        class1_dir = os.path.join(root_dir, 'class_1')
        if os.path.exists(class1_dir):
            class1_files = [os.path.join(class1_dir, f) for f in os.listdir(class1_dir) 
                           if f.endswith('.dcm')]
            self.samples.extend([(path, 1) for path in class1_files])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Use preprocess_dicom function for preprocessing
        try:
            # Preprocess DICOM
            processed_img = preprocess_dicom(img_path, normalize_size=self.img_size)
            
            # Convert to tensor and normalize to [0,1]
            img_tensor = torch.from_numpy(processed_img.transpose(2, 0, 1)).float() / 255.0
            
            # Apply additional transforms if provided
            if self.transform:
                img_tensor = self.transform(img_tensor)
            
            # Return image and label
            return img_tensor, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a placeholder on error
            placeholder = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return placeholder, torch.tensor(label, dtype=torch.long)
        
class YOLOPredictor:
    def __init__(self, det_model_name, seg_model_name):
        """
        Initialize with both detection and segmentation models (both required)
        
        Args:
            det_model: YOLO model for object detection
            seg_model: YOLO model for segmentation
        """
        self.det_model = YOLO(det_model_name)
        self.seg_model = YOLO(seg_model_name)
    
    def get_detection_data(self, images, conf=0.25):
        """
        Process images and return detection data suitable for DetectionToEfficientNet
        
        Args:
            images: List of image paths or numpy arrays
            conf: Confidence threshold
            
        Returns:
            List of tensors with detection data [x, y, w, h, conf, cls]
        """
        # Run inference with detection model
        results = self.det_model(images, conf=conf)
        
        # Format detections for DetectionToEfficientNet
        batch_detections = []
        
        for result in results:
            # Get boxes (normalized xywh format)
            if result.boxes.xywh.shape[0] > 0:
                # Get xywh, confidence, and class
                xywh = result.boxes.xywh.cpu() / torch.tensor([result.orig_shape[1], result.orig_shape[0], 
                                                               result.orig_shape[1], result.orig_shape[0]])
                conf = result.boxes.conf.cpu().unsqueeze(1)
                cls = result.boxes.cls.cpu().unsqueeze(1)
                
                # Combine into [x, y, w, h, conf, cls]
                detections = torch.cat((xywh, conf, cls), dim=1)
            else:
                detections = torch.zeros((0, 6))
                
            batch_detections.append(detections)
            
        return batch_detections
    
    def get_segmentation_data(self, images, conf=0.25):
        """
        Process images and return segmentation data suitable for SegmentationToEfficientNet
        
        Args:
            images: List of image paths or numpy arrays
            conf: Confidence threshold
            
        Returns:
            List of tensors with segmentation data [cls, x1, y1, x2, y2, ...]
        """
        # Run inference with segmentation model
        results = self.seg_model(images, conf=conf)
        
        # Format segmentations for SegmentationToEfficientNet
        batch_segmentations = []
        
        for result in results:
            image_segmentations = []
            
            # Check if any masks were found
            if hasattr(result, 'masks') and result.masks is not None:
                for i, mask in enumerate(result.masks.data):
                    # Get class
                    cls = result.boxes.cls[i].item()
                    
                    # Get polygon points from mask
                    # Extract contours from the mask
                    mask_np = mask.cpu().numpy()
                    contours, _ = cv2.findContours((mask_np * 255).astype(np.uint8), 
                                                  cv2.RETR_EXTERNAL, 
                                                  cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Process largest contour (simplify it to reduce points)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        # Format for YOLO: [cls, x1, y1, x2, y2, ...]
                        points = approx.reshape(-1) / np.array([result.orig_shape[1], result.orig_shape[0]] * (len(approx.reshape(-1)) // 2))
                        seg_data = torch.tensor([cls] + points.tolist())
                        image_segmentations.append(seg_data)
            
            # If no masks, add empty tensor
            if not image_segmentations:
                batch_segmentations.append(torch.zeros((0, 1)))
            else:
                batch_segmentations.append(torch.stack(image_segmentations))
        
        return batch_segmentations
    
    def process_batch(self, images, conf=0.25):
        """
        Process a batch of images with both detection and segmentation models
        
        Args:
            images: List of image paths or numpy arrays
            conf: Confidence threshold
            
        Returns:
            Tuple of (detection_data, segmentation_data)
        """
        detection_data = self.get_detection_data(images, conf=conf)
        segmentation_data = self.get_segmentation_data(images, conf=conf)
        return (detection_data, segmentation_data)

class PreprocessorLayer(nn.Module):
    """
    PyTorch layer for preprocessing DICOM images with specialized
    windowing and CLAHE enhancement for stroke detection
    """
    def __init__(self, output_size=(640, 640)):
        """
        Initialize the preprocessor layer
        
        Args:
            output_size: Tuple (height, width) for output image size
        """
        super(PreprocessorLayer, self).__init__()
        self.output_size = output_size
        
        # Window settings for brain CT optimized for stroke detection
        self.window_settings = [
            {"center": 35, "width": 45},   # Stroke window - for early ischemic changes
            {"center": 40, "width": 80},   # Standard brain window - for brain parenchyma
            {"center": 80, "width": 200},  # Hemorrhage window - for detecting blood
        ]
        
    def forward(self, x):
        """
        Process batch of DICOM images or paths
        
        Args:
            x: List of DICOM file paths, or batch of already loaded pixel data
               If tensor, expected shape is [batch_size, H, W]
               
        Returns:
            torch.Tensor of shape [batch_size, 3, H, W] with processed images
        """
        if isinstance(x, list):
            # Process list of file paths
            batch_size = len(x)
            processed_batch = []
            
            for dicom_path in x:
                processed_img = self.process_single_dicom(dicom_path)
                processed_batch.append(processed_img)
                
            # Stack tensors along batch dimension
            return torch.stack(processed_batch)
        
        elif isinstance(x, torch.Tensor):
            # Process batch tensor
            batch_size = x.shape[0]
            device = x.device
            processed_batch = []
            
            # Process each image in the batch
            for i in range(batch_size):
                # Convert single image to numpy
                img_np = x[i].detach().cpu().numpy()
                
                # Apply preprocessing
                img_channels = self.apply_windows_and_clahe(img_np)
                
                # Convert back to tensor
                img_tensor = torch.tensor(img_channels, dtype=torch.float32, device=device)
                processed_batch.append(img_tensor)
                
            return torch.stack(processed_batch)
        
        else:
            raise ValueError("Input must be either a list of DICOM paths or a batch tensor")
            
    def process_single_dicom(self, dicom_path_or_data):
        """Process a single DICOM file or dataset"""
        # Load DICOM if path is provided
        if isinstance(dicom_path_or_data, str):
            dicom_data = pydicom.dcmread(dicom_path_or_data)
        else:
            dicom_data = dicom_path_or_data
        
        # Extract pixel array
        dicom_array = dicom_data.pixel_array
        
        # Normalize to Hounsfield units if needed
        if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
            dicom_array = dicom_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
        
        # Apply windowing and CLAHE
        processed_img = self.apply_windows_and_clahe(dicom_array)
        
        # Convert to PyTorch tensor [3, H, W]
        processed_tensor = torch.tensor(processed_img, dtype=torch.float32)
        
        # Normalize to [0, 1] range
        processed_tensor = processed_tensor / 255.0
        
        return processed_tensor
            
    def apply_windows_and_clahe(self, dicom_array):
        """Apply windowing and CLAHE to a single numpy array"""
        channels = []
        for window in self.window_settings:
            # Apply windowing
            img_min = window["center"] - window["width"] // 2
            img_max = window["center"] + window["width"] // 2
            windowed = np.clip(dicom_array, img_min, img_max)
            windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            
            # Apply CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(windowed)
            
            channels.append(enhanced)
        
        # Combine 3 channels
        multichannel_image = np.stack(channels)
        
        # Resize if needed
        if self.output_size:
            resized_channels = []
            for i in range(3):
                resized = cv2.resize(channels[i], 
                                   (self.output_size[1], self.output_size[0]), 
                                   interpolation=cv2.INTER_AREA)
                resized_channels.append(resized)
            multichannel_image = np.stack(resized_channels)
        
        return multichannel_image

class CombinedToEfficientNet(nn.Module):
    def __init__(self, pretrained=True, freeze_base=True):
        super(CombinedToEfficientNet, self).__init__()
        
        # Load pretrained EfficientNetB7 model
        self.efficientnet = models.efficientnet_b7(pretrained=pretrained)
        
        # Get the feature output size (typically 2560 for EfficientNet-B7)
        self.feature_size = self.efficientnet.classifier[1].in_features
        
        # Remove the classifier to get features
        self.efficientnet.classifier = nn.Identity()
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.efficientnet.parameters():
                param.requires_grad = False
    
    def detection_to_heatmap(self, detections, image_size=(640, 640)):
        """
        Convert YOLO detections to heatmap
        
        Args:
            detections: List of detections, each being [x, y, w, h, conf, class]
                        where x,y,w,h are normalized coordinates (0-1)
            image_size: Target image size (height, width)
            
        Returns:
            Tensor of shape (batch_size, 3, height, width) for detection channel
        """
        batch_size = len(detections)
        height, width = image_size
        
        # Create empty heatmaps for detection channel
        heatmaps = torch.zeros(batch_size, 3, height, width, device=detections[0].device 
                              if len(detections) > 0 and isinstance(detections[0], torch.Tensor) 
                              else 'cpu')
        
        for b, detection_list in enumerate(detections):
            # Skip if no detections
            if len(detection_list) == 0:
                continue
                
            # Process each detection
            for det in detection_list:
                # YOLO format: [x_center, y_center, width, height, confidence, class]
                x_center, y_center, box_width, box_height, confidence = det[:5]
                
                # Convert normalized coordinates to pixel values
                x_center = int(x_center * width)
                y_center = int(y_center * height)
                box_width = int(box_width * width)
                box_height = int(box_height * height)
                
                # Calculate box boundaries
                x1 = max(0, int(x_center - box_width / 2))
                y1 = max(0, int(y_center - box_height / 2))
                x2 = min(width - 1, int(x_center + box_width / 2))
                y2 = min(height - 1, int(y_center + box_height / 2))
                
                # Fill the box with confidence value
                heatmaps[b, :, y1:y2+1, x1:x2+1] = confidence
        
        return heatmaps
        
    def segmentation_to_heatmap(self, segmentations, image_size=(640, 640)):
        """
        Convert YOLO segmentations to heatmap
        
        Args:
            segmentations: List of segmentations, each containing polygons
                           where each polygon is [class_id, x1, y1, x2, y2, ...]
                           with x,y being normalized coordinates (0-1)
            image_size: Target image size (height, width)
            
        Returns:
            Tensor of shape (batch_size, 3, height, width) for segmentation channel
        """
        batch_size = len(segmentations)
        height, width = image_size
        
        # Create empty heatmaps for the batch
        heatmaps = torch.zeros(batch_size, 3, height, width, device=segmentations[0].device 
                               if len(segmentations) > 0 and isinstance(segmentations[0], torch.Tensor) 
                               else 'cpu')
        
        # Process each image in the batch
        for b, segmentation_list in enumerate(segmentations):
            # Skip if no segmentations
            if len(segmentation_list) == 0:
                continue
            
            # Create a CPU numpy array for OpenCV operations
            heatmap_np = np.zeros((height, width), dtype=np.float32)
            
            # Process each segmentation polygon
            for seg in segmentation_list:
                # Extract class_id and points
                class_id = seg[0]
                points = seg[1:].reshape(-1, 2)
                
                # Convert normalized points to pixel coordinates
                points_px = points.clone()
                points_px[:, 0] *= width
                points_px[:, 1] *= height
                points_px = points_px.cpu().numpy().astype(np.int32)
                
                # Get confidence (use 1.0 or last element if available)
                confidence = 1.0
                if hasattr(seg, 'confidence'):
                    confidence = seg.confidence
                
                # Draw filled polygon with confidence value
                cv2.fillPoly(heatmap_np, [points_px], confidence)
            
            # Convert back to tensor and assign to all channels
            heatmap_tensor = torch.from_numpy(heatmap_np).to(heatmaps.device)
            heatmaps[b, 0, :, :] = heatmap_tensor
            heatmaps[b, 1, :, :] = heatmap_tensor
            heatmaps[b, 2, :, :] = heatmap_tensor
        
        return heatmaps

    def forward(self, detection_segmentation_tuple):
        """
        Forward pass: Combine detection and segmentation data into a single tensor,
        then feed to EfficientNet
        
        Args:
            detection_segmentation_tuple: Tuple containing:
                - List of detections per image in batch
                - List of segmentations per image in batch
            
        Returns:
            Feature vector from EfficientNet (2560-dim)
        """
        # Unpack tuple
        detections, segmentations = detection_segmentation_tuple
        
        # Convert detections and segmentations to heatmaps
        detection_heatmaps = self.detection_to_heatmap(detections)
        segmentation_heatmaps = self.segmentation_to_heatmap(segmentations)
        
        # Combine the heatmaps:
        # - Use detection for Red channel
        # - Use segmentation for Green channel
        # - Use zeros for Blue channel
        batch_size = detection_heatmaps.shape[0]
        height, width = 640, 640
        
        combined_heatmaps = torch.zeros(batch_size, 3, height, width, 
                                      device=detection_heatmaps.device)
        
        # Red channel - Detection
        combined_heatmaps[:, 0, :, :] = detection_heatmaps[:, 0, :, :]
        
        # Green channel - Segmentation
        combined_heatmaps[:, 1, :, :] = segmentation_heatmaps[:, 0, :, :]
        
        # Blue channel - You could leave as zeros or combine them
        combined_heatmaps[:, 2, :, :] = detection_heatmaps[:, 0, :, :] * segmentation_heatmaps[:, 0, :, :]
        
        # Pass through EfficientNet
        features = self.efficientnet(combined_heatmaps)
        
        return features
    
class BinaryClassificationHead(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(BinaryClassificationHead, self).__init__()
        
        # Input size is fixed at 2560 (EfficientNet feature output)
        in_features = 2560
        
        # Define layers for binary classification
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 1)  # Single output for binary classification
        
    def forward(self, x):
        # First block
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second block
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Output layer
        x = self.dropout3(x)
        x = self.fc3(x)
        
        # No activation here - use with nn.BCEWithLogitsLoss for better numerical stability
        return x
    
class BaksiDetection(nn.Module):
    """
    Complete stroke detection model that combines preprocessing, YOLO detection & segmentation,
    EfficientNet feature extraction, and binary classification
    """
    def __init__(self, det_model_path, seg_model_path, pretrained=True, device='cuda'):
        super(BaksiDetection, self).__init__()
        
        # Initialize components
        self.preprocessor = PreprocessorLayer(output_size=(640, 640))
        self.yolo_predictor = YOLOPredictor(det_model_name=det_model_path, seg_model_name=seg_model_path)
        self.feature_extractor = CombinedToEfficientNet(pretrained=pretrained, freeze_base=True)
        self.classifier = BinaryClassificationHead(dropout_rate=0.3)
        self.device = device
        
        # Move model to device
        self.to(device)
        
        # Freeze all components except the classification head
        self._freeze_components()
    
    def _freeze_components(self):
        """Freeze all components except the classification head"""
        # Freeze preprocessor
        for param in self.preprocessor.parameters():
            param.requires_grad = False
        
        # YOLO predictor doesn't have parameters to freeze in PyTorch sense
        
        # Freeze feature extractor (although it's already frozen in its constructor)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Make sure classifier is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass through the full model
        
        Args:
            x: Input, can be either:
               - List of DICOM file paths
               - Batch tensor of DICOM pixel arrays
               
        Returns:
            Classification logits (0 = no stroke, 1 = stroke)
        """
        # 1. Preprocess DICOM images to get CT with windowing & CLAHE applied
        processed_images = self.preprocessor(x)
        
        # 2. Convert processed tensor images to numpy arrays for YOLO
        image_list = [img.permute(1, 2, 0).cpu().numpy() * 255 for img in processed_images]
        
        # 3. Run YOLO detection and segmentation on the images
        yolo_outputs = self.yolo_predictor.process_batch(image_list)
        
        # 4. Feed YOLO outputs to EfficientNet to extract features
        features = self.feature_extractor(yolo_outputs)
        
        # 5. Classify features with the binary classification head
        logits = self.classifier(features)
        
        return logits

    def train_step(self, batch, criterion, optimizer):
        """
        Perform a single training step
        
        Args:
            batch: Tuple of (inputs, labels)
            criterion: Loss function (BCEWithLogitsLoss recommended)
            optimizer: Optimizer for classifier parameters
            
        Returns:
            Dictionary with loss and accuracy metrics
        """
        self.train()
        inputs, labels = batch
        
        # Move labels to device
        labels = labels.to(self.device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self(inputs)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        predictions = (torch.sigmoid(outputs) >= 0.5).long()
        correct = (predictions.squeeze() == labels).sum().item()
        total = labels.size(0)
        
        return {
            'loss': loss.item(),
            'accuracy': correct / total
        }

    def evaluate(self, dataloader, criterion):
        """
        Evaluate the model on a dataset
        
        Args:
            dataloader: DataLoader with validation/test data
            criterion: Loss function (same as used in training)
            
        Returns:
            Dictionary with loss and accuracy metrics
        """
        self.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                
                # Save predictions and labels
                predictions = (torch.sigmoid(outputs) >= 0.5).cpu().numpy()
                all_predictions.extend(predictions.squeeze())
                all_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
        
        # Calculate metrics
        accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy
        }

    def predict(self, inputs):
        """
        Make predictions on new data
        
        Args:
            inputs: List of DICOM paths or tensor of DICOM images
            
        Returns:
            Dictionary with logits, probabilities and binary predictions
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self(inputs)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= 0.5).long()
            
        return {
            'logits': outputs.cpu(),
            'probabilities': probabilities.cpu(), 
            'predictions': predictions.cpu()
        }