import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pydicom
import cv2
from torch.utils.data import Dataset
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
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
        self.preprocessor = PreprocessorLayer(output_size=img_size)
        
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
        
        try:
            img_tensor = self.preprocessor.process_single_dicom(img_path)
            
            if self.transform:
                img_tensor = self.transform(img_tensor)
            
            return img_tensor, torch.tensor(label, dtype=torch.float)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            placeholder = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return placeholder, torch.tensor(label, dtype=torch.float)
        
class YOLOPredictor:
    def __init__(self, det_model_name, seg_model_name):
        """
        Initialize with both detection and segmentation models.
        """
        self.det_model = YOLO(det_model_name)
        self.seg_model = YOLO(seg_model_name)
    
    def get_detection_data(self, images, conf=0.001):
        """
        Process images and return detection data formatted for the feature extractor.
        """
        results = self.det_model(images, conf=conf, verbose=False)
        batch_detections = []
        for result in results:
            if result.boxes.xywh.shape[0] > 0:
                xywh = result.boxes.xywh.cpu() / torch.tensor(
                    [result.orig_shape[1], result.orig_shape[0],
                     result.orig_shape[1], result.orig_shape[0]]
                )
                conf_tensor = result.boxes.conf.cpu().unsqueeze(1)
                cls_tensor = result.boxes.cls.cpu().unsqueeze(1)
                detections = torch.cat((xywh, conf_tensor, cls_tensor), dim=1)
            else:
                detections = torch.zeros((0, 6))
            batch_detections.append(detections)
        return batch_detections
    
    def get_segmentation_data(self, images, conf=0.001):
        """
        Process images and return segmentation data from YOLO model
        
        Args:
            images: List of image paths or numpy arrays
            conf: Confidence threshold
            
        Returns:
            Raw segmentation results from YOLO model
        """

        results = self.seg_model(images, conf=conf, verbose=False)
        return results
    
    def process_batch(self, images, conf=0.001):
        """
        Process a batch of images with both detection and segmentation models.
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
               If tensor, expected shape is [batch_size, H, W] or [batch_size, C, H, W]
               
        Returns:
            torch.Tensor of shape [batch_size, 3, H, W] with processed images
        """
        if isinstance(x, list):
            processed_batch = [self.process_single_dicom(dicom_path) for dicom_path in x]
            return torch.stack(processed_batch)
        
        elif isinstance(x, torch.Tensor):
            if x.dim() == 4 and x.shape[1] == 3:
                return x
            
            processed_batch = []
            for img in x:
                processed_batch.append(self.process_single_tensor(img))
            return torch.stack(processed_batch)
        
        else:
            raise ValueError("Input must be either a list of DICOM paths or a batch tensor")
            
    def process_single_dicom(self, dicom_path):
        """Process a single DICOM file"""
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            dicom_array = dicom_data.pixel_array
            
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                dicom_array = dicom_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
            
            processed_img = self.apply_windows_and_clahe(dicom_array)
            
            processed_tensor = torch.tensor(processed_img, dtype=torch.float32)
            
            processed_tensor = processed_tensor / 255.0
            
            return processed_tensor
        
        except Exception as e:
            print(f"Error processing DICOM file {dicom_path}: {e}")
            return torch.zeros((3, self.output_size[0], self.output_size[1]), dtype=torch.float32)
            
    def process_single_tensor(self, img_tensor):
        """Process a single tensor image"""
        try:
            if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:
                return img_tensor
            
            if img_tensor.dim() > 2:
                img_tensor = img_tensor.mean(dim=0) if img_tensor.dim() == 3 else img_tensor.squeeze()
                
            img_np = img_tensor.detach().cpu().numpy()
            processed_img = self.apply_windows_and_clahe(img_np)
            processed_tensor = torch.tensor(processed_img, dtype=torch.float32, device=img_tensor.device)
            processed_tensor = processed_tensor / 255.0
            return processed_tensor
        except Exception as e:
            print(f"Error processing tensor image: {e}")
            return torch.zeros((3, self.output_size[0], self.output_size[1]), dtype=torch.float32, device=img_tensor.device)
            
    def apply_windows_and_clahe(self, dicom_array):
        """Apply windowing and CLAHE to a single numpy array"""
        if dicom_array.ndim > 2:
            dicom_array = np.mean(dicom_array, axis=0) if dicom_array.ndim == 3 else dicom_array.squeeze()
        
        channels = []
        for window in self.window_settings:
            try:
                img_min = window["center"] - window["width"] // 2
                img_max = window["center"] + window["width"] // 2
                windowed = np.clip(dicom_array, img_min, img_max)
                windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                
                if self.output_size:
                    windowed = cv2.resize(windowed, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_AREA)
                
                if windowed.ndim != 2:
                    windowed = windowed.squeeze()
                
                enhanced = clahe.apply(windowed)
                channels.append(enhanced)
            except Exception as e:
                print(f"Error in window {window}: {e}")
                placeholder = np.zeros(self.output_size, dtype=np.uint8)
                channels.append(placeholder)
        
        multichannel_image = np.stack(channels)
        
        return multichannel_image

class SingleEfficientNetExtractor(nn.Module):
    """
    Single EfficientNetB7 feature extraction pathway that processes YOLO detections, segmentations, and DICOM data
    """
    def __init__(self, pretrained=True, freeze_base=True, device='cuda'):
        super(SingleEfficientNetExtractor, self).__init__()
        self.device = device
        
        weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        self.yolo_stream = efficientnet_b7(weights=weights)
        
        if freeze_base:
            self.freeze_base()
        
        self.yolo_stream.features[0][0] = self._modify_first_conv(self.yolo_stream.features[0][0], in_channels=5)
        
        self.feature_size = self.yolo_stream.classifier[1].in_features
        
        self.yolo_stream.classifier = nn.Identity()

    def freeze_base(self):
        """Freeze the base EfficientNet model."""
        for param in self.yolo_stream.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        """Unfreeze only the last two layers of the EfficientNet model."""
        for name, param in self.yolo_stream.named_parameters():
            if 'features.8' in name or 'features.9' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def _modify_first_conv(self, conv, in_channels):
        """Replace a Conv2d layer with a new one expecting in_channels channels.
           New channels are initialized as the average of the original channels.
        """
        old_weight = conv.weight.data
        out_channels = conv.out_channels
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        bias_flag = conv.bias is not None

        new_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=bias_flag)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_weight  
            if in_channels > 3:
                avg = old_weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:in_channels, :, :] = avg.repeat(1, in_channels-3, 1, 1)
            if bias_flag:
                new_conv.bias = conv.bias
        return new_conv
        
    def detection_to_heatmap(self, detections, image_size=(640, 640), batch_size=None):
        if batch_size is None:
            batch_size = len(detections)
        heatmaps = torch.zeros((batch_size, 1, image_size[0], image_size[1]), device=self.device)
        
        if not detections or len(detections) == 0:
            return heatmaps

        for img_idx in range(batch_size):
            img_detections = detections[img_idx]
            if img_detections.ndim < 2:
                continue
            for box_idx in range(img_detections.shape[0]):
                if img_detections.shape[1] < 5:
                    continue
                box = img_detections[box_idx, :4]
                conf = img_detections[box_idx, 4]
                x1 = int((box[0] * image_size[1]).clamp(0, image_size[1]-1))
                y1 = int((box[1] * image_size[0]).clamp(0, image_size[0]-1))
                x2 = int((box[2] * image_size[1]).clamp(0, image_size[1]-1))
                y2 = int((box[3] * image_size[0]).clamp(0, image_size[0]-1))
                if x2 > x1 and y2 > y1:
                    heatmaps[img_idx, 0, y1:y2, x1:x2] += conf
        max_val = heatmaps.max()
        if max_val > 0:
            heatmaps /= max_val
        return heatmaps

    def segmentation_to_heatmap(self, segmentations, image_size=(640, 640), batch_size=None):
        """Convert YOLO segmentation masks to a heatmap tensor"""
        if batch_size is None:
            batch_size = len(segmentations)
            
        heatmaps = torch.zeros((batch_size, 1, image_size[0], image_size[1]), device=self.device)
        
        if not segmentations or len(segmentations) == 0:
            return heatmaps
            
        for img_idx in range(min(batch_size, len(segmentations))):
            result = segmentations[img_idx]
            
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                mask_tensor = result.masks.data
                
                if len(mask_tensor) > 0:
                    mask = mask_tensor[0]
                    
                    mask_resized = torch.nn.functional.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=image_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                    
                    heatmaps[img_idx, 0] = mask_resized
        
        max_val = heatmaps.max()
        if max_val > 0:
            heatmaps = heatmaps / max_val
            
        return heatmaps

    def create_combined_tensor(self, detection_segmentation_tuple, dicom_tensor):
        detections, segmentations = detection_segmentation_tuple
        batch_size = dicom_tensor.shape[0]
        det_heatmap = self.detection_to_heatmap(detections, image_size=(dicom_tensor.shape[2], dicom_tensor.shape[3]), batch_size=batch_size)
        seg_heatmap = self.segmentation_to_heatmap(segmentations, image_size=(dicom_tensor.shape[2], dicom_tensor.shape[3]), batch_size=batch_size)
        
        if dicom_tensor.dim() == 3:
            dicom_tensor = dicom_tensor.unsqueeze(0)
            
        combined = torch.cat([dicom_tensor, det_heatmap, seg_heatmap], dim=1)
        return combined

    def forward(self, inputs):
        """Forward pass through single EfficientNet pathway
        
        Args:
            inputs: Tuple of (yolo_outputs, dicom_tensor)
                - yolo_outputs: Tuple of (detections, segmentations)
                - dicom_tensor: Preprocessed DICOM image tensor [batch, 3, H, W]
        
        Returns:
            Feature vector from the EfficientNet model
        """
        yolo_outputs, dicom_tensor = inputs
        
        device = next(self.parameters()).device
        
        if dicom_tensor.device != device:
            dicom_tensor = dicom_tensor.to(device)
            
        stream_input = self.create_combined_tensor(yolo_outputs, dicom_tensor)
        
        stream_input = stream_input.to(device, dtype=torch.float)
        
        features = self.yolo_stream(stream_input)
        
        return features

class EnhancedClassificationHead(nn.Module):
    """
    Enhanced classification head with strong regularization to prevent overfitting
    """
    def __init__(self, in_features=2560, dropout_rate=0.5):
        super(EnhancedClassificationHead, self).__init__()
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.dropout4(x)
        x = self.fc4(x)
        
        return x

class BaksiDetection(nn.Module):
    """
    Complete stroke detection model with single EfficientNet architecture
    """
    def __init__(self, det_model_path, seg_model_path, pretrained=True, device='cuda', **kwargs):
        """
        Initialize the model with backward compatibility for train.py
        
        Args:
            det_model_path: Path to YOLO detection model
            seg_model_path: Path to YOLO segmentation model
            pretrained: Whether to use pretrained EfficientNet weights
            device: Device to run the model on
            **kwargs: Extra parameters for compatibility with train.py
        """
        super(BaksiDetection, self).__init__()
        
        self.preprocessor = PreprocessorLayer(output_size=(640, 640))
        self.yolo_predictor = YOLOPredictor(det_model_name=det_model_path, seg_model_name=seg_model_path)
        self.feature_extractor = SingleEfficientNetExtractor(pretrained=pretrained, freeze_base=True)
        self.classifier = EnhancedClassificationHead(in_features=self.feature_extractor.feature_size)
        self.device = device
        
        self.to(device)
    
    def forward(self, x):
        """
        Forward pass through the full model
        
        Args:
            x: Input, can be either:
            - List of DICOM file paths
            - Batch tensor of DICOM pixel arrays
            - Dictionary with 'windowed_img' for compatibility with train.py
                
        Returns:
            Classification logits (0 = no stroke, 1 = stroke)
        """
        if isinstance(x, dict) and 'windowed_img' in x:
            dicom_tensor = x['windowed_img']
            
            image_list = [img.permute(1, 2, 0).cpu().numpy() * 255 for img in dicom_tensor]
        else:
            dicom_tensor = self.preprocessor(x)

            image_list = [img.permute(1, 2, 0).cpu().numpy() * 255 for img in dicom_tensor]
        
        yolo_outputs = self.yolo_predictor.process_batch(image_list)
        
        features = self.feature_extractor((yolo_outputs, dicom_tensor))
        
        logits = self.classifier(features)
        
        return logits
  
    def freeze_backbone(self):
        """Freeze the EfficientNet backbones."""
        self.feature_extractor.freeze_base()

    def unfreeze_backbone(self):
        """Unfreeze only the last two layers of the EfficientNet backbones."""
        self.feature_extractor.unfreeze_base()    
    
    def train_step(self, batch, criterion, optimizer):
        """Perform a single training step"""
        self.train()
        inputs, labels = batch
        
        labels = labels.to(self.device)
        
        optimizer.zero_grad()
        
        outputs = self(inputs)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        predictions = (torch.sigmoid(outputs) >= 0.5).long()
        correct = (predictions.squeeze() == labels).sum().item()
        total = labels.size(0)
        
        return {
            'loss': loss.item(),
            'accuracy': correct / total
        }

    def evaluate(self, dataloader, criterion):
        """Evaluate the model on a dataset"""
        self.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                labels = labels.to(self.device)
                
                outputs = self(inputs)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                
                predictions = (torch.sigmoid(outputs) >= 0.5).cpu().numpy()
                all_predictions.extend(predictions.squeeze())
                all_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
        
        accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy
        }

    def predict(self, inputs):
        """Make predictions on new data"""
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