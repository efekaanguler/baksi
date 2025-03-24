# Baksi - Stroke Detection Model

This repository contains the code and model architecture for the stroke detection system developed by Team Baksi for the Teknofest 2025 Artificial Intelligence in Healthcare Competition.

## Model Architecture

The model is designed to detect strokes from DICOM images using a multi-step approach. The architecture consists of the following components:

1. **Preprocessing Step**: 
   - The DICOM file undergoes three different windowing operations, and the results are mapped to three separate channels.
2. **YOLO11L & YOLO11L-SEG**:
   - These models process the windowed images for object detection and segmentation.
3. **Rebuilding Greyscale Images**:
   - The original unwindowed 8-bit image and the confidence scores from YOLO models are used to reconstruct greyscale images.
4. **EfficientNet-B7**:
   - The rebuilt greyscale images are combined into three channels and fed into the EfficientNet-B7 model for feature extraction.
5. **Classification Head**:
   - The EfficientNet output is passed through a classification head to perform binary stroke classification.

The model leverages both classification and segmentation techniques to accurately identify stroke regions in medical images.

## Team Baksi

Team Baksi is composed of dedicated members with diverse expertise in AI, medical research, and data preparation. The team members are:

- **Efe Kaan Güler** - Team Leader and Architecture Design
- **Berkay Özoğlu** - Literature Review and Medical Guidance
- **İzgi Su Göğüs** - Literature Review and Medical Guidance
- **İbrahim Akpınar** - Dataset Acquisition and YOLO Training Data Preparation

## Project Overview

This project was developed for the Teknofest 2025 Artificial Intelligence in Healthcare Competition. The goal of the competition is to create innovative AI solutions to improve healthcare outcomes. Our model aims to assist medical professionals in the early detection of strokes, potentially saving lives and improving patient outcomes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

