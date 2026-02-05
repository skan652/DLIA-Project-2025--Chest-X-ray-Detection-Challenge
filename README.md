# Chest X-ray Pathology Detection

A deep learning project for detecting pathological findings in chest X-ray images using object detection techniques.

## 🎯 Project Overview

This project implements an automated system for identifying and localizing pathological findings on chest X-ray images using transfer learning with Faster R-CNN. The model is fine-tuned on the NIH ChestXray dataset to detect multiple pathology classes with bounding box predictions.

## 🔬 Methodology

### Model Architecture

- **Base Model**: Faster R-CNN with ResNet-50 FPN (Feature Pyramid Network) backbone
- **Transfer Learning**: Pre-trained on COCO dataset
- **Fine-tuning**: Adapted for medical imaging domain on chest X-ray annotations

### Training Configuration

- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Batch Size**: 4
- **Epochs**: 2
- **Mixed Precision Training**: Implemented using CUDA AMP for faster training
- **Image Resolution**: 1024×1024

### Evaluation Metric

- **mAP@0.5**: Mean Average Precision at IoU threshold of 0.5

## 📊 Dataset

- **Source**: NIH ChestXray dataset
- **Format**: PNG chest X-ray images with bounding box annotations
- **Task**: Multi-class object detection
- **Annotations Include**:
  - Image ID
  - Bounding box coordinates (x_min, y_min, x_max, y_max)
  - Pathology labels
  - Confidence scores

## 🛠️ Technical Stack

### Core Libraries

- **PyTorch** & **TorchVision**: Deep learning framework and computer vision tools
- **Ultralytics**: YOLO implementations
- **Pandas** & **NumPy**: Data manipulation and numerical computing
- **Matplotlib** & **Seaborn**: Data visualization

### Additional Tools

- **PIL/OpenCV**: Image processing
- **Albumentations**: Advanced image augmentation
- **scikit-learn**: Machine learning utilities
- **TorchMetrics**: Model evaluation metrics
- **pycocotools**: COCO dataset utilities

## 📁 Project Structure

```text
.
├── dlia.ipynb    # Main notebook with complete pipeline
├── README.md                  # This file
└── data/                      # Dataset directory (not included)
    ├── train/                 # Training images
    ├── test/                  # Test images
    ├── train.csv             # Training annotations
    └── ID_to_Image_Mapping.csv # Image ID mapping
```

## 🚀 Getting Started

### Prerequisites

```bash
# CUDA-enabled GPU recommended for training
# Python 3.8+
```

### Installation

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install detection and ML libraries
pip install ultralytics pycocotools torchmetrics

# Install data science stack
pip install pandas numpy matplotlib seaborn scikit-learn tqdm

# Install image processing libraries
pip install Pillow opencv-python albumentations
```

### Running the Notebook

1. Open `dlia-project-2025.ipynb` in Jupyter or Kaggle
2. Configure the data paths:

   ```python
   BASE_PATH = "/path/to/your/dataset"
   ```

3. Run cells sequentially to:
   - Load and explore data
   - Set up the model
   - Train the detector
   - Evaluate performance
   - Generate predictions
   - Create submission file

## 📈 Pipeline Workflow

1. **Data Exploration**: Analyze dataset statistics and class distributions
2. **Data Loading**: Custom PyTorch Dataset class for X-ray images
3. **Model Setup**: Initialize Faster R-CNN with custom number of classes
4. **Training**: Fine-tune with mixed precision training
5. **Evaluation**: Compute mAP metrics on validation set
6. **Visualization**: Display predictions vs ground truth
7. **Analysis**: Examine confidence score distributions
8. **Submission**: Generate predictions for test set

## 🎨 Visualization Features

The notebook includes:

- **Prediction Visualization**: Side-by-side comparison of ground truth (green) vs predictions (red)
- **Confidence Distribution**: Histogram of prediction confidence scores
- **Detection Statistics**: Number of predictions per image
- **Sample Analysis**: Random image sampling for qualitative assessment

## 📊 Output Format

Submission CSV format:

```text
ID, image_id, x_min, y_min, x_max, y_max, confidence, label
```

## 🔍 Key Features

- ✅ Transfer learning from COCO to medical imaging domain
- ✅ Mixed precision training for efficiency
- ✅ Comprehensive evaluation metrics
- ✅ Visual analysis tools
- ✅ Confidence threshold filtering
- ✅ Proper handling of missing predictions
- ✅ Multi-worker data loading for performance

## 📝 Notes

- The model uses a confidence threshold of 0.05 for filtering predictions
- For images with no confident predictions, a dummy low-confidence box is added
- Training is optimized with persistent workers and pin memory for faster data loading
- The notebook is designed to run on Kaggle with GPU acceleration

## 🎯 Performance Considerations

- **GPU Memory**: Batch size of 4 is optimal for most GPUs
- **Training Time**: ~2 epochs typically sufficient for initial fine-tuning
- **Inference**: Model runs in evaluation mode for predictions
- **Data Pipeline**: Utilizes 8 workers for efficient data loading

## 🔗 Dependencies

See installation section above for complete list. Key requirements:

- PyTorch >= 2.0
- TorchVision >= 0.15
- CUDA Toolkit (for GPU acceleration)
- Ultralytics YOLOv8 (optional, for comparison)

## 📄 License

This project is for educational and research purposes.

---

**Note**: This project was developed as part of the DLIA Project 2025 for medical imaging analysis.
