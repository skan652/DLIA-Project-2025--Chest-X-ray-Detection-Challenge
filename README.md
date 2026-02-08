# Chest X-ray Pathology Detection

A deep learning project for detecting pathological findings in chest X-ray images using advanced object detection techniques with extensive optimization for medical imaging.

## 🎯 Project Overview

This project implements a highly optimized automated system for identifying and localizing pathological findings on chest X-ray images using transfer learning with Faster R-CNN. The model is fine-tuned on the NIH ChestXray dataset with a fully unfrozen backbone to detect multiple pathology classes with bounding box predictions. Key innovations include Test Time Augmentation, dynamic threshold optimization, and medical imaging-specific adaptations.

## 🔬 Methodology

### Model Architecture

- **Base Model**: Faster R-CNN with ResNet-50 FPN V2 (Feature Pyramid Network V2) backbone
- **Transfer Learning**: Pre-trained on COCO dataset
- **Backbone Strategy**: **Fully Unfrozen (5 layers)** - Critical for learning medical imaging textures
- **Resolution**: Fixed 1024×1024 (forced min/max size to prevent downscaling)
- **NMS Threshold**: 0.3 (lower threshold for better recall)

### Training Configuration

- **Optimizer**: SGD with momentum 0.9 (optimal for object detection)
- **Base Learning Rate**: 5e-3 with 5-epoch warmup
- **LR Schedule**: Cosine Annealing (eta_min=1e-6)
- **Batch Size**: 4 per step
- **Effective Batch Size**: 16 (using gradient accumulation)
- **Epochs**: 60 maximum with Early Stopping (patience=15)
- **Gradient Clipping**: max_norm=1.0
- **Weight Decay**: 1e-4
- **Image Resolution**: 1024×1024 (no downscaling)

### Data Augmentation

- **Library**: Albumentations (synced bounding box & image transforms)
- **Transforms**:
  - Horizontal Flip (p=0.5)
  - ShiftScaleRotate (shift=0.0625, scale=0.1, rotate=±15°, p=0.5)
  - RandomBrightnessContrast (p=0.5)
  - CoarseDropout/Random Erasing (p=0.2)

### Test Time Augmentation (TTA)

- **Multi-Scale**: 0.9x (zoomed out), 1.0x (base), 1.1x (zoomed in)
- **Horizontal Flip**: Mirror image detection
- **Color Jitter**: Brightness variation handling
- **Ensemble**: NMS-based prediction fusion (IoU=0.3)

### Evaluation Metric

- **Primary**: mAP@0.5 (Mean Average Precision at IoU=0.5)
- **Dynamic Threshold**: Confidence threshold optimized on validation set

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
- **Albumentations**: Advanced image augmentation with bounding box synchronization
- **TorchMetrics**: Model evaluation metrics (MeanAveragePrecision)
- **Pandas** & **NumPy**: Data manipulation and numerical computing
- **Matplotlib**: Data visualization

### Additional Tools

- **PIL**: Image loading and processing
- **pycocotools**: COCO dataset utilities
- **scikit-learn**: Train/validation splitting
- **tqdm**: Progress bars for training monitoring

## 📁 Project Structure

```text
.
├── DLIA.ipynb                 # Main notebook with complete pipeline
├── README.md                  # This file
└── data/                      # Dataset directory (not included)
    ├── train/                 # Training images (PNG format, 1024x1024)
    ├── test/                  # Test images (PNG format, 1024x1024)
    ├── train.csv              # Training annotations with bounding boxes
    └── ID_to_Image_Mapping.csv # Image ID to filename mapping
```

## 🚀 Getting Started

### Prerequisites

- CUDA-enabled GPU strongly recommended for training
- Python 3.8+
- Tested on Kaggle with T4 x 2 GPU

### Installation

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install detection and evaluation libraries
pip install pycocotools torchmetrics

# Install data science stack
pip install pandas numpy matplotlib scikit-learn tqdm

# Install image processing libraries
pip install Pillow albumentations
```

### Running the Notebook

1. Open `DLIA.ipynb` in Jupyter or Kaggle
2. Configure the data paths:

   ```python
   BASE_PATH = "/path/to/your/dataset"
   ```

3. Run cells sequentially to:
   - Load and explore data
   - Set up custom Dataset with Albumentations transforms
   - Initialize Faster R-CNN with fully unfrozen backbone
   - Train with early stopping and gradient accumulation
   - Evaluate performance with mAP metrics
   - Optimize confidence threshold on validation set
   - Apply Test Time Augmentation for inference
   - Generate predictions with ensemble
   - Create submission file

## 📈 Pipeline Workflow

1. **Data Exploration**: Analyze dataset statistics and class distributions
2. **Data Loading**: Custom PyTorch Dataset with Albumentations augmentation
3. **Model Setup**: Initialize Faster R-CNN FPN V2 with 5 unfrozen backbone layers
4. **Training**: 60-epoch training with early stopping (patience=15)
   - 5-epoch warmup with gradual LR increase
   - Cosine annealing LR schedule after warmup
   - Gradient accumulation for effective batch size of 16
   - Gradient clipping for training stability
5. **Evaluation**: Compute mAP@0.5 metrics on validation set
6. **Threshold Optimization**: Find optimal confidence threshold (tested 0.05-0.40)
7. **Visualization**: Display predictions vs ground truth with bounding boxes
8. **Test Time Augmentation**: Multi-scale (0.9x, 1.0x, 1.1x) + flip + color jitter
9. **Submission**: Generate predictions with exactly 1 per image (170 unique IDs)

## 🎨 Visualization Features

The notebook includes:

- **Prediction Visualization**: Side-by-side comparison of ground truth (green) vs predictions (red)
- **Confidence Distribution**: Histogram of prediction confidence scores
- **Detection Statistics**: Number of predictions per image analysis
- **Sample Analysis**: Random image sampling for qualitative assessment
- **Multi-Sample Grid**: 2×2 grid visualization for batch analysis

## 📊 Output Format

Submission CSV format:

```text
ID, image_id, x_min, y_min, x_max, y_max, confidence, label
```

## 🔍 Key Features

### Medical Imaging Adaptations

- ✅ **Fully unfrozen backbone** - Critical for learning X-ray texture features
- ✅ **Fixed 1024×1024 resolution** - No downscaling for maximum detail preservation
- ✅ **Albumentations pipeline** - Synchronized bounding box & image transforms

### Advanced Training Techniques

- ✅ **SGD with momentum 0.9** - Optimal for object detection tasks
- ✅ **Gradient accumulation** - Effective batch size of 16 for stability
- ✅ **Early stopping** - Prevents overfitting with patience monitoring
- ✅ **Cosine annealing** - Smooth learning rate decay after warmup
- ✅ **Gradient clipping** - Training stability with max_norm=1.0

### Inference Optimization

- ✅ **Test Time Augmentation** - Multi-scale (0.9x, 1.0x, 1.1x) ensemble
- ✅ **Dynamic threshold optimization** - Validated on held-out data
- ✅ **Lower NMS threshold** (0.3) - Better recall for medical findings
- ✅ **Multi-worker data loading** - Persistent workers with pin memory

### Quality Assurance

- ✅ **Proper data splitting** - No leakage between train/val sets
- ✅ **Comprehensive evaluation** - mAP@0.5 and per-class metrics
- ✅ **Visual analysis tools** - Ground truth vs prediction comparison
- ✅ **Submission validation** - Exactly 170 unique IDs, proper format

## 📝 Notes

- The model uses a **dynamically optimized confidence threshold** validated on the validation set
- For images with no confident predictions, a dummy low-confidence box is added to maintain submission format
- Training utilizes **persistent workers** and **pin memory** for faster data loading
- The notebook is designed to run on **Kaggle with GPU acceleration** (tested on T4 x 2 GPU)
- **Albumentations** is used instead of torchvision transforms to properly sync bounding box transformations
- **Fully unfrozen backbone** allows the model to learn medical imaging-specific features (soft tissue patterns)
- **Test Time Augmentation** with multi-scale helps detect pathologies of varying sizes
- Submission format constraint: **Exactly 1 prediction per image** (170 unique test IDs)

## 🎯 Performance Considerations

- **GPU Memory**: Batch size of 4 with gradient accumulation (effective 16) optimal for most GPUs
- **Training Time**: ~60 epochs maximum, typically stops early around epoch 30-40
- **Inference**: Model runs in evaluation mode; TTA adds ~5x inference time but improves accuracy
- **Data Pipeline**: Utilizes 4 workers with persistent workers for efficient loading
- **Resolution**: Fixed 1024×1024 (forced min/max) to prevent downscaling artifacts
- **Backbone Training**: All 5 ResNet layers unfrozen for medical domain adaptation
- **Learning Rate**: Aggressive 5e-3 with warmup compensates for unfrozen backbone

## 🔗 Dependencies

See installation section above for complete list. Key requirements:

- PyTorch >= 2.0
- TorchVision >= 0.15
- Albumentations >= 1.3
- TorchMetrics >= 1.0
- CUDA Toolkit (for GPU acceleration)
- pycocotools (for mAP computation)

## 🏆 Model Innovations

This implementation includes several key innovations for medical imaging:

1. **Unfrozen Backbone Strategy**: Unlike standard transfer learning that freezes early layers, this model trains all 5 ResNet layers to learn X-ray-specific texture features
2. **Multi-Scale TTA**: Combines predictions at 0.9x (zoomed out), 1.0x (base), and 1.1x (zoomed in) scales to handle variable pathology sizes
3. **Albumentations Integration**: Fixes bounding box synchronization issues present in standard torchvision transforms
4. **Dynamic Threshold Optimization**: Confidence threshold validated on held-out data rather than using arbitrary values
5. **Gradient Accumulation**: Achieves stable training with effective batch size of 16 despite GPU memory constraints

## 📄 License

This project is for educational and research purposes.

## 🎓 Project Context

Developed as part of the **DLIA Project 2025** (Deep Learning for Image Analysis) for automated chest X-ray pathology detection using state-of-the-art object detection techniques adapted for medical imaging.

### Team Members

- **Skander Adam Afi**
- **Mamadi Keita**
- **Mahdi Znaidi**

## 📊 Results

The model achieves strong performance through:

- Comprehensive data augmentation preventing overfitting
- Fully unfrozen backbone learning medical imaging features
- Test Time Augmentation ensemble improving robustness
- Dynamic threshold optimization maximizing validation mAP

Performance metrics (mAP@0.5) are computed on the validation set and reported in the notebook evaluation cells.

---

**Note**: This implementation prioritizes medical imaging best practices, including maintaining high resolution (1024×1024), unfreezing the backbone for domain adaptation, and using Test Time Augmentation for robust predictions.
