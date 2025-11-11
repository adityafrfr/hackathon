# Safety Equipment Detection

A dual-framework object detection project for identifying safety equipment in industrial environments using both **YOLO (Ultralytics)** and **TensorFlow/Keras** implementations.

## 🎯 Overview

This project detects 7 classes of safety equipment:
- OxygenTank
- NitrogenTank
- FirstAidBox
- FireAlarm
- SafetySwitchPanel
- EmergencyPhone
- FireExtinguisher

The project provides two distinct training pipelines:
1. **YOLO (YOLOv8)** - Using Ultralytics framework
2. **TensorFlow Custom** - Custom YOLO-like architecture with MobileNetV2 backbone

## 📋 Requirements

```bash
pip install -r requirements.txt
```

### Key Dependencies
- TensorFlow 2.20.0
- Ultralytics 8.3.226
- PyTorch 2.9.0
- OpenCV 4.12.0.88
- NumPy 2.2.6
- Keras 3.12.0

## 📁 Project Structure

```
.
├── train.py              # YOLO training script
├── traintf.py            # TensorFlow custom model training
├── predict.py            # YOLO inference script
├── predict_tf.py         # TensorFlow inference script
├── visualize.py          # Dataset visualization tool
├── classes.txt           # Class names
├── yolo_params.yaml      # Dataset configuration
├── requirements.txt      # Python dependencies
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── runs/                 # YOLO training outputs
├── tf_runs/              # TensorFlow training outputs
│   ├── train/
│   ├── validation/
│   └── weights/
└── predictions/          # Inference outputs
```

## 🚀 Usage

### 1. Training

#### YOLO Training

```bash
python train.py --epochs 10 --lr0 0.0001 --lrf 0.0001 --mosaic 0.4 --optimizer AdamW
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 10)
- `--mosaic`: Mosaic augmentation probability (default: 0.4)
- `--optimizer`: Optimizer choice (default: AdamW)
- `--momentum`: SGD momentum (default: 0.9)
- `--lr0`: Initial learning rate (default: 0.0001)
- `--lrf`: Final learning rate (default: 0.0001)
- `--single_cls`: Single class training mode (default: False)

**Output:** Trained weights saved in `runs/detect/trainN/weights/best.pt`

#### TensorFlow Training

```bash
python traintf.py
```

**Key Features:**
- **VRAM-Friendly**: 4GB VRAM optimized with gradient accumulation
- **Image Size**: 512x512
- **Grid Size**: 13x13
- **Batch Size**: 2 (with 8-step accumulation = effective batch of 16)
- **Epochs**: 200 with early stopping (patience: 30)
- **Backbone**: MobileNetV2 (ImageNet pretrained)
- **Learning Rate**: Warmup + Cosine decay schedule
- **Augmentation**: Random flip, brightness, contrast

**Configuration** (in `traintf.py`):
```python
IMG_SIZE = 512
GRID_SIZE = 13
BATCH_SIZE = 2
ACCUM_STEPS = 8
EPOCHS = 200
WARMUP_EPOCHS = 5
```

**Output:** 
- Best model: `tf_runs/weights/best.keras`
- Final model: `tf_runs/weights/final.keras`
- TensorBoard logs: `tf_runs/`

**Monitor Training:**
```bash
tensorboard --logdir tf_runs
```

### 2. Inference

#### YOLO Inference

```bash
python predict.py
```

This script will:
1. Automatically detect available trained models in `runs/detect/`
2. Let you select which training run to use
3. Run inference on test images
4. Save predictions to `predictions/` (images + labels)
5. Display validation metrics

#### TensorFlow Inference

```bash
python predict_tf.py
```

**Configuration:**
```python
MODEL_PATH = "tf_runs/weights/best.keras"  # or final.keras
TEST_DIR = "dataset/test/images"
CONF_THRES = 0.5
```

Outputs annotated images to `predictions/` directory.

### 3. Dataset Visualization

```bash
python visualize.py
```

**Interactive Controls:**
- `d` - Next image
- `a` - Previous image
- `t` - Switch to training set
- `v` - Switch to validation set
- `q` or `ESC` - Quit

## 📊 Dataset Format

### YOLO Label Format
Each `.txt` file contains bounding boxes in normalized format:
```
class_id x_center y_center width height
```

Where all coordinates are normalized (0-1) relative to image dimensions.

### Configuration File (`yolo_params.yaml`)
```yaml
train: /path/to/dataset/train
val: /path/to/dataset/val
test: /path/to/dataset/test
nc: 7
names: ['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm', 
        'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']
```

## 🏗️ Model Architectures

### YOLO (YOLOv8s)
- Parameters: ~3M
- Architecture: CSPDarknet backbone with PAN neck
- Detection head: Anchor-free with decoupled heads
- GFLOPs: 8.2

### TensorFlow Custom
- Backbone: MobileNetV2 (ImageNet pretrained)
- Architecture: Single-shot detection on 13x13 grid
- Output: (batch, 13, 13, 5+num_classes)
  - Box coords: tx, ty (sigmoid)
  - Box size: tw, th (sigmoid)
  - Objectness: 1 channel (sigmoid)
  - Class logits: 7 channels (raw)
- Loss: Custom YOLO v1-style loss
  - Box regression (L1)
  - Objectness (BCE)
  - Classification (CCE)

### Loss Components
```python
total_loss = λ_box × box_loss 
           + λ_obj × obj_loss 
           + λ_noobj × noobj_loss 
           + λ_cls × cls_loss

# Defaults:
λ_box = 5.0
λ_obj = 1.0
λ_noobj = 0.5
λ_cls = 1.0
```

## 🔧 Advanced Features

### TensorFlow Training Features
1. **Gradient Accumulation**: Simulates larger batch sizes on limited VRAM
2. **Learning Rate Scheduling**: Warmup (5 epochs) + Cosine decay
3. **Progressive Unfreezing**: Backbone unfrozen after warmup
4. **Early Stopping**: Patience of 30 epochs on validation loss
5. **ReduceLROnPlateau**: Halves LR after 10 epochs without improvement
6. **TensorBoard Integration**: Real-time training visualization
7. **Mixed Precision Ready**: Optional FP16 training (set `MIXED_PRECISION=True`)

### Custom Callbacks
- **UnfreezeCallback**: Automatically unfreezes backbone after warmup
- **ModelCheckpoint**: Saves best model based on validation loss
- **EarlyStopping**: Prevents overfitting
- **ReduceLROnPlateau**: Adaptive learning rate

## 🎮 Training Tips

### YOLO
- **Mosaic augmentation**: Keep at 0.4-0.7 (not 1.0)
- **Learning rates**: Start with lr0=0.0001, gradually decay
- **Optimizer**: AdamW works best for this dataset

### TensorFlow
- **Memory issues**: Reduce `BATCH_SIZE` or `IMG_SIZE`
- **Faster training**: Increase `BATCH_SIZE` if VRAM allows
- **Better accuracy**: Increase `IMG_SIZE` to 640
- **Convergence issues**: Check warmup duration and base LR

## 📈 Model Serialization

Both models use Keras 3 serialization with custom objects:

```python
# Saving (automatic in training scripts)
model.save("path/to/model.keras")

# Loading (for inference)
model = tf.keras.models.load_model(
    "path/to/model.keras",
    custom_objects={"split_activation": split_activation}
)
```

## 🐛 Troubleshooting

### CUDA/GPU Issues
```python
# Already handled in scripts:
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
tf.config.optimizer.set_jit(False)
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)
```

### OOM (Out of Memory)
- Reduce `BATCH_SIZE` in `traintf.py`
- Reduce `IMG_SIZE` to 416 or 320
- Increase `ACCUM_STEPS` to maintain effective batch size

### Poor Detection Performance
- Check dataset quality with `visualize.py`
- Ensure labels are normalized (0-1 range)
- Verify class distribution is balanced
- Increase training epochs
- Try different confidence thresholds during inference

## 📝 Notes

- **YOLO models** are stored in `.pt` format (PyTorch)
- **TensorFlow models** use `.keras` format (Keras 3)
- Test set evaluation is automatic during YOLO inference
- TensorBoard logs are separate for train/validation
- Predictions include both visualized images and label files

## 🔬 Experimental Notes

From `train.py`:
> Mixup boosts validation prediction but reduces test prediction  
> Mosaic shouldn't be 1.0

## 📄 License

This project uses:
- Ultralytics YOLO (AGPL-3.0)
- TensorFlow (Apache 2.0)
- MobileNetV2 (Apache 2.0)

## 🤝 Contributing

To add new classes:
1. Update `classes.txt`
2. Update `yolo_params.yaml` (nc and names)
3. Retrain both models

## 📧 Contact

For issues or questions, please open an issue in the repository.

---

**Happy Detecting! 🔍🦺**
