# Breast Cancer Histopathology Classifier (PyTorch)

A **binary image classifier** for breast cancer histopathology images using **PyTorch**.  
The model distinguishes between **benign** and **malignant** tissue samples, with support for data augmentation, GPU acceleration, and experiment tracking via **Weights & Biases (W&B)**.

---

## 🚀 Features

- ✅ Binary classification: `benign` vs `malignant`  
- ✅ Train **with or without augmentation**  
- ✅ Two model options:
  - **SimpleCNN** (custom lightweight model)
  - **ResNet-18** (transfer learning with ImageNet weights)
- ✅ Automatic GPU usage (`cuda` if available)  
- ✅ Logs metrics to **Weights & Biases (W&B)**  
- ✅ Evaluation metrics: Accuracy, Precision, Recall, F1 Score, and AUC-ROC  

---

## 📂 Dataset Structure

Your dataset directory should look like this:

```
Dataset_2_breast_cancer_histopathology_400X/
├─ train/
│  ├─ benign/
│  └─ malignant/
└─ test/
   ├─ benign/
   └─ malignant/
```

> 📝 The loader automatically ignores `.ipynb_checkpoints` folders.

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install torch torchvision scikit-learn numpy wandb
```

*(For CUDA, choose the appropriate PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/)).*

---

## ▶️ How to Run

1. **Login to W&B** (only once):

```bash
wandb login
```

2. **Run training** (example using ResNet-18 with augmentation):

```bash
python train.py   --data_dir Dataset_2_breast_cancer_histopathology_400X   --model resnet18   --img_size 224   --batch_size 32   --epochs 10   --augment true   --project breast-cancer-classifier   --run_name vgg_19_bn-run1-augmentation
```

### Optional Arguments

| Argument | Description | Default |
|-----------|--------------|----------|
| `--data_dir` | Path to dataset | *required* |
| `--model` | `simplecnn` or `resnet18` | `resnet18` |
| `--img_size` | Image size | `224` |
| `--batch_size` | Batch size | `32` |
| `--epochs` | Number of training epochs | `10` |
| `--augment` | Enable data augmentation | `true` |
| `--project` | W&B project name | `breast-cancer-classifier` |
| `--run_name` | W&B run name | `experiment-1` |

---

## 🧩 Data Transformations

### Without Augmentation (for testing)
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
```

### With Augmentation (for training)
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
```

---

## 🧠 Model Options

### 1️⃣ SimpleCNN (from scratch)
A lightweight CNN trained from scratch:
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
```

### 2️⃣ ResNet-18 (Transfer Learning)
Uses pretrained ImageNet weights:
```python
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1),
    nn.Sigmoid()
)
```

---

## ⚡ Training Details

- **Device:** `cuda` if available, else `cpu`  
- **Loss:** Binary Cross-Entropy (`BCELoss`)  
- **Optimizer:** Adam (`lr=0.001`)  
- **Batch Size:** 32  
- **Epochs:** 10  
- **Threshold:** 0.5 (for binary decision)

Each epoch logs:
- Training loss
- Training accuracy  
to **Weights & Biases (W&B)**

Example log:
```
Epoch 1/10 - Train loss: 31.8232, Train acc: 67.39%
```

---

## 🧪 Evaluation Metrics

After training, the model is evaluated on the test set using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **AUC-ROC**

These metrics are printed and logged to W&B as:

| Metric | Key in W&B |
|---------|-------------|
| Accuracy | `test_accuracy` |
| Precision | `test_precision` |
| Recall | `test_recall` |
| F1 Score | `test_f1` |
| AUC-ROC | `test_auc` |

---

## 📊 Weights & Biases Integration

The script initializes W&B like this:

```python
wandb.init(project="breast-cancer-classifier", name="vgg_19_bn-run1-augmentation")
```
## 👤 Author

**Talat Can Atılgan, MSc., PSPO**  
Product Owner
📍 Luleå University of Technology  
