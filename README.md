# Amazon from Space: Satellite Image Recognition in the Amazon Basin

This repository contains a clean, professional, and highly modular **PyTorch** solution for multi-label satellite image recognition and classification in the Amazon basin. The goal is to classify satellite images into multiple categories to track human activities (agriculture, mining, roads, habitation) and natural features (primary forest, rivers, haze, clouds).

The dataset is available on Kaggle under the name **`planets-dataset`**.

---

## 🚀 Project Architecture

The project is designed from the beginning using a modular architecture oriented toward production:

```
Amazon_from_space/
├── README.md                          # Project overview, architecture, and experimental results
├── Amazon_ResNet50_Pipeline.ipynb     # Interactive Jupyter notebook for demonstration and inference
├── amazon_resnet50_v1.pth             # Trained model weights in PyTorch state_dict format
└── src/                               # Centralized source code package
    ├── __init__.py                    # Python package initialization
    ├── config.py                      # Dynamic path configuration and hyperparameters
    ├── dataset.py                     # AmazonDataset class and transformation pipeline
    ├── model.py                       # Network architecture (ResNet50 backbone with adapted FC layer)
    ├── train.py                       # Training and validation loops with F1-Score evaluation
    └── utils.py                       # Visualization and inference helper functions
```

### Module Descriptions:
1. **`src/config.py`**: To manage paths dynamically. To default to the Kaggle environment path `/kaggle/input/...` if detected, otherwise falling back to configurable local paths. It centralizes constants such as `BATCH_SIZE = 32`, `LEARNING_RATE = 1e-3`, and `NUM_CLASSES = 17`.
2. **`src/dataset.py`**: To build the data pipeline via `AmazonDataset` (inheriting from `torch.utils.data.Dataset`), to manage image transformations (Resize to 224x224, ImageNet normalization), and to handle multi-label One-Hot encoding.
3. **`src/model.py`**: To define the `AmazonResnet` class, based on a ResNet50 model with pre-trained weights (`ResNet50_Weights.DEFAULT`). It allows freezing the feature extractor (`freeze_backbone=True`) and replaces the final linear layer for 17 classes.
4. **`src/train.py`**: To implement the training and validation loops. It evaluates the **F1-Score** using a `samples` average and a decision threshold of 0.5.
5. **`src/utils.py`**: To encapsulate helper functions for plotting satellite images using matplotlib and executing inference on test samples.

---

## 📊 Experimental Results and Metrics

The model was trained using the **BCEWithLogitsLoss** criterion (loss function for multi-label classification) and the **Adam** optimizer with a learning rate of `1e-3`.

### 1. Experimental Results with Local Validation (80% Train, 20% Val Split)
To validate the model's generalization capability, the dataset was split into 32,383 training samples and 8,096 validation samples:

| Epoch | Training Loss | Validation Loss | Local Validation F1-Score |
| :---: | :-----------: | :-------------: | :-----------------------: |
| **1** | 0.1241        | 0.1136          | **0.8792**                |
| **2** | 0.1032        | 0.1045          | **0.8866**                |

### 2. Pre-trained Weights and Avoiding Re-training
To avoid re-training the neural network (a computationally intensive process requiring a GPU), it is possible to load the pre-trained weights file: **`amazon_resnet50_v1.pth`** (trained in the cloud). 
With these weights loaded, the model achieves a **88.66%** F1-Score on the local validation set and a validation loss of **0.1045**.

### 3. Inference Samples on the Test Set
To run inference on samples from the test set (40,669 total images) using an activation threshold of `0.5`:
* **Image: `test_37288.jpg`**
  * *Predictions:* `['clear', 'primary', 'water']`
* **Image: `test_9357.jpg`**
  * *Predictions:* `['clear', 'primary']`
* **Image: `test_40104.jpg`**
  * *Predictions:* `['agriculture', 'clear', 'primary']`

---

## 🛠️ How to Use This Project

### Installing Prerequisites
To install the required dependencies in a local or Kaggle environment:
```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib pillow
```

### Running on Kaggle (or local environments with GPU and data)
1. To open the interactive notebook [Amazon_ResNet50_Pipeline.ipynb](Amazon_ResNet50_Pipeline.ipynb).
2. To configure the `AMAZON_DATA_DIR` environment variable to use a custom location for the dataset (by default, the project searches for the dataset in the Kaggle path).
3. To load the pre-trained weights from `amazon_resnet50_v1.pth` to run direct inference on new images and visualize predictions without re-training.

---

## 📄 License
This project is open-source and licensed under the MIT License.
