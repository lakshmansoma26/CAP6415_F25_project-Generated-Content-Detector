# CAP6415_F25_project-Generated-Content-Detector
## Abstract
In this project, a reliable system that can differentiate between artificial intelligence-generated and real images is developed.  
It combines a convolutional neural network-based deep learning detector with traditional image forensics features (such as color statistics, edges, and frequency-domain artifacts).  
Reaching >80% detection accuracy, examining the detector's generalization across various image generators, and assessing its resilience to common distortions like blur and JPEG compression are the objectives.



## Framework
- **Language:** Python 3.10+
- **Libraries:** PyTorch, Torchvision, OpenCV, NumPy, scikit-learn, Matplotlib, Pillow, grad-cam
- **Environment:** Linux/Mac/Windows (development on Windows 10)

## Repository Structure
```text
data/       - datasets (not included in repo)
src/        - source code (data loading, models, evaluation)
results/    - figures, metrics, and saved models
logs/       - weekly development logs



## Week 2 – Classical Baseline

- Dataset: RealArt (real) vs AiArtData (AI-generated) from the cashbowman collection,
  split into 775 train images (347 real, 428 fake) and 195 test images (87 real, 108 fake).
- Feature vector (99D):
  - 3 × 32-bin BGR color histograms
  - Sobel edge magnitude mean and std
  - High-frequency FFT energy ratio
- Models:
  - Logistic Regression (class-balanced)
  - Linear SVM (class-balanced)

Test-set performance (195 images):

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 0.667    |
| Linear SVM          | 0.662    |

Confusion matrices are saved in `results/cm_lr.png` and `results/cm_svm.png`.  
These baselines show that simple global statistics are not sufficient for reliable AI-art detection, motivating CNN-based detectors and more forensic-style features in later weeks.



## Week 3 – Forensic Feature Engineering & Ablation

In Week 3, classical baselines were extended with digital-forensics
inspired features.

 1. 8×8 DCT Block Statistics (Successful)

For each grayscale image, 8×8 DCT blocks were computed and the following
statistics were extracted:

- Mean and std of high-frequency coefficients
- Energy of mid-frequency and high-frequency bands

This produced an 8-dimensional forensic descriptor, increasing the
feature dimension from 99 → 107.

Test-set performance (195 images):

| Model               | Week 2 (Classical) | Week 3 (Classical + DCT) |
|---------------------|--------------------|---------------------------|
| Logistic Regression | 0.667              | ~0.677                    |
| Linear SVM          | 0.662              | ~0.708                |

Metrics are saved in `results/week3_dct_baselines.csv`.

### 2. LBP-on-Laplacian Texture Histogram (Negative Result)

A 256-D LBP histogram over a Laplacian texture map was also tested,
increasing the total feature dimension to 363. However, this did not
improve performance:

| Model               | DCT-only | DCT + LBP |
|---------------------|----------|-----------|
| Logistic Regression | ~0.677   | ~0.667    |
| Linear SVM          | ~0.708   | ~0.672    |

Metrics are saved in `results/week3_forensic_full_baselines.csv`.  
This feature is kept as an ablation result, and the default classical
pipeline uses only the DCT-augmented features, which showed the best
performance.



## Week 4 – Robustness to Distortions

In Week 4, the robustness of the classical DCT-augmented detector was
evaluated under common real-world distortions. The Logistic Regression
and Linear SVM models were trained once on clean training data and then
tested on distorted versions of the test images.

### Distortion Scenarios

The following perturbations were applied to the test set:

- clean – no distortion 
- jpeg_q50 – JPEG compression with quality = 50
- jpeg_q30 – JPEG compression with quality = 30
- blur_5 – Gaussian blur, kernel size 5
- blur_9 – Gaussian blur, kernel size 9
- noise_10 – additive Gaussian noise, σ = 10
- noise_20 – additive Gaussian noise, σ = 20

### Robustness Results (Test Set, 195 Images)

| Scenario  | LR Accuracy | SVM Accuracy |
|----------|-------------|--------------|
| clean    | 0.677       | 0.708        |
| jpeg_q50 | 0.667       | 0.656        |
| jpeg_q30 | 0.662       | 0.646        |
| blur_5   | 0.677       | 0.667        |
| blur_9   | 0.682       | 0.672        |
| noise_10 | 0.574       | 0.590        |
| noise_20 | 0.569       | 0.574        |

Results are stored in `results/week4_robustness.csv`.

### Summary

- The detector is relatively robust to moderate JPEG compression and
  mild Gaussian blur, with only small drops in accuracy compared to the
  clean condition.
- Performance degrades significantly under additive Gaussian noise,
  revealing a key weakness of the current classical + DCT feature
  pipeline.
- These robustness measurements provide a baseline that will be
  compared against future CNN-based detectors and hybrid forensic +
  CNN models in later weeks.




  ## Week 5 – CNN-Based Detector (ResNet-18) and Inference

In Week 5, a deep learning model was introduced to significantly improve
AI-generated image detection performance and make the system usable on
individual images.

### ResNet-18 Training

A ResNet-18 model (pretrained on ImageNet) was fine-tuned on the
cashbowman dataset:

- Train / test split: `data/raw/train` and `data/raw/test`  
  (`real` vs `fake` folders).
- Data augmentation:
  - Resize to 256×256
  - RandomResizedCrop(224)
  - Random horizontal flip
  - Normalization with ImageNet mean and std
- Optimizer: Adam (lr = 1e-4)
- Loss: CrossEntropyLoss
- Epochs: 15, batch size: 32

Best test accuracy: **~0.8564**, compared to ~0.71 for the classical + DCT SVM baseline.

Training history is saved in:
- `results/week5_cnn_resnet18_metrics.csv`

The best model checkpoint is saved in:
- `results/models/resnet18_best.pth`

### Single-Image Prediction (CLI Tool)

To make the system more practical, a single-image inference script was
added:

python src/predict_cnn.py --image path/to/image.jpg

