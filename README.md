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

**Test-set performance (195 images):**

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 0.667    |
| Linear SVM          | 0.662    |

Confusion matrices are saved in `results/cm_lr.png` and `results/cm_svm.png`.  
These baselines show that simple global statistics are not sufficient for reliable AI-art detection, motivating CNN-based detectors and more forensic-style features in later weeks.