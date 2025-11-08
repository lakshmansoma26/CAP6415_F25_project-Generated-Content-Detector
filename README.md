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