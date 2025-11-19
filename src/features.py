import cv2
import numpy as np

from forensics_dct import dct_block_features
from forensics_lbp import lbp_laplacian_features


def load_bgr(path: str) -> np.ndarray:
 
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def rgb_histogram(img: np.ndarray, bins: int = 32) -> np.ndarray:

    feats = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feats.append(hist)
    return np.concatenate(feats).astype(np.float32)


def edge_stats(img: np.ndarray) -> np.ndarray:
  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    return np.array([mag.mean(), mag.std()], dtype=np.float32)


def fft_highfreq_ratio(img: np.ndarray, low_freq_ratio: float = 0.1) -> np.ndarray:
  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    ry, rx = int(h * low_freq_ratio / 2), int(w * low_freq_ratio / 2)

    low_region = mag[cy-ry:cy+ry, cx-rx:cx+rx]
    low_energy = low_region.sum()
    total_energy = mag.sum() + 1e-8
    high_energy = total_energy - low_energy
    ratio = high_energy / total_energy

    return np.array([ratio], dtype=np.float32)


def extract_features(path: str) -> np.ndarray:
    
    img = load_bgr(path)

    h = rgb_histogram(img)          # 96 dims
    e = edge_stats(img)             # 2 dims
    f = fft_highfreq_ratio(img)     # 1 dim
    dct_feats = dct_block_features(img)  # 8 dims
    return np.concatenate([h, e, f, dct_feats]).astype(np.float32)


if __name__ == "__main__":
    
    import os

    real_dir = os.path.join("data", "raw", "train", "real")
    first_image = None
    for fname in os.listdir(real_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            first_image = os.path.join(real_dir, fname)
            break

    print("Testing feature extraction on:", first_image)
    feats = extract_features(first_image)
    print("Feature vector shape:", feats.shape)
    print("First 10 values:", feats[:10])
