import cv2
import numpy as np


def lbp_laplacian_features(img: np.ndarray) -> np.ndarray:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Laplacian to emphasize edges/texture
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap = cv2.normalize(lap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    lap = lap.astype(np.uint8)

    # Downsize to limit computation
    lap = cv2.resize(lap, (128, 128), interpolation=cv2.INTER_AREA)

    h, w = lap.shape
    lbp = np.zeros_like(lap, dtype=np.uint8)

    # 8-neighbor LBP with stride 2 
    for y in range(1, h - 1, 2):
        for x in range(1, w - 1, 2):
            center = lap[y, x]
            code = 0
            code |= (lap[y - 1, x - 1] > center) << 7
            code |= (lap[y - 1, x    ] > center) << 6
            code |= (lap[y - 1, x + 1] > center) << 5
            code |= (lap[y,     x + 1] > center) << 4
            code |= (lap[y + 1, x + 1] > center) << 3
            code |= (lap[y + 1, x    ] > center) << 2
            code |= (lap[y + 1, x - 1] > center) << 1
            code |= (lap[y,     x - 1] > center) << 0
            lbp[y, x] = code

    # Histogram over LBP codes 
    mask = np.zeros_like(lbp, dtype=bool)
    mask[1:h - 1:2, 1:w - 1:2] = True

    hist, _ = np.histogram(lbp[mask].ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-8)
    return hist


if __name__ == "__main__":
    import os

    real_dir = os.path.join("data", "raw", "train", "real")
    first_image = None
    for fname in os.listdir(real_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            first_image = os.path.join(real_dir, fname)
            break

    img = cv2.imread(first_image)
    feats = lbp_laplacian_features(img)
    print("LBP feature length:", len(feats))
    print("Sum of histogram:", feats.sum())
