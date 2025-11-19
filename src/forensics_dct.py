import cv2
import numpy as np


def dct_block_features(img: np.ndarray, block_size: int = 8) -> np.ndarray:
  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Ensure divisible by block_size
    h_pad = h - h % block_size
    w_pad = w - w % block_size
    gray = gray[:h_pad, :w_pad]

    blocks_h = h_pad // block_size
    blocks_w = w_pad // block_size

    hf_means = []
    hf_stds = []
    mid_energy = []
    high_energy = []

    for by in range(blocks_h):
        for bx in range(blocks_w):
            block = gray[
                by * block_size:(by + 1) * block_size,
                bx * block_size:(bx + 1) * block_size
            ].astype(np.float32)

            # Apply 2D DCT
            dct = cv2.dct(block)

            # Define frequency masks
            mid_mask = np.zeros_like(dct, dtype=np.uint8)
            high_mask = np.zeros_like(dct, dtype=np.uint8)

            # Mid frequency band (diagonal middle)
            mid_mask[2:6, 2:6] = 1

            # High frequency band (corners)
            high_mask[6:, 6:] = 1

            dct_mid = dct[mid_mask == 1]
            dct_high = dct[high_mask == 1]

            # Extract stats
            hf_means.append(np.mean(dct_high))
            hf_stds.append(np.std(dct_high))
            mid_energy.append(np.sum(dct_mid ** 2))
            high_energy.append(np.sum(dct_high ** 2))

    # Aggregate features
    features = np.array([
        np.mean(hf_means), np.std(hf_means),
        np.mean(hf_stds), np.std(hf_stds),
        np.mean(mid_energy), np.std(mid_energy),
        np.mean(high_energy), np.std(high_energy)
    ], dtype=np.float32)

    return features


if __name__ == "__main__":
    import os

    test_path = os.listdir("data/raw/train/real")[0]
    img = cv2.imread(os.path.join("data/raw/train/real", test_path))

    feats = dct_block_features(img)
    print("DCT feature vector length:", len(feats))
    print("DCT features:", feats)
