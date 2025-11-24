import os
from typing import List, Callable, Tuple

import cv2
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from data_loader import get_image_paths_and_labels
from features import rgb_histogram, edge_stats, fft_highfreq_ratio, load_bgr
from forensics_dct import dct_block_features




def extract_features_from_img(img: np.ndarray) -> np.ndarray:
    """
    Mirror of the current classical + DCT pipeline, but takes an image
    array instead of a path.
    """
    h = rgb_histogram(img)              # 96 dims
    e = edge_stats(img)                 # 2 dims
    f = fft_highfreq_ratio(img)         # 1 dim
    dct_feats = dct_block_features(img) # 8 dims

    return np.concatenate([h, e, f, dct_feats]).astype(np.float32)




def apply_jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode(".jpg", img, encode_param)
    if not result:
        raise RuntimeError("Failed to encode image for JPEG compression")
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return decimg


def apply_gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    # ksize must be odd
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy



SCENARIOS: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("clean", lambda x: x),
    ("jpeg_q50", lambda x: apply_jpeg_compression(x, 50)),
    ("jpeg_q30", lambda x: apply_jpeg_compression(x, 30)),
    ("blur_5", lambda x: apply_gaussian_blur(x, 5)),
    ("blur_9", lambda x: apply_gaussian_blur(x, 9)),
    ("noise_10", lambda x: apply_gaussian_noise(x, 10.0)),
    ("noise_20", lambda x: apply_gaussian_noise(x, 20.0)),
]




def build_feature_matrix_from_paths(paths: List[str]) -> np.ndarray:
    feats = []
    for i, p in enumerate(paths):
        try:
            img = load_bgr(p)
            f = extract_features_from_img(img)
            feats.append(f)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(paths)} images...")
    return np.vstack(feats)


def build_feature_matrix_with_distortion(
    paths: List[str],
    distortion_fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    feats = []
    for i, p in enumerate(paths):
        try:
            img = load_bgr(p)
            img_distorted = distortion_fn(img)
            f = extract_features_from_img(img_distorted)
            feats.append(f)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(paths)} images...")
    return np.vstack(feats)


def evaluate_robustness():
    train_root = os.path.join("data", "raw", "train")
    test_root = os.path.join("data", "raw", "test")

    train_paths, train_labels = get_image_paths_and_labels(train_root)
    test_paths, test_labels = get_image_paths_and_labels(test_root)

    print(f"Train images: {len(train_paths)}")
    print(f"Test images:  {len(test_paths)}")

    
    print("\n[+] Extracting CLEAN train features...")
    X_train = build_feature_matrix_from_paths(train_paths)
    y_train = np.array(train_labels[: X_train.shape[0]])

    print("\n[+] Training baseline models on CLEAN train features...")
    lr = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced")
    svm = LinearSVC(class_weight="balanced")

    lr.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    results = []

    
    for name, distortion_fn in SCENARIOS:
        print(f"\n[SCENARIO] {name}")
        print("[+] Extracting distorted test features...")
        X_test = build_feature_matrix_with_distortion(test_paths, distortion_fn)
        y_test = np.array(test_labels[: X_test.shape[0]])

        # Logistic Regression
        y_pred_lr = lr.predict(X_test)
        acc_lr = accuracy_score(y_test, y_pred_lr)

        # Linear SVM
        y_pred_svm = svm.predict(X_test)
        acc_svm = accuracy_score(y_test, y_pred_svm)

        print(f"  LR  accuracy: {acc_lr:.4f}")
        print(f"  SVM accuracy: {acc_svm:.4f}")

        results.append({
            "scenario": name,
            "accuracy_lr": acc_lr,
            "accuracy_svm": acc_svm,
        })

    # save results
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    out_path = os.path.join("results", "week4_robustness.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved robustness results to {out_path}")
    print(df)


if __name__ == "__main__":
    evaluate_robustness()
