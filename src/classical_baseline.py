import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data_loader import get_image_paths_and_labels
from features import extract_features


def build_feature_matrix(paths: List[str]) -> np.ndarray:
  
    feats = []
    for i, p in enumerate(paths):
        try:
            f = extract_features(p)
            feats.append(f)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(paths)} images...")
    return np.vstack(feats)


def plot_and_save_confusion_matrix(cm: np.ndarray, title: str, out_path: str):
   
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Write counts in cells
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def run_baselines():
    train_root = os.path.join("data", "raw", "train")
    test_root = os.path.join("data", "raw", "test")

    train_paths, train_labels = get_image_paths_and_labels(train_root)
    test_paths, test_labels = get_image_paths_and_labels(test_root)

    print(f"Train images: {len(train_paths)}")
    print(f"Test images:  {len(test_paths)}")

    # ----- Feature extraction -----
    print("\n[+] Extracting train features...")
    X_train = build_feature_matrix(train_paths)
    y_train = np.array(train_labels[: X_train.shape[0]])

    print("\n[+] Extracting test features...")
    X_test = build_feature_matrix(test_paths)
    y_test = np.array(test_labels[: X_test.shape[0]])

    print(f"\nFeature dimension: {X_train.shape[1]}")

    # Logistic Regression
    
    print("\n[LOGISTIC REGRESSION]")
    lr = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    cm_lr = confusion_matrix(y_test, y_pred_lr)

    print(f"Accuracy: {acc_lr:.4f}")
    print("Confusion matrix:\n", cm_lr)
    print(
        "Classification report:\n",
        classification_report(y_test, y_pred_lr, target_names=["real", "fake"]),
    )

    # Save LR confusion matrix
    plot_and_save_confusion_matrix(
        cm_lr,
        title="Logistic Regression – Confusion Matrix",
        out_path=os.path.join("results", "cm_lr.png"),
    )

    
    # Linear SVM
    
    print("\n[LINEAR SVM]")
    svm = LinearSVC(class_weight="balanced")
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    cm_svm = confusion_matrix(y_test, y_pred_svm)

    print(f"Accuracy: {acc_svm:.4f}")
    print("Confusion matrix:\n", cm_svm)
    print(
        "Classification report:\n",
        classification_report(y_test, y_pred_svm, target_names=["real", "fake"]),
    )

    # Save SVM confusion matrix
    plot_and_save_confusion_matrix(
        cm_svm,
        title="Linear SVM – Confusion Matrix",
        out_path=os.path.join("results", "cm_svm.png"),
    )


    os.makedirs("results", exist_ok=True)

    data = {
        "model": ["logistic_regression", "linear_svm"],
        "accuracy": [acc_lr, acc_svm],
    }

    df = pd.DataFrame(data)
    out_path = os.path.join("results", "week3_dct_baselines.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved baseline metrics to {out_path}")


if __name__ == "__main__":
    run_baselines()
