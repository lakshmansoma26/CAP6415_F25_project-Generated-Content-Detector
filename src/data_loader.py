import os

def get_image_paths_and_labels(root):
    image_paths = []
    labels = []

    for label_name in ["real", "fake"]:
        folder = os.path.join(root, label_name)
        if not os.path.exists(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue

        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(folder, filename))
                labels.append(0 if label_name == "real" else 1)

    return image_paths, labels


if __name__ == "__main__":
    train_root = os.path.join("data", "raw", "train")
    test_root  = os.path.join("data", "raw", "test")

    train_paths, train_labels = get_image_paths_and_labels(train_root)
    test_paths,  test_labels  = get_image_paths_and_labels(test_root)

    print("=== Data Loader Verification ===")
    print(f"Train images: {len(train_paths)}")
    print(f"  Real: {sum(1 for l in train_labels if l == 0)}")
    print(f"  Fake: {sum(1 for l in train_labels if l == 1)}")
    print()
    print(f"Test images: {len(test_paths)}")
    print(f"  Real: {sum(1 for l in test_labels if l == 0)}")
    print(f"  Fake: {sum(1 for l in test_labels if l == 1)}")
