import os
import shutil
import random

random.seed(42)

def split_folder(src, train_dst, test_dst, split_ratio=0.8):
    images = [f for f in os.listdir(src)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(images) == 0:
        raise RuntimeError(f"No images found in {src}")

    print(f"Found {len(images)} images in {src}")

    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)
    train_files = images[:split_idx]
    test_files = images[split_idx:]

    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(test_dst, exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(src, f),
                    os.path.join(train_dst, f))

    for f in test_files:
        shutil.copy(os.path.join(src, f),
                    os.path.join(test_dst, f))

    print(f"{src} -> {train_dst}: {len(train_files)} train")
    print(f"{src} -> {test_dst}:  {len(test_files)} test")


if __name__ == "__main__":
    base_all = os.path.join("data", "raw", "all")

    # Your actual dataset names
    real_src = os.path.join(base_all, "real")
    fake_src = os.path.join(base_all, "fake")

    train_real = os.path.join("data", "raw", "train", "real")
    test_real  = os.path.join("data", "raw", "test", "real")

    train_fake = os.path.join("data", "raw", "train", "fake")
    test_fake  = os.path.join("data", "raw", "test", "fake")

    split_folder(real_src, train_real, test_real, split_ratio=0.8)
    split_folder(fake_src, train_fake, test_fake, split_ratio=0.8)

    print("Done splitting dataset.")
