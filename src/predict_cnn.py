import argparse
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int = 2):
    # Same architecture as in cnn_resnet.py
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    transform = get_test_transform()
    tensor = transform(img).unsqueeze(0)  # shape (1, C, H, W)
    return tensor, img


def infer(image_path: str,
          weights_path: str = "results/models/resnet18_best.pth",
          data_root: str = "data/raw"):
    # Determine class order from training folders
    train_dir = os.path.join(data_root, "train")
    classes = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )
    # Example: ['fake', 'real']
    print(f"[INFO] Class order from train folder: {classes}")

    model = build_model(num_classes=len(classes)).to(DEVICE)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tensor, pil_img = load_image(image_path)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_class = classes[pred_idx]
        pred_prob = float(probs[pred_idx])

    print(f"\nImage: {image_path}")
    print(f"Prediction: {pred_class.upper()} (class index {pred_idx})")
    print(f"Confidence: {pred_prob:.4f}")
    print("\nClass probabilities:")
    for idx, cls_name in enumerate(classes):
        print(f"  {cls_name}: {probs[idx]:.4f}")

    return pred_class, pred_prob, dict(zip(classes, probs))


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether an image is AI-generated or real using ResNet-18."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="results/models/resnet18_best.pth",
        help="Path to the trained ResNet-18 weights.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Root folder containing train/ and test/ directories.",
    )

    args = parser.parse_args()

    infer(
        image_path=args.image,
        weights_path=args.weights,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
