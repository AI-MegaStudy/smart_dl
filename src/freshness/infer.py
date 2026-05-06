from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from config import DEFAULT_FRUIT_TYPE, DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_VERSION
from dataset import build_transforms
from model import load_checkpoint
from opencv_features import extract_features
from scoring import FreshnessResult, calculate_freshness_score, make_shipping_decision


def predict_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    fruit_type: str = DEFAULT_FRUIT_TYPE,
    device_name: str | None = None,
) -> FreshnessResult:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(str(checkpoint_path), device)
    labels = checkpoint["labels"]
    image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    model_version = checkpoint.get("model_version", DEFAULT_MODEL_VERSION)

    transform = build_transforms(image_size=image_size, train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    confidence, index = torch.max(probabilities, dim=0)
    predicted_grade = labels[int(index.item())]
    model_confidence = round(float(confidence.item()), 4)

    features = extract_features(image_path, image_size)
    freshness_score = calculate_freshness_score(
        predicted_grade=predicted_grade,
        color_score=features["color_score"],
        roundness_score=features["roundness_score"],
        bruise_probability=features["bruise_probability"],
    )
    shipping_decision = make_shipping_decision(freshness_score, features["bruise_probability"])

    return FreshnessResult(
        fruit_type=fruit_type,
        predicted_grade=predicted_grade,
        freshness_score=freshness_score,
        color_score=features["color_score"],
        roundness_score=features["roundness_score"],
        bruise_probability=features["bruise_probability"],
        shipping_decision=shipping_decision,
        model_confidence=model_confidence,
        model_version=model_version,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run freshness inference for one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fruit-type", default=DEFAULT_FRUIT_TYPE)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_image(args.image, args.checkpoint, args.fruit_type, args.device)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))

