from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


KEY_FIELDS = [
    "model_decision",
    "action_required",
    "retake_reason",
    "angle_label",
    "angle_confidence",
    "view_confidence_threshold",
    "model_grade",
    "grade_confidence",
    "grade_confidence_threshold",
    "freshness_score",
]


def build_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/owner/quality-inspections"):
        return base_url
    return f"{base_url}/owner/quality-inspections"


def post_image(base_url: str, image_path: Path, timeout: int) -> requests.Response:
    endpoint = build_endpoint(base_url)
    with image_path.open("rb") as file:
        return requests.post(
            endpoint,
            files={"image": (image_path.name, file, "application/octet-stream")},
            timeout=timeout,
        )


def print_summary(payload: dict) -> None:
    data = payload.get("data")
    if not isinstance(data, dict):
        return

    print("\nsummary")
    print("-" * 40)
    for key in KEY_FIELDS:
        if key in data:
            print(f"{key}: {data[key]}")
    if payload.get("message"):
        print(f"message: {payload['message']}")
    print("-" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send one apple image to the Kaggle FastAPI ngrok endpoint.",
    )
    parser.add_argument(
        "--url",
        required=True,
        help="ngrok public URL, e.g. https://xxxx.ngrok-free.app",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Local image path to upload.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")

    endpoint = build_endpoint(args.url)
    print(f"POST {endpoint}")
    print(f"image: {image_path}")

    response = post_image(args.url, image_path, args.timeout)
    print(f"status_code: {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        print(response.text)
        response.raise_for_status()
        return

    print_summary(data)
    print("\nfull_response")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    response.raise_for_status()


if __name__ == "__main__":
    main()
