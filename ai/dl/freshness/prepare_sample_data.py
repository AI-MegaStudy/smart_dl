from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

from config import DATA_ROOT, DEFAULT_FRUIT_TYPE, TRAIN_RATIO, VALID_RATIO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OBJECT_ID_PATTERN = re.compile(r"apple_fuji_[LMS]_(\d+)-\d+_")


def parse_mapping(mapping_text: str) -> dict[str, str]:
    mapping = {}
    for item in mapping_text.split(","):
        source, target = item.split("=")
        mapping[source.strip()] = target.strip()
    return mapping


def resolve_source_root(source_root: Path) -> Path:
    if source_root.name == "apple_fuji":
        return source_root
    candidates = sorted(p for p in source_root.rglob("apple_fuji") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"Could not find apple_fuji folder under: {source_root}")
    return max(candidates, key=count_images_under)


def count_images_under(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def object_id_from_name(image_path: Path) -> str:
    match = OBJECT_ID_PATTERN.search(image_path.name)
    if match:
        return match.group(1)
    return image_path.stem


def split_images(images: list[Path], args, rng: random.Random) -> dict[str, list[Path]]:
    if not args.group_split:
        rng.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * args.train_ratio)
        n_valid = int(n_total * args.valid_ratio)
        return {
            "train": images[:n_train],
            "valid": images[n_train : n_train + n_valid],
            "test": images[n_train + n_valid :],
        }

    groups: dict[str, list[Path]] = {}
    for image in images:
        groups.setdefault(object_id_from_name(image), []).append(image)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)
    n_total = len(group_ids)
    n_train = int(n_total * args.train_ratio)
    n_valid = int(n_total * args.valid_ratio)
    if n_total >= 3:
        n_valid = max(1, n_valid)
        n_train = min(n_train, n_total - n_valid - 1)
    split_group_ids = {
        "train": group_ids[:n_train],
        "valid": group_ids[n_train : n_train + n_valid],
        "test": group_ids[n_train + n_valid :],
    }
    return {split: [image for group_id in ids for image in groups[group_id]] for split, ids in split_group_ids.items()}


def prepare_sample_data(args):
    source_root = resolve_source_root(Path(args.source_root))
    output_root = Path(args.output_root)
    mapping = parse_mapping(args.mapping)
    rng = random.Random(args.seed)

    if args.overwrite:
        for split in ("train", "valid", "test"):
            fruit_folder = output_root / split / args.fruit_type
            if fruit_folder.exists():
                shutil.rmtree(fruit_folder)

    copied = {"train": 0, "valid": 0, "test": 0}
    for source_label, target_label in mapping.items():
        source_folder = source_root / f"apple_fuji_{source_label}"
        if not source_folder.exists():
            raise FileNotFoundError(f"Source label folder does not exist: {source_folder}")

        images = sorted(p for p in source_folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        splits = split_images(images, args, rng)

        for split, images_for_split in splits.items():
            target_folder = output_root / split / args.fruit_type / target_label
            target_folder.mkdir(parents=True, exist_ok=True)
            for image_path in images_for_split:
                target_name = f"{source_label}_{image_path.name}"
                target_path = target_folder / target_name
                if target_path.exists() and not args.overwrite:
                    continue
                shutil.copy2(image_path, target_path)
                copied[split] += 1

    print("Prepared sample dataset.")
    print(f"source_root={source_root}")
    print(f"output_root={output_root}")
    print(f"mapping={mapping}")
    print(f"group_split={args.group_split}")
    print(f"copied={copied}")
    print()
    print("Label guide from the QC image manual:")
    print("  L -> A: special grade")
    print("  M -> B: high grade")
    print("  S -> C: normal grade")
    print("This dataset is a quality-grade dataset. In this MVP, quality grade is used as a freshness proxy.")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare AI-Hub style apple sample images for ImageFolder training.")
    parser.add_argument("--source-root", default=str(Path("Data") / "Sample"))
    parser.add_argument("--output-root", default=str(DATA_ROOT))
    parser.add_argument("--fruit-type", default=DEFAULT_FRUIT_TYPE)
    parser.add_argument("--mapping", default="L=A,M=B,S=C", help="Example: L=A,M=B,S=C")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--valid-ratio", type=float, default=VALID_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--group-split", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_sample_data(parse_args())
