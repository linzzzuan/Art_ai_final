"""Prepare AffectNet dataset: remap labels, skip Contempt, split into project format.

Reads from:   backend/data/AffectNetCustom/{train,test}/{0..7}/
Writes to:    backend/datasets/real/{train,val}/{0_angry,1_disgust,...,6_neutral}/

Usage:
    cd backend
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --source data/AffectNetCustom --target datasets/real --copy
"""

import argparse
import shutil
from pathlib import Path

# AffectNet standard: 0=Neutral, 1=Happy, 2=Sad, 3=Surprise, 4=Fear, 5=Disgust, 6=Anger, 7=Contempt
# Project format:     0=angry,   1=disgust, 2=fear, 3=happy,   4=sad, 5=surprise, 6=neutral
AFFECTNET_TO_PROJECT = {
    0: "6_neutral",
    1: "3_happy",
    2: "4_sad",
    3: "5_surprise",
    4: "2_fear",
    5: "1_disgust",
    6: "0_angry",
    # 7 = Contempt → skipped
}

SPLIT_MAP = {
    "train": "train",
    "test": "val",
}


def prepare(source: Path, target: Path, use_copy: bool = False):
    action = shutil.copy2 if use_copy else shutil.move
    action_name = "Copying" if use_copy else "Moving"

    total = 0
    for src_split, dst_split in SPLIT_MAP.items():
        src_dir = source / src_split
        if not src_dir.exists():
            print(f"  [SKIP] {src_dir} does not exist")
            continue

        for affectnet_idx, project_dir in AFFECTNET_TO_PROJECT.items():
            src_class = src_dir / str(affectnet_idx)
            if not src_class.exists():
                print(f"  [SKIP] {src_class} does not exist")
                continue

            dst_class = target / dst_split / project_dir
            dst_class.mkdir(parents=True, exist_ok=True)

            images = [
                f for f in src_class.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            ]
            print(f"  {action_name} {len(images):>5} images: {src_split}/{affectnet_idx} -> {dst_split}/{project_dir}")

            for img in images:
                action(str(img), str(dst_class / img.name))
                total += 1

    print(f"\nDone! Total images processed: {total}")

    # Print summary
    print("\n=== Dataset Summary ===")
    for split in ["train", "val"]:
        split_dir = target / split
        if not split_dir.exists():
            continue
        print(f"\n{split}/")
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                count = len([f for f in cls_dir.iterdir() if f.is_file()])
                print(f"  {cls_dir.name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Prepare AffectNet dataset for project")
    parser.add_argument("--source", type=str, default="data/AffectNetCustom",
                        help="Source dataset path (default: data/AffectNetCustom)")
    parser.add_argument("--target", type=str, default="datasets/real",
                        help="Target dataset path (default: datasets/real)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of moving them")
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)

    if not source.exists():
        print(f"Error: Source directory does not exist: {source}")
        return

    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Mode:   {'copy' if args.copy else 'move'}")
    print()

    prepare(source, target, use_copy=args.copy)


if __name__ == "__main__":
    main()
