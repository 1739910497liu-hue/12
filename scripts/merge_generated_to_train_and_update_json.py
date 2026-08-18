import argparse
import glob
import json
import re
import shutil
from pathlib import Path


def next_image_index(folder, cache):
    if folder in cache:
        return cache[folder]

    max_idx = -1
    for path in folder.glob("image-*.*"):
        match = re.match(r"image-(\d+)\.[^.]+$", path.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    cache[folder] = max_idx + 1
    return cache[folder]


def build_caption(prompt, params):
    if params is None or len(params) != 4:
        return prompt or ""

    voltage, current, speed, gas_flow = [float(value) for value in params]
    return (
        f"{prompt or ''} "
        f"Process parameters: voltage {voltage:g} V, current {current:g} A, "
        f"travel speed {speed:g} cm/min, shielding gas flow {gas_flow:g} L/min."
    ).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Merge generated grid samples into their source welding-process folders and update train_text.json."
    )
    parser.add_argument("--train-root", default="datasets/ss304/train")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--pattern", default="grid_params_*/manifest.jsonl")
    parser.add_argument("--json-path", default="datasets/text/train_text.json")
    parser.add_argument("--backup", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    outputs_root = Path(args.outputs_root)
    json_path = Path(args.json_path)

    with open(json_path, "r", encoding="utf-8") as file:
        train_text = json.load(file)

    backup_path = json_path.with_suffix(json_path.suffix + ".bak")
    if args.backup and not args.dry_run and not backup_path.exists():
        shutil.copy2(json_path, backup_path)

    index_cache = {}
    copied = 0
    missing_manifest_items = 0
    skipped_missing_images = 0
    overwritten_json_keys = 0

    if Path(args.pattern).is_absolute():
        manifest_paths = [Path(path) for path in glob.glob(args.pattern)]
    else:
        manifest_paths = list(outputs_root.glob(args.pattern))

    for manifest_path in sorted(manifest_paths):
        generated_dir = manifest_path.parent

        with open(manifest_path, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                item = json.loads(line)
                output = item.get("output")
                source_relpath = item.get("source_relpath")

                if not output or not source_relpath:
                    missing_manifest_items += 1
                    continue

                src_img = generated_dir / output
                if not src_img.exists():
                    skipped_missing_images += 1
                    continue

                source_parent = Path(source_relpath).parent
                dst_dir = train_root / source_parent
                idx = next_image_index(dst_dir, index_cache)
                dst_img = dst_dir / f"image-{idx:04d}{src_img.suffix.lower()}"
                while dst_img.exists():
                    idx += 1
                    dst_img = dst_dir / f"image-{idx:04d}{src_img.suffix.lower()}"

                index_cache[dst_dir] = idx + 1
                rel_key = str(dst_img.relative_to(train_root)).replace("\\", "/")

                if rel_key in train_text:
                    overwritten_json_keys += 1

                train_text[rel_key] = build_caption(item.get("prompt", ""), item.get("numeric_params"))

                if not args.dry_run:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_img, dst_img)

                copied += 1

    if not args.dry_run:
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(train_text, file, ensure_ascii=False, indent=2)

    print(f"copied={copied}")
    print(f"missing_manifest_items={missing_manifest_items}")
    print(f"skipped_missing_images={skipped_missing_images}")
    print(f"overwritten_json_keys={overwritten_json_keys}")
    print(f"updated_json={json_path}")
    if args.backup:
        print(f"backup_json={backup_path}")
    if args.dry_run:
        print("dry_run=true")


if __name__ == "__main__":
    main()
