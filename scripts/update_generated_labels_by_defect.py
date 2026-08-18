import argparse
import glob
import json
import re
import shutil
from pathlib import Path
from collections import defaultdict


def defect_from_name(name):
    match = re.search(r"_defect_(.+)$", name)
    if not match:
        return None
    return match.group(1).replace("_", " ").strip()


def defect_from_prompt(prompt):
    match = re.search(r"defect\s+type\s+(.+?)(?:\.|,|;|$)", str(prompt), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().lower()


def normalize_relpath(path):
    return str(path).replace("\\", "/").lstrip("./")


DEFECT_TO_ID = {
    "burn through": 0,
    "contamination": 1,
    "good weld": 2,
    "high travel speed": 3,
    "lack of fusion": 4,
    "lack of shielding gas": 5,
}


def build_label(defect, prompt=None, params=None, label_format="caption"):
    defect = str(defect).strip().lower().replace("_", " ")
    if label_format == "id":
        if defect not in DEFECT_TO_ID:
            raise ValueError(f"Unknown defect label: {defect!r}. Known labels: {sorted(DEFECT_TO_ID)}")
        return DEFECT_TO_ID[defect]

    if label_format == "defect":
        return defect

    base = f"Categories: welding type TIG, material SS304, defect type {defect}."
    if params is None or len(params) != 4:
        return base

    voltage, current, speed, gas_flow = [float(value) for value in params]
    return (
        f"{base} "
        f"Process parameters: voltage {voltage:g} V, current {current:g} A, "
        f"travel speed {speed:g} cm/min, shielding gas flow {gas_flow:g} L/min."
    )


def main():
    parser = argparse.ArgumentParser(description="Update generated image labels in train JSON by defect type.")
    parser.add_argument("--train-root", default="datasets/ss304/train")
    parser.add_argument("--json-path", default="datasets/ss304/train/train.json")
    parser.add_argument("--outputs-root", default="datasets/ss304")
    parser.add_argument("--pattern", default="grid_params_*/manifest.jsonl")
    parser.add_argument("--original-json", default=None, help="Pre-merge JSON backup. Used to identify generated image keys.")
    parser.add_argument("--label-format", choices=["caption", "defect", "id"], default="caption")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    json_path = Path(args.json_path)
    outputs_root = Path(args.outputs_root)

    with open(json_path, "r", encoding="utf-8") as file:
        labels = json.load(file)
    original_labels = None
    if args.original_json:
        with open(args.original_json, "r", encoding="utf-8") as file:
            original_labels = json.load(file)

    if Path(args.pattern).is_absolute():
        manifest_paths = [Path(path) for path in glob.glob(args.pattern)]
    else:
        manifest_paths = list(outputs_root.glob(args.pattern))

    updated = 0
    not_found = 0
    no_defect = 0

    if original_labels is not None:
        generated_keys_by_folder = defaultdict(list)
        for key in labels:
            if key not in original_labels:
                generated_keys_by_folder[normalize_relpath(Path(key).parent)].append(key)
        for folder_keys in generated_keys_by_folder.values():
            folder_keys.sort()

        generated_items_by_folder = defaultdict(list)
        for manifest_path in sorted(manifest_paths):
            generated_dir = manifest_path.parent
            defect = defect_from_name(generated_dir.name)
            with open(manifest_path, "r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    defect_value = defect or defect_from_prompt(item.get("prompt", ""))
                    source_relpath = item.get("source_relpath")
                    if not defect_value:
                        no_defect += 1
                        continue
                    if not source_relpath:
                        not_found += 1
                        continue
                    folder = normalize_relpath(Path(source_relpath).parent)
                    generated_items_by_folder[folder].append((defect_value, item))

        for folder, generated_items in generated_items_by_folder.items():
            keys = generated_keys_by_folder.get(folder, [])
            for key, (defect_value, item) in zip(keys, generated_items):
                labels[key] = build_label(defect_value, item.get("prompt", ""), item.get("numeric_params"), args.label_format)
                updated += 1
            if len(keys) < len(generated_items):
                not_found += len(generated_items) - len(keys)

        if not args.dry_run:
            backup_path = json_path.with_suffix(json_path.suffix + ".defect_update.bak")
            if not backup_path.exists():
                shutil.copy2(json_path, backup_path)
            with open(json_path, "w", encoding="utf-8") as file:
                json.dump(labels, file, ensure_ascii=False, indent=2)

        print(f"updated={updated}")
        print(f"not_found={not_found}")
        print(f"no_defect={no_defect}")
        print(f"json_path={json_path}")
        print(f"label_format={args.label_format}")
        print(f"original_json={args.original_json}")
        if args.dry_run:
            print("dry_run=true")
        return

    for manifest_path in sorted(manifest_paths):
        generated_dir = manifest_path.parent
        defect = defect_from_name(generated_dir.name)

        with open(manifest_path, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                defect_value = defect or defect_from_prompt(item.get("prompt", ""))
                if not defect_value:
                    no_defect += 1
                    continue

                source_relpath = item.get("source_relpath")
                if not source_relpath:
                    not_found += 1
                    continue

                source_parent = Path(source_relpath).parent
                dst_dir = train_root / source_parent
                candidates = sorted(dst_dir.glob("image-*.*"))
                if not candidates:
                    not_found += 1
                    continue

                # Generated images were appended after original files; update newest entries that
                # currently have the same generated prompt/parameter caption.
                expected_label = build_label(defect_value, item.get("prompt", ""), item.get("numeric_params"), args.label_format)
                matched_key = None
                for candidate in reversed(candidates):
                    rel_key = normalize_relpath(candidate.relative_to(train_root))
                    if rel_key in labels and (
                        labels[rel_key] == expected_label
                        or str(labels[rel_key]).find("defect type") >= 0
                        or str(labels[rel_key]).find("Process parameters") >= 0
                    ):
                        matched_key = rel_key
                        break

                if matched_key is None:
                    not_found += 1
                    continue

                labels[matched_key] = expected_label
                updated += 1

    if not args.dry_run:
        backup_path = json_path.with_suffix(json_path.suffix + ".defect_update.bak")
        if not backup_path.exists():
            shutil.copy2(json_path, backup_path)
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(labels, file, ensure_ascii=False, indent=2)

    print(f"updated={updated}")
    print(f"not_found={not_found}")
    print(f"no_defect={no_defect}")
    print(f"json_path={json_path}")
    print(f"label_format={args.label_format}")
    if args.dry_run:
        print("dry_run=true")


if __name__ == "__main__":
    main()
