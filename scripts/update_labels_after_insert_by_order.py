import argparse
import glob
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


DEFECT_TO_ID = {
    "burn through": 0,
    "contamination": 1,
    "good weld": 2,
    "high travel speed": 3,
    "lack of fusion": 4,
    "lack of shielding gas": 5,
}


def normalize(path):
    return str(path).replace("\\", "/").lstrip("./")


def image_index(name):
    match = re.search(r"image-(\d+)", str(name))
    return int(match.group(1)) if match else 10**12


def defect_from_name(name):
    match = re.search(r"_defect_(.+)$", name)
    if not match:
        return None
    return match.group(1).replace("_", " ").strip().lower()


def defect_from_prompt(prompt):
    match = re.search(r"defect\s+type\s+(.+?)(?:\.|,|;|$)", str(prompt), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).replace("_", " ").strip().lower()


def get_manifest_paths(outputs_root, pattern):
    if Path(pattern).is_absolute():
        return [Path(path) for path in glob.glob(pattern)]
    return list(Path(outputs_root).glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description="Update generated numeric labels after images were inserted and renumbered, by folder/order."
    )
    parser.add_argument("--train-root", default="datasets/ss304/train")
    parser.add_argument("--json-path", default="datasets/ss304/train/train.json")
    parser.add_argument("--outputs-root", default="datasets/ss304")
    parser.add_argument("--pattern", default="grid_params_*/manifest.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    json_path = Path(args.json_path)
    with open(json_path, "r", encoding="utf-8") as file:
        labels = json.load(file)

    planned_by_folder = defaultdict(list)
    no_defect = 0
    missing = 0
    for manifest in sorted(get_manifest_paths(args.outputs_root, args.pattern)):
        defect = defect_from_name(manifest.parent.name)
        with open(manifest, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                source_relpath = item.get("source_relpath")
                if not source_relpath:
                    missing += 1
                    continue
                defect_value = defect or defect_from_prompt(item.get("prompt", ""))
                if defect_value not in DEFECT_TO_ID:
                    no_defect += 1
                    continue
                folder = normalize(Path(source_relpath).parent)
                source_idx = image_index(Path(source_relpath).name)
                planned_by_folder[folder].append((source_idx, DEFECT_TO_ID[defect_value]))

    updated = 0
    not_found = 0
    for folder, planned in sorted(planned_by_folder.items()):
        folder_path = train_root / folder
        if not folder_path.exists():
            not_found += len(planned)
            continue
        planned.sort(key=lambda row: row[0])
        image_keys = [
            normalize(path.relative_to(train_root))
            for path in sorted(folder_path.glob("image-*.*"), key=lambda p: image_index(p.name))
            if normalize(path.relative_to(train_root)) in labels
        ]
        if len(image_keys) < len(planned):
            not_found += len(planned) - len(image_keys)
            target_keys = image_keys
        else:
            target_keys = image_keys[-len(planned):]
        for key, (_source_idx, label_id) in zip(target_keys, planned):
            labels[key] = label_id
            updated += 1

    if not args.dry_run:
        backup = json_path.with_suffix(json_path.suffix + ".before_order_label_update.bak")
        if not backup.exists():
            shutil.copy2(json_path, backup)
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(labels, file, ensure_ascii=False, indent=2)

    print(f"updated={updated}")
    print(f"not_found={not_found}")
    print(f"missing_manifest_items={missing}")
    print(f"no_defect={no_defect}")
    print(f"json_path={json_path}")
    if args.dry_run:
        print("dry_run=true")


if __name__ == "__main__":
    main()
