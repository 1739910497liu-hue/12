import argparse
import glob
import json
import re
import shutil
import tempfile
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


def normalize_relpath(path):
    return str(path).replace("\\", "/").lstrip("./")


def image_index(path):
    match = re.match(r"image-(\d+)\.[^.]+$", path.name)
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


def manifest_paths(outputs_root, pattern):
    if Path(pattern).is_absolute():
        return [Path(path) for path in glob.glob(pattern)]
    return list(Path(outputs_root).glob(pattern))


def source_relative_to_train(item, train_root):
    source_image = item.get("source_image")
    if source_image:
        source_path = Path(source_image)
        train_root_norm = normalize_relpath(train_root)
        source_norm = normalize_relpath(source_path)
        marker = train_root_norm + "/"
        if marker in source_norm:
            return Path(source_norm.split(marker, 1)[1])
        parts = source_path.parts
        if "train" in parts:
            idx = parts.index("train")
            return Path(*parts[idx + 1:])
    source_relpath = item.get("source_relpath")
    return Path(source_relpath) if source_relpath else None


def load_generated_items(outputs_root, pattern, train_root):
    by_folder = defaultdict(list)
    missing = 0
    for manifest in sorted(manifest_paths(outputs_root, pattern)):
        gen_dir = manifest.parent
        defect = defect_from_name(gen_dir.name)
        with open(manifest, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                output = item.get("output")
                source_relpath = source_relative_to_train(item, train_root)
                if not output or source_relpath is None:
                    missing += 1
                    continue
                src_img = gen_dir / output
                if not src_img.exists():
                    missing += 1
                    continue
                source_parent = normalize_relpath(Path(source_relpath).parent)
                source_name = Path(source_relpath).name
                defect_value = defect or defect_from_prompt(item.get("prompt", ""))
                label = DEFECT_TO_ID.get(defect_value)
                if label is None:
                    missing += 1
                    continue
                by_folder[source_parent].append(
                    {
                        "source_name": source_name,
                        "source_index": image_index(Path(source_name)),
                        "src_img": src_img,
                        "label": label,
                    }
                )
    for items in by_folder.values():
        items.sort(key=lambda row: (row["source_index"], row["src_img"].name))
    return by_folder, missing


def main():
    parser = argparse.ArgumentParser(
        description="Insert generated images after their source images, renumber image-xxxx files, and update train JSON."
    )
    parser.add_argument("--train-root", default="datasets/ss304/train")
    parser.add_argument("--json-path", default="datasets/ss304/train/train.json")
    parser.add_argument("--outputs-root", default="datasets/ss304")
    parser.add_argument("--pattern", default="grid_params_*/manifest.jsonl")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    json_path = Path(args.json_path)

    with open(json_path, "r", encoding="utf-8") as file:
        labels = json.load(file)

    generated_by_folder, missing = load_generated_items(args.outputs_root, args.pattern, train_root)

    new_labels = dict(labels)
    total_inserted = 0
    total_renamed = 0
    touched_folders = 0

    for folder_rel, generated_items in sorted(generated_by_folder.items()):
        folder = train_root / folder_rel
        if not folder.exists():
            continue

        originals = sorted(
            [path for path in folder.iterdir() if path.is_file() and re.match(r"image-\d+\.[^.]+$", path.name)],
            key=image_index,
        )
        inserts_after = defaultdict(list)
        for item in generated_items:
            inserts_after[item["source_name"]].append(item)

        ordered_entries = []
        for original in originals:
            old_key = normalize_relpath(original.relative_to(train_root))
            ordered_entries.append({"path": original, "label": labels.get(old_key), "generated": False})
            for item in inserts_after.get(original.name, []):
                ordered_entries.append({"path": item["src_img"], "label": item["label"], "generated": True})

        if len(ordered_entries) == len(originals):
            continue

        touched_folders += 1
        total_inserted += len(ordered_entries) - len(originals)

        if args.dry_run:
            total_renamed += len(ordered_entries)
            continue

        backup_json = json_path.with_suffix(json_path.suffix + ".before_insert_renumber.bak")
        if not backup_json.exists():
            shutil.copy2(json_path, backup_json)

        temp_dir = Path(tempfile.mkdtemp(prefix="renumber_", dir=str(folder)))
        try:
            folder_new_labels = {}
            for offset, entry in enumerate(ordered_entries):
                new_name = f"image-{args.start_index + offset:04d}{entry['path'].suffix.lower()}"
                temp_path = temp_dir / new_name
                shutil.copy2(entry["path"], temp_path)
                new_key = normalize_relpath((folder / new_name).relative_to(train_root))
                folder_new_labels[new_key] = entry["label"]

            for original in originals:
                original.unlink()
            for temp_path in sorted(temp_dir.iterdir(), key=image_index):
                shutil.move(str(temp_path), str(folder / temp_path.name))

            old_folder_prefix = normalize_relpath(Path(folder_rel)) + "/"
            for key in list(new_labels):
                if normalize_relpath(key).startswith(old_folder_prefix):
                    del new_labels[key]
            new_labels.update(folder_new_labels)
            total_renamed += len(ordered_entries)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    if not args.dry_run:
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(new_labels, file, ensure_ascii=False, indent=2)

    print(f"touched_folders={touched_folders}")
    print(f"inserted_generated_images={total_inserted}")
    print(f"renumbered_images={total_renamed}")
    print(f"missing_manifest_items={missing}")
    print(f"updated_json={json_path}")
    if args.dry_run:
        print("dry_run=true")


if __name__ == "__main__":
    main()
