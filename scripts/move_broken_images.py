import argparse
import shutil
from pathlib import Path


def parse_report(report_path):
    paths = []
    with open(report_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path = line.split("\t", 1)[0].strip()
            if path:
                paths.append(Path(path))
    return paths


def choose_relative_path(path, dataset_root):
    try:
        return path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        parts = path.parts
        for marker in ("ss304", "datasets"):
            if marker in parts:
                idx = parts.index(marker)
                return Path(*parts[idx:])
        return Path(path.name)


def main():
    parser = argparse.ArgumentParser(description="Move broken images listed in a report file.")
    parser.add_argument("--report", default="outputs/broken_images.txt")
    parser.add_argument("--dataset-root", default="datasets/ss304")
    parser.add_argument("--broken-root", default="datasets/ss304/broken")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report_path = Path(args.report)
    dataset_root = Path(args.dataset_root)
    broken_root = Path(args.broken_root)

    if not report_path.exists():
        raise FileNotFoundError("Report file not found: {}".format(report_path))

    paths = parse_report(report_path)
    moved = 0
    missing = 0
    for path in paths:
        if not path.exists():
            missing += 1
            print("[MISSING] {}".format(path))
            continue

        rel = choose_relative_path(path, dataset_root)
        dst = broken_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        print("[MOVE] {} -> {}".format(path, dst))
        if not args.dry_run:
            shutil.move(str(path), str(dst))
        moved += 1

    print("listed_images: {}".format(len(paths)))
    print("moved_images: {}".format(moved))
    print("missing_images: {}".format(missing))
    print("broken_root: {}".format(broken_root))


if __name__ == "__main__":
    main()
