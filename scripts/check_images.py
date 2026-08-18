import argparse
from pathlib import Path

from PIL import Image, ImageFile


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def iter_images(root):
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
            yield path


def check_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
        return None
    except Exception as exc:
        return "{}: {}".format(type(exc).__name__, exc)


def main():
    parser = argparse.ArgumentParser(description="Find unreadable or broken image files.")
    parser.add_argument("roots", nargs="+", help="Image folders to scan.")
    parser.add_argument("--out", default="outputs/broken_images.txt")
    args = parser.parse_args()

    broken = []
    total = 0
    for root in args.roots:
        for path in iter_images(root):
            total += 1
            error = check_image(path)
            if error is not None:
                broken.append((path, error))
                print("[BROKEN] {} | {}".format(path, error))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for path, error in broken:
            f.write("{}\t{}\n".format(path, error))

    print("checked_images: {}".format(total))
    print("broken_images: {}".format(len(broken)))
    print("report: {}".format(out_path))


if __name__ == "__main__":
    main()
