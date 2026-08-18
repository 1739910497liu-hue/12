import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resize_image(src, dst, size, mode=None):
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)
        if mode:
            image = image.convert(mode)
        resized = image.resize((size, size), Image.BICUBIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {}
        if dst.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = 95
        resized.save(dst, **save_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Resize all dataset images to the same square size.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--mode", default=None, choices=[None, "RGB", "L"], help="Optional PIL color mode.")
    parser.add_argument("--copy-non-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    resized = 0
    skipped = 0
    copied = 0

    for src in input_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(input_root)
        dst = output_root / rel
        if src.suffix.lower() in IMAGE_EXTS:
            if dst.exists() and not args.overwrite:
                skipped += 1
                continue
            resize_image(src, dst, args.size, args.mode)
            resized += 1
        elif args.copy_non_images:
            if dst.exists() and not args.overwrite:
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    print(f"input_root={input_root}")
    print(f"output_root={output_root}")
    print(f"size={args.size}")
    print(f"resized={resized}")
    print(f"copied_non_images={copied}")
    print(f"skipped_existing={skipped}")


if __name__ == "__main__":
    main()
