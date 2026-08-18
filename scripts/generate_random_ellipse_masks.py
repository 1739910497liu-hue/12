import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(indir):
    root = Path(indir)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMG_EXTENSIONS
        and not p.stem.endswith("_mask")
    )


def random_ellipse_mask(width, height, rng, args):
    scale_x = width / 256.0
    scale_y = height / 256.0
    min_rx = args.min_radius_x * scale_x
    max_rx = args.max_radius_x * scale_x
    min_ry = args.min_radius_y * scale_y
    max_ry = args.max_radius_y * scale_y

    radius_x = rng.uniform(min_rx, max_rx)
    radius_y = rng.uniform(min_ry, max_ry)

    if args.random_center:
        margin_x = min(radius_x + args.margin * scale_x, max(width / 2.0 - 1.0, 0.0))
        margin_y = min(radius_y + args.margin * scale_y, max(height / 2.0 - 1.0, 0.0))
        center_x = rng.uniform(margin_x, max(margin_x, width - 1 - margin_x))
        center_y = rng.uniform(margin_y, max(margin_y, height - 1 - margin_y))
    else:
        center_x = args.center_x * (width - 1)
        center_y = args.center_y * (height - 1)

    angle_deg = rng.uniform(args.angle_min_deg, args.angle_max_deg) if args.random_angle else args.angle_deg
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    yy, xx = np.mgrid[0:height, 0:width]
    x = xx.astype(np.float32) - center_x
    y = yy.astype(np.float32) - center_y
    x_rot = x * cos_a + y * sin_a
    y_rot = -x * sin_a + y * cos_a
    mask = ((x_rot / radius_x) ** 2 + (y_rot / radius_y) ** 2) <= 1.0
    return (mask.astype(np.uint8) * 255), {
        "center_x": center_x,
        "center_y": center_y,
        "radius_x": radius_x,
        "radius_y": radius_y,
        "angle_deg": angle_deg,
    }


def save_pair(image_path, outdir, mask, index=None, copy_image=True):
    image_path = Path(image_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if index is None:
        out_image = outdir / image_path.name
        out_mask = outdir / "{}_mask.png".format(image_path.stem)
    else:
        stem = "{}_{:03d}".format(image_path.stem, index)
        out_image = outdir / "{}{}".format(stem, image_path.suffix.lower())
        out_mask = outdir / "{}_mask.png".format(stem)

    if copy_image:
        shutil.copy2(image_path, out_image)
    Image.fromarray(mask, mode="L").save(out_mask)
    return out_image, out_mask


def main():
    parser = argparse.ArgumentParser(description="Generate random white ellipse masks for SS304 inpainting inference.")
    parser.add_argument("--image", default=None, help="Single source image.")
    parser.add_argument("--indir", default=None, help="Folder of source images.")
    parser.add_argument("--outdir", default="data/test_pairs", help="Output folder for image/mask pairs.")
    parser.add_argument("--count", type=int, default=1, help="Masks per image.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-radius-x", type=float, default=42.0)
    parser.add_argument("--max-radius-x", type=float, default=110.0)
    parser.add_argument("--min-radius-y", type=float, default=28.0)
    parser.add_argument("--max-radius-y", type=float, default=88.0)
    parser.add_argument("--random-center", action="store_true", default=True)
    parser.add_argument("--fixed-center", dest="random_center", action="store_false")
    parser.add_argument("--center-x", type=float, default=0.5, help="Normalized center x if --fixed-center is used.")
    parser.add_argument("--center-y", type=float, default=0.5, help="Normalized center y if --fixed-center is used.")
    parser.add_argument("--random-angle", action="store_true", default=True)
    parser.add_argument("--fixed-angle", dest="random_angle", action="store_false")
    parser.add_argument("--angle-deg", type=float, default=0.0)
    parser.add_argument("--angle-min-deg", type=float, default=0.0)
    parser.add_argument("--angle-max-deg", type=float, default=360.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--no-copy-image", action="store_true", help="Only save masks, do not copy images.")
    args = parser.parse_args()

    if args.image is None and args.indir is None:
        raise ValueError("Pass either --image or --indir.")

    images = [Path(args.image)] if args.image is not None else list_images(args.indir)
    if not images:
        raise RuntimeError("No images found.")

    rng = np.random.default_rng(args.seed)
    total = 0
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size

        for i in range(args.count):
            mask, params = random_ellipse_mask(width, height, rng, args)
            index = None if args.count == 1 else i
            out_image, out_mask = save_pair(
                image_path,
                args.outdir,
                mask,
                index=index,
                copy_image=not args.no_copy_image,
            )
            total += 1
            print(
                "saved: {} | center=({:.1f},{:.1f}) radius=({:.1f},{:.1f}) angle={:.1f}".format(
                    out_mask,
                    params["center_x"],
                    params["center_y"],
                    params["radius_x"],
                    params["radius_y"],
                    params["angle_deg"],
                )
            )
            if not args.no_copy_image:
                print("paired image: {}".format(out_image))

    print("Generated {} mask(s) under {}".format(total, args.outdir))


if __name__ == "__main__":
    main()
