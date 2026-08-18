import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def make_base_image(size):
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[..., 0] = np.clip(80 + xx * 0.45, 0, 255)
    base[..., 1] = np.clip(90 + yy * 0.35, 0, 255)
    base[..., 2] = np.clip(120 + (xx + yy) * 0.18, 0, 255)
    return base


def load_base_image(path, size):
    if path is None:
        return make_base_image(size)
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    return np.array(image)


def make_heat_source_mask(size):
    with open(PROJECT_ROOT / "configs" / "autoencoder" / "autoencoder_kl_32x32x4_large2_train.yaml", "r") as f:
        config = yaml.safe_load(f)
    params = config["data"]["params"]["train"]["params"]
    heat_params = params.get("heat_source_params", {})
    threshold = float(params.get("heat_source_threshold", 0.1))
    q = float(heat_params.get("Q", 1.0))
    f_front = float(heat_params.get("f_front", 0.5))
    f_back = float(heat_params.get("f_back", 0.5))
    alpha_front = float(heat_params.get("alpha_front", 48.0))
    beta_front = float(heat_params.get("beta_front", 24.0))
    alpha_back = float(heat_params.get("alpha_back", 72.0))
    beta_back = float(heat_params.get("beta_back", 32.0))
    center_x = float(heat_params.get("center_x", 0.5)) * (size - 1)
    center_y = float(heat_params.get("center_y", 0.5)) * (size - 1)

    yy, xx = np.mgrid[0:size, 0:size]
    x = xx.astype(np.float32) - center_x
    y = yy.astype(np.float32) - center_y
    front = (q * f_front / (np.pi * alpha_front * beta_front)) * np.exp(
        -3.0 * ((x ** 2) / (alpha_front ** 2) + (y ** 2) / (beta_front ** 2))
    )
    back = (q * f_back / (np.pi * alpha_back * beta_back)) * np.exp(
        -3.0 * ((x ** 2) / (alpha_back ** 2) + (y ** 2) / (beta_back ** 2))
    )
    mask = front + back
    mask = mask / max(float(mask.max()), 1e-8)
    return (mask >= threshold).astype(np.float32)


def make_random_mask(size):
    with open(PROJECT_ROOT / "configs" / "autoencoder" / "random_thick_256.yaml", "r") as f:
        config = yaml.safe_load(f)
    mask_kwargs = dict(config["mask_generator_kwargs"])
    variants_n = mask_kwargs.pop("variants_n", 5)
    min_mask_area = mask_kwargs.pop("min_mask_area", 0.0)
    max_mask_area = mask_kwargs.pop("max_mask_area", 1.0)
    max_attempts = mask_kwargs.pop("max_attempts", 20)

    fallback = None
    for _ in range(max_attempts):
        masks = [make_mixed_random_mask(size, mask_kwargs) for _ in range(variants_n)]
        mask = masks[np.random.choice(len(masks))]
        fallback = mask
        area = float(mask.mean())
        if min_mask_area <= area <= max_mask_area:
            return mask
    return fallback


def make_irregular_mask(size, kwargs):
    min_times = int(kwargs.get("min_times", 1))
    max_times = int(kwargs.get("max_times", 5))
    max_width = int(kwargs.get("max_width", 100))
    max_len = int(kwargs.get("max_len", 200))
    max_angle = int(kwargs.get("max_angle", 4))

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        x = int(np.random.randint(size))
        y = int(np.random.randint(size))
        for _ in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = 10 + np.random.randint(max_len)
            width = 5 + np.random.randint(max_width)
            x2 = int(np.clip(x + length * np.sin(angle), 0, size - 1))
            y2 = int(np.clip(y + length * np.cos(angle), 0, size - 1))
            draw.line((x, y, x2, y2), fill=255, width=width)
            radius = max(1, width // 2)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
            x, y = x2, y2
    return np.asarray(mask, dtype=np.float32) / 255.0


def make_box_mask(size, kwargs):
    margin = int(kwargs.get("margin", 10))
    bbox_min_size = int(kwargs.get("bbox_min_size", 30))
    bbox_max_size = int(kwargs.get("bbox_max_size", 100))
    min_times = int(kwargs.get("min_times", 1))
    max_times = int(kwargs.get("max_times", 3))

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    bbox_max_size = min(bbox_max_size, size - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for _ in range(times):
        box_w = int(np.random.randint(bbox_min_size, bbox_max_size + 1))
        box_h = int(np.random.randint(bbox_min_size, bbox_max_size + 1))
        x = int(np.random.randint(margin, size - margin - box_w + 1))
        y = int(np.random.randint(margin, size - margin - box_h + 1))
        draw.rectangle((x, y, x + box_w, y + box_h), fill=255)
    return np.asarray(mask, dtype=np.float32) / 255.0


def make_mixed_random_mask(size, kwargs):
    irregular_proba = float(kwargs.get("irregular_proba", 0.0))
    box_proba = float(kwargs.get("box_proba", 0.0))
    total = irregular_proba + box_proba
    if total <= 0:
        return np.zeros((size, size), dtype=np.float32)
    if np.random.random() < irregular_proba / total:
        return make_irregular_mask(size, kwargs.get("irregular_kwargs", {}))
    return make_box_mask(size, kwargs.get("box_kwargs", {}))


def overlay_mask(image, mask):
    overlay = image.copy().astype(np.float32)
    color = np.array([255, 64, 32], dtype=np.float32)
    alpha = 0.55
    overlay[mask > 0.5] = overlay[mask > 0.5] * (1 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def mask_to_rgb(mask):
    return np.repeat((mask[..., None] * 255).astype(np.uint8), 3, axis=2)


def add_label(tile, title, area):
    image = Image.fromarray(tile)
    draw = ImageDraw.Draw(image)
    label_h = 28
    canvas = Image.new("RGB", (image.width, image.height + label_h), (28, 28, 28))
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    text = "{}  area={:.1f}%".format(title, area * 100)
    draw.text((8, 7), text, fill=(245, 245, 245), font=ImageFont.load_default())
    return np.array(canvas)


def make_contact_sheet(rows, gap=8):
    row_images = []
    for row in rows:
        width = sum(tile.shape[1] for tile in row) + gap * (len(row) - 1)
        height = max(tile.shape[0] for tile in row)
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        x = 0
        for tile in row:
            canvas[:tile.shape[0], x:x + tile.shape[1]] = tile
            x += tile.shape[1] + gap
        row_images.append(canvas)

    width = max(row.shape[1] for row in row_images)
    height = sum(row.shape[0] for row in row_images) + gap * (len(row_images) - 1)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = 0
    for row in row_images:
        canvas[y:y + row.shape[0], :row.shape[1]] = row
        y += row.shape[0] + gap
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize SS304 training mask types.")
    parser.add_argument("--image", default=None, help="Optional image used as overlay background.")
    parser.add_argument("--outdir", default="outputs/mask_visualization")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--random-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    np.random.seed(args.seed)
    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    base = load_base_image(args.image, args.size)
    masks = [("heat_source", make_heat_source_mask(args.size))]
    masks.append(("full_Cf_r", np.ones((args.size, args.size), dtype=np.float32)))
    for i in range(args.random_samples):
        masks.append(("random_{:02d}".format(i + 1), make_random_mask(args.size)))

    rows = []
    for name, mask in masks:
        area = float(mask.mean())
        mask_img = mask_to_rgb(mask)
        overlay_img = overlay_mask(base, mask)
        Image.fromarray(mask_img).save(outdir / "{}_mask.png".format(name))
        Image.fromarray(overlay_img).save(outdir / "{}_overlay.png".format(name))
        rows.append([
            add_label(mask_img, "{} mask".format(name), area),
            add_label(overlay_img, "{} overlay".format(name), area),
        ])

    sheet = make_contact_sheet(rows)
    sheet_path = outdir / "mask_types_overview.png"
    Image.fromarray(sheet).save(sheet_path)
    print("saved: {}".format(sheet_path))
    print("outdir: {}".format(outdir))


if __name__ == "__main__":
    main()
