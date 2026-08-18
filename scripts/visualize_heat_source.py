import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_heat_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    params = config["data"]["params"]["train"]["params"]
    heat_params = params.get("heat_source_params", {})
    return {
        "threshold": float(params.get("heat_source_threshold", 0.1)),
        "Q": float(heat_params.get("Q", 1.0)),
        "f_front": float(heat_params.get("f_front", 0.5)),
        "f_back": float(heat_params.get("f_back", 0.5)),
        "alpha_front": float(heat_params.get("alpha_front", 48.0)),
        "beta_front": float(heat_params.get("beta_front", 24.0)),
        "alpha_back": float(heat_params.get("alpha_back", 72.0)),
        "beta_back": float(heat_params.get("beta_back", 32.0)),
        "center_x": float(heat_params.get("center_x", 0.5)),
        "center_y": float(heat_params.get("center_y", 0.5)),
        "random_angle": bool(heat_params.get("random_angle", False)),
        "angle_deg": float(heat_params.get("angle_deg", 0.0)),
        "angle_min_deg": float(heat_params.get("angle_min_deg", -45.0)),
        "angle_max_deg": float(heat_params.get("angle_max_deg", 45.0)),
    }


def sample_angle(cfg):
    if cfg["random_angle"]:
        return float(np.random.uniform(cfg["angle_min_deg"], cfg["angle_max_deg"]))
    return cfg["angle_deg"]


def make_heat_source(size, cfg, angle_deg=None):
    if angle_deg is None:
        angle_deg = sample_angle(cfg)
    center_x = cfg["center_x"] * (size - 1)
    center_y = cfg["center_y"] * (size - 1)
    yy, xx = np.mgrid[0:size, 0:size]
    x = xx.astype(np.float32) - center_x
    y = yy.astype(np.float32) - center_y
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_rot = x * cos_a + y * sin_a
    y_rot = -x * sin_a + y * cos_a

    front = (cfg["Q"] * cfg["f_front"] / (np.pi * cfg["alpha_front"] * cfg["beta_front"])) * np.exp(
        -3.0 * ((x_rot ** 2) / (cfg["alpha_front"] ** 2) + (y_rot ** 2) / (cfg["beta_front"] ** 2))
    )
    back = (cfg["Q"] * cfg["f_back"] / (np.pi * cfg["alpha_back"] * cfg["beta_back"])) * np.exp(
        -3.0 * ((x_rot ** 2) / (cfg["alpha_back"] ** 2) + (y_rot ** 2) / (cfg["beta_back"] ** 2))
    )
    heat = front + back
    heat = heat / max(float(heat.max()), 1e-8)
    mask = (heat >= cfg["threshold"]).astype(np.float32)
    return heat, mask, angle_deg


def make_base_image(size):
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[..., 0] = np.clip(70 + xx * 0.50, 0, 255)
    base[..., 1] = np.clip(80 + yy * 0.40, 0, 255)
    base[..., 2] = np.clip(110 + (xx + yy) * 0.20, 0, 255)
    return base


def heat_to_rgb(heat):
    heat = np.clip(heat, 0, 1)
    rgb = np.zeros((*heat.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(2.5 * heat, 0, 1)
    rgb[..., 1] = np.clip(2.5 * (heat - 0.25), 0, 1)
    rgb[..., 2] = np.clip(1.5 * (1 - heat), 0, 1)
    return (rgb * 255).astype(np.uint8)


def mask_to_rgb(mask):
    return np.repeat((mask[..., None] * 255).astype(np.uint8), 3, axis=2)


def overlay_mask(base, mask):
    out = base.astype(np.float32).copy()
    red = np.array([255, 58, 35], dtype=np.float32)
    alpha = 0.58
    out[mask > 0.5] = out[mask > 0.5] * (1 - alpha) + red * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def bounding_box(mask):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_bbox(tile, box):
    image = Image.fromarray(tile)
    if box is not None:
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline=(0, 255, 120), width=3)
    return np.array(image)


def add_label(tile, title, detail=""):
    label_h = 38
    image = Image.fromarray(tile)
    canvas = Image.new("RGB", (image.width, image.height + label_h), (28, 28, 28))
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 6), title, fill=(245, 245, 245), font=font)
    if detail:
        draw.text((8, 22), detail, fill=(210, 210, 210), font=font)
    return np.array(canvas)


def make_profile_plot(values, threshold, width, height, title):
    margin_l, margin_r, margin_t, margin_b = 34, 10, 18, 24
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x0, y0 = margin_l, margin_t
    x1, y1 = margin_l + plot_w, margin_t + plot_h

    draw.rectangle((x0, y0, x1, y1), outline=(90, 90, 90), width=1)
    ty = y1 - threshold * plot_h
    draw.line((x0, ty, x1, ty), fill=(220, 70, 55), width=2)

    pts = []
    for i, val in enumerate(values):
        x = x0 + i * plot_w / max(1, len(values) - 1)
        y = y1 - float(val) * plot_h
        pts.append((x, y))
    draw.line(pts, fill=(30, 90, 190), width=2)
    font = ImageFont.load_default()
    draw.text((8, 3), title, fill=(20, 20, 20), font=font)
    draw.text((x0 + 4, ty - 12), "threshold", fill=(180, 40, 35), font=font)
    draw.text((4, y0 - 3), "1.0", fill=(70, 70, 70), font=font)
    draw.text((4, y1 - 8), "0", fill=(70, 70, 70), font=font)
    return np.array(canvas)


def make_contact_sheet(tiles, cols=2, gap=10):
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    row_imgs = []
    for row in rows:
        width = sum(tile.shape[1] for tile in row) + gap * (len(row) - 1)
        height = max(tile.shape[0] for tile in row)
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        x = 0
        for tile in row:
            canvas[:tile.shape[0], x:x + tile.shape[1]] = tile
            x += tile.shape[1] + gap
        row_imgs.append(canvas)

    width = max(row.shape[1] for row in row_imgs)
    height = sum(row.shape[0] for row in row_imgs) + gap * (len(row_imgs) - 1)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = 0
    for row in row_imgs:
        canvas[y:y + row.shape[0], :row.shape[1]] = row
        y += row.shape[0] + gap
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize heat-source mask size and intensity content.")
    parser.add_argument("--config", default="configs/autoencoder/autoencoder_kl_32x32x4_large2_train.yaml")
    parser.add_argument("--outdir", default="outputs/mask_visualization")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    np.random.seed(args.seed)
    config_path = PROJECT_ROOT / args.config
    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_heat_config(config_path)
    heat, mask, angle_deg = make_heat_source(args.size, cfg)
    box = bounding_box(mask)
    area = float(mask.mean())
    if box is None:
        width = height = 0
    else:
        width = box[2] - box[0] + 1
        height = box[3] - box[1] + 1

    base = make_base_image(args.size)
    heat_rgb = heat_to_rgb(heat)
    mask_rgb = mask_to_rgb(mask)
    overlay_rgb = overlay_mask(base, mask)
    bbox_rgb = draw_bbox(mask_rgb, box)

    center_y = int(round(cfg["center_y"] * (args.size - 1)))
    center_x = int(round(cfg["center_x"] * (args.size - 1)))
    h_profile = make_profile_plot(heat[center_y, :], cfg["threshold"], args.size, 128, "horizontal heat profile")
    v_profile = make_profile_plot(heat[:, center_x], cfg["threshold"], args.size, 128, "vertical heat profile")

    detail = "area={:.1f}% bbox={}x{} angle={:.1f} threshold={:.2f}".format(
        area * 100, width, height, angle_deg, cfg["threshold"]
    )
    tiles = [
        add_label(heat_rgb, "continuous heat intensity", "white/red is hottest, blue is cooler"),
        add_label(mask_rgb, "binary heat-source mask", detail),
        add_label(overlay_rgb, "mask overlay", "white region is applied as mask=1"),
        add_label(bbox_rgb, "measured mask size", "green box: {}x{} px".format(width, height)),
        add_label(h_profile, "horizontal center profile", "alpha/back controls width"),
        add_label(v_profile, "vertical center profile", "beta/back controls height"),
    ]

    for i in range(max(0, args.samples - 1)):
        sample_heat, sample_mask, sample_angle_deg = make_heat_source(args.size, cfg)
        sample_box = bounding_box(sample_mask)
        sample_area = float(sample_mask.mean())
        if sample_box is None:
            sample_width = sample_height = 0
        else:
            sample_width = sample_box[2] - sample_box[0] + 1
            sample_height = sample_box[3] - sample_box[1] + 1
        sample_detail = "area={:.1f}% bbox={}x{} angle={:.1f}".format(
            sample_area * 100, sample_width, sample_height, sample_angle_deg
        )
        tiles.append(add_label(mask_to_rgb(sample_mask), "random angle sample {}".format(i + 2), sample_detail))
        tiles.append(add_label(overlay_mask(base, sample_mask), "sample {} overlay".format(i + 2), sample_detail))

    sheet = make_contact_sheet(tiles, cols=2)
    overview_path = outdir / "heat_source_size_content.png"
    Image.fromarray(sheet).save(overview_path)
    Image.fromarray(heat_rgb).save(outdir / "heat_source_intensity.png")
    Image.fromarray(mask_rgb).save(outdir / "heat_source_binary_mask.png")
    Image.fromarray(overlay_rgb).save(outdir / "heat_source_overlay.png")

    print("saved: {}".format(overview_path))
    print("area_percent: {:.2f}".format(area * 100))
    print("bbox_width_px: {}".format(width))
    print("bbox_height_px: {}".format(height))
    print("threshold: {:.4f}".format(cfg["threshold"]))


if __name__ == "__main__":
    main()
