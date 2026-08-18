import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calc_fid_ssim import (  # noqa: E402
    calculate_fid,
    calculate_inception_score,
    calculate_pair_metrics,
    list_images,
)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate laserwelding generated-image metrics: FID, IS, PSNR, SSIM, MAE and LPIPS."
    )
    parser.add_argument("--result-dir", default="outputs/laserwelding_stage2",
                        help="Inference output folder containing inputs/ and samples/.")
    parser.add_argument("--real", default=None, help="Ground-truth/input image folder. Defaults to result-dir/inputs.")
    parser.add_argument("--fake", default=None, help="Generated image folder. Defaults to result-dir/samples.")
    parser.add_argument("--out-json", default=None, help="Defaults to result-dir/metrics.json.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--pair-mode", default="name", choices=["name", "sorted"])
    parser.add_argument("--lpips-net", default="vgg", choices=["vgg", "alex", "squeeze"])
    parser.add_argument("--is-splits", type=int, default=10)
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-is", action="store_true")
    parser.add_argument("--skip-pair", action="store_true")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    real_dir = Path(args.real) if args.real else result_dir / "inputs"
    fake_dir = Path(args.fake) if args.fake else result_dir / "samples"
    out_json = Path(args.out_json) if args.out_json else result_dir / "metrics.json"

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    real_files = list_images(real_dir)
    fake_files = list_images(fake_dir)

    if not real_files:
        raise RuntimeError("No real/input images found under {}".format(real_dir))
    if not fake_files:
        raise RuntimeError("No generated images found under {}".format(fake_dir))

    results = {
        "result_dir": str(result_dir),
        "real": str(real_dir),
        "fake": str(fake_dir),
        "real_images": len(real_files),
        "fake_images": len(fake_files),
        "device": str(device),
        "image_size": args.image_size,
        "pair_mode": args.pair_mode,
    }

    print("real_images: {}".format(len(real_files)))
    print("fake_images: {}".format(len(fake_files)))
    print("device: {}".format(device))

    if not args.skip_fid:
        if len(real_files) < 2 or len(fake_files) < 2:
            raise RuntimeError("FID needs at least two images in each folder.")
        fid = calculate_fid(real_files, fake_files, device, args.batch_size, args.dims, args.image_size)
        results["fid"] = fid
        print("FID: {:.6f}".format(fid))

    if not args.skip_is:
        is_score = calculate_inception_score(fake_files, device, args.batch_size, 299, args.is_splits)
        results["is_mean"] = is_score["mean"]
        results["is_std"] = is_score["std"]
        print("IS_mean: {:.6f}".format(is_score["mean"]))
        print("IS_std: {:.6f}".format(is_score["std"]))

    if not args.skip_pair:
        pair_metrics = calculate_pair_metrics(
            real_dir,
            fake_dir,
            device,
            args.batch_size,
            args.image_size,
            args.pair_mode,
            args.lpips_net,
        )
        results.update(pair_metrics)
        print("PSNR_mean: {:.6f}".format(pair_metrics["psnr_mean"]))
        print("PSNR_std: {:.6f}".format(pair_metrics["psnr_std"]))
        print("SSIM_mean: {:.6f}".format(pair_metrics["ssim_mean"]))
        print("SSIM_std: {:.6f}".format(pair_metrics["ssim_std"]))
        print("MAE_mean: {:.6f}".format(pair_metrics["mae_mean"]))
        print("MAE_std: {:.6f}".format(pair_metrics["mae_std"]))
        print("LPIPS_mean: {:.6f}".format(pair_metrics["lpips_mean"]))
        print("LPIPS_std: {:.6f}".format(pair_metrics["lpips_std"]))
        print("paired_images: {}".format(pair_metrics["pairs"]))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved metrics JSON: {}".format(out_json))


if __name__ == "__main__":
    main()
