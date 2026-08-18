import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ID_TO_DEFECT = {
    0: "burn through",
    1: "contamination",
    2: "good weld",
    3: "high travel speed",
    4: "lack of fusion",
    5: "lack of shielding gas",
}


def safe_name(value):
    return str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")


def main():
    parser = argparse.ArgumentParser(description="Write commands to generate minority defect classes per process folder.")
    parser.add_argument("--json-path", default="datasets/ss304/train/train.json")
    parser.add_argument("--train-root", default="datasets/ss304/train")
    parser.add_argument("--out-script", default="scripts/run_generate_minority_defects.sh")
    parser.add_argument("--target-per-class", type=int, default=50)
    parser.add_argument("--max-generate-per-class", type=int, default=50)
    parser.add_argument("--include-zero-classes", action="store_true", help="Also generate classes with zero existing samples.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/latent-diffusion/ss304-ldm-text-numeric-vq-no-y2h.yaml")
    parser.add_argument("--ckpt", default="/root/autodl-tmp/TIGLDM6000/logs/ss304_text_physical_router_no_y2h/2026-08-10T01-06-00_ss304-ldm-text-numeric-vq/checkpoints/last-v1.ckpt")
    parser.add_argument("--first-stage-ckpt", default="/root/autodl-tmp/TIGLDM6000/logs/ss304_stage1_vqgan/2026-08-08T10-52-02_autoencoder_kl_32x32x4_large2_train/checkpoints/last.ckpt")
    parser.add_argument("--default-params", default="17.2,200,19,30")
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as file:
        labels = json.load(file)

    counts = defaultdict(Counter)
    for relpath, label in labels.items():
        if not isinstance(label, int):
            continue
        folder = str(Path(relpath).parent).replace("\\", "/")
        counts[folder][label] += 1

    commands = [
        "#!/usr/bin/env bash",
        "set -e",
        "cd /root/autodl-tmp/TIGLDM6000",
        "export OMP_NUM_THREADS=1",
        "export PYTHONPATH=/root/autodl-tmp/TIGLDM6000",
        "",
    ]

    total_jobs = 0
    total_images = 0
    for folder, counter in sorted(counts.items()):
        for label_id, defect in ID_TO_DEFECT.items():
            current = counter.get(label_id, 0)
            if current == 0 and not args.include_zero_classes:
                continue
            if current >= args.target_per_class:
                continue
            count = min(args.target_per_class - current, args.max_generate_per_class)
            if count <= 0:
                continue
            input_root = str(Path(args.train_root) / folder).replace("\\", "/")
            outdir = f"datasets/ss304/minority_generated/{safe_name(folder)}_defect_{safe_name(defect)}"
            commands.append(
                "python scripts/infer_ss304_text_stage2.py "
                f"--config {args.config} "
                f"--ckpt {args.ckpt} "
                f"--first-stage-ckpt {args.first_stage_ckpt} "
                f"--input-root \"{input_root}\" "
                "--validation-log-images-mode "
                f"--prompt \"Categories: welding type TIG, material SS304, defect type {defect}.\" "
                f"--params {args.default_params} "
                f"--outdir \"{outdir}\" "
                f"--count {count} "
                f"--batch-size {args.batch_size} "
                f"--ddim-steps {args.ddim_steps} "
                "--ddim-eta 0.0 "
                f"--seed {args.seed} "
                "--overwrite"
            )
            total_jobs += 1
            total_images += count

    out_path = Path(args.out_script)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"folders={len(counts)}")
    print(f"generation_jobs={total_jobs}")
    print(f"planned_images={total_images}")
    print(f"saved_script={out_path}")


if __name__ == "__main__":
    main()
