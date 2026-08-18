# TIGLDM6000 Laserwelding

This repository is an Asymmetric VQGAN / latent diffusion project adapted for
laser welding image generation. The current local workflow supports:

- Stage 1: train a laserwelding first-stage VQGAN / `VQModelWithMask`.
- Stage 2: train a laserwelding latent diffusion model.
- Stage 2 text+numeric: condition generation on defect category text plus
  welding parameters.
- DDIM benchmark: fixed GPU, batch size, 256x256 output, DDIM 50 steps, and
  average single-image generation time.

## Environment

Install dependencies from the project root:

```cmd
pip install -r requirements.txt
pip install wandb
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
```

For Python 3.12 environments, use `requirements-py312.txt` if the default
requirements file is not compatible.

## Data Layout

Expected laserwelding image data:

```text
H:/数据集/laserwelding/train
H:/数据集/laserwelding/train/train.json
H:/数据集/laserwelding/validation
H:/数据集/laserwelding/validation/validation.json
```

Expected text and numeric condition files:

```text
H:/数据集/JSON/text/train_text_categories_label_corrected_original_format.json
H:/数据集/JSON/text/train_text_actual_parameters_original_format.json
```

Category label format:

```json
{
  "激光焊接数据集/优质焊接/功率1280W 速度24mm.s 气流量  10 无明显缺陷/file_0613.jpg": "Categories: welding type laser, material HC420, defect type good weld."
}
```

Numeric parameter format:

```json
{
  "激光焊接数据集/优质焊接/功率1280W 速度24mm.s 气流量  10 无明显缺陷/file_0613.jpg": "Parameters: Power 1280W, speed 24mm/s, gas flow 10L/min."
}
```

For the text+numeric laserwelding model, numeric parameters are encoded as:

```text
[1, power_W, speed_mm_s, gas_flow_L_min]
```

The leading `1` is used as a dummy voltage so the existing physical encoder's
internal `V * I` feature equals laser power.

## Stage 1: Laserwelding VQGAN

Run from the project root:

```cmd
cd /d H:\LDM\热源模型\6000\VQGAN\TIGLDM600\TIGLDM6000 && python main.py --base configs/autoencoder/autoencoder_kl_32x32x4_large2_laserwelding_train.yaml --train true --name laserwelding_first_stage --logdir logs --gpus 0 --epoch 12
```

If the process is killed during model initialization or training, use the
lower-memory command:

```cmd
cd /d H:\LDM\热源模型\6000\VQGAN\TIGLDM600\TIGLDM6000 && python main.py --base configs/autoencoder/autoencoder_kl_32x32x4_large2_laserwelding_train.yaml --train true --name laserwelding_first_stage_bs1 --logdir logs --gpus 0 --epoch 12 data.params.batch_size=1 data.params.num_workers=0 lightning.trainer.accumulate_grad_batches=8
```

The trained checkpoint is usually saved under:

```text
logs/Base/<stage1-run-name>/checkpoints/last.ckpt
```

## Stage 2: Unconditional Laserwelding LDM

Use this when you only need image generation without text or numeric
conditioning:

```cmd
cd /d H:\LDM\热源模型\6000\VQGAN\TIGLDM600\TIGLDM6000 && python main.py --base configs/latent-diffusion/laserwelding-ldm-vq-lite.yaml --train true --name laserwelding_stage2 --logdir logs --gpus 0 --epoch 12 model.params.first_stage_config.params.ckpt_path=logs/Base/你的laserwelding第一阶段目录/checkpoints/last.ckpt data.params.batch_size=1 data.params.num_workers=0 lightning.trainer.accumulate_grad_batches=8
```

Replace the first-stage checkpoint path with your actual Stage 1 checkpoint.

## Stage 2: Text+Numeric Laserwelding LDM

Use this config for defect-category text plus welding numeric parameters:

```text
configs/latent-diffusion/laserwelding-ldm-text-numeric-vq-lite.yaml
```

Training command:

```cmd
cd /d H:\LDM\热源模型\6000\VQGAN\TIGLDM600\TIGLDM6000 && python main.py --base configs/latent-diffusion/laserwelding-ldm-text-numeric-vq-lite.yaml --train true --name laserwelding_stage2_text_numeric --logdir logs --gpus 0 --epoch 12 model.params.first_stage_config.params.ckpt_path=logs/Base/你的laserwelding第一阶段目录/checkpoints/last.ckpt model.params.cond_stage_config.params.version=models/clip/clip-vit-base-patch32 data.params.batch_size=1 data.params.num_workers=0 lightning.trainer.accumulate_grad_batches=8
```

The text encoder expects a local CLIP folder at:

```text
models/clip/clip-vit-base-patch32
```

If CLIP is stored elsewhere, override:

```text
model.params.cond_stage_config.params.version=你的CLIP路径
```

## DDIM Benchmark

The inference script can report average generation time per image. Timing covers
`sample_log` plus first-stage decode, and excludes image loading/saving I/O.

Fixed setup:

- GPU: `CUDA_VISIBLE_DEVICES=0`
- batch size: `--batch-size 1`
- output resolution: 256x256
- DDIM steps: `--ddim-steps 50`

Command:

```cmd
cd /d H:\LDM\热源模型\6000\VQGAN\TIGLDM600\TIGLDM6000 && set CUDA_VISIBLE_DEVICES=0 && python scripts\infer_laserwelding_stage2.py --config configs/latent-diffusion/laserwelding-ldm-vq-lite.yaml --ckpt logs/Base/你的第二阶段目录/checkpoints/last.ckpt --first-stage-ckpt logs/Base/你的第一阶段目录/checkpoints/last.ckpt --input-root H:\数据集\laserwelding\validation --input-json H:\数据集\laserwelding\validation\validation.json --outdir outputs\laserwelding_benchmark --count 64 --batch-size 1 --ddim-steps 50 --device cuda --overwrite --benchmark --warmup-batches 1
```

The final output includes:

```text
Benchmark: gpu=cuda, batch_size=1, resolution=256x256, ddim_steps=50, warmup_batches=1, timed_images=63, total_generation_seconds=..., avg_seconds_per_image=...
```

## Useful Configs

```text
configs/autoencoder/autoencoder_kl_32x32x4_large2_laserwelding_train.yaml
configs/latent-diffusion/laserwelding-ldm-vq-lite.yaml
configs/latent-diffusion/laserwelding-ldm-text-numeric-vq-lite.yaml
configs/latent-diffusion/ss304-ldm-text-numeric-vq.yaml
configs/latent-diffusion/ss304-ldm-text-numeric-vq-no-y2h.yaml
```

## Useful Scripts

```text
scripts/infer_first_stage_vqgan.py
scripts/infer_laserwelding_stage2.py
scripts/infer_ss304_text_stage2.py
scripts/calc_laserwelding_metrics.py
scripts/extract_welding_numeric_params.py
scripts/correct_welding_categories_from_label_json.py
```

## Troubleshooting

If training prints `Killed`, reduce memory pressure:

```cmd
data.params.batch_size=1 data.params.num_workers=0 lightning.trainer.accumulate_grad_batches=8
```

If PyTorch Lightning does not accept `--gpus 0`, use:

```cmd
--devices 0 --accelerator gpu
```

If a path containing Chinese characters fails in YAML, use forward slashes
(`H:/数据集/...`) or override the path directly on the command line.

## Original Project

This codebase builds on Asymmetric VQGAN and Stable Diffusion components:

- Paper: [Designing a Better Asymmetric VQGAN for StableDiffusion](https://arxiv.org/abs/2306.04632)
- Stable Diffusion: https://github.com/runwayml/stable-diffusion

## BibTeX

```bibtex
@misc{zhu2023designing,
      title={Designing a Better Asymmetric VQGAN for StableDiffusion},
      author={Zixin Zhu and Xuelu Feng and Dongdong Chen and Jianmin Bao and Le Wang and Yinpeng Chen and Lu Yuan and Gang Hua},
      year={2023},
      eprint={2306.04632},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
