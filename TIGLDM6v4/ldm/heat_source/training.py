"""Single-GPU, resumable two-stage training and generation for the study."""
import contextlib
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

from .data import HeatDataset, stable_seed
from .models import HeatVQ, HeatDiffusion, PatchDiscriminator

VQ_KEYS = ("image_size", "factor", "embed_dim", "codebook_size", "channels", "res_blocks", "fusion", "perturbation")


def load_checkpoint(path, device="cpu"):
    # These checkpoints are produced by this runner, including optimizer/RNG state.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_identity(directory):
    return {name: file_hash(Path(directory) / name) for name in ("train.jsonl", "val.jsonl", "test.jsonl", "numeric_stats.json")}


def datasets(config, data_root, manifests, seed, stage):
    manifests = Path(manifests)
    stats = json.loads((manifests / "numeric_stats.json").read_text(encoding="utf-8"))
    condition = config["condition"] if stage == 2 else "none"
    return [HeatDataset(manifests / (split + ".jsonl"), data_root, config["image_size"],
                        config["perturbation"], seed, split == "train", condition, stats,
                        config.get("caption_field", "caption"), config.get("prior_source", "analytic"),
                        include_raw_prior=config.get("include_raw_prior", False))
            for split in ("train", "val", "test")]


def autocast(config, device):
    if config["precision"] == "fp32":
        return contextlib.nullcontext()
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 needs a CUDA device with BF16 support; use --precision fp32.")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


class BatchStream:
    def __init__(self, dataset, batch_size, workers, seed, state=None):
        if len(dataset) < batch_size:
            raise ValueError("Training set smaller than batch_size.")
        self.dataset, self.batch_size, self.workers, self.seed = dataset, batch_size, workers, seed
        self.state = dict(state or {"epoch": 0, "batch": 0})
        self.iterator = None

    def __next__(self):
        if self.iterator is None:
            self.dataset.epoch = self.state["epoch"]
            generator = torch.Generator().manual_seed(self.seed + self.state["epoch"])
            order = torch.randperm(len(self.dataset), generator=generator).tolist()
            start = self.state["batch"] * self.batch_size
            loader = DataLoader(Subset(self.dataset, order[start:]), batch_size=self.batch_size,
                                num_workers=self.workers, drop_last=True, pin_memory=True,
                                generator=generator, persistent_workers=False)
            self.iterator = iter(loader)
        try:
            batch = next(self.iterator)
            self.state["batch"] += 1
            return batch
        except StopIteration:
            self.state = {"epoch": self.state["epoch"] + 1, "batch": 0}
            self.iterator = None
            return next(self)


def write_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def make_vq(checkpoint, config, device):
    state = load_checkpoint(checkpoint)
    if state.get("stage") != 1:
        raise ValueError("--first-stage must be a stage-1 heat-source checkpoint.")
    differences = [key for key in VQ_KEYS if state["config"][key] != config[key]]
    saved_fusion = state["config"].get("decoder_fusion", state["config"]["fusion"])
    requested_fusion = config.get("decoder_fusion", config["fusion"])
    if saved_fusion != requested_fusion:
        differences.append("decoder_fusion")
    if state["config"].get("prior_source", "analytic") != config.get("prior_source", "analytic"):
        differences.append("prior_source")
    for key, default in (("decoder_gate_mode", "mask"), ("decoder_gate_init", 0.5),
                         ("include_raw_prior", False), ("decoder_gate_alpha_init", 1.0),
                         ("decoder_gate_alpha_scale", 10000.0), ("decoder_gate_bias_init", -1.0)):
        if state["config"].get(key, default) != config.get(key, default):
            differences.append(key)
    if differences:
        raise ValueError("First-stage mismatch in {}. Each f/K/fusion/perturbation needs its own training.".format(differences))
    vq = HeatVQ(state["config"]).to(device)
    vq.load_state_dict(state["model"], strict=True)
    return vq.eval().requires_grad_(False), state


@torch.no_grad()
def fit_latent_scale(vq, train, config, device):
    loader = DataLoader(Subset(train, range(min(len(train), config["latent_scale_images"]))),
                        batch_size=config["batch_size"], shuffle=False, num_workers=0)
    count, total, squares = 0, 0., 0.
    for batch in loader:
        latent = vq.encode(batch["image"].to(device))[0].double()
        count += latent.numel()
        total += latent.sum().item()
        squares += latent.square().sum().item()
    variance = max(squares / count - (total / count) ** 2, 1e-12)
    return variance ** -0.5


@torch.no_grad()
def validate(model, vq, dataset, config, stage, scale, device, seed):
    model.eval()
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["workers"])
    generator = torch.Generator(device=device).manual_seed(seed + 100000)
    total, count = 0., 0
    usage = torch.zeros(config["codebook_size"], device=device) if stage == 1 else None
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(config, device):
            if stage == 1:
                reconstruction, _, indices = model(batch["image"], batch["template"])
                loss = F.l1_loss(reconstruction.float(), batch["image"])
                usage += torch.bincount(indices, minlength=len(usage))
            else:
                latent = vq.encode(batch["image"])[0] * scale
                loss = model.loss(latent, batch["template"], batch["numeric"], batch["caption"], generator)
        total += loss.item() * len(batch["image"])
        count += len(batch["image"])
    result = {"val_l1" if stage == 1 else "val_noise_l1": total / count}
    if usage is not None:
        probabilities = usage / usage.sum()
        result["codebook_utilization"] = float((usage > 0).float().mean())
        result["codebook_perplexity"] = float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))
    model.train()
    return result


def train(config, stage, data_root, manifests, out, seed, device, first_stage=None, clip_path=None, resume=False):
    seed_everything(seed)
    device, out = torch.device(device), Path(out)
    trainset, valset, _ = datasets(config, data_root, manifests, seed, stage)
    identity = manifest_identity(manifests)
    existing = out / "last.pt"
    if existing.exists() and not resume:
        raise FileExistsError("Run exists; use --resume or a new output directory: {}".format(out))
    if resume and not existing.is_file():
        raise FileNotFoundError("Cannot resume: {}".format(existing))
    out.mkdir(parents=True, exist_ok=True)
    model = HeatVQ(config) if stage == 1 else HeatDiffusion(config, clip_path)
    model = model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["stage{}_lr".format(stage)], betas=(.5, .9) if stage == 1 else (.9, .999), weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step + 1) / max(1, config["warmup_steps"])))
    discriminator, perceptual, disc_optimizer, vq = None, None, None, None
    first_hash, scale = None, 1.
    if stage == 1:
        discriminator = PatchDiscriminator().to(device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config["stage1_lr"], betas=(.5, .9))
        if config["perceptual_weight"]:
            import lpips
            perceptual = lpips.LPIPS(net="vgg").to(device).eval().requires_grad_(False)
    else:
        if not first_stage:
            raise ValueError("Stage 2 requires --first-stage from the matching experiment and seed.")
        vq, vq_state = make_vq(first_stage, config, device)
        if vq_state["seed"] != seed or vq_state["manifest_hashes"] != identity:
            raise ValueError("First stage has a different seed or data manifests.")
        if vq_state["step"] != config["stage1_steps"]:
            raise ValueError("Stage-1 training budget is incomplete or differs from this experiment.")
        first_hash = file_hash(first_stage)
        scale = fit_latent_scale(vq, trainset, config, device)
    generator = torch.Generator(device=device).manual_seed(seed + 200000)
    step, best, stream_state, previous_wall = 0, float("inf"), None, 0.
    if resume:
        saved = load_checkpoint(existing)
        if saved["config"] != config or saved["seed"] != seed or saved["stage"] != stage or saved["manifest_hashes"] != identity:
            raise ValueError("Resume configuration/seed/stage/manifests differ from checkpoint.")
        if saved["first_stage_hash"] != first_hash:
            raise ValueError("First-stage weights changed since this run started.")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if discriminator is not None:
            discriminator.load_state_dict(saved["discriminator"])
            disc_optimizer.load_state_dict(saved["disc_optimizer"])
        step, best, stream_state, scale = saved["step"], saved["best"], saved["stream"], saved["latent_scale"]
        previous_wall = saved.get("training_wall_seconds", 0.)
        generator.set_state(saved["generator_rng"])
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    stream = BatchStream(trainset, config["batch_size"], config["workers"], seed, stream_state)
    metadata = {"config": config, "seed": seed, "stage": stage, "manifest_hashes": identity,
                "torch": torch.__version__, "device": str(device), "first_stage_hash": first_hash,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "effective_batch_size": config["batch_size"] * config["accumulate"],
                "latent_scale": scale}
    (out / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    model.train()
    budget = config["stage{}_steps".format(stage)]
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while step < budget:
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if disc_optimizer is not None:
            disc_optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0., "discriminator_loss": 0.}
        for _ in range(config["accumulate"]):
            batch = to_device(next(stream), device)
            with autocast(config, device):
                if stage == 1:
                    discriminator.requires_grad_(False)
                    reconstruction, quant_loss, _ = model(batch["image"], batch["template"])
                    loss = F.l1_loss(reconstruction.float(), batch["image"]) + config["codebook_weight"] * quant_loss
                    if perceptual is not None:
                        loss = loss + config["perceptual_weight"] * perceptual(reconstruction.float(), batch["image"]).mean()
                    if step >= config["disc_start"]:
                        loss = loss - config["disc_weight"] * discriminator(reconstruction).mean()
                else:
                    with torch.no_grad():
                        latent = vq.encode(batch["image"])[0] * scale
                    loss = model.loss(latent, batch["template"], batch["numeric"], batch["caption"], generator)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss at optimizer step {}".format(step))
            (loss / config["accumulate"]).backward()
            totals["loss"] += loss.item() / config["accumulate"]
            if stage == 1 and step >= config["disc_start"]:
                discriminator.requires_grad_(True)
                with autocast(config, device):
                    real, fake = discriminator(batch["image"]), discriminator(reconstruction.detach())
                    disc_loss = .5 * (F.relu(1 - real).mean() + F.relu(1 + fake).mean())
                (disc_loss / config["accumulate"]).backward()
                totals["discriminator_loss"] += disc_loss.item() / config["accumulate"]
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if stage == 1 and step >= config["disc_start"]:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config["clip_grad"], error_if_nonfinite=True)
            disc_optimizer.step()
        step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        totals["optimizer_step_seconds"] = time.perf_counter() - step_start
        improved = False
        if step % config["validate_every"] == 0 or step == budget:
            metrics = validate(model, vq, valset, config, stage, scale, device, seed)
            totals.update(metrics)
            score = metrics["val_l1" if stage == 1 else "val_noise_l1"]
            improved, best = score < best, min(best, score)
        if step % config["log_every"] == 0 or "val_l1" in totals or "val_noise_l1" in totals:
            totals.update(step=step, seconds_since_start=time.perf_counter() - start_time,
                          lr=optimizer.param_groups[0]["lr"],
                          peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(totals) + "\n")
            print(json.dumps(totals), flush=True)
        if step % config["save_every"] == 0 or step == budget or improved:
            payload = dict(metadata, step=step, best=best, model=model.state_dict(), optimizer=optimizer.state_dict(),
                           scheduler=scheduler.state_dict(), stream=stream.state, latent_scale=scale,
                           training_wall_seconds=previous_wall + time.perf_counter() - start_time,
                           peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                           generator_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                           numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                           cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                           discriminator=discriminator.state_dict() if discriminator is not None else None,
                           disc_optimizer=disc_optimizer.state_dict() if disc_optimizer is not None else None)
            write_checkpoint(existing, payload)
            if improved:
                write_checkpoint(out / "best.pt", payload)
    return existing


def save_image(tensor, path):
    array = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path)


@torch.no_grad()
def sample_run(checkpoint, first_stage, data_root, manifests, out, clip_path, device, sample_seed=2026, limit=None,
               sampling_steps=None):
    state = load_checkpoint(checkpoint)
    config, seed, stage = state["config"], state["seed"], state["stage"]
    sample_steps = config["ddim_steps"] if sampling_steps is None else sampling_steps
    if limit is not None and limit < 1:
        raise ValueError("Sample limit must be positive.")
    if stage == 2 and not 2 <= sample_steps <= config["timesteps"]:
        raise ValueError("Sampling steps must lie in [2, timesteps].")
    if manifest_identity(manifests) != state["manifest_hashes"]:
        raise ValueError("Evaluation manifests differ from training run.")
    device, out = torch.device(device), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty sample output directory to avoid mixing runs: " + str(out))
    real, fake = out / "real", out / "samples"
    real.mkdir(parents=True, exist_ok=True)
    fake.mkdir(parents=True, exist_ok=True)
    testset = datasets(config, data_root, manifests, seed, stage)[2]
    if stage == 1:
        model = HeatVQ(config).to(device).eval()
        model.load_state_dict(state["model"], strict=True)
        vq = None
    else:
        if not first_stage or file_hash(first_stage) != state["first_stage_hash"]:
            raise ValueError("Sampling needs the exact first-stage checkpoint used during training.")
        vq, _ = make_vq(first_stage, config, device)
        model = HeatDiffusion(config, clip_path).to(device).eval()
        model.load_state_dict(state["model"], strict=True)
    # One item per generator seed makes sampling invariant to batch-size changes.
    loader = DataLoader(testset, batch_size=1, shuffle=False)
    elapsed = []
    count = 0
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        batch = to_device(batch, device)
        generator = torch.Generator(device=device).manual_seed(stable_seed(sample_seed, 0, batch["id"][0]))
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with autocast(config, device):
            if stage == 1:
                generated = model(batch["image"], batch["template"])[0]
            else:
                latent = model.sample(batch["template"], batch["numeric"], batch["caption"], sample_steps, generator)
                generated = vq.decode(latent / state["latent_scale"], batch["template"], requantize=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
        save_image(batch["image"][0], real / (batch["id"][0] + ".png"))
        save_image(generated[0], fake / (batch["id"][0] + ".png"))
        count += 1
    info = {"experiment": config["experiment"], "stage": stage, "training_seed": seed,
            "sample_seed": sample_seed, "images": count, "sampling_steps": sample_steps if stage == 2 else None,
            "checkpoint_sha256": file_hash(checkpoint), "first_stage_hash": state["first_stage_hash"],
            "manifest_hashes": state["manifest_hashes"], "config": config,
            "parameters": state["parameters"], "trainable_parameters": state["trainable_parameters"],
            "training_wall_seconds": state.get("training_wall_seconds"),
            "peak_memory_bytes": state.get("peak_memory_bytes"),
            "mean_seconds_per_image": float(np.mean(elapsed)),
            "steady_seconds_per_image": float(np.mean(elapsed[1:])) if len(elapsed) > 1 else None,
            "paired_metrics_valid": stage == 1}
    (out / "sampling.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
