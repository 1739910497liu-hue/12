"""Generation-only physical conditioning; target RGB never enters the decoder."""
import math
import torch
from torch import nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.model import Encoder, Decoder, nonlinearity
from ldm.modules.diffusionmodules.openaimodel import UNetModel


def fuse_physics(features, physical, mask, mode):
    if mode == "off":
        return features
    if mode == "gated":
        return features + mask * physical
    if mode == "plain":
        return features + physical
    if mode == "convex":
        return mask * physical + (1 - mask) * features
    raise ValueError("Unknown fusion: {}".format(mode))


class PhysicalDecoder(Decoder):
    """Prior fusion by addition, concatenation, residual or convex gating.

    This adapts the old inpainting mask-gate idea to generation. It is NOT the
    old inpainting decoder or a claim that the previous checkpoint used it.
    """
    def __init__(self, fusion="gated", gate_mode="mask", gate_init=0.5,
                 gate_alpha_init=1., gate_alpha_scale=10000., gate_bias_init=-1., **kwargs):
        super().__init__(**kwargs)
        if fusion not in ("off", "plain", "gated", "concat", "convex"):
            raise ValueError("Unknown decoder fusion: " + str(fusion))
        self.fusion = fusion
        if gate_mode not in ("mask", "fixed", "scalar", "spatial_sigmoid"):
            raise ValueError("Unknown decoder gate mode: " + str(gate_mode))
        if gate_mode == "spatial_sigmoid" and fusion != "convex":
            raise ValueError("decoder_gate_mode='spatial_sigmoid' requires decoder_fusion='convex'.")
        if gate_mode in ("fixed", "scalar") and fusion not in ("gated", "convex"):
            raise ValueError("Fixed/scalar decoder gates require fusion='gated' or 'convex'.")
        if not math.isfinite(gate_init) or not 0 < gate_init < 1:
            raise ValueError("decoder_gate_init must be finite and strictly between 0 and 1.")
        self.gate_mode, self.gate_init = gate_mode, float(gate_init)
        if not math.isfinite(gate_alpha_init):
            raise ValueError("decoder_gate_alpha_init must be finite.")
        if not math.isfinite(gate_alpha_scale) or gate_alpha_scale <= 0:
            raise ValueError("decoder_gate_alpha_scale must be finite and positive.")
        if not math.isfinite(gate_bias_init):
            raise ValueError("decoder_gate_bias_init must be finite.")
        self.gate_alpha_scale = float(gate_alpha_scale)
        channels = [level.block[0].in_channels for level in self.up]
        self.physical_projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, c, 1))
            for c in channels])
        self.physical_projections.requires_grad_(fusion != "off")
        # Deterministic initialization consumes no RNG: shared weights retain
        # the same initialization in G1/G2/G3 when the seed is held fixed.
        if gate_mode == "scalar":
            logit = math.log(self.gate_init / (1 - self.gate_init))
            self.gate_logits = nn.Parameter(torch.full((len(channels),), logit))
        else:
            # None adds no state-dict keys, keeping legacy G3 checkpoints valid.
            self.register_parameter("gate_logits", None)
        if gate_mode == "spatial_sigmoid":
            # These are per-resolution parameters. Fixed unit scaling changes
            # optimizer parameterization, never the raw prior per sample.
            # Defaults are an engineering initialization, not a paper claim:
            # alpha = 10000 * alpha_raw, with alpha_raw=1 and bias=-1.
            self.gate_alpha_raw = nn.Parameter(torch.full((len(channels),), float(gate_alpha_init)))
            self.gate_bias = nn.Parameter(torch.full((len(channels),), float(gate_bias_init)))
        else:
            self.register_parameter("gate_alpha_raw", None)
            self.register_parameter("gate_bias", None)
        # An empty ModuleList adds no keys to existing checkpoint formats.
        self.concat_projections = nn.ModuleList()
        if fusion == "concat":
            # Preserve the RNG used by subsequent modules (e.g. discriminator).
            # [I, I] makes this learned 2C->C convolution start as H + P.
            with torch.random.fork_rng(devices=[]):
                for channels_at_level in channels:
                    projection = nn.Conv2d(2 * channels_at_level, channels_at_level, 1)
                    with torch.no_grad():
                        projection.weight.zero_()
                        projection.bias.zero_()
                        identity = torch.eye(channels_at_level)
                        projection.weight[:, :channels_at_level, 0, 0].copy_(identity)
                        projection.weight[:, channels_at_level:, 0, 0].copy_(identity)
                    self.concat_projections.append(projection)
        self.capture_fusion_stats = False
        self.fusion_stats = []
        self.capture_gate_maps = False
        self.gate_maps = []

    def get_gate(self, prior, level):
        """Effective fusion gate, broadcast over feature channels."""
        if self.fusion == "concat":
            raise ValueError("Concatenation fusion has no scalar/spatial gate map.")
        if self.gate_mode == "spatial_sigmoid":
            self._validate_raw_prior(prior)
            alpha = self.gate_alpha_scale * self.gate_alpha_raw[level]
            return torch.sigmoid(alpha * prior[:, 2:3] + self.gate_bias[level])
        mask = prior[:, 1:2]
        if self.fusion == "off":
            return torch.zeros_like(mask)
        if self.fusion == "plain":
            return torch.ones_like(mask)
        if self.gate_mode == "fixed":
            return torch.full_like(mask, self.gate_init)
        if self.gate_mode == "scalar":
            return self.gate_logits[level].sigmoid().expand_as(mask)
        return mask

    @staticmethod
    def _validate_raw_prior(template):
        if template.ndim != 4 or template.shape[1] != 3:
            raise ValueError("spatial_sigmoid requires a 4D, three-channel template "
                             "[normalized heat, binary mask, raw analytic heat]; "
                             "set include_raw_prior=True.")

    def forward(self, z, template):
        self.fusion_stats = []
        self.gate_maps = []
        if self.gate_mode == "spatial_sigmoid":
            self._validate_raw_prior(template)
        h = self.conv_in(z)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h, None)), None)
        for level in reversed(range(self.num_resolutions)):
            if self.fusion != "off":
                prior = F.interpolate(template, size=h.shape[-2:], mode="nearest")
                # Preserve the existing conditional features; raw channel 3
                # is used exclusively by the new continuous spatial gate.
                physical = self.physical_projections[level](prior[:, :2])
                gate = None if self.fusion == "concat" else self.get_gate(prior, level)
                if self.capture_gate_maps and gate is not None:
                    self.gate_maps.append({"level": level, "gate": gate.detach().float().cpu().clone()})
                if self.capture_fusion_stats:
                    with torch.no_grad():
                        active = prior[:, 1:2]
                        a, b = (h.float() * active).flatten(1), (physical.float() * active).flatten(1)
                        stats = {"level": level,
                            "masked_cosine": F.cosine_similarity(a, b, dim=1).mean().item(),
                            "physical_to_latent_norm": (b.norm(dim=1) / a.norm(dim=1).clamp_min(1e-8)).mean().item()}
                        if gate is not None:
                            stats.update(gate_mean=gate.float().mean().item(),
                                         gate_min=gate.float().min().item(), gate_max=gate.float().max().item())
                        self.fusion_stats.append(stats)
                if self.fusion == "concat":
                    h = self.concat_projections[level](torch.cat((h, physical), dim=1))
                else:
                    h = fuse_physics(h, physical, gate, self.fusion)
            for block in range(self.num_res_blocks + 1):
                h = self.up[level].block[block](h, None)
                if self.up[level].attn:
                    h = self.up[level].attn[block](h)
            if level:
                h = self.up[level].upsample(h)
        return self.conv_out(nonlinearity(self.norm_out(h)))


class ChunkedVectorQuantizer(nn.Module):
    def __init__(self, size, dim, beta=0.25):
        super().__init__()
        self.embedding = nn.Embedding(size, dim)
        nn.init.uniform_(self.embedding.weight, -1 / size, 1 / size)
        self.beta = beta

    def forward(self, z):
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, z.shape[1])
        indices = []
        # Keep the nearest-neighbour search in FP32 even during mixed precision.
        with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
            weights = self.embedding.weight.float()
            for vectors in flat.float().split(1024):
                best = torch.full((len(vectors),), float("inf"), device=z.device)
                ids = torch.zeros(len(vectors), dtype=torch.long, device=z.device)
                for start in range(0, len(weights), 4096):
                    code = weights[start:start + 4096]
                    distances = vectors.square().sum(1, keepdim=True) + code.square().sum(1) - 2 * vectors @ code.t()
                    values, candidate = distances.min(1)
                    update = values < best
                    ids = torch.where(update, candidate + start, ids)
                    best = torch.minimum(best, values)
                indices.append(ids)
        indices = torch.cat(indices)
        quant = self.embedding(indices).view(z.shape[0], z.shape[2], z.shape[3], z.shape[1]).permute(0, 3, 1, 2).contiguous()
        loss = F.mse_loss(quant.float(), z.detach().float()) + self.beta * F.mse_loss(z.float(), quant.detach().float())
        quant = z + (quant.to(z.dtype) - z).detach()
        return quant, loss, indices


class HeatVQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        if config.get("decoder_gate_mode", "mask") == "spatial_sigmoid" and not config.get("include_raw_prior", False):
            raise ValueError("decoder_gate_mode='spatial_sigmoid' requires include_raw_prior=True.")
        f = config["factor"]
        multipliers = {4: [1, 2, 4], 8: [1, 2, 4, 4], 16: [1, 2, 4, 4, 4],
                       32: [1, 2, 4, 4, 4, 4]}
        if f not in multipliers:
            raise ValueError("Supported VQ factors are 4, 8, 16 and 32.")
        if config["image_size"] % f:
            raise ValueError("image_size must be divisible by the VQ factor.")
        mult = multipliers[f]
        dd = dict(ch=config["channels"], ch_mult=mult, num_res_blocks=config["res_blocks"],
                  attn_resolutions=[], resolution=config["image_size"], in_channels=3,
                  out_ch=3, z_channels=config["embed_dim"], dropout=0., attn_type="vanilla")
        self.encoder = Encoder(double_z=False, **dd)
        self.quant_conv = nn.Conv2d(config["embed_dim"], config["embed_dim"], 1)
        self.quantize = ChunkedVectorQuantizer(config["codebook_size"], config["embed_dim"])
        self.post_quant_conv = nn.Conv2d(config["embed_dim"], config["embed_dim"], 1)
        self.decoder = PhysicalDecoder(fusion=config.get("decoder_fusion", config["fusion"]),
                                       gate_mode=config.get("decoder_gate_mode", "mask"),
                                       gate_init=config.get("decoder_gate_init", 0.5),
                                       gate_alpha_init=config.get("decoder_gate_alpha_init", 1.),
                                       gate_alpha_scale=config.get("decoder_gate_alpha_scale", 10000.),
                                       gate_bias_init=config.get("decoder_gate_bias_init", -1.), **dd)

    def encode(self, image):
        return self.quantize(self.quant_conv(self.encoder(image)))

    def decode(self, latent, template, requantize=False):
        if requantize:
            latent = self.quantize(latent)[0]
        return self.decoder(self.post_quant_conv(latent), template)

    def forward(self, image, template):
        latent, loss, indices = self.encode(image)
        return self.decode(latent, template), loss, indices


class ConditionEncoder(nn.Module):
    """Fixed token count and UNet architecture across the four branch settings."""
    def __init__(self, mode, clip_path, context_dim=512):
        super().__init__()
        self.mode, self.context_dim = mode, context_dim
        self.null_token = nn.Parameter(torch.zeros(1, 1, context_dim))
        self.numeric = nn.ModuleList([nn.Sequential(nn.Linear(1, context_dim), nn.SiLU(),
                                                   nn.Linear(context_dim, context_dim)) for _ in range(3)])
        self.numeric.requires_grad_(mode in ("numeric", "both"))
        self.clip = None
        if mode in ("text", "both"):
            from transformers import CLIPTokenizer, CLIPTextModel
            if not clip_path:
                raise ValueError("Text experiments require --clip-path to a local CLIP directory.")
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
            self.clip = CLIPTextModel.from_pretrained(clip_path, local_files_only=True).eval().requires_grad_(False)
            if self.clip.config.hidden_size != context_dim:
                raise ValueError("CLIP hidden dimension must match context_dim (default 512, ViT-B/32).")

    def train(self, mode=True):
        super().train(mode)
        if self.clip is not None:
            self.clip.eval()
        return self

    def forward(self, numeric, captions):
        batch = len(numeric)
        text = self.null_token.expand(batch, 77, -1)
        number = self.null_token.expand(batch, 3, -1)
        if self.mode in ("text", "both"):
            tokens = self.tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
            tokens = {k: v.to(numeric.device) for k, v in tokens.items()}
            with torch.no_grad():
                text = self.clip(**tokens).last_hidden_state
        if self.mode in ("numeric", "both"):
            number = torch.stack([layer(numeric[:, i:i+1]) for i, layer in enumerate(self.numeric)], dim=1)
        return torch.cat((text, number), dim=1)


class HeatDiffusion(nn.Module):
    def __init__(self, config, clip_path=None):
        super().__init__()
        self.config = dict(config)
        dim = config["embed_dim"]
        factor = config["factor"]
        if factor <= 0 or config["image_size"] % factor:
            raise ValueError("image_size must be divisible by a positive factor.")
        if not config["unet_mult"]:
            raise ValueError("unet_mult must contain at least one level.")
        latent_side = config["image_size"] // factor
        unet_divisor = 2 ** (len(config["unet_mult"]) - 1)
        if latent_side < unet_divisor or latent_side % unet_divisor:
            raise ValueError(
                "Latent side {} is incompatible with {} U-Net levels (requires a positive multiple of {}). "
                "For 256px images and factor=32 use unet_mult=[1, 2, 2, 4].".format(
                    latent_side, len(config["unet_mult"]), unet_divisor))
        # Instantiate identical shared weights before condition-specific modules.
        self.unet = UNetModel(image_size=latent_side,
                              in_channels=dim + (2 if config.get("diffusion_conditioning") == "concat" else 0),
                              out_channels=dim, model_channels=config["unet_channels"],
                              num_res_blocks=config["unet_res_blocks"], channel_mult=config["unet_mult"],
                              attention_resolutions=config["attention_resolutions"], num_heads=config["heads"],
                              use_scale_shift_norm=True, resblock_updown=True,
                              use_spatial_transformer=True, context_dim=config["context_dim"],
                              transformer_depth=1, use_checkpoint=False)
        self.physical = nn.Sequential(nn.Conv2d(2, dim, 3, padding=1), nn.SiLU(), nn.Conv2d(dim, dim, 1))
        self.physical.requires_grad_(config["fusion"] != "off" and config.get("diffusion_conditioning") != "concat")
        self.conditioner = ConditionEncoder(config["condition"], clip_path, config["context_dim"])
        betas = torch.linspace(config["beta_start"] ** 0.5, config["beta_end"] ** 0.5,
                               config["timesteps"], dtype=torch.float64).square()
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, dim=0).float())

    def predict(self, noisy, timestep, template, context):
        if self.config["fusion"] == "off":
            features = noisy
        else:
            # Optional raw channel is decoder-only. Slice before resizing so
            # diffusion conditioning follows the original two-channel path.
            prior = F.interpolate(template[:, :2], size=noisy.shape[-2:], mode="nearest")
            if self.config.get("diffusion_conditioning") == "concat":
                features = torch.cat((noisy, prior), dim=1)
            else:
                features = fuse_physics(noisy, self.physical(prior), prior[:, 1:2], self.config["fusion"])
        return self.unet(features, timestep, context=context)

    def loss(self, latent, template, numeric, captions, generator=None):
        t = torch.randint(len(self.alpha_bar), (len(latent),), device=latent.device, generator=generator)
        noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator)
        alpha = self.alpha_bar[t, None, None, None]
        noisy = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise
        prediction = self.predict(noisy, t, template, self.conditioner(numeric, captions))
        return F.l1_loss(prediction.float(), noise.float())

    @torch.no_grad()
    def sample(self, template, numeric, captions, steps=50, generator=None):
        if not 2 <= steps <= len(self.alpha_bar):
            raise ValueError("DDIM steps must be between 2 and timesteps.")
        size = self.config["image_size"] // self.config["factor"]
        z = torch.randn((len(template), self.config["embed_dim"], size, size),
                        device=template.device, generator=generator)
        context = self.conditioner(numeric, captions)
        times = torch.linspace(len(self.alpha_bar) - 1, 0, steps, device=z.device).long()
        for index, t in enumerate(times):
            timestep = t.expand(len(z))
            eps = self.predict(z, timestep, template, context)
            a = self.alpha_bar[t]
            previous = self.alpha_bar[times[index + 1]] if index + 1 < steps else z.new_tensor(1.)
            clean = (z - (1 - a).sqrt() * eps) / a.sqrt()
            z = previous.sqrt() * clean + (1 - previous).sqrt() * eps
        return z


class PatchDiscriminator(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, channels, 4, 2, 1), nn.LeakyReLU(.2),
                                    nn.Conv2d(channels, channels * 2, 4, 2, 1), nn.GroupNorm(32, channels * 2), nn.LeakyReLU(.2),
                                    nn.Conv2d(channels * 2, channels * 4, 3, 1, 1), nn.GroupNorm(32, channels * 4), nn.LeakyReLU(.2),
                                    nn.Conv2d(channels * 4, 1, 3, 1, 1))

    def forward(self, image):
        return self.layers(image)
