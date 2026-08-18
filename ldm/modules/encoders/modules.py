import torch
import torch.nn as nn
from functools import partial
from pathlib import Path
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


def load_openai_clip():
    try:
        import clip
    except ImportError as exc:
        raise ImportError(
            "OpenAI CLIP is required for FrozenCLIPTextEmbedder/FrozenClipImageEmbedder. "
            "Install it with: pip install git+https://github.com/openai/CLIP.git"
        ) from exc
    return clip


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77,
                 tokenizer_name="bert-base-uncased", local_files_only=False):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained(
            tokenizer_name,
            local_files_only=local_files_only,
        )
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda", use_tokenizer=True, embedding_dropout=0.0,
                 tokenizer_name="bert-base-uncased", local_files_only=False):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(
                vq_interface=False,
                max_length=max_seq_len,
                tokenizer_name=tokenizer_name,
                local_files_only=local_files_only,
            )
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(
            self,
            version="openai/clip-vit-large-patch14",
            device="cuda",
            max_length=77,
            freeze=True,
            local_files_only=False,
    ):
        super().__init__()
        if local_files_only:
            model_dir = Path(version)
            required_files = ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt")
            missing_files = [name for name in required_files if not (model_dir / name).is_file()]
            has_weights = (model_dir / "pytorch_model.bin").is_file() or (model_dir / "model.safetensors").is_file()
            if not model_dir.is_dir():
                raise FileNotFoundError(
                    "Local CLIP directory not found: {}. Download the full HuggingFace model folder "
                    "openai/clip-vit-base-patch32 to this path, or change cond_stage_config.params.version.".format(model_dir)
                )
            if missing_files or not has_weights:
                if not has_weights:
                    missing_files.append("pytorch_model.bin or model.safetensors")
                raise FileNotFoundError(
                    "Local CLIP directory is incomplete: {}. Missing: {}".format(
                        model_dir,
                        ", ".join(missing_files),
                    )
                )
        self.tokenizer = CLIPTokenizer.from_pretrained(version, local_files_only=local_files_only)
        self.transformer = CLIPTextModel.from_pretrained(version, local_files_only=local_files_only)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class NumericConditionMLP(nn.Module):
    def __init__(self, input_dim=4, embed_dim=512, hidden_dim=512, num_groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, y):
        return self.net(y.float())


class CCGANShortNumericEncoder(nn.Module):
    """CcGAN model_y2h-style short encoder for continuous numeric labels."""
    def __init__(self, input_dim=4, dim_embed=128, num_groups=8):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, dim_embed),
            nn.GroupNorm(num_groups, dim_embed),
            nn.ReLU(),
            nn.Linear(dim_embed, dim_embed),
            nn.GroupNorm(num_groups, dim_embed),
            nn.ReLU(),
            nn.Linear(dim_embed, dim_embed),
            nn.GroupNorm(num_groups, dim_embed),
            nn.ReLU(),
            nn.Linear(dim_embed, dim_embed),
            nn.GroupNorm(num_groups, dim_embed),
            nn.ReLU(),
            nn.Linear(dim_embed, dim_embed),
            nn.ReLU(),
        )

    def forward(self, y):
        return self.main(y.float().view(y.shape[0], -1))


class ScalarTokenMLP(nn.Module):
    def __init__(self, token_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, x):
        return self.net(x.view(-1, 1).float())


class PhysicalExpert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DecoupledDynamicPhysicalEncoder(nn.Module):
    """
    Decoupled physical parameter encoder:
    I/V/v/Q scalar MLP tokens -> dynamic router -> expert fusion -> physical tokens.

    The dataset stores numeric_params as [voltage_v, current_a, speed_cm_min, gas_flow_l_min].
    Internally we reorder them to [I, V, v, Q] to match the welding-physics notation.
    """
    def __init__(
            self,
            numeric_dim=4,
            embed_dim=512,
            scalar_hidden_dim=256,
            expert_hidden_dim=1024,
            num_experts=4,
            num_physical_tokens=4,
            parameter_normalizer=(22.0, 350.0, 75.0, 40.0),
            normalize_parameters=False,
            include_decoupled_tokens=False,
            use_physics_features=True,
            short_encoder_type="none",
            short_encoder_ckpt=None,
            short_encoder_dim=128,
            short_encoder_uses_normalized=True,
            freeze_short_encoder=True,
            num_groups=8,
    ):
        super().__init__()
        if numeric_dim != 4:
            raise ValueError("DecoupledDynamicPhysicalEncoder expects 4 parameters: V, I, v, Q.")
        self.embed_dim = embed_dim
        self.num_experts = int(num_experts)
        self.num_physical_tokens = int(num_physical_tokens)
        self.normalize_parameters = bool(normalize_parameters)
        self.include_decoupled_tokens = bool(include_decoupled_tokens)
        self.use_physics_features = bool(use_physics_features)
        self.short_encoder_type = str(short_encoder_type or "none").lower()
        self.short_encoder_uses_normalized = bool(short_encoder_uses_normalized)

        normalizer = torch.tensor(parameter_normalizer, dtype=torch.float32).view(1, 4)
        self.register_buffer("parameter_normalizer", normalizer, persistent=False)

        self.scalar_mlps = nn.ModuleDict({
            "I": ScalarTokenMLP(embed_dim, scalar_hidden_dim),
            "V": ScalarTokenMLP(embed_dim, scalar_hidden_dim),
            "v": ScalarTokenMLP(embed_dim, scalar_hidden_dim),
            "Q": ScalarTokenMLP(embed_dim, scalar_hidden_dim),
        })

        physics_dim = 6 if self.use_physics_features else 0
        if self.use_physics_features:
            self.physics_projector = nn.Sequential(
                nn.Linear(physics_dim, embed_dim),
                nn.SiLU(),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
            )
            physics_context_dim = embed_dim
        else:
            self.physics_projector = None
            physics_context_dim = 0

        if self.short_encoder_type in ("ccgan_y2h", "ccgan", "y2h"):
            self.short_encoder = CCGANShortNumericEncoder(
                input_dim=numeric_dim,
                dim_embed=short_encoder_dim,
                num_groups=num_groups,
            )
            self.short_projector = nn.Sequential(
                nn.Linear(short_encoder_dim, embed_dim),
                nn.SiLU(),
                nn.LayerNorm(embed_dim),
            )
            if short_encoder_ckpt not in (None, "", "None", "none", "null"):
                self.load_short_encoder(short_encoder_ckpt)
            if freeze_short_encoder:
                self.short_encoder.eval()
                for param in self.short_encoder.parameters():
                    param.requires_grad = False
            short_context_dim = embed_dim
        elif self.short_encoder_type in ("none", "false", "no"):
            self.short_encoder = None
            self.short_projector = None
            short_context_dim = 0
        else:
            raise ValueError(f"Unknown short_encoder_type: {short_encoder_type}")

        self.base_context_dim = 4 * embed_dim + physics_context_dim + short_context_dim
        self.router = nn.Sequential(
            nn.Linear(self.base_context_dim, expert_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(expert_hidden_dim),
            nn.Linear(expert_hidden_dim, self.num_experts),
        )
        self.experts = nn.ModuleList([
            PhysicalExpert(
                input_dim=self.base_context_dim,
                output_dim=self.num_physical_tokens * embed_dim,
                hidden_dim=expert_hidden_dim,
            )
            for _ in range(self.num_experts)
        ])

    def load_short_encoder(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"CcGAN short encoder checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = checkpoint.get("net_state_dict", checkpoint)
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            cleaned[key] = value
        missing, unexpected = self.short_encoder.load_state_dict(cleaned, strict=False)
        print(
            f"Loaded CcGAN short encoder from {ckpt_path}; "
            f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
        )

    def _prepare_params(self, numeric):
        if not torch.is_tensor(numeric):
            numeric = torch.as_tensor(numeric)
        raw = numeric.float().view(numeric.shape[0], 4)
        denom = torch.clamp(self.parameter_normalizer.to(raw.device, raw.dtype), min=1e-6)
        normalized = raw / denom
        encoded = normalized if self.normalize_parameters else raw

        # raw/encoded input order is [V, I, v, Q]; token order is [I, V, v, Q].
        raw_v = raw[:, 0]
        raw_i = raw[:, 1]
        raw_speed = raw[:, 2]
        raw_q = raw[:, 3]
        enc_v = encoded[:, 0]
        enc_i = encoded[:, 1]
        enc_speed = encoded[:, 2]
        enc_q = encoded[:, 3]
        return raw, normalized, encoded, (enc_i, enc_v, enc_speed, enc_q), (raw_i, raw_v, raw_speed, raw_q)

    def _physics_features(self, raw_tuple):
        raw_i, raw_v, raw_speed, raw_q = raw_tuple
        power = raw_i * raw_v
        heat_input = power / torch.clamp(raw_speed, min=1e-6)
        features = torch.stack([raw_i, raw_v, raw_speed, raw_q, power, heat_input], dim=1)
        return features

    def forward(self, numeric):
        raw, normalized, encoded, encoded_tuple, raw_tuple = self._prepare_params(numeric)
        enc_i, enc_v, enc_speed, enc_q = encoded_tuple

        e_i = self.scalar_mlps["I"](enc_i)
        e_v = self.scalar_mlps["V"](enc_v)
        e_speed = self.scalar_mlps["v"](enc_speed)
        e_q = self.scalar_mlps["Q"](enc_q)
        decoupled_tokens = torch.stack([e_i, e_v, e_speed, e_q], dim=1)

        contexts = [decoupled_tokens.flatten(1)]
        if self.use_physics_features:
            physics_context = self.physics_projector(self._physics_features(raw_tuple).to(raw.device, raw.dtype))
            contexts.append(physics_context)
        if self.short_encoder is not None:
            short_input = normalized if self.short_encoder_uses_normalized else encoded
            short_context = self.short_projector(self.short_encoder(short_input))
            contexts.append(short_context)

        context = torch.cat(contexts, dim=1)
        expert_weights = torch.softmax(self.router(context), dim=1)
        expert_outputs = torch.stack([expert(context) for expert in self.experts], dim=1)
        fused = torch.sum(expert_outputs * expert_weights[:, :, None], dim=1)
        physical_tokens = fused.view(-1, self.num_physical_tokens, self.embed_dim)

        if self.include_decoupled_tokens:
            physical_tokens = torch.cat([decoupled_tokens, physical_tokens], dim=1)
        return physical_tokens


class DecoupledDynamicPhysicalCLIPEmbedder(AbstractEncoder):
    """Frozen CLIP text tokens fused with decoupled dynamic physical tokens."""
    def __init__(
            self,
            version="openai/clip-vit-base-patch32",
            device="cuda",
            max_length=77,
            freeze_text=True,
            local_files_only=False,
            numeric_key="numeric_params",
            text_key="caption",
            numeric_dim=4,
            embed_dim=512,
            scalar_hidden_dim=256,
            expert_hidden_dim=1024,
            num_experts=4,
            num_physical_tokens=4,
            parameter_normalizer=(22.0, 350.0, 75.0, 40.0),
            normalize_parameters=False,
            include_decoupled_tokens=False,
            use_physics_features=True,
            short_encoder_type="none",
            short_encoder_ckpt=None,
            short_encoder_dim=128,
            short_encoder_uses_normalized=True,
            freeze_short_encoder=True,
            num_groups=8,
    ):
        super().__init__()
        self.text_encoder = FrozenCLIPEmbedder(
            version=version,
            device=device,
            max_length=max_length,
            freeze=freeze_text,
            local_files_only=local_files_only,
        )
        self.device = device
        self.numeric_key = numeric_key
        self.text_key = text_key
        self.physical_encoder = DecoupledDynamicPhysicalEncoder(
            numeric_dim=numeric_dim,
            embed_dim=embed_dim,
            scalar_hidden_dim=scalar_hidden_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            num_physical_tokens=num_physical_tokens,
            parameter_normalizer=parameter_normalizer,
            normalize_parameters=normalize_parameters,
            include_decoupled_tokens=include_decoupled_tokens,
            use_physics_features=use_physics_features,
            short_encoder_type=short_encoder_type,
            short_encoder_ckpt=short_encoder_ckpt,
            short_encoder_dim=short_encoder_dim,
            short_encoder_uses_normalized=short_encoder_uses_normalized,
            freeze_short_encoder=freeze_short_encoder,
            num_groups=num_groups,
        )

    def forward(self, batch):
        if isinstance(batch, dict):
            text = batch[self.text_key]
            numeric = batch[self.numeric_key]
        else:
            text = batch
            numeric = None

        if isinstance(text, tuple):
            text = list(text)
        if isinstance(text, list):
            text = [str(item) for item in text]
        elif not isinstance(text, str):
            text = str(text)

        text_tokens = self.text_encoder(text)
        if numeric is None:
            return text_tokens

        if not torch.is_tensor(numeric):
            numeric = torch.as_tensor(numeric)
        numeric = numeric.to(device=text_tokens.device, dtype=text_tokens.dtype)
        physical_tokens = self.physical_encoder(numeric).to(device=text_tokens.device, dtype=text_tokens.dtype)
        return torch.cat([text_tokens, physical_tokens], dim=1)

    def encode(self, batch):
        return self(batch)


class FrozenCLIPNumericEmbedder(AbstractEncoder):
    """Frozen CLIP text tokens plus trainable numeric-parameter tokens."""
    def __init__(
            self,
            version="openai/clip-vit-base-patch32",
            device="cuda",
            max_length=77,
            freeze_text=True,
            local_files_only=False,
            numeric_key="numeric_params",
            text_key="caption",
            numeric_dim=4,
            embed_dim=512,
            hidden_dim=512,
            numeric_embed_dim=128,
            numeric_encoder_type="ccgan_y2h",
            numeric_encoder_ckpt=None,
            freeze_numeric_encoder=False,
            num_numeric_tokens=1,
            num_groups=8,
    ):
        super().__init__()
        self.text_encoder = FrozenCLIPEmbedder(
            version=version,
            device=device,
            max_length=max_length,
            freeze=freeze_text,
            local_files_only=local_files_only,
        )
        self.device = device
        self.numeric_key = numeric_key
        self.text_key = text_key
        self.num_numeric_tokens = int(num_numeric_tokens)
        self.numeric_encoder_type = numeric_encoder_type
        if numeric_encoder_type == "ccgan_y2h":
            self.numeric_encoder = CCGANShortNumericEncoder(
                input_dim=numeric_dim,
                dim_embed=numeric_embed_dim,
                num_groups=num_groups,
            )
            self.numeric_projector = nn.Sequential(
                nn.Linear(numeric_embed_dim, embed_dim * self.num_numeric_tokens),
                nn.ReLU(),
            )
        elif numeric_encoder_type == "mlp":
            self.numeric_encoder = NumericConditionMLP(
                input_dim=numeric_dim,
                embed_dim=embed_dim * self.num_numeric_tokens,
                hidden_dim=hidden_dim,
                num_groups=num_groups,
            )
            self.numeric_projector = nn.Identity()
        else:
            raise ValueError(f"Unknown numeric_encoder_type: {numeric_encoder_type}")

        if numeric_encoder_ckpt not in (None, "", "None", "none", "null"):
            self.load_numeric_encoder(numeric_encoder_ckpt)

        if freeze_numeric_encoder:
            self.numeric_encoder.eval()
            for param in self.numeric_encoder.parameters():
                param.requires_grad = False

    def load_numeric_encoder(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"CcGAN numeric encoder checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = checkpoint.get("net_state_dict", checkpoint)
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            cleaned[key] = value
        missing, unexpected = self.numeric_encoder.load_state_dict(cleaned, strict=False)
        print(
            f"Loaded CcGAN numeric encoder from {ckpt_path}; "
            f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
        )

    def forward(self, batch):
        if isinstance(batch, dict):
            text = batch[self.text_key]
            numeric = batch[self.numeric_key]
        else:
            text = batch
            numeric = None

        text_tokens = self.text_encoder(text)
        if numeric is None:
            return text_tokens

        if not torch.is_tensor(numeric):
            numeric = torch.as_tensor(numeric)
        numeric = numeric.to(device=text_tokens.device, dtype=text_tokens.dtype)
        numeric_tokens = self.numeric_projector(self.numeric_encoder(numeric))
        numeric_tokens = numeric_tokens.view(numeric_tokens.shape[0], self.num_numeric_tokens, text_tokens.shape[-1])
        return torch.cat([text_tokens, numeric_tokens], dim=1)

    def encode(self, batch):
        return self(batch)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        clip = load_openai_clip()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.clip = clip
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = self.clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        clip = load_openai_clip()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)
