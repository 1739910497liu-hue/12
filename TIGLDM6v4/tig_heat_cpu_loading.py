"""Strict, low-peak-memory CPU loading for TIG HeatVQ and HeatDiffusion.

PyTorch >= 2.1 can construct on meta and assign checkpoint tensors directly.
Pass a CPU state dictionary loaded with mmap=True when the checkpoint format
supports it. This helper does not load checkpoint files or change their data.
Older PyTorch uses explicitly reported ordinary construction and strict copy.
"""
from collections.abc import Mapping
import contextlib
import inspect
from pathlib import Path
import re

import torch
from torch import nn


def supports_meta_assignment():
    version = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    return bool(version and tuple(map(int, version.groups())) >= (2, 1)
                and "assign" in inspect.signature(nn.Module.load_state_dict).parameters)


def _validate_state(model, state_dict):
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Expected a nonempty model state dictionary.")
    if any(not isinstance(key, str) for key in state_dict):
        raise ValueError("All state-dictionary keys must be strings.")
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(state_dict)), sorted(set(state_dict) - set(expected))
    if missing or unexpected:
        raise ValueError("Strict checkpoint key mismatch: missing={}, unexpected={}".format(
            missing[:12], unexpected[:12]))
    errors = []
    for key, value in state_dict.items():
        reference = expected[key]
        if not torch.is_tensor(value) or not torch.is_tensor(reference):
            errors.append("non-tensor entry " + key)
        elif value.device.type != "cpu":
            errors.append("checkpoint entry is not on CPU: " + key)
        elif value.is_complex():
            errors.append("complex checkpoint entry cannot be normalized to FP32: " + key)
        elif tuple(value.shape) != tuple(reference.shape):
            errors.append("shape mismatch {}: expected {}, found {}".format(
                key, tuple(reference.shape), tuple(value.shape)))
        elif reference.is_floating_point() and not value.is_floating_point():
            errors.append("floating model entry has non-floating checkpoint dtype: " + key)
        elif not reference.is_floating_point() and value.dtype != reference.dtype:
            errors.append("non-floating buffer dtype mismatch: " + key)
    if errors:
        raise ValueError("Strict checkpoint validation failed: " + "; ".join(errors[:12]))


def _repair_clip_position_ids(model):
    repaired = []
    for module_name, module in model.named_modules():
        for name, value in list(module._buffers.items()):
            if value is None or value.device.type != "meta":
                continue
            known_clip = (name == "position_ids"
                          and module.__class__.__name__ == "CLIPTextEmbeddings"
                          and module.__class__.__module__.startswith("transformers.models.clip.")
                          and name in module._non_persistent_buffers_set)
            embedding = getattr(module, "position_embedding", None)
            if not known_clip or not isinstance(embedding, nn.Embedding):
                continue
            if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] != embedding.num_embeddings or value.dtype != torch.long:
                raise ValueError("Unexpected CLIP position_ids shape/dtype: " + module_name)
            module._buffers[name] = torch.arange(
                embedding.num_embeddings, dtype=torch.long, device="cpu").unsqueeze(0)
            repaired.append((module_name + "." if module_name else "") + name)
    return repaired


def instantiate_and_load(factory, state_dict, allow_meta=True):
    """Return (CPU FP32 model, loading_report), with strict validation.

    factory is a zero-argument callable. Unsupported PyTorch versions select
    normal construction up front; errors during meta construction/loading are
    never hidden by retrying with a different, larger-memory strategy.
    """
    use_meta = bool(allow_meta and supports_meta_assignment())
    mode = torch.device("meta") if use_meta else contextlib.nullcontext()
    with mode:
        model = factory()
    if not isinstance(model, nn.Module):
        raise TypeError("factory must return torch.nn.Module.")
    _validate_state(model, state_dict)
    requires_grad = {name: value.requires_grad for name, value in model.named_parameters()}
    source_dtypes = sorted({str(value.dtype) for value in state_dict.values()})
    source_cast_needed = any(value.is_floating_point() and value.dtype != torch.float32
                             for value in state_dict.values())
    # Assigning an nn.Parameter instance directly can make subsequent .float()
    # rebind its data in the supplied mapping; plain detached tensors avoid that.
    parameter_values = any(isinstance(value, nn.Parameter) for value in state_dict.values())
    supplied = state_dict
    if parameter_values:
        supplied = state_dict.copy()
        for key, value in supplied.items():
            if isinstance(value, nn.Parameter):
                supplied[key] = value.detach()
        if hasattr(state_dict, "_metadata") and hasattr(supplied, "__dict__"):
            supplied._metadata = state_dict._metadata
    if use_meta:
        model.load_state_dict(supplied, strict=True, assign=True)
    else:
        model.load_state_dict(supplied, strict=True)
    # PyTorch 2.1's assign=True wraps ordinary tensors with Parameter(tensor)
    # and defaults requires_grad to True. Restore the architecture's flags
    # explicitly rather than assuming the behavior of newer PyTorch releases.
    assigned_parameters = dict(model.named_parameters())
    if set(assigned_parameters) != set(requires_grad):
        raise ValueError("Loading changed the parameter registry or parameter sharing.")
    for name, parameter in assigned_parameters.items():
        parameter.requires_grad_(requires_grad[name])
    repaired = _repair_clip_position_ids(model)
    remaining_meta = ["parameter:" + name for name, value in model.named_parameters() if value.device.type == "meta"]
    remaining_meta += ["buffer:" + name for name, value in model.named_buffers() if value.device.type == "meta"]
    if remaining_meta:
        raise ValueError("Uninitialized meta tensors remain; refusing incomplete model: " + ", ".join(remaining_meta[:12]))
    non_cpu = [name for name, value in list(model.named_parameters()) + list(model.named_buffers()) if value.device.type != "cpu"]
    if non_cpu:
        raise ValueError("CPU loader encountered non-CPU tensors: " + ", ".join(non_cpu[:12]))
    cast_needed = any(value.is_floating_point() and value.dtype != torch.float32
                      for value in list(model.parameters()) + list(model.buffers()))
    model.float()
    after_grad = {name: value.requires_grad for name, value in model.named_parameters()}
    if after_grad != requires_grad:
        raise ValueError("Loading changed parameter trainability; refusing model.")
    invalid_float = [name for name, value in list(model.named_parameters()) + list(model.named_buffers())
                     if value.is_floating_point() and value.dtype != torch.float32]
    if invalid_float:
        raise ValueError("CPU model was not normalized to FP32: " + ", ".join(invalid_float[:12]))
    return model, {
        "strategy": "meta_assign" if use_meta else "normal_strict_copy",
        "meta_requested": bool(allow_meta), "meta_supported": supports_meta_assignment(),
        "strict": True, "loaded_entries": len(state_dict),
        "source_dtypes": source_dtypes, "model_floating_dtype": "torch.float32",
        "cast_to_fp32": bool(cast_needed or source_cast_needed),
        "repaired_nonpersistent_buffers": repaired,
        "memory_note": ("Checkpoint tensor storage is assigned directly; CPU mmap tensors are reused where dtype is FP32. "
                        "Floating dtype conversion allocates new FP32 storage." if use_meta else
                        "This PyTorch/configuration uses ordinary allocation plus strict copy; checkpoint and model storage coexist during loading."),
    }


@contextlib.contextmanager
def clip_architecture_only(clip_path):
    """Read only local CLIP config; complete text weights must come from checkpoint.

    The temporary classmethod patch is intended for isolated loading workers and
    is restored even if construction fails. Tokenizer loading is unchanged.
    """
    root = Path(clip_path).expanduser().resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError("Missing local CLIP config: " + str(root / "config.json"))
    from transformers import CLIPTextConfig, CLIPTextModel
    absent = object()
    original_descriptor = CLIPTextModel.__dict__.get("from_pretrained", absent)

    @classmethod
    def architecture(cls, path, *args, **kwargs):
        if args or set(kwargs) != {"local_files_only"} or kwargs["local_files_only"] is not True:
            raise ValueError("Architecture-only CLIP loading expects local_files_only=True and no weight-loading options.")
        if Path(path).expanduser().resolve() != root:
            raise ValueError("Unexpected CLIP path in architecture-only model construction.")
        config = CLIPTextConfig.from_pretrained(str(root), local_files_only=True)
        return cls(config)

    CLIPTextModel.from_pretrained = architecture
    try:
        yield
    finally:
        if original_descriptor is absent:
            delattr(CLIPTextModel, "from_pretrained")
        else:
            CLIPTextModel.from_pretrained = original_descriptor


def load_heat_model(kind, config, modelstate, clip_path=None, allow_meta=True):
    """Build a HeatVQ or HeatDiffusion on CPU and strictly load its model state.

    kind accepts 1/'stage1'/'vq' or 2/'stage2'/'diffusion'. The caller must have
    added the intended project root to sys.path before calling this function.
    """
    from ldm.heat_source.models import HeatVQ, HeatDiffusion
    if kind in (1, "stage1", "vq"):
        model, report = instantiate_and_load(lambda: HeatVQ(config), modelstate, allow_meta)
        report["kind"] = "stage1"
        report["clip_weights_source"] = None
        return model, report
    if kind not in (2, "stage2", "diffusion"):
        raise ValueError("Unknown heat model kind: " + str(kind))
    uses_clip = config.get("condition") in ("text", "both")
    if uses_clip and not clip_path:
        raise ValueError("Text-conditioned HeatDiffusion requires a local CLIP path.")
    context = clip_architecture_only(clip_path) if uses_clip else contextlib.nullcontext()
    with context:
        model, report = instantiate_and_load(
            lambda: HeatDiffusion(config, str(clip_path) if clip_path else None), modelstate, allow_meta)
    report["kind"] = "stage2"
    report["clip_weights_source"] = "strict training checkpoint; local config/tokenizer only" if uses_clip else None
    return model, report
