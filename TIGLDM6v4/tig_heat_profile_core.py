"""Component arithmetic accounting for loaded TIG HeatVQ/HeatDiffusion models.

No checkpoints, model packages or datasets are imported here. The caller builds
and strictly loads models and supplies an already prepared, batch-one condition.
First-stage-only checkpoints receive parameter counts, not DDIM FLOPs.
"""
import contextlib
import math

import torch

from tig_major_flops import profile_major_flops


def _submodule(model, path):
    for component in path.split("."):
        if model is None:
            return None
        model = getattr(model, component, None)
    return model


def _unique_parameters(modules):
    values = {}
    for module in modules:
        if module is not None:
            values.update((id(value), value) for value in module.parameters())
    return list(values.values())


def heat_parameter_report(vq, diffusion=None):
    parameters = _unique_parameters((vq, diffusion))
    total = sum(value.numel() for value in parameters)
    trainable = sum(value.numel() for value in parameters if value.requires_grad)
    groups = {
        "vq": (vq,),
        "vq_encoder": (_submodule(vq, "encoder"),),
        "vq_quant_conv": (_submodule(vq, "quant_conv"),),
        "vq_codebook": (_submodule(vq, "quantize"),),
        "vq_post_quant_conv": (_submodule(vq, "post_quant_conv"),),
        "vq_decoder": (_submodule(vq, "decoder"),),
        "diffusion": (diffusion,),
        "unet": (_submodule(diffusion, "unet"),),
        "diffusion_physical": (_submodule(diffusion, "physical"),),
        "conditioner": (_submodule(diffusion, "conditioner"),),
        "clip": (_submodule(diffusion, "conditioner.clip"),),
        "numeric_encoders": (_submodule(diffusion, "conditioner.numeric"),),
        "denoiser": (_submodule(diffusion, "unet"), _submodule(diffusion, "physical")),
    }
    result = {
        "total_parameters": total, "total_parameters_M": total / 1e6,
        "trainable_parameters": trainable, "trainable_parameters_M": trainable / 1e6,
        "frozen_parameters": total - trainable,
    }
    for name, modules in groups.items():
        count = sum(value.numel() for value in _unique_parameters(modules))
        result[name + "_parameters"] = count
        result[name + "_parameters_M"] = count / 1e6
    result["scope"] = (
        "Unique nn.Parameters in the loaded VQ and diffusion modules; excludes buffers, "
        "optimizer state, training discriminator and LPIPS. Total includes the resident "
        "VQ encoder and inactive registered branches. Trainable means current requires_grad, "
        "not historical training metadata. Subgroup counts overlap and must not be summed; "
        "denoiser means registered UNet plus diffusion physical projection."
    )
    return result


def _math_attention():
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False)
        return contextlib.nullcontext()


@contextlib.contextmanager
def _profiling_mode(vq, diffusion, device, precision):
    modules = {id(module): module for root in (vq, diffusion) if root is not None
               for module in root.modules()}
    modes = [(module, module.training) for module in modules.values()]
    configurations = {}
    clip = _submodule(diffusion, "conditioner.clip")
    if clip is not None:
        for module in clip.modules():
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "output_attentions"):
                configurations[id(config)] = (config, config.output_attentions)
    try:
        for root in (vq, diffusion):
            if root is not None:
                root.eval()
        # Pinned Transformers CLIPSdpaAttention falls back to the original
        # eager attention when output_attentions=True. The weights are unchanged.
        for config, _ in configurations.values():
            config.output_attentions = True
        amp = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
               if precision == "bf16" else contextlib.nullcontext())
        with torch.no_grad(), amp, _math_attention():
            yield
    finally:
        for config, previous in configurations.values():
            config.output_attentions = previous
        for module, previous in modes:
            module.training = previous


def _finite(value, name):
    if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Non-finite or invalid {} during complexity profiling.".format(name))


def count_heat_models(vq, diffusion, config, template=None, numeric=None,
                      captions=None, steps=50, latent_scale=1.0, precision="fp32"):
    """Return parameter/component FLOP reports; never train or sample all steps.

    Models and condition tensors must already be on the same device. FLOPs are
    for one image, including the nearest-codebook matrix search during decoding.
    This function does not count a VQ encoder pass during stage-two generation.
    Set diffusion=None for first-stage-only parameter accounting.
    """
    if vq is None:
        raise ValueError("A loaded first-stage VQ model is required.")
    parameters = heat_parameter_report(vq, diffusion)
    summary = {
        "Params_total_M": parameters["total_parameters_M"],
        "Params_trainable_M": parameters["trainable_parameters_M"],
        "Params_denoiser_M": parameters["denoiser_parameters_M"] if diffusion is not None else None,
        "FLOPs_condition_encoder_G": None,
        "FLOPs_denoiser_single_step_G": None,
        "MACs_denoiser_single_step_G": None,
        "FLOPs_decode_with_requantization_G": None,
        "FLOPs_full_generation_estimate_G": None,
    }
    report = {"format": "tig_heat_complexity_core_v1", "parameters": parameters,
              "flops": None, "summary": summary, "status": "stage1_only"}
    if diffusion is None:
        report["reason"] = "Only a stage-one autoencoder is available; no diffusion-generation FLOPs are reported."
        return report
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16.")
    if type(steps) is not int or not 2 <= steps <= len(diffusion.alpha_bar):
        raise ValueError("DDIM steps must be in [2, len(alpha_bar)].")
    if isinstance(latent_scale, bool) or not math.isfinite(float(latent_scale)) or float(latent_scale) <= 0:
        raise ValueError("latent_scale must be finite and positive.")
    if not torch.is_tensor(template) or template.ndim != 4 or template.shape[0] != 1:
        raise ValueError("A BCHW, batch-one template is required.")
    if not torch.is_tensor(numeric) or tuple(numeric.shape) != (1, 3) or numeric.device != template.device:
        raise ValueError("numeric must have shape [1,3] on the template device.")
    if not isinstance(captions, (list, tuple)) or len(captions) != 1 or not isinstance(captions[0], str):
        raise ValueError("captions must contain one string.")
    size, factor, dim = config["image_size"], config["factor"], config["embed_dim"]
    if any(type(value) is not int or value <= 0 for value in (size, factor, dim)) or size % factor:
        raise ValueError("Invalid image_size/factor/embed_dim.")
    if tuple(template.shape[-2:]) != (size, size):
        raise ValueError("Template spatial dimensions differ from image_size.")
    device = template.device
    if any(value.device != device for value in _unique_parameters((vq, diffusion))):
        raise ValueError("All model parameters must be on the template device.")
    _finite(template, "template")
    _finite(numeric, "numeric conditions")
    shape = (1, dim, size // factor, size // factor)
    # Fixed local RNG leaves the caller's RNG state untouched; values affect
    # outputs but not the dense operation-count shapes in these models.
    latent = torch.randn(shape, device=device,
                         generator=torch.Generator(device=device).manual_seed(2026))
    with _profiling_mode(vq, diffusion, device, precision):
        context, condition = profile_major_flops(lambda: diffusion.conditioner(numeric, captions))
        _finite(context, "context")
        timestep = torch.full((1,), len(diffusion.alpha_bar) - 1, device=device, dtype=torch.long)
        prediction, denoiser = profile_major_flops(
            lambda: diffusion.predict(latent, timestep, template, context))
        _finite(prediction, "noise prediction")
        del prediction
        timestep.zero_()
        prediction, endpoint = profile_major_flops(
            lambda: diffusion.predict(latent, timestep, template, context))
        _finite(prediction, "endpoint noise prediction")
        del prediction
        if denoiser["flops"] != endpoint["flops"]:
            raise ValueError("Denoising FLOPs change with timestep; static-step estimate is invalid.")
        decoded, decoder = profile_major_flops(
            lambda: vq.decode(latent / float(latent_scale), template, requantize=True))
        _finite(decoded, "requantized decoded image")
        del decoded
    full = condition["flops"] + steps * denoiser["flops"] + decoder["flops"]
    report.update(status="stage2_profiled", batch_size=1, image_size=size,
                  latent_shape=list(shape), precision=precision, device=str(device))
    report["flops"] = {
        "condition_encoder": condition,
        "denoiser_single_step": denoiser,
        "decode_with_requantization": decoder,
        "endpoint_step_flops_equal": True, "steps": steps, "cfg_scale": 1.,
        "full_generation_major_flops_estimate": full,
        "formula": "condition_once + steps * predict_once + requantize_and_decode_once",
        "scope": "Convolution/matrix products only, FMA=2. Decoder includes codebook nearest-neighbor matrix products and physical decoder projections; distance reductions, comparisons, gates, norms, activations, softmax and sampler elementwise updates are excluded. VQ encoder is not executed for generation.",
    }
    summary.update(
        FLOPs_condition_encoder_G=condition["flops"] / 1e9,
        FLOPs_denoiser_single_step_G=denoiser["flops"] / 1e9,
        MACs_denoiser_single_step_G=denoiser["flops"] / 2e9,
        FLOPs_decode_with_requantization_G=decoder["flops"] / 1e9,
        FLOPs_full_generation_estimate_G=full / 1e9)
    return report
