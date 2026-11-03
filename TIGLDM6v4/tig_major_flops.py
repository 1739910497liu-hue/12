"""Count dense convolution/matrix FLOPs without optional profiling packages.

This is a deliberately restricted arithmetic count, not a count of all machine
instructions. One multiply-accumulate is counted as two FLOPs. Bias additions,
normalization, activation, softmax, indexing and elementwise operations are
excluded and enumerated in the report. Use dense/native attention for profiling;
unrecognized fused attention, matrix and convolution operations are rejected.

The caller chooses device, precision, eval mode and torch.no_grad(). In
particular, this module does not change attention backends or model weights.
"""
from collections import Counter
from functools import reduce
from operator import mul

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def prod(values):
    return reduce(mul, values, 1)


class UnsupportedMajorOperation(RuntimeError):
    """An operation could hide arithmetic that this counter cannot measure."""


def _argument(func, args, kwargs, name, fallback_index=None, default=None):
    """Read dispatcher arguments without relying on one PyTorch call style."""
    if name in kwargs:
        return kwargs[name]
    schema = getattr(func, "_schema", None)
    if schema is not None:
        for index, argument in enumerate(schema.arguments):
            if argument.name == name:
                return args[index] if index < len(args) else argument.default_value
        if fallback_index is not None and fallback_index < len(schema.arguments):
            argument = schema.arguments[fallback_index]
            if argument.name in kwargs:
                return kwargs[argument.name]
    if fallback_index is not None and fallback_index < len(args):
        return args[fallback_index]
    return default


def _name(func):
    schema = getattr(func, "_schema", None)
    return schema.name if schema is not None else str(func)


def _validate_matrix(left, right, batch=False):
    dimensions = 3 if batch else 2
    if left.ndim != dimensions or right.ndim != dimensions:
        raise ValueError("Unexpected matrix dimensions in FLOPs profiling.")
    if left.shape[-1] != right.shape[-2]:
        raise ValueError("Incompatible matrix dimensions in FLOPs profiling.")
    if batch and left.shape[0] != right.shape[0]:
        raise ValueError("Batched matrix dimensions disagree.")


class MajorFlopCounter(TorchDispatchMode):
    """Observe the primitive operations executed during a function call.

    Composite Linear/matmul/einsum operations normally become mm/addmm/bmm in
    TorchDispatchMode. Direct linear/matmul kernels are supported as well, but
    the counter never counts both a composite and its decomposition.
    """

    _COUNTED = {
        "aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm",
        "aten::linear", "aten::matmul", "aten::mv", "aten::dot",
        "aten::convolution", "aten::_convolution",
        "aten::conv1d", "aten::conv2d", "aten::conv3d",
    }
    _MAJOR_MARKERS = (
        "matmul", "addmm", "baddbmm", "einsum", "linear",
        "attention", "flash", "gemm", "scaled_mm", "multi_dot",
        "tensordot", "addmv", "addr", "outer", "bilinear",
    )

    def __init__(self):
        super().__init__()
        self.by_operator = Counter()
        self.operator_calls = Counter()
        self.excluded_operator_calls = Counter()

    @classmethod
    def _reject_unrecognized(cls, name):
        namespace, separator, operation = name.partition("::")
        # A custom kernel can contain an entire attention/convolution module
        # under an opaque name. Do not silently accept any custom namespace.
        if not separator or namespace not in {"aten", "prim", "prims"}:
            return True
        lower = operation.lower()
        # Type conversions and bilinear image resizing are not matrix kernels.
        if lower.startswith(("convert_element_type", "upsample_")):
            return False
        if "convolution" in lower or lower.startswith(("conv", "_conv")):
            return True
        if any(marker in lower for marker in cls._MAJOR_MARKERS):
            return True
        # Cover specialized/mutating matrix families, including sparse kernels.
        return lower in {"mm", "bmm", "mv", "dot", "vdot"} or any(
            marker in lower for marker in ("_mm", "mm_", "_bmm", "bmm_", "_mv", "mv_")
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = _name(func)
        if name not in self._COUNTED and self._reject_unrecognized(name):
            raise UnsupportedMajorOperation(
                "Cannot count major arithmetic in {}. Use a dense/native math "
                "attention backend for FLOPs profiling or add an explicit, "
                "tested formula; partial FLOPs will not be reported.".format(name)
            )
        output = func(*args, **kwargs)
        if name not in self._COUNTED:
            self.excluded_operator_calls[name] += 1
            return output
        value = self._flops(name, func, args, kwargs, output)
        self.by_operator[name] += int(value)
        self.operator_calls[name] += 1
        return output

    @staticmethod
    def _flops(name, func, args, kwargs, output):
        def argument(key, index=None, default=None):
            return _argument(func, args, kwargs, key, index, default)

        if name in {"aten::mm", "aten::bmm"}:
            left, right = argument("self", 0), argument("mat2", 1)
            _validate_matrix(left, right, batch=(name == "aten::bmm"))
            return 2 * output.numel() * left.shape[-1]
        if name in {"aten::addmm", "aten::baddbmm"}:
            left = argument("mat1" if name == "aten::addmm" else "batch1", 1)
            right = argument("mat2" if name == "aten::addmm" else "batch2", 2)
            _validate_matrix(left, right, batch=(name == "aten::baddbmm"))
            # Bias and alpha/beta scaling are outside the declared scope.
            return 2 * output.numel() * left.shape[-1]
        if name == "aten::linear":
            value, weight = argument("input", 0), argument("weight", 1)
            if weight.ndim != 2 or value.shape[-1] != weight.shape[1]:
                raise ValueError("Unexpected Linear shapes in FLOPs profiling.")
            return 2 * output.numel() * weight.shape[1]
        if name in {"aten::matmul", "aten::mv", "aten::dot"}:
            left = argument("self", 0)
            right = argument("other" if name != "aten::mv" else "vec", 1)
            if left.ndim < 1 or right.ndim < 1:
                raise ValueError("Unexpected matmul shapes in FLOPs profiling.")
            contraction = right.shape[-2] if right.ndim >= 2 else right.shape[-1]
            if left.shape[-1] != contraction:
                raise ValueError("Incompatible matmul shapes in FLOPs profiling.")
            return 2 * output.numel() * contraction
        value, weight = argument("input", 0), argument("weight", 1)
        if value.ndim < 3 or weight.ndim != value.ndim or output.ndim != value.ndim:
            raise ValueError("Unexpected convolution shapes in FLOPs profiling.")
        if name in {"aten::convolution", "aten::_convolution"}:
            transposed = bool(argument("transposed", 6, False))
            groups = int(argument("groups", 8, 1))
        else:
            transposed = False
            groups = int(argument("groups", 6, 1))
        if groups < 1 or value.shape[1] % groups or output.shape[1] % groups:
            raise ValueError("Invalid grouped-convolution shapes.")
        if transposed:
            if weight.shape[0] != value.shape[1] or weight.shape[1] * groups != output.shape[1]:
                raise ValueError("Transposed convolution weight/channel mismatch.")
        elif weight.shape[0] != output.shape[1] or weight.shape[1] * groups != value.shape[1]:
            raise ValueError("Convolution weight/channel mismatch.")
        # The weight already contains Cin/groups (or Cout/groups for transpose):
        # dividing by groups again would undercount. This dense convention counts
        # the full kernel including padded spatial locations.
        spatial = value.shape[2:] if transposed else output.shape[2:]
        return 2 * int(value.shape[0]) * int(weight.numel()) * prod(spatial)

    def report(self):
        total = int(sum(self.by_operator.values()))
        return {
            "format": "tig_major_flops_v1",
            "backend": "torch.utils._python_dispatch.TorchDispatchMode",
            "torch_version": torch.__version__,
            "flops": total,
            "macs": total // 2,
            "gflops": total / 1e9,
            "gmacs": total / 2e9,
            "multiply_accumulate_flops": 2,
            "by_operator": dict(sorted(self.by_operator.items())),
            "operator_calls": dict(sorted(self.operator_calls.items())),
            "excluded_operator_calls": dict(sorted(self.excluded_operator_calls.items())),
            "scope": "Dense convolution and matrix products, including attention QK and probability-V products; full padded kernels; FMA=2. Excludes bias/scaling additions, normalization, softmax, activations, elementwise arithmetic, indexing and memory operations. MACs=FLOPs/2 within this restricted scope.",
        }


def profile_major_flops(function):
    """Run a zero-argument callable once and return (output, JSON-safe report)."""
    counter = MajorFlopCounter()
    with counter:
        output = function()
    return output, counter.report()
