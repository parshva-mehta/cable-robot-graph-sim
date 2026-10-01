import torch

from utilities.misc_utils import DEFAULT_DTYPE


def zeros(shape, dtype=None, device=None, ref_tensor=None):
    if ref_tensor is None and (dtype is None or device is None):
        raise Exception("Need to specify either ref tensor or (dtype and device)")

    if ref_tensor is not None:
        dtype = ref_tensor.dtype
        device = ref_tensor.device

    return torch.zeros(shape, dtype=dtype, device=device)


def ones(shape, dtype=None, device=None, ref_tensor=None):
    if ref_tensor is None and (dtype is None or device is None):
        raise Exception("Need to specify either ref tensor or (dtype and device)")

    if ref_tensor is not None:
        dtype = ref_tensor.dtype
        device = ref_tensor.device

    return torch.ones(shape, dtype=dtype, device=device)


def tensorify(non_tensor, dtype=None, reshape=None):
    dtype = dtype if dtype else DEFAULT_DTYPE
    ten = torch.tensor(non_tensor, dtype=dtype)
    reshape = reshape if reshape else ten.shape
    return ten.reshape(reshape)


def safe_norm(tensor, dim=1):
    return tensor / torch.clamp_min(tensor.norm(dim=dim, keepdim=True), 1e-5)


def move_stray_tensors(module: torch.nn.Module, device) -> int:
    """Move plain tensor attributes that ``nn.Module.to()`` leaves behind.

    ``nn.Module.to()`` only relocates registered parameters and buffers.  Much of
    this codebase stores tensors as ordinary attributes instead (robot pose
    state, cached masks and edge templates, LSTM/control history, message-passing
    scratch buffers).  Each class carries a hand-written ``to()`` listing its own
    attributes, so any tensor attribute added later silently stays on the source
    device — harmless on CPU-only runs, but a hard error on MPS ("Passed CPU
    tensor to MPS op") and a silent host/device copy on CUDA.

    This sweeps ``module`` and all submodules and relocates anything still on the
    wrong device, so the explicit ``to()`` chain no longer has to be exhaustive.

    Args:
        module: Root module to sweep.
        device: Target device.

    Returns:
        Number of tensors moved.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    moved = 0
    for _, mod in module.named_modules():
        registered = set(dict(mod.named_buffers(recurse=False))) \
                     | set(dict(mod.named_parameters(recurse=False)))
        for attr, val in list(vars(mod).items()):
            if not isinstance(val, torch.Tensor) or attr in registered:
                continue
            if val.device.type == device.type:
                continue
            setattr(mod, attr, val.to(device))
            moved += 1
    return moved
