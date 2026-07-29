from __future__ import annotations

import torch

SM120: tuple[int, int] = (12, 0)
SM89: tuple[int, int] = (8, 9)


def _inventory() -> str:
    if not torch.cuda.is_available():
        return "  (no CUDA devices visible)"
    lines: list[str] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        major, minor = torch.cuda.get_device_capability(i)
        lines.append(
            f"  cuda:{i}  {props.name}  sm_{major}{minor}  "
            f"{props.total_memory / 1024**3:.2f} GiB"
        )
    return "\n".join(lines)


def get_device(min_capability: tuple[int, int] = SM120) -> torch.device:
    """Return the first CUDA device meeting min_capability, else raise.

    Never address a device by a hardcoded index. Enumeration order on this
    machine changed once already (setting CUDA_DEVICE_ORDER=PCI_BUS_ID moved the
    5090 from cuda:1 to cuda:0) and can change again on a driver update or slot
    change. See EXP-001.
    """
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(i) >= min_capability:
            return torch.device(f"cuda:{i}")
    raise RuntimeError(
        f"No CUDA device with capability >= {min_capability}. Visible devices:\n"
        f"{_inventory()}"
    )


def require_cuda(min_capability: tuple[int, int] = SM120) -> torch.device:
    """Hard guard for every GPU-requiring entry point.

    The `base` conda env carries a CPU-only torch build, so code run outside
    `retention` would otherwise execute silently on CPU: correct but 100x slow in
    Phase 0, and an invisible confound in Phase 1. Crash instead.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Likely running in `base` (CPU-only torch) "
            "instead of the `retention` env. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}"
        )
    return get_device(min_capability)


def describe_device(device: torch.device) -> dict[str, object]:
    """Device facts for the run manifest."""
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device!r}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    major, minor = torch.cuda.get_device_capability(index)
    return {
        "index": index,
        "name": props.name,
        "capability": f"{major}.{minor}",
        "total_memory_bytes": props.total_memory,
        "multi_processor_count": props.multi_processor_count,
    }
