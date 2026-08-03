from __future__ import annotations

import os

import torch

SM120: tuple[int, int] = (12, 0)
SM89: tuple[int, int] = (8, 9)
SM80: tuple[int, int] = (8, 0)

#: Environment override for the capability floor, e.g. `AR_MIN_CAPABILITY=8.0`.
#:
#: The sm_120 default is a property of the machine this project was developed on, not
#: of the science: the 5090 produces garbage under a pre-cu128 torch, so the floor is a
#: guard against that specific footgun. Reproducing on Ampere/Ada/Hopper is legitimate
#: and needs a lower floor. This is an explicit opt-in rather than a silent fallback --
#: the run still raises if no device clears whatever floor is in force, and the resolved
#: device and capability are recorded in every manifest.
CAPABILITY_ENV = "AR_MIN_CAPABILITY"


def _env_capability() -> tuple[int, int] | None:
    raw = os.environ.get(CAPABILITY_ENV)
    if not raw:
        return None
    try:
        major, _, minor = raw.strip().partition(".")
        return (int(major), int(minor or 0))
    except ValueError as exc:
        raise ValueError(
            f"{CAPABILITY_ENV}={raw!r} is not a capability like '8.0' or '12.0'."
        ) from exc


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
    """Return the largest-memory CUDA device meeting min_capability, else raise.

    Never address a device by a hardcoded index. Enumeration order on this
    machine changed once already (setting CUDA_DEVICE_ORDER=PCI_BUS_ID moved the
    5090 from cuda:1 to cuda:0) and can change again on a driver update or slot
    change. See EXP-001.

    Among qualifying devices the largest-memory one wins, so 8B BF16 loads land on the
    32 GB card without anything naming an index; ties break on the lower index so the
    choice stays deterministic and reproducible.

    `AR_MIN_CAPABILITY` overrides the floor (see CAPABILITY_ENV). It is an explicit
    opt-in for other hardware, never an automatic relaxation.
    """
    floor = _env_capability() or min_capability
    qualifying = [
        i for i in range(torch.cuda.device_count())
        if torch.cuda.get_device_capability(i) >= floor
    ]
    if not qualifying:
        raise RuntimeError(
            f"No CUDA device with capability >= {floor}. Visible devices:\n"
            f"{_inventory()}\n"
            f"If your GPU is older than sm_{floor[0]}{floor[1]} and you know your torch "
            f"build matches it, set {CAPABILITY_ENV} (e.g. {CAPABILITY_ENV}=8.0)."
        )
    best = max(
        qualifying,
        key=lambda i: (torch.cuda.get_device_properties(i).total_memory, -i),
    )
    return torch.device(f"cuda:{best}")


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
