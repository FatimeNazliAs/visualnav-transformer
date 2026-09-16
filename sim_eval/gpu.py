"""Pin this process to one GPU, and refuse to run if the pin did not take.

`cukurovaai` is shared. Taking a GPU that was not allotted to us is the one
mistake in this workstream that costs somebody else their work, and it is
invisible when it happens: the render succeeds and the frame looks perfect.

It is also not hypothetical. P0's first smoke run set CUDA_VISIBLE_DEVICES=1
and rendered on GPU 0 anyway, because iGibson enumerates render devices through
the driver, which does not consult CUDA_VISIBLE_DEVICES. It was caught only
because nvidia-smi happened to be under observation at that moment.

So the rule lives here, in the process, rather than in whichever wrapper script
the caller remembered to use:

    gpu = select_gpu()                 # before importing igibson
    ...
    verify_renderer(renderer, gpu)     # after the renderer exists

Two environment variables are involved and they are NOT interchangeable:

    CUDA_VISIBLE_DEVICES  restricts torch (the policy, from P1 on). It *remaps*
                          indices, so torch sees the chosen GPU as cuda:0.
    GIBSON_DEVICE_ID      picks the GPU that EGL renders on. It is a physical
                          device minor and is deliberately never remapped.
"""

import os
from pathlib import Path

RENDER_DEVICE_ENV = "GIBSON_DEVICE_ID"
TORCH_DEVICE_ENV = "CUDA_VISIBLE_DEVICES"
WORKSTREAM_ENV = "SIM_GPU"


class GpuSelectionError(RuntimeError):
    """Raised when the GPU to use cannot be established, or the pin failed."""


def device_exists(index):
    """True if /dev/nvidia<index> is present.

    The node's minor number *is* the value GIBSON_DEVICE_ID expects, so this
    checks exactly the right thing and needs neither torch nor nvidia-smi.
    """
    return Path("/dev/nvidia{}".format(index)).exists()


def _read_index(environ, name):
    """Parse one env var as a single GPU index, or None if unset/empty."""
    raw = environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise GpuSelectionError(
            "{}={!r} is not a single GPU index.".format(name, raw))


def resolve_gpu(explicit=None, environ=None):
    """Decide which physical GPU to use, without touching the environment.

    Precedence: explicit argument > GIBSON_DEVICE_ID > SIM_GPU. There is no
    final fallback on purpose — defaulting to 0 is the exact failure this
    module exists to prevent, and 0 is the GPU that also drives the X server.

    Reads from `environ` (default: the real one) and writes nothing, so the
    precedence rules are testable on a machine with no GPU at all.
    """
    environ = os.environ if environ is None else environ

    if explicit is not None:
        return int(explicit)

    for name in (RENDER_DEVICE_ENV, WORKSTREAM_ENV):
        index = _read_index(environ, name)
        if index is not None:
            return index

    raise GpuSelectionError(
        "No GPU was specified. Check `nvidia-smi` for a free one, then either "
        "run ./sim_eval/run_p0_smoke_test.sh (which sets this for you) or "
        "export {}=<index>. Refusing to guess: the guess would be GPU 0, which "
        "also drives this machine's display.".format(WORKSTREAM_ENV))


def select_gpu(explicit=None):
    """Resolve the GPU and pin this process to it. Returns the index.

    Must be called BEFORE importing igibson or torch: both read these variables
    when they initialise, not when they are used.
    """
    gpu = resolve_gpu(explicit)

    if not device_exists(gpu):
        raise GpuSelectionError(
            "GPU {} was requested but /dev/nvidia{} does not exist. Either the "
            "index is wrong or the container was started without access to it."
            .format(gpu, gpu))

    os.environ[RENDER_DEVICE_ENV] = str(gpu)
    os.environ[TORCH_DEVICE_ENV] = str(gpu)
    return gpu


def verify_renderer(renderer, expected_gpu):
    """Fail if the renderer did not land on the GPU we asked for.

    The pin is a request, not a guarantee: iGibson falls back to device 0 with
    only a log line if it dislikes the index. Checking closes the gap between
    "we asked" and "it happened".
    """
    actual = getattr(renderer, "device_minor", None)
    if actual is None:
        raise GpuSelectionError(
            "Renderer exposes no device_minor, so the GPU pin cannot be "
            "verified. Refusing to continue on a shared machine.")
    if int(actual) != int(expected_gpu):
        raise GpuSelectionError(
            "GPU pin failed: asked for GPU {}, renderer is on GPU {}. Stopping "
            "rather than risk another user's GPU.".format(expected_gpu, actual))
    return int(actual)
