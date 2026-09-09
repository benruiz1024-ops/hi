# LILY ACTUALLY-WORKS LTX 2B — T4 compatibility launcher
# The full studio is preserved in LILY_ACTUALLY_WORKS_LTX2B_CORE.py.
# This launcher applies Turing/T4-safe attention settings before executing it.

import urllib.request
from pathlib import Path

CORE_URL = "https://raw.githubusercontent.com/benruiz1024-ops/hi/main/LILY_ACTUALLY_WORKS_LTX2B_CORE.py"
DST = Path("/kaggle/tmp/LILY_ACTUALLY_WORKS_LTX2B_CORE.py")
DST.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(CORE_URL, DST)
code = DST.read_text(encoding="utf-8")

# Kaggle T4 is NVIDIA Turing (sm_75). Some newer PyTorch fused SDPA/Flash
# kernels are built only for newer architectures and fail with
# "no kernel image is available for execution on the device".
old_backend = '''torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.backends.cuda.enable_flash_sdp(True)
except Exception:
    pass
'''
new_backend = '''torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
# T4/sm_75 compatibility: force the portable PyTorch math attention backend.
try:
    torch.backends.cuda.enable_flash_sdp(False)
except Exception:
    pass
try:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
except Exception:
    pass
try:
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass
print("T4 SAFE ATTENTION: fused Flash/mem-efficient/cuDNN SDPA disabled; math SDPA enabled", flush=True)
'''
if old_backend not in code:
    raise RuntimeError("T4 patch failed: attention backend block was not found in core studio.")
code = code.replace(old_backend, new_backend, 1)

# Also keep T5 itself on its eager attention implementation rather than any
# Transformers SDPA fast path that may select an incompatible fused kernel.
old_t5 = '''TEXT_ENCODER = T5EncoderModel.from_pretrained(
    TEXT_REPO,
    subfolder="text_encoder",
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
).eval()'''
new_t5 = '''TEXT_ENCODER = T5EncoderModel.from_pretrained(
    TEXT_REPO,
    subfolder="text_encoder",
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
).eval()'''
if old_t5 not in code:
    raise RuntimeError("T4 patch failed: T5 loading block was not found in core studio.")
code = code.replace(old_t5, new_t5, 1)

# Make any remaining PyTorch scaled-dot-product attention calls obey the math
# backend during generation. This is slower than Flash attention but works on T4.
exec(compile(code, str(DST), "exec"))
