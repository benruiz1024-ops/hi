# Lily Wan 2.2 Kaggle Studio

A complete replacement of `LILY_WAN22_DUAL_T4_STUDIO.ipynb`: one executable cell, one inference architecture, and no previous-notebook downloads or source patches.

## Run from an iPhone

1. In Kaggle, create a notebook and use **File → Import Notebook → GitHub** with `benruiz1024-ops/hi`. Select [`LILY_WAN22_DUAL_T4_STUDIO.ipynb`](https://github.com/benruiz1024-ops/hi/blob/main/LILY_WAN22_DUAL_T4_STUDIO.ipynb).
2. In notebook Settings, enable **Internet** and select **GPU T4 ×2** if available. **P100** is handled automatically.
3. Run the one code cell. No terminal commands, code edits, model downloads by hand, or backend launches are needed.
4. Wait for **[5/5] READY** and open the printed Gradio share link in Safari.
5. Upload an image, describe its movement, select a preset, and press **Generate video**. The video appears with a **Save MP4** download.

Keep the Kaggle session running. Kaggle still imposes its own GPU/session quotas; a share link does not keep the GPU allocated indefinitely. Stop the Kaggle session when finished.

## Architecture and upstream choices

The path is **image → official Wan 2.2 image-conditioning node → native UniPC sampling → tiled VAE decode → FFmpeg MP4**. Gradio is the only web interface. ComfyUI's Python inference core runs inside the notebook process: no ComfyUI server, workflow JSON, custom-node discovery, frontend installation, or HTTP job polling.

| Considered | Decision |
| --- | --- |
| Official native Wan | Its default setup includes FlashAttention and more dependencies; the reference 5B configuration targets 24 GB GPUs. |
| Diffusers | Supports the model, but the official snapshot includes a full-precision transformer and totals about 32 GiB. Smaller conversions or additional quantization add integration work. |
| Native ComfyUI core | Selected: official compact weights total 16.9 GiB, established CPU offloading, and direct access to the official image-conditioning node. |
| Other wrappers/accelerators | Not included; baseline generation needs no RIFE, upscaler, MagCache, LoRA, or extra model. |

Upstream was checked for this rebuild on 2026-09-08. The inference core deliberately uses the verified **v0.3.59** revision `72212fef660bcd7d9702fa52011d089c027a64d8`, rather than following a moving branch. It already implements this exact 5B workflow and avoids the newer core's additional compiled kernel dependencies. No upstream source is modified. This is an integration pin, not a claim that v0.3.59 is the newest release.

Sources: [official Wan repository](https://github.com/Wan-Video/Wan2.2), [Diffusers Wan APIs](https://huggingface.co/docs/diffusers/en/api/pipelines/wan), [official Diffusers file inventory](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/tree/b8fff7315c768468a5333511427288870b2e9635), [ComfyUI's official Wan 2.2 workflow](https://docs.comfy.org/tutorials/video/wan/wan2_2), [pinned image-conditioning node](https://github.com/Comfy-Org/ComfyUI/blob/72212fef660bcd7d9702fa52011d089c027a64d8/comfy_extras/nodes_wan.py), [pinned model manager](https://github.com/Comfy-Org/ComfyUI/blob/72212fef660bcd7d9702fa52011d089c027a64d8/comfy/model_management.py).

## Hardware and presets

**GPU 0 performs inference. GPU 1 remains unused.** Two T4s do not form a single 32 GB GPU, and this notebook does not claim a multi-GPU speedup. A single T4 uses the same T4 presets.

The transformer uses FP16 with native low-VRAM offloading. The text encoder uses scaled FP8 **storage**, with upstream FP32 arithmetic; this does not require native FP8 matrix multiplication. The VAE uses FP32 with spatial and temporal tiles. T4 uses PyTorch attention; P100 uses split attention with upcasting. Neither profile requires CUDA BF16 or FlashAttention.

CPU offloading requires **at least 21 GiB available system RAM**, normally a fresh Kaggle GPU session with about 29 GB RAM. This check happens before weights download. Other GPU names use the conservative preset family; GPUs below 14 GiB actual VRAM are rejected early.

| GPU | Preset | Landscape pixels | Frames | Steps | Duration at 24 fps |
| --- | --- | --- | --- | --- | --- |
| T4 | ⚡ TURBO | 448 × 256 | 49 | 12 | 2.04 s |
| T4 | ✨ NORMAL | 512 × 288 | 81 | 20 | 3.38 s |
| T4 | 👑 MAX | 576 × 320 | 121 | 30 | 5.04 s |
| P100 | ⚡ TURBO | 320 × 192 | 33 | 12 | 1.38 s |
| P100 | ✨ NORMAL | 384 × 224 | 49 | 20 | 2.04 s |
| P100 | 👑 MAX | 448 × 256 | 81 | 30 | 3.38 s |

Portrait swaps the dimensions. Square and Match image stay within the preset's pixel budget and use multiples of 32. Frame counts follow `4k + 1`. Aspect-ratio choices can center-crop the source; they do not stretch it. Match image limits extreme ratios to 1:2–2:1.

TURBO is a short, reduced-step draft preset, **not a distilled four-step model**. NORMAL trades more time for detail and motion; MAX is slower and longer. These sub-720p sizes trade quality for feasibility on 16 GB GPUs. Low-step outputs can have artifacts. No unmeasured speed promise: generation can take many minutes, especially on P100. The UI reports actual settings, step progress, elapsed time, and seed. These limits are conservative engineering choices, not measured peak-VRAM guarantees.

**P100 limitation:** some current Kaggle Torch/CUDA builds cannot execute on Pascal/sm_60. The notebook tests actual CUDA kernels before installing application packages or downloading weights. If this fails, select T4 or a Kaggle environment image with P100 support. It cannot repair an incompatible GPU binary without replacing the Torch/CUDA stack, which it deliberately preserves. See the [Kaggle upstream issue](https://github.com/Kaggle/docker-python/issues/1546) and [Kaggle image repository](https://github.com/Kaggle/docker-python).

## Startup, dependencies, and storage

- **[1/5] ENVIRONMENT:** hardware/disk report; CUDA kernel tests; audited dependency plan; pinned source checkout; real native-node preflight; real H.264 encoding smoke test.
- **[2/5] MODEL:** pinned repository metadata, filenames, exact sizes, and SHA-256 checks; disk budget; resumable file downloads.
- **[3/5] BACKEND:** load the transformer, text encoder, and VAE, or reuse the already loaded components.
- **[4/5] UI:** start the Gradio interface and share tunnel.
- **[5/5] READY:** print the usable link and output directory.

Compatible installed packages are reused. Missing/incompatible application dependencies use verified current pins, including Gradio 6.26.0, Transformers 5.16.1, Hugging Face Hub 1.30.0, and torchsde 0.2.6. These versions use a consistent Hugging Face dependency family and support current Pillow versions. Gradio 6's CSS is passed to `launch`, as its current API requires. `torchsde` is a small import dependency of the native sampler; no SDE-specific model is downloaded.

Pip first performs a dry-run with exact constraints on existing Torch, torchvision, torchaudio, CUDA/NVIDIA packages, NumPy, SciPy, Pillow, and distributions already imported by the kernel. The plan is checked again; actual installation uses **`--no-deps`** and only the approved versions. There is no venv, blind upgrade, uninstall, or kernel restart. Conflicting protected packages produce an early `FAILED: ENVIRONMENT`; a standard fresh Kaggle GPU session is required in that case. It does not attempt to repair arbitrary pre-existing dependency damage.

Only three weights are downloaded from [Comfy-Org's official repackaging](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/c4f60d30c55a624e35427060fdd217579a6c1d77):

| File | Size |
| --- | --- |
| `wan2.2_ti2v_5B_fp16.safetensors` | 9,999,658,848 bytes |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6,735,906,897 bytes |
| `wan2.2_vae.safetensors` | 1,409,400,960 bytes |
| **Total** | **18,144,966,705 bytes / 16.90 GiB** |

The download check requires only remaining bytes, plus **1.5 GiB reserve** for temporary work and outputs. Downloads write one `.part` file and atomically rename it; full-size temporary duplicates and separate Hugging Face weight caches are not created. Interrupted files resume. Completed files get exact-size, SHA-256, and safetensors checks. A small verification record avoids rehashing an unchanged file on every rerun.

Everything this notebook owns is under `/kaggle/working/lily_wan22_studio/`:

- `models/`: the single canonical weight location, partial downloads, verification records.
- `ComfyUI/`: one pinned inference source checkout; no models stored inside it.
- `outputs/`: completed MP4s, encoded as **H.264 / yuv420p / faststart**, silent at 24 fps.
- `temporary/`: short command/encoding logs, install plan, and Gradio's temporary upload/display files.

Exact matching weights from the old `/kaggle/working/ComfyUI/models/` are validated and reused through symlinks. Other prior directories are reported and preserved. Cleanup removes only this notebook's abandoned encoding files/incomplete source clone; partial model downloads and completed videos are preserved. Gradio expires its own temporary copies after a day. A new Kaggle session may lose working storage, so save wanted MP4s to your phone before stopping it.

## Troubleshooting

| Message or problem | What to do |
| --- | --- |
| `FAILED: INSUFFICIENT DISK SPACE` | Read Required/Available. Remove unneeded files through Kaggle or start a fresh session. The notebook does not delete arbitrary user data or shared caches. |
| `FAILED: ENVIRONMENT` | Read the short cause. Enable GPU/Internet, use a fresh standard image, or select T4 if P100's CUDA kernel test fails. Imported-package conflicts are detected before changes; no notebook edits are required. |
| `FAILED: MODEL DOWNLOAD` | Re-enable Internet or wait for Hugging Face to recover, then rerun the same cell. Network interruptions retry three times; `.part` files remain resumable. |
| `FAILED: MODEL VALIDATION` | A checksum mismatch removes only the bad canonical file/link. Rerun to fetch a clean copy. Other validated weights are reused. |
| `FAILED: CUDA OUT OF MEMORY` | The notebook first unloads GPU residency and retries once with smaller dimensions and at most 33 frames. If both attempts fail, choose TURBO and close other GPU workloads or start a fresh session. |
| P100 assigned instead of T4 | Conservative P100 presets and split attention are automatic. P100 requires an installed Torch build supporting its GPU, as explained above. |
| `FAILED: BACKEND STARTUP` / missing share link | Check the reported cause and Internet. Rerun to close the old interface and attempt a new share tunnel, reusing models. Kaggle/network restrictions or Gradio tunnel outages cannot be fixed by notebook code. No READY message is printed without a share URL. |
| `FAILED: GENERATION` | Check the short UI status; upload an image, enter a prompt, and try TURBO with another seed. Prompts are limited to 2,000 characters each. |
| `FAILED: VIDEO ENCODE` | The status includes a short FFmpeg log tail. Check free disk and retry. The startup smoke test already checks the encoder before weights download. |

Rerunning the cell is safe once an active generation finishes: the current UI is closed, models/components are reused, the checkout is not updated, and compatible packages are not reinstalled. All completed videos remain in `outputs/`.

## Validation and honest limits

**22 automated development checks passed**: notebook JSON/schema and compilation; JSON-literal guards; native node/API preflight; real H.264 encoding and faststart; encoding-failure cleanup; partial and interrupted HTTP download recovery; servers ignoring Range or returning an invalid range; checksums; completed-download reuse; disk and cleanup safeguards; preset dimensions; OOM retry limits; protected installation plans; simulated T4/P100 detection; and UI construction/share-failure handling. Checks use CPU hardware and simulated failure conditions, not full model weights.

The current pinned package combination also passed an actual dependency dry-run twice without reinstalling packages, native ComfyUI imports, and a live local Gradio HTTP check. A tiny randomly initialized Wan 2.2 transformer passed real UniPC sampling on CPU with finite output; this is an API/shape smoke test, not a visual-quality test. Each of the three upstream weight URLs was checked with an actual small HTTP Range request against its pinned revision and exact byte count.

**A full 5B generation has not been run on Kaggle T4 or P100 here.** Visual quality, exact runtime, peak GPU/system memory, iPhone playback on a physical device, and a Kaggle-hosted public Gradio tunnel remain unverified. The notebook contains runtime preflight and clear errors for these environmental limits; it does not claim guaranteed error-free or instant generation.
