# LILY ACTUALLY-WORKS LTX 2B — Kaggle T4/T4x2 Image-to-Video Studio
# One purpose: upload an image, prompt motion, get a real MP4 in a Gradio page.

import os, sys, subprocess, shutil, urllib.request, zipfile, gc, time, random
from pathlib import Path

# ---------- Storage ----------
TMP = Path("/kaggle/tmp")
WORK = Path("/kaggle/working")
TMP.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(TMP / "hf")
os.environ["HF_HUB_CACHE"] = str(TMP / "hf" / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(TMP / "hf" / "transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def sh(*args, cwd=None):
    cmd = [str(x) for x in args]
    print(">", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)

# ---------- GPU check ----------
import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU is attached. In Kaggle: Settings → Accelerator → GPU T4 x2 "
        "(best) or GPU T4, then restart the session and Run All."
    )

GPU_NAMES = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
GPU_GB = [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)
          for i in range(torch.cuda.device_count())]
print("CUDA:", list(zip(GPU_NAMES, GPU_GB)), flush=True)

# ---------- Known-compatible userspace ----------
sh(
    sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "diffusers==0.33.1",
    "transformers==4.51.3",
    "huggingface-hub==0.30.2",
    "hf-xet>=1.1.5",
    "accelerate==1.6.0",
    "safetensors>=0.4.5",
    "sentencepiece>=0.2.0",
    "einops>=0.8.0",
    "timm>=1.0.0",
    "imageio[ffmpeg]>=2.34.0",
    "av>=13.0.0",
    "gradio==5.31.0",
)

# ---------- Pinned LTX source ----------
PIN = "4b2d053057623ddd4d0a1d3e9cd28890e9ef487f"
LTX_DIR = TMP / "LTX-Video"
if not (LTX_DIR / "ltx_video").exists():
    shutil.rmtree(LTX_DIR, ignore_errors=True)
    zip_path = TMP / "ltx.zip"
    print("Downloading pinned LTX source...", flush=True)
    urllib.request.urlretrieve(
        f"https://github.com/Lightricks/LTX-Video/archive/{PIN}.zip",
        zip_path,
    )
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(TMP)
    extracted = TMP / f"LTX-Video-{PIN}"
    extracted.rename(LTX_DIR)
    zip_path.unlink(missing_ok=True)

sh(sys.executable, "-m", "pip", "install", "-q", "-e", str(LTX_DIR), "--no-deps")
if str(LTX_DIR) not in sys.path:
    sys.path.insert(0, str(LTX_DIR))

# ---------- Imports after install ----------
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import T5EncoderModel, T5Tokenizer

from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.inference import calculate_padding, prepare_conditioning
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.backends.cuda.enable_flash_sdp(True)
except Exception:
    pass

VIDEO_DEVICE = torch.device("cuda:0")
DUAL_GPU = torch.cuda.device_count() >= 2
TEXT_DEVICE = torch.device("cuda:1") if DUAL_GPU else torch.device("cpu")
DTYPE = torch.float16

print("Video model device:", VIDEO_DEVICE, flush=True)
print("Text encoder device:", TEXT_DEVICE, flush=True)

# ---------- Download + load 2B distilled ----------
MODEL_REPO = "Lightricks/LTX-Video"
MODEL_FILE = "ltxv-2b-0.9.6-distilled-04-25.safetensors"
TEXT_REPO = "PixArt-alpha/PixArt-XL-2-1024-MS"

print("Downloading LTX 2B distilled checkpoint (first run only)...", flush=True)
CKPT = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, repo_type="model")
print("Checkpoint ready:", CKPT, flush=True)

with safe_open(CKPT, framework="pt") as f:
    metadata = f.metadata() or {}
import json as _json
_allowed = _json.loads(metadata.get("config", "{}")).get("allowed_inference_steps", None)

print("Loading transformer on GPU 0...", flush=True)
TRANSFORMER = Transformer3DModel.from_pretrained(CKPT).to(dtype=DTYPE)
TRANSFORMER = TRANSFORMER.to(VIDEO_DEVICE).eval()

print("Loading VAE on GPU 0...", flush=True)
VAE = CausalVideoAutoencoder.from_pretrained(CKPT).to(dtype=DTYPE)
VAE = VAE.to(VIDEO_DEVICE).eval()

print("Loading T5 text encoder" + (" on GPU 1..." if DUAL_GPU else " in CPU RAM..."), flush=True)
TOKENIZER = T5Tokenizer.from_pretrained(TEXT_REPO, subfolder="tokenizer")
TEXT_ENCODER = T5EncoderModel.from_pretrained(
    TEXT_REPO,
    subfolder="text_encoder",
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
).eval()
TEXT_ENCODER = TEXT_ENCODER.to(TEXT_DEVICE)

SCHEDULER = RectifiedFlowScheduler.from_pretrained(CKPT)
PATCHIFIER = SymmetricPatchifier(patch_size=1)

# Keep text encoder outside the pipeline so __call__ never tries to move it onto GPU 0.
PIPE = LTXVideoPipeline(
    tokenizer=TOKENIZER,
    text_encoder=None,
    vae=VAE,
    transformer=TRANSFORMER,
    scheduler=SCHEDULER,
    patchifier=PATCHIFIER,
    prompt_enhancer_image_caption_model=None,
    prompt_enhancer_image_caption_processor=None,
    prompt_enhancer_llm_model=None,
    prompt_enhancer_llm_tokenizer=None,
    allowed_inference_steps=_allowed,
)

OUT = WORK / "lily_ltx_outputs"
OUT.mkdir(parents=True, exist_ok=True)
PROMPT_CACHE = {}
FPS = 20

def gpu_status():
    bits = []
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        bits.append(f"GPU{i} {GPU_NAMES[i]}: {free_b/2**30:.1f}/{total_b/2**30:.1f} GB free")
    return " | ".join(bits)

print("MODEL READY.", gpu_status(), flush=True)

# ---------- Prompt encoding ----------
def encode_prompt(prompt):
    key = prompt.strip()
    if key in PROMPT_CACHE:
        e, m = PROMPT_CACHE[key]
        return e.to(VIDEO_DEVICE), m.to(VIDEO_DEVICE)

    # On T4x2, T5 lives permanently on GPU1.
    # On one T4, temporarily swap the 2B transformer out so T5 can encode on GPU0.
    global TRANSFORMER, TEXT_ENCODER
    if DUAL_GPU:
        enc_device = TEXT_DEVICE
    else:
        PIPE.transformer = PIPE.transformer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        TEXT_ENCODER = TEXT_ENCODER.to(VIDEO_DEVICE)
        enc_device = VIDEO_DEVICE

    try:
        PIPE.text_encoder = TEXT_ENCODER
        with torch.inference_mode():
            embeds, mask, _, _ = PIPE.encode_prompt(
                prompt=key,
                do_classifier_free_guidance=False,
                negative_prompt="",
                num_images_per_prompt=1,
                device=enc_device,
                text_encoder_max_tokens=128,
            )
        e_cpu = embeds.detach().to("cpu", dtype=DTYPE)
        m_cpu = mask.detach().to("cpu")
    finally:
        PIPE.text_encoder = None
        if not DUAL_GPU:
            TEXT_ENCODER = TEXT_ENCODER.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            PIPE.transformer = PIPE.transformer.to(VIDEO_DEVICE).eval()

    if len(PROMPT_CACHE) >= 6:
        PROMPT_CACHE.pop(next(iter(PROMPT_CACHE)))
    PROMPT_CACHE[key] = (e_cpu, m_cpu)
    return e_cpu.to(VIDEO_DEVICE), m_cpu.to(VIDEO_DEVICE)

def ensure_8n1(n):
    n = max(9, int(n))
    return ((n - 2) // 8 + 1) * 8 + 1

def orientation(path):
    with Image.open(path) as im:
        w, h = im.size
    r = w / max(h, 1)
    if r < 0.84:
        return "portrait"
    if r > 1.19:
        return "landscape"
    return "square"

PRESETS = {
    "⚡ FAST — target <3 min": {
        "portrait": (256, 448), "landscape": (448, 256), "square": (320, 320), "frames": 49
    },
    "✨ BALANCED": {
        "portrait": (320, 576), "landscape": (576, 320), "square": (448, 448), "frames": 49
    },
    "💎 LARGER — slower": {
        "portrait": (384, 640), "landscape": (640, 384), "square": (512, 512), "frames": 65
    },
}

def generate(image_path, prompt, preset_name, motion_freedom, seed):
    import gradio as gr

    if not image_path:
        raise gr.Error("Upload an image first.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Type what you want to happen.")

    try:
        seed = int(seed)
    except Exception:
        seed = -1
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    preset = PRESETS[preset_name]
    kind = orientation(image_path)
    width, height = preset[kind]
    frames = ensure_8n1(preset["frames"])

    # LTX wants multiples of 32; our presets already are, but keep this bulletproof.
    hp = ((height + 31) // 32) * 32
    wp = ((width + 31) // 32) * 32
    padding = calculate_padding(height, width, hp, wp)

    stamp = f"{int(time.time())}_{seed}"
    input_copy = OUT / f"input_{stamp}.png"
    with Image.open(image_path) as im:
        im.convert("RGB").save(input_copy)

    out_path = OUT / f"LILY_LTX_{stamp}.mp4"
    t0 = time.time()

    try:
        # Make sure generation model is back on GPU0 after any previous CPU offload.
        PIPE.transformer = PIPE.transformer.to(VIDEO_DEVICE).eval()
        PIPE.vae = PIPE.vae.to(VIDEO_DEVICE).eval()
        gc.collect()
        torch.cuda.empty_cache()

        prompt_embeds, prompt_mask = encode_prompt(prompt)

        cond = prepare_conditioning(
            conditioning_media_paths=[str(input_copy)],
            conditioning_strengths=[1.0],
            conditioning_start_frames=[0],
            height=height,
            width=width,
            num_frames=frames,
            padding=padding,
            pipeline=PIPE,
        )

        gen = torch.Generator(device=VIDEO_DEVICE).manual_seed(seed)

        with torch.inference_mode():
            result = PIPE(
                prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
                negative_prompt="",
                negative_prompt_embeds=None,
                negative_prompt_attention_mask=None,
                height=hp,
                width=wp,
                num_frames=frames,
                frame_rate=FPS,
                num_inference_steps=8,
                guidance_scale=1.0,
                stg_scale=0.0,
                rescaling_scale=1.0,
                skip_layer_strategy=SkipLayerStrategy.AttentionValues,
                generator=gen,
                output_type="pt",
                conditioning_items=cond,
                decode_timestep=0.05,
                decode_noise_scale=0.025,
                image_cond_noise_scale=float(motion_freedom),
                mixed_precision=False,
                offload_to_cpu=True,   # frees GPU0 before the expensive VAE decode
                enhance_prompt=False,
                stochastic_sampling=True,
                is_video=True,
                vae_per_channel_normalize=True,
                device=VIDEO_DEVICE,
            ).images

        # Crop back from any defensive padding.
        left, right_pad, top, bottom_pad = padding
        bottom = result.shape[3] - bottom_pad if bottom_pad else result.shape[3]
        right = result.shape[4] - right_pad if right_pad else result.shape[4]
        result = result[:, :, :frames, top:bottom, left:right]

        arr = result[0].permute(1, 2, 3, 0).detach().cpu().float().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        del result, prompt_embeds, prompt_mask, cond
        gc.collect()
        torch.cuda.empty_cache()

        with imageio.get_writer(
            out_path,
            fps=FPS,
            codec="libx264",
            macro_block_size=None,
            ffmpeg_log_level="error",
        ) as writer:
            for frame in arr:
                writer.append_data(frame)
        del arr

        # Restore transformer to GPU0 now, not on the user's next click.
        PIPE.transformer = PIPE.transformer.to(VIDEO_DEVICE).eval()
        elapsed = time.time() - t0
        status = (
            f"✅ REAL MP4 GENERATED in {elapsed:.1f}s | "
            f"{width}×{height} | {frames} frames | seed {seed}\n{gpu_status()}"
        )
        return str(out_path), status, seed

    except torch.cuda.OutOfMemoryError:
        try:
            PIPE.transformer = PIPE.transformer.to("cpu")
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        raise gr.Error(
            "GPU ran out of memory. Use ⚡ FAST. If Kaggle offers T4 x2, select it and restart."
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"{type(e).__name__}: {e}")

# ---------- Gradio ----------
import gradio as gr

with gr.Blocks(title="Lily LTX 2B Video") as demo:
    gr.Markdown(
        "# 🎥 Lily LTX 2B — actual image → video\n"
        "Upload → describe motion → Generate. The first startup downloads the model; generations after that reuse it."
    )
    gr.Markdown("**GPU:** " + gpu_status())

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", label="Starting image")
            prompt = gr.Textbox(
                label="Motion prompt",
                lines=4,
                placeholder="She smiles, sways naturally, turns slightly toward the camera...",
            )
            preset = gr.Radio(
                choices=list(PRESETS.keys()),
                value="⚡ FAST — target <3 min",
                label="Speed / size",
            )
            motion = gr.Slider(
                minimum=0.0, maximum=0.25, value=0.08, step=0.01,
                label="Motion freedom (lower = cling closer to the starting image)"
            )
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            go = gr.Button("GENERATE VIDEO", variant="primary")
        with gr.Column():
            video = gr.Video(label="Generated MP4")
            status = gr.Textbox(label="Status", lines=3)

    go.click(
        fn=generate,
        inputs=[image, prompt, preset, motion, seed],
        outputs=[video, status, seed],
    )

demo.queue(default_concurrency_limit=1, max_size=4)
demo.launch(
    share=True,
    server_name="0.0.0.0",
    show_error=True,
)
