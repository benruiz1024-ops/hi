# ============================================================
# LILY LTX 2B DISTILLED — KAGGLE FAST I2V STUDIO
# Cell 1/2: install + load once
# ============================================================

import os, sys, subprocess, shutil, gc, json, math, time, random
from pathlib import Path

# ---------- GPU sanity check ----------
try:
    import torch
except Exception:
    torch = None

if torch is None or not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU is visible to this notebook. In Kaggle open Settings → Accelerator → GPU, "
        "then restart the session and Run All."
    )

GPU_NAMES = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
GPU_MEM = [
    round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 1)
    for i in range(torch.cuda.device_count())
]
print("✅ CUDA visible:", list(zip(GPU_NAMES, GPU_MEM)))
print("ℹ️ This studio uses cuda:0 for one generation. A second T4 is not required.")

# ---------- Stable environment ----------
ROOT = Path("/kaggle/working")
LTX_REPO = ROOT / "LTX-Video"
PIN = "4b2d053057623ddd4d0a1d3e9cd28890e9ef487f"

os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/hf_cache/transformers"
os.environ["HF_HUB_CACHE"] = "/kaggle/working/hf_cache/hub"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def run(cmd, cwd=None):
    print(">", " ".join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)), cwd=cwd)

# Pin the userspace libs that match this LTX-Video snapshot.
run([
    sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "diffusers==0.33.1",
    "transformers==4.51.3",
    "huggingface-hub==0.30.2",
    "accelerate==1.6.0",
    "safetensors>=0.4.5",
    "sentencepiece>=0.2.0",
    "imageio[ffmpeg]>=2.34.0",
    "av>=13.0.0",
    "timm>=1.0.0",
    "gradio==5.31.0"
])

if not LTX_REPO.exists():
    run(["git", "clone", "https://github.com/Lightricks/LTX-Video.git", str(LTX_REPO)])
run(["git", "fetch", "origin", PIN], cwd=LTX_REPO)
run(["git", "checkout", "--force", PIN], cwd=LTX_REPO)
run([sys.executable, "-m", "pip", "install", "-q", "-e", str(LTX_REPO), "--no-deps"])

# Make repo imports deterministic in the current kernel.
if str(LTX_REPO) not in sys.path:
    sys.path.insert(0, str(LTX_REPO))

# Re-import after pip operations.
import torch
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
if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except Exception:
        pass

DEVICE = torch.device("cuda:0")
DTYPE = torch.float16

MODEL_REPO = "Lightricks/LTX-Video"
MODEL_FILE = "ltxv-2b-0.9.6-distilled-04-25.safetensors"
TEXT_REPO = "PixArt-alpha/PixArt-XL-2-1024-MS"

print("\n⬇️ Downloading the distilled 2B checkpoint if it is not cached...")
CKPT = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, repo_type="model")
print("✅ LTX checkpoint:", CKPT)

with safe_open(CKPT, framework="pt") as f:
    metadata = f.metadata() or {}
cfg = json.loads(metadata.get("config", "{}"))
ALLOWED_STEPS = cfg.get("allowed_inference_steps", None)

print("\n🧠 Loading LTX transformer on GPU (FP16)...")
TRANSFORMER = Transformer3DModel.from_pretrained(CKPT).to(dtype=DTYPE)
TRANSFORMER = TRANSFORMER.to(DEVICE).eval()
gc.collect()
torch.cuda.empty_cache()

print("🎞️ Loading LTX VAE on GPU (FP16)...")
VAE = CausalVideoAutoencoder.from_pretrained(CKPT).to(dtype=DTYPE)
VAE = VAE.to(DEVICE).eval()
gc.collect()
torch.cuda.empty_cache()

print("📝 Loading PixArt T5 text encoder in CPU RAM (FP16 storage)...")
TOKENIZER = T5Tokenizer.from_pretrained(TEXT_REPO, subfolder="tokenizer")
TEXT_ENCODER = T5EncoderModel.from_pretrained(
    TEXT_REPO,
    subfolder="text_encoder",
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
).cpu().eval()

SCHEDULER = RectifiedFlowScheduler.from_pretrained(CKPT)
PATCHIFIER = SymmetricPatchifier(patch_size=1)

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
    allowed_inference_steps=ALLOWED_STEPS,
)

PROMPT_CACHE = {}
OUTPUT_DIR = ROOT / "ltx_outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def gpu_status():
    free_b, total_b = torch.cuda.mem_get_info(0)
    return f"{GPU_NAMES[0]} — {free_b/2**30:.1f} GB free / {total_b/2**30:.1f} GB"

print("\n✅ MODEL READY")
print("GPU:", gpu_status())

# ============================================================
# Mobile-friendly Gradio studio
# ============================================================

import traceback
import gradio as gr

NATIVE_FPS = 20
HUMAN_PREFIX = (
    "Realistic continuous human motion. Preserve the same person's facial identity, "
    "facial proportions, hairstyle, body proportions, clothing, and scene. "
    "Natural anatomy, coherent hands and limbs, stable face, temporally consistent motion. "
)

PRESETS = {
    "⚡ Turbo — 4 steps / smallest": {
        "portrait": (288, 512), "landscape": (512, 288), "square": (384, 384), "steps": 4
    },
    "✨ Balanced — 8 steps / fast": {
        "portrait": (320, 576), "landscape": (576, 320), "square": (448, 448), "steps": 8
    },
    "💎 Quality — 8 steps / larger": {
        "portrait": (384, 640), "landscape": (640, 384), "square": (512, 512), "steps": 8
    },
}

DURATIONS = {
    "2.4 sec — fastest": 49,
    "3.2 sec": 65,
    "4.0 sec": 81,
}

def orientation_for(image_path):
    with Image.open(image_path) as im:
        w, h = im.size
    ratio = w / max(h, 1)
    if ratio < 0.86:
        return "portrait"
    if ratio > 1.16:
        return "landscape"
    return "square"

def encode_prompt_gpu_swap(prompt, max_tokens=128):
    cache_key = (prompt, int(max_tokens))
    if cache_key in PROMPT_CACHE:
        embeds, mask = PROMPT_CACHE[cache_key]
        return embeds.to(DEVICE), mask.to(DEVICE)

    PIPE.transformer = PIPE.transformer.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    try:
        TEXT_ENCODER.to(DEVICE)
        PIPE.text_encoder = TEXT_ENCODER
        with torch.inference_mode():
            embeds, mask, _, _ = PIPE.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                negative_prompt="",
                num_images_per_prompt=1,
                device=DEVICE,
                text_encoder_max_tokens=int(max_tokens),
            )
        embeds = embeds.detach().to("cpu", dtype=DTYPE)
        mask = mask.detach().to("cpu")
    finally:
        PIPE.text_encoder = None
        TEXT_ENCODER.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        PIPE.transformer = PIPE.transformer.to(DEVICE)
        PIPE.transformer.eval()

    if len(PROMPT_CACHE) >= 4:
        PROMPT_CACHE.pop(next(iter(PROMPT_CACHE)))
    PROMPT_CACHE[cache_key] = (embeds, mask)
    return embeds.to(DEVICE), mask.to(DEVICE)

def ensure_8n1(n):
    n = int(n)
    return ((n - 2) // 8 + 1) * 8 + 1

def even(n):
    return max(2, int(n) // 2 * 2)

def ffmpeg_finish(native_path, out_path, long_side, out_fps, smooth):
    reader = imageio.get_reader(native_path)
    try:
        first = reader.get_data(0)
        h, w = first.shape[:2]
    finally:
        reader.close()

    long_side = int(long_side)
    if w >= h:
        ow = even(long_side)
        oh = even(round(h * ow / w))
    else:
        oh = even(long_side)
        ow = even(round(w * oh / h))

    if smooth:
        vf = (
            f"scale={ow}:{oh}:flags=lanczos,"
            f"minterpolate=fps={int(out_fps)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        )
    else:
        vf = f"scale={ow}:{oh}:flags=lanczos,fps={int(out_fps)}"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(native_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.check_call(cmd)
    return str(out_path)

def generate(
    image_path,
    prompt,
    preset_name,
    duration_name,
    human_fidelity,
    identity_strength,
    output_long_side,
    output_fps,
    smooth_fps,
    seed,
):
    if not image_path:
        raise gr.Error("Upload an image first.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter what you want the person/scene to do.")

    t0 = time.time()
    preset = PRESETS[preset_name]
    orient = orientation_for(image_path)
    width, height = preset[orient]
    steps = int(preset["steps"])
    frames = ensure_8n1(DURATIONS[duration_name])

    try:
        seed = int(seed)
    except Exception:
        seed = -1
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    full_prompt = (HUMAN_PREFIX + prompt) if human_fidelity else prompt

    height_padded = ((height - 1) // 32 + 1) * 32
    width_padded = ((width - 1) // 32 + 1) * 32
    frames_padded = ensure_8n1(frames)
    padding = calculate_padding(height, width, height_padded, width_padded)

    stamp = f"{int(time.time())}_{seed}"
    input_copy = OUTPUT_DIR / f"input_{stamp}.png"
    with Image.open(image_path) as im:
        im.convert("RGB").save(input_copy)

    native_path = OUTPUT_DIR / f"native_{stamp}.mp4"
    final_path = OUTPUT_DIR / f"LILY_LTX_{stamp}.mp4"

    try:
        prompt_embeds, prompt_mask = encode_prompt_gpu_swap(full_prompt, max_tokens=128)

        conditioning_items = prepare_conditioning(
            conditioning_media_paths=[str(input_copy)],
            conditioning_strengths=[1.0],
            conditioning_start_frames=[0],
            height=height,
            width=width,
            num_frames=frames,
            padding=padding,
            pipeline=PIPE,
        )

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        with torch.inference_mode():
            images = PIPE(
                prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
                negative_prompt="",
                negative_prompt_embeds=None,
                negative_prompt_attention_mask=None,
                height=height_padded,
                width=width_padded,
                num_frames=frames_padded,
                frame_rate=NATIVE_FPS,
                num_inference_steps=steps,
                guidance_scale=1.0,
                stg_scale=0.0,
                rescaling_scale=1.0,
                skip_layer_strategy=SkipLayerStrategy.AttentionValues,
                generator=generator,
                output_type="pt",
                conditioning_items=conditioning_items,
                decode_timestep=0.05,
                decode_noise_scale=0.025,
                image_cond_noise_scale=float(identity_strength),
                mixed_precision=False,
                offload_to_cpu=False,
                enhance_prompt=False,
                stochastic_sampling=True,
                is_video=True,
                vae_per_channel_normalize=True,
                device=DEVICE,
            ).images

        pad_left, pad_right, pad_top, pad_bottom = padding
        bottom = images.shape[3] - pad_bottom if pad_bottom else images.shape[3]
        right = images.shape[4] - pad_right if pad_right else images.shape[4]
        images = images[:, :, :frames, pad_top:bottom, pad_left:right]

        video_np = images[0].permute(1, 2, 3, 0).detach().cpu().float().numpy()
        video_np = np.clip(video_np * 255.0, 0, 255).astype(np.uint8)
        del images, prompt_embeds, prompt_mask
        gc.collect()
        torch.cuda.empty_cache()

        with imageio.get_writer(
            native_path,
            fps=NATIVE_FPS,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            for frame in video_np:
                writer.append_data(frame)
        del video_np

        ffmpeg_finish(
            native_path=native_path,
            out_path=final_path,
            long_side=int(output_long_side),
            out_fps=int(output_fps),
            smooth=bool(smooth_fps),
        )

        elapsed = time.time() - t0
        status = (
            f"✅ Done in {elapsed:.1f}s | native {width}×{height}, {frames} frames, "
            f"{steps} steps @ {NATIVE_FPS}fps → {int(output_long_side)}px long side @ {int(output_fps)}fps | "
            f"seed {seed} | {gpu_status()}"
        )
        return str(final_path), str(final_path), status

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        raise gr.Error(
            "GPU ran out of memory. Switch to Balanced or Turbo, shorten the clip, then try again."
        )
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        traceback.print_exc()
        raise gr.Error(f"Generation failed: {type(e).__name__}: {e}")

CSS = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
#go_btn { min-height: 58px; font-size: 20px; font-weight: 700; }
"""

with gr.Blocks(title="Lily LTX 2B Fast I2V", css=CSS) as demo:
    gr.Markdown(
        """
# 🌸 Lily’s LTX 2B Fast Image → Video
**Upload image → distilled LTX native clip → cheap upscale/FPS → MP4**

Default **Balanced** is the speed/quality sweet spot.  
For faces/bodies, leave **Human fidelity assist** on and identity noise around **0.05**.
"""
    )

    with gr.Row():
        image_in = gr.Image(
            label="1. Reference image",
            type="filepath",
            height=360,
        )
        with gr.Column():
            prompt_in = gr.Textbox(
                label="2. Motion prompt",
                placeholder="She turns toward the camera, smiles softly, then takes two natural steps forward...",
                lines=5,
            )
            preset_in = gr.Dropdown(
                list(PRESETS.keys()),
                value="✨ Balanced — 8 steps / fast",
                label="Native quality",
            )
            duration_in = gr.Dropdown(
                list(DURATIONS.keys()),
                value="2.4 sec — fastest",
                label="Native clip length",
            )

    with gr.Accordion("Fidelity + output settings", open=False):
        human_in = gr.Checkbox(value=True, label="Human fidelity assist")
        identity_in = gr.Slider(
            0.02, 0.15, value=0.05, step=0.01,
            label="Image conditioning noise — lower = tighter reference identity"
        )
        long_in = gr.Dropdown(
            [720, 960, 1280], value=960,
            label="Final MP4 long side (cheap ffmpeg upscale)"
        )
        fps_in = gr.Dropdown(
            [24, 30, 48], value=30,
            label="Final FPS"
        )
        smooth_in = gr.Checkbox(
            value=False,
            label="Optical-flow FPS smoothing (prettier motion, slower CPU postprocess)"
        )
        seed_in = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")

    go = gr.Button("✨ GENERATE VIDEO", variant="primary", elem_id="go_btn")

    video_out = gr.Video(label="Result", format="mp4")
    file_out = gr.File(label="Download MP4")
    status_out = gr.Textbox(label="Status", interactive=False)

    go.click(
        fn=generate,
        inputs=[
            image_in, prompt_in, preset_in, duration_in, human_in,
            identity_in, long_in, fps_in, smooth_in, seed_in
        ],
        outputs=[video_out, file_out, status_out],
    )

demo.queue(default_concurrency_limit=1, max_size=8)
demo.launch(
    share=True,
    server_name="0.0.0.0",
    show_error=True,
)
