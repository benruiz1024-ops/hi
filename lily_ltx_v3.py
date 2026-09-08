import os, sys, subprocess, pathlib, shutil, gc, json, time, random, importlib.util

TMP_ROOT = pathlib.Path('/kaggle/tmp/lily-ltx-v3')
ROOT = TMP_ROOT / 'LTX-Video'
HF_CACHE = TMP_ROOT / 'hf-cache'
TMP_ROOT.mkdir(parents=True, exist_ok=True)
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(HF_CACHE)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(HF_CACHE / 'hub')
os.environ['TRANSFORMERS_CACHE'] = str(HF_CACHE)
os.environ['HF_HUB_DISABLE_XET'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'


def run(cmd, label):
    print(f'\n▶ {label}')
    print(' '.join(cmd))
    return subprocess.run(cmd, check=True)


def gpu_name_shell():
    try:
        return subprocess.check_output(
            ['nvidia-smi','--query-gpu=name','--format=csv,noheader'], text=True
        ).splitlines()[0].strip()
    except Exception:
        return 'Unknown NVIDIA GPU'


def setup_environment():
    gpu = gpu_name_shell()
    print(f'GPU detected: {gpu}')
    print(f"Scratch free: {shutil.disk_usage('/kaggle/tmp').free/1024**3:.1f} GB")

    if 'P100' in gpu.upper():
        print('\n⚠️ P100 detected. Kaggle default PyTorch cannot execute on P100.')
        print('Installing a compatible CUDA 11.8 PyTorch build. DO NOT STOP THIS CELL — this can take a few minutes.')
        run([
            sys.executable,'-m','pip','install','--disable-pip-version-check','--progress-bar','on',
            '--upgrade','--force-reinstall','torch==2.7.1+cu118','torchvision==0.22.1+cu118',
            '--index-url','https://download.pytorch.org/whl/cu118'
        ], 'Installing P100-compatible PyTorch')
    else:
        print('✅ GPU is not P100; keeping Kaggle PyTorch for fastest startup.')

    if not ROOT.exists():
        run(['git','clone','--depth','1','https://github.com/Lightricks/LTX-Video.git',str(ROOT)], 'Cloning LTX-Video')
    os.chdir(ROOT)

    if importlib.util.find_spec('ltx_video') is None:
        print('\nInstalling LTX dependencies. DO NOT PRESS STOP while pip is working.')
        run([
            sys.executable,'-m','pip','install','--disable-pip-version-check','--progress-bar','on',
            '-e','.[inference]','gradio>=5,<7','imageio-ffmpeg','hf_xet'
        ], 'Installing LTX + UI dependencies')
    else:
        print('✅ LTX package already installed; skipping pip install.')


def launch_app():
    import torch, numpy as np, imageio, gradio as gr
    from pathlib import Path
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from transformers import T5EncoderModel, T5Tokenizer
    from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
    from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
    from ltx_video.models.transformers.transformer3d import Transformer3DModel
    from ltx_video.schedulers.rf import RectifiedFlowScheduler
    from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
    from ltx_video.inference import calculate_padding, prepare_conditioning
    from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        print('❌ CUDA is unavailable. In Kaggle choose Settings → Accelerator → GPU and rerun.')
        return

    GPU_NAME = torch.cuda.get_device_name(0)
    CC = torch.cuda.get_device_capability(0)
    print(f'PyTorch: {torch.__version__} | CUDA runtime: {torch.version.cuda} | arch list: {torch.cuda.get_arch_list()}')

    try:
        x = torch.ones(1, device='cuda'); torch.cuda.synchronize(); del x
    except Exception as e:
        print(f'❌ GPU/PyTorch preflight failed cleanly: {type(e).__name__}: {str(e).splitlines()[0]}')
        print('Restart the Kaggle session and rerun V3. If possible choose T4 x2 for the fastest path.')
        return

    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
    DTYPE = torch.bfloat16 if CC[0] >= 8 else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = CC[0] >= 8
    torch.backends.cudnn.allow_tf32 = CC[0] >= 8
    torch.backends.cudnn.benchmark = True
    print(f'✅ CUDA preflight passed: {GPU_NAME} | {VRAM_GB:.1f}GB | {DTYPE}')

    MODEL_REPO='Lightricks/LTX-Video'
    MODEL_FILE='ltxv-2b-0.9.8-distilled.safetensors'
    TEXT_REPO='PixArt-alpha/PixArt-XL-2-1024-MS'
    TEXT_MAX_TOKENS=96

    print('\nDownloading/checking the 2B distilled checkpoint…')
    CKPT=hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    with safe_open(CKPT, framework='pt') as f:
        metadata=f.metadata() or {}; cfg=json.loads(metadata.get('config','{}'))
    allowed_steps=cfg.get('allowed_inference_steps', None)

    print('Loading LTX transformer on GPU…')
    transformer=Transformer3DModel.from_pretrained(CKPT).to(device='cuda', dtype=DTYPE).eval()
    print('Loading LTX VAE on GPU…')
    vae=CausalVideoAutoencoder.from_pretrained(CKPT).to(device='cuda', dtype=DTYPE).eval()
    if hasattr(vae,'enable_tiling'): vae.enable_tiling()
    scheduler=RectifiedFlowScheduler.from_pretrained(CKPT)
    tokenizer=T5Tokenizer.from_pretrained(TEXT_REPO, subfolder='tokenizer')

    print('Loading T5-XXL text encoder on CPU (FP16)…')
    text_encoder_cpu=T5EncoderModel.from_pretrained(
        TEXT_REPO, subfolder='text_encoder', torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to('cpu').eval()
    for p in text_encoder_cpu.parameters(): p.requires_grad_(False)

    pipe=LTXVideoPipeline(
        transformer=transformer, patchifier=SymmetricPatchifier(patch_size=1),
        text_encoder=None, tokenizer=tokenizer, scheduler=scheduler, vae=vae,
        prompt_enhancer_image_caption_model=None, prompt_enhancer_image_caption_processor=None,
        prompt_enhancer_llm_model=None, prompt_enhancer_llm_tokenizer=None,
        allowed_inference_steps=allowed_steps
    )
    gc.collect(); torch.cuda.empty_cache()
    print('✅ MODEL IS HOT. The expensive setup is finished.')

    OUT=Path('/kaggle/working/lily_videos'); OUT.mkdir(parents=True, exist_ok=True)
    TURBO=[1.0,0.9812,0.9094,0.7250]
    FAST=[1.0,0.9875,0.9750,0.9094,0.7250]
    HQ=[1.0,0.9937,0.9875,0.9812,0.9750,0.9094,0.7250]
    PRESETS={
        '⚡ TURBO — 512×320 / 4 steps':(512,320,TURBO),
        '🔥 BALANCED — 576×320 / 5 steps':(576,320,FAST),
        '✨ QUALITY — 640×384 / 7 steps':(640,384,HQ),
    }
    DEFAULT='⚡ TURBO — 512×320 / 4 steps' if CC[0] < 8 else '🔥 BALANCED — 576×320 / 5 steps'
    FPS=13
    FRAMES={'5 seconds':65,'8 seconds':105,'10 seconds':129}
    prompt_cache={}

    def encode_prompt(prompt):
        key=str(prompt).strip()
        if key in prompt_cache:
            emb,mask=prompt_cache[key]
            return emb,mask,0.0,True
        toks=tokenizer(pipe._text_preprocessing(key), padding='max_length', max_length=TEXT_MAX_TOKENS,
                       truncation=True, add_special_tokens=True, return_tensors='pt')
        t=time.perf_counter()
        with torch.inference_mode():
            emb=text_encoder_cpu(input_ids=toks.input_ids, attention_mask=toks.attention_mask).last_hidden_state
        elapsed=time.perf_counter()-t
        if len(prompt_cache)>=3: prompt_cache.pop(next(iter(prompt_cache)))
        prompt_cache[key]=(emb, toks.attention_mask)
        return emb,toks.attention_mask,elapsed,False

    def save_mp4(images,path):
        vid=images[0].permute(1,2,3,0).detach().float().cpu().numpy()
        vid=np.clip(vid*255.0,0,255).astype(np.uint8)
        w=imageio.get_writer(str(path),fps=FPS,codec='libx264',ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart'],macro_block_size=None)
        for frame in vid: w.append_data(frame)
        w.close()

    def render(prompt,image_path,preset,duration,seed):
        width,height,timesteps=PRESETS[preset]
        num_frames=FRAMES[duration]
        seed=int(seed or 0) or random.randint(1,2147483647)
        emb_cpu,mask_cpu,enc_time,cached=encode_prompt(prompt)
        prompt_embeds=emb_cpu.to('cuda',dtype=DTYPE); prompt_mask=mask_cpu.to('cuda')
        conditioning=None
        if image_path:
            padding=calculate_padding(height,width,height,width)
            conditioning=prepare_conditioning(
                conditioning_media_paths=[str(image_path)], conditioning_strengths=[1.0],
                conditioning_start_frames=[0], height=height,width=width,num_frames=num_frames,
                padding=padding,pipeline=pipe)
        gen=torch.Generator(device='cuda').manual_seed(seed)
        path=OUT/f'lily_{int(time.time())}_{seed}.mp4'
        torch.cuda.empty_cache(); t=time.perf_counter()
        with torch.inference_mode():
            result=pipe(
                prompt=None, negative_prompt=None,
                prompt_embeds=prompt_embeds,prompt_attention_mask=prompt_mask,
                negative_prompt_embeds=None,negative_prompt_attention_mask=None,
                height=height,width=width,num_frames=num_frames,frame_rate=FPS,
                timesteps=timesteps,guidance_scale=1.0,stg_scale=0.0,rescaling_scale=1.0,
                skip_block_list=[42],skip_layer_strategy=SkipLayerStrategy.AttentionValues,
                generator=gen,output_type='pt',conditioning_items=conditioning,
                decode_timestep=0.05,decode_noise_scale=0.025,stochastic_sampling=False,
                is_video=True,vae_per_channel_normalize=True,image_cond_noise_scale=0.15,
                mixed_precision=False,offload_to_cpu=False,device='cuda',enhance_prompt=False,
                text_encoder_max_tokens=TEXT_MAX_TOKENS)
        render_time=time.perf_counter()-t
        save_mp4(result.images,path)
        del result,conditioning,prompt_embeds,prompt_mask
        gc.collect(); torch.cuda.empty_cache()
        cache_note='cached prompt' if cached else f'prompt {enc_time:.1f}s'
        return str(path), f'✅ {duration} • {width}×{height} • {len(timesteps)} steps • {cache_note} + render {render_time:.1f}s • seed {seed} • {GPU_NAME}'

    def ui_generate(prompt,image,preset,duration,seed):
        if not prompt or not str(prompt).strip(): return None,'❌ Write a prompt first.'
        try:
            return render(prompt,image,preset,duration,seed)
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            if preset!='⚡ TURBO — 512×320 / 4 steps':
                try: return render(prompt,image,'⚡ TURBO — 512×320 / 4 steps',duration,seed)
                except Exception: pass
            return None,'❌ GPU memory ran out. Restart session and use TURBO 5 seconds.'
        except Exception as e:
            gc.collect(); torch.cuda.empty_cache()
            return None,f'❌ Generation stopped cleanly: {type(e).__name__}: {str(e).splitlines()[0][:220]}'

    with gr.Blocks(title='Lily Video Studio V3') as demo:
        gr.Markdown(f'## ⚡ Lily Video Studio V3\n**GPU:** {GPU_NAME}  •  **Recommended:** {DEFAULT}\n\n2B distilled • model stays loaded • 5/8/10 second clips')
        with gr.Row():
            with gr.Column():
                image=gr.Image(type='filepath',label='Reference image (optional)',sources=['upload'])
                prompt=gr.Textbox(label='What happens?',lines=4,placeholder='Describe the motion, subject, camera, and lighting…')
                preset=gr.Radio(list(PRESETS.keys()),value=DEFAULT,label='Speed / quality')
                duration=gr.Radio(list(FRAMES.keys()),value='5 seconds',label='Length')
                seed=gr.Number(value=0,precision=0,label='Seed (0 = random)')
                go=gr.Button('GENERATE ⚡',variant='primary')
            with gr.Column():
                video=gr.Video(label='Result',autoplay=True)
                status=gr.Markdown('Ready.')
        go.click(ui_generate,[prompt,image,preset,duration,seed],[video,status])
    demo.queue(default_concurrency_limit=1)
    demo.launch(share=True,debug=False,show_error=False)


def main():
    setup_environment()
    launch_app()

try:
    main()
except KeyboardInterrupt:
    print('\n⏹ Setup was manually stopped. Nothing is corrupted. Run the V3 cell again and let the install finish.')
except Exception as e:
    print(f'\n❌ V3 stopped cleanly: {type(e).__name__}: {str(e).splitlines()[0]}')
    print('No traceback spam. Restart the Kaggle session and rerun V3; T4 x2 is the preferred accelerator.')
