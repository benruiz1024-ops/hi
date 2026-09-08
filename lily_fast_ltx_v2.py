import os, sys, subprocess, pathlib, shutil, gc, json, time, random

# ---------- DISK-SAFE SETUP ----------
TMP_ROOT = pathlib.Path('/kaggle/tmp/lily-ltx')
ROOT = TMP_ROOT / 'LTX-Video'
HF_CACHE = TMP_ROOT / 'hf-cache'
TMP_ROOT.mkdir(parents=True, exist_ok=True)
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(HF_CACHE)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(HF_CACHE / 'hub')
os.environ['TRANSFORMERS_CACHE'] = str(HF_CACHE)
os.environ['HF_HUB_DISABLE_XET'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
print(f"Scratch free: {shutil.disk_usage('/kaggle/tmp').free/1024**3:.1f} GB")
print(f"Working free: {shutil.disk_usage('/kaggle/working').free/1024**3:.1f} GB")
print('Large model files will use /kaggle/tmp ✅')
if not ROOT.exists(): subprocess.run(['git','clone','--depth','1','https://github.com/Lightricks/LTX-Video.git',str(ROOT)],check=True)
os.chdir(ROOT)
subprocess.run([sys.executable,'-m','pip','install','-q','-e','.[inference]','gradio>=5,<7','imageio-ffmpeg','hf_xet'],check=True)

import torch, numpy as np, imageio, gradio as gr
from pathlib import Path
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import T5EncoderModel,T5Tokenizer
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.inference import calculate_padding,prepare_conditioning
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
assert torch.cuda.is_available(),'No GPU. In Kaggle: Settings → Accelerator → GPU.'
GPU_NAME=torch.cuda.get_device_name(0); CC=torch.cuda.get_device_capability(0)
VRAM_GB=torch.cuda.get_device_properties(0).total_memory/1024**3
DTYPE=torch.bfloat16 if CC[0]>=8 else torch.float16
torch.backends.cuda.matmul.allow_tf32=CC[0]>=8; torch.backends.cudnn.allow_tf32=CC[0]>=8
print(f'GPU: {GPU_NAME} | {VRAM_GB:.1f} GB | compute {CC[0]}.{CC[1]} | video dtype: {DTYPE}')

MODEL_REPO='Lightricks/LTX-Video'; MODEL_FILE='ltxv-2b-0.9.8-distilled.safetensors'; TEXT_REPO='PixArt-alpha/PixArt-XL-2-1024-MS'; TEXT_MAX_TOKENS=128
print('Downloading/checking 2B distilled checkpoint…')
CKPT=hf_hub_download(repo_id=MODEL_REPO,filename=MODEL_FILE)
with safe_open(CKPT,framework='pt') as f:
    metadata=f.metadata() or {}; checkpoint_config=json.loads(metadata.get('config','{}'))
allowed_steps=checkpoint_config.get('allowed_inference_steps',None)
print('Loading LTX transformer on GPU…'); transformer=Transformer3DModel.from_pretrained(CKPT).to(device='cuda',dtype=DTYPE).eval()
print('Loading LTX VAE on GPU…'); vae=CausalVideoAutoencoder.from_pretrained(CKPT).to(device='cuda',dtype=DTYPE).eval()
if hasattr(vae,'enable_tiling'): vae.enable_tiling()
scheduler=RectifiedFlowScheduler.from_pretrained(CKPT)
tokenizer=T5Tokenizer.from_pretrained(TEXT_REPO,subfolder='tokenizer')
print('Loading T5-XXL text encoder on CPU in FP16…')
text_encoder_cpu=T5EncoderModel.from_pretrained(TEXT_REPO,subfolder='text_encoder',torch_dtype=torch.float16,low_cpu_mem_usage=True).to('cpu').eval()
for p in text_encoder_cpu.parameters(): p.requires_grad_(False)
pipe=LTXVideoPipeline(transformer=transformer,patchifier=SymmetricPatchifier(patch_size=1),text_encoder=None,tokenizer=tokenizer,scheduler=scheduler,vae=vae,prompt_enhancer_image_caption_model=None,prompt_enhancer_image_caption_processor=None,prompt_enhancer_llm_model=None,prompt_enhancer_llm_tokenizer=None,allowed_inference_steps=allowed_steps)
gc.collect(); torch.cuda.empty_cache(); print('✅ MODEL IS HOT — downloads are done.')

# Keep prompt embeddings on CPU until all companion tensors exist. This avoids launching
# an extra CUDA zeros_like kernel on older Kaggle GPUs before the pipeline starts.
def encode_prompt_cpu(prompt):
    cleaned=pipe._text_preprocessing(str(prompt).strip())
    toks=tokenizer(cleaned,padding='max_length',max_length=TEXT_MAX_TOKENS,truncation=True,add_special_tokens=True,return_tensors='pt')
    ids=toks.input_ids; mask=toks.attention_mask
    start=time.perf_counter()
    with torch.inference_mode(): emb=text_encoder_cpu(input_ids=ids,attention_mask=mask).last_hidden_state
    return emb,mask,time.perf_counter()-start

OUTPUT_DIR=Path('/kaggle/working/lily_videos'); OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
FAST_TIMESTEPS=[1.0,.9937,.9875,.9812,.975,.9094,.725]; TURBO_TIMESTEPS=[1.0,.9812,.9094,.725]
PRESETS={'⚡ TURBO — 512×320 / 4 steps':(512,320,TURBO_TIMESTEPS),'🔥 FAST — 640×384 / 7 steps':(640,384,FAST_TIMESTEPS),'✨ BETTER — 704×416 / 7 steps':(704,416,FAST_TIMESTEPS)}
FPS=15; DURATION_FRAMES={'5 seconds':73,'10 seconds':153}
def save_mp4(images,out_path):
    vid=images[0].permute(1,2,3,0).detach().float().cpu().numpy(); vid=np.clip(vid*255,0,255).astype(np.uint8)
    w=imageio.get_writer(str(out_path),fps=FPS,codec='libx264',ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart'],macro_block_size=None)
    for frame in vid: w.append_data(frame)
    w.close()
def generate_video(prompt,image_path,preset,duration,seed):
    if not prompt or not str(prompt).strip(): raise ValueError('Write a prompt first.')
    width,height,timesteps=PRESETS[preset]; num_frames=DURATION_FRAMES[duration]; seed=int(seed or 0)
    if seed<=0: seed=random.randint(1,2147483647)
    print('Encoding prompt on CPU…')
    pe_cpu,pm_cpu,enc_time=encode_prompt_cpu(prompt)
    # Create legacy negative placeholders on CPU, then transfer everything together.
    ne_cpu=torch.zeros(pe_cpu.shape,dtype=pe_cpu.dtype,device='cpu'); nm_cpu=torch.zeros(pm_cpu.shape,dtype=pm_cpu.dtype,device='cpu')
    prompt_embeds=pe_cpu.to('cuda',dtype=DTYPE); prompt_mask=pm_cpu.to('cuda'); neg_embeds=ne_cpu.to('cuda',dtype=DTYPE); neg_mask=nm_cpu.to('cuda')
    del pe_cpu,pm_cpu,ne_cpu,nm_cpu; gc.collect()
    padding=calculate_padding(height,width,height,width); conditioning_items=None
    if image_path:
        conditioning_items=prepare_conditioning(conditioning_media_paths=[str(image_path)],conditioning_strengths=[1.0],conditioning_start_frames=[0],height=height,width=width,num_frames=num_frames,padding=padding,pipeline=pipe)
    generator=torch.Generator(device='cuda').manual_seed(seed); out_path=OUTPUT_DIR/f'lily_{int(time.time())}_{seed}.mp4'; start=time.perf_counter()
    with torch.inference_mode():
        result=pipe(prompt=None,negative_prompt=None,prompt_embeds=prompt_embeds,prompt_attention_mask=prompt_mask,negative_prompt_embeds=neg_embeds,negative_prompt_attention_mask=neg_mask,height=height,width=width,num_frames=num_frames,frame_rate=FPS,timesteps=timesteps,guidance_scale=1.0,stg_scale=0.0,rescaling_scale=1.0,skip_block_list=[42],skip_layer_strategy=SkipLayerStrategy.AttentionValues,generator=generator,output_type='pt',conditioning_items=conditioning_items,decode_timestep=.05,decode_noise_scale=.025,stochastic_sampling=False,is_video=True,vae_per_channel_normalize=True,image_cond_noise_scale=.15,mixed_precision=False,offload_to_cpu=False,device='cuda',enhance_prompt=False,text_encoder_max_tokens=TEXT_MAX_TOKENS)
    render_time=time.perf_counter()-start; save_mp4(result.images,out_path)
    del result,conditioning_items,prompt_embeds,prompt_mask,neg_embeds,neg_mask; gc.collect(); torch.cuda.empty_cache()
    return str(out_path),f'✅ {duration} • {width}×{height} • {len(timesteps)} steps • prompt {enc_time:.1f}s + render {render_time:.1f}s • seed {seed} • {GPU_NAME}'
def ui_generate(prompt,image,preset,duration,seed):
    try: return generate_video(prompt,image,preset,duration,seed)
    except Exception as e: raise gr.Error(f'{type(e).__name__}: {e}')
with gr.Blocks(title='Lily Fast Video Studio V2') as demo:
    gr.Markdown('## ⚡ Lily Fast Video Studio V2\nKaggle-safe build. T5 stays on CPU; LTX stays on GPU.')
    with gr.Row():
        with gr.Column():
            image=gr.Image(type='filepath',label='Reference image (optional)',sources=['upload']); prompt=gr.Textbox(label='What happens?',lines=4,placeholder='Describe motion and camera movement…'); preset=gr.Radio(list(PRESETS.keys()),value='🔥 FAST — 640×384 / 7 steps',label='Speed / quality'); duration=gr.Radio(['5 seconds','10 seconds'],value='5 seconds',label='Length'); seed=gr.Number(value=0,precision=0,label='Seed (0 = random)'); go=gr.Button('GENERATE ⚡',variant='primary')
        with gr.Column(): video=gr.Video(label='Result',autoplay=True); status=gr.Markdown('Ready.')
    go.click(ui_generate,[prompt,image,preset,duration,seed],[video,status])
demo.queue(default_concurrency_limit=1); demo.launch(share=True,debug=True)
