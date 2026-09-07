from pathlib import Path
import os, sys, shutil, json, time
from PIL import Image, ImageDraw

ROOT = Path('/kaggle/working/Wan2GP')
DATA = Path('/kaggle/temp/Wan2GP-data')
CKPTS = DATA / 'ckpts'
CACHE = DATA / 'cache'
OUTPUTS = Path('/kaggle/working/Wan2GP-outputs')
WARM = Path('/kaggle/working/lily_h3_prewarm')

if not (ROOT / '.git').exists(): raise RuntimeError('WanGP is missing. Kaggle reset; start from Cell 1.')
for p in (CKPTS,CACHE,OUTPUTS,WARM): p.mkdir(parents=True,exist_ok=True)
os.environ['HF_HOME']=str(CACHE/'huggingface'); os.environ['HUGGINGFACE_HUB_CACHE']=str(CACHE/'huggingface'/'hub'); os.environ['TRANSFORMERS_CACHE']=str(CACHE/'huggingface'/'transformers'); os.environ['TORCH_HOME']=str(CACHE/'torch'); os.environ['XDG_CACHE_HOME']=str(CACHE/'.cache'); os.environ['WAN_CACHE_DIR']=str(CACHE); os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

def free_gib(): return shutil.disk_usage('/kaggle').free/1024**3
print(f'💾 Free disk: {free_gib():.1f} GiB')

# Kaggle T4 workaround: current H3 spatial depthwise conv can make cuDNN report
# "GET was unable to find an engine". Disabling cuDNN only for this one conv2d
# lets PyTorch use its CUDA fallback while leaving the rest of H3/cuDNN alone.
vdn=ROOT/'models/minimax_h3/vdn_attention.py'
text=vdn.read_text()
old='        volume = F.conv2d(volume, getattr(self, f"{name}_sp").weight, padding=2, groups=channels)'
new='''        # Lily/Kaggle T4: bypass cuDNN for this depthwise 5x5 op only.\n        # T4 + Kaggle cuDNN can fail engine selection here; PyTorch CUDA fallback works.\n        with torch.backends.cudnn.flags(enabled=False):\n            volume = F.conv2d(volume, getattr(self, f"{name}_sp").weight, padding=2, groups=channels)'''
if old in text:
    vdn.write_text(text.replace(old,new)); print('🩹 Applied H3 T4 depthwise-conv patch.')
elif 'Lily/Kaggle T4' in text:
    print('🩹 H3 T4 patch already applied.')
else:
    raise RuntimeError('WanGP H3 source changed upstream; refusing to patch an unknown code shape.')

# If the previous attempt already fetched H3, availability should now be local and no giant redownload occurs.
sys.path.insert(0,str(ROOT))
from shared.api import init
session=init(root=ROOT,output_dir=OUTPUTS,cli_args=['--profile','5','--attention','sdpa'],console_output=True)

def choose_h3():
    c=[]
    for m in session.list_model_defs(main_output='video',inputs='image'):
        blob=json.dumps(m,default=str).lower(); mid=str(m.get('model_type',''))
        if 'minimax' not in blob or 'h3' not in blob: continue
        score=(100 if 'fl2va' in blob else 0)+(50 if 'pruned' in blob else 0)-(80 if 'ref2va' in blob else 0)-(40 if 'pdd' in blob else 0)
        if any(x in blob for x in ('voice','audio_only','refiner','outpaint','viggle')): score-=100
        c.append((score,mid,m))
    if not c: raise RuntimeError('No H3 I2V model found.')
    c.sort(key=lambda x:(x[0],x[1]),reverse=True); _,mid,m=c[0]
    if 'fl2va' not in json.dumps(m,default=str).lower(): raise RuntimeError(f'Could not safely identify H3 FL2VA: {mid}')
    return mid

model=choose_h3(); print('🌷 Selected:',model)
av=session.get_model_availability(model); print('Availability:',av)
if isinstance(av,dict) and av.get('available') is False:
    print('📦 Previous run did not finish all H3 files; WanGP will resume only missing files.')

dummy=WARM/'warmup.png'; Image.new('RGB',(832,480),(242,242,242)).save(dummy)
defaults=session.get_default_settings(model); settings=dict(defaults)
settings.update({'model_type':model,'prompt':'very subtle natural movement, stable composition; quiet ambience, no speech','image_start':str(dummy),'image_prompt_type':'S','video_length':'5s','duration_seconds':5,'batch_size':1,'seed':123})
if 'num_inference_steps' in settings:
    try: settings['num_inference_steps']=min(int(settings['num_inference_steps']),15)
    except Exception: pass
print('🧪 Retrying H3 verification with T4 patch. Existing checkpoint files are preserved.')
started=time.time(); result=session.submit_task(settings).result(); elapsed=int(time.time()-started)
if not result.success:
    msg=' • '.join(getattr(e,'message',str(e)) for e in (getattr(result,'errors',[]) or [])) or 'Generation failed.'
    raise RuntimeError(msg)
files=[str(p) for p in (getattr(result,'generated_files',[]) or [])]
print(f'✅ H3 VERIFIED in {elapsed}s.')
print('Availability after:',session.get_model_availability(model))
for f in files:
    try:
        p=Path(f).resolve()
        if p.is_file() and OUTPUTS.resolve() in p.parents: p.unlink()
    except Exception: pass
try: session.close()
except Exception: pass
print(f'💾 Free disk after verification: {free_gib():.1f} GiB')
print('🎉 PATCHED H3 IS READY. Run Cell 4.')
