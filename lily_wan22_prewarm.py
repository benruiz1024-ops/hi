from pathlib import Path
import os,sys,shutil,time
from PIL import Image
ROOT=Path('/kaggle/working/Wan2GP'); DATA=Path('/kaggle/temp/Wan2GP-data'); CACHE=DATA/'cache'; OUTPUTS=Path('/kaggle/working/Wan2GP-outputs'); WARM=Path('/kaggle/working/lily_wan22_warmup')
for p in (CACHE,OUTPUTS,WARM): p.mkdir(parents=True,exist_ok=True)
os.environ['WAN_CACHE_DIR']=str(CACHE); os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
sys.path.insert(0,str(ROOT)); from shared.api import init
session=init(root=ROOT,output_dir=OUTPUTS,cli_args=['--profile','5','--attention','sdpa'],console_output=True)
MODEL='i2v_2_2'; print('🌊 Wan 2.2 I2V availability before:',session.get_model_availability(MODEL))
defaults=session.get_default_settings(MODEL); print('Native defaults:',defaults.get('resolution'),defaults.get('num_inference_steps'),'steps')
dummy=WARM/'warmup.png'; Image.new('RGB',(832,480),(240,240,240)).save(dummy)
s=dict(defaults); s.update({'model_type':MODEL,'prompt':'very subtle natural movement, gentle breeze, stable composition','image_start':str(dummy),'image_prompt_type':'S','video_length':'2s','duration_seconds':2,'batch_size':1,'seed':123})
# Verification only: use a small step count; actual Studio defaults are independent.
if 'num_inference_steps' in s:
 try:s['num_inference_steps']=min(int(s['num_inference_steps']),8)
 except:pass
print('📦 Downloading + verifying ONLY Wan 2.2 I2V. Do not interrupt.')
start=time.time(); result=session.submit_task(s).result()
if not result.success:
 msg=' • '.join(getattr(e,'message',str(e)) for e in (getattr(result,'errors',[]) or [])) or 'Wan 2.2 verification failed'; raise RuntimeError(msg)
print(f'✅ WAN 2.2 VERIFIED in {int(time.time()-start)}s'); print('Availability after:',session.get_model_availability(MODEL))
for f in (getattr(result,'generated_files',[]) or []):
 try:
  p=Path(f).resolve()
  if p.is_file() and OUTPUTS.resolve() in p.parents:p.unlink()
 except:pass
try:session.close()
except:pass
print('🎉 WAN 2.2 IS READY — run Cell 4.')
