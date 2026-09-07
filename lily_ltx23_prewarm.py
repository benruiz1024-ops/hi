from pathlib import Path
import os,sys,time
from PIL import Image
ROOT=Path('/kaggle/working/Wan2GP');DATA=Path('/kaggle/temp/Wan2GP-data');CACHE=DATA/'cache';OUTPUTS=Path('/kaggle/working/Wan2GP-outputs');WARM=Path('/kaggle/working/lily_ltx_warmup')
for p in (CACHE,OUTPUTS,WARM):p.mkdir(parents=True,exist_ok=True)
os.environ['WAN_CACHE_DIR']=str(CACHE);os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True';sys.path.insert(0,str(ROOT))
from shared.api import init
session=init(root=ROOT,output_dir=OUTPUTS,cli_args=['--profile','5','--attention','sdpa'],console_output=True)
# Prefer current 1.1 model id, but discover it rather than betting the notebook on a name.
models=session.list_model_defs(family='ltx2',main_output='video',inputs='image')
def score(m):
 t=(m.get('model_type','')+' '+m.get('name','')+' '+str(m.get('description',''))).lower();return (200 if 'distilled_1_1' in t or 'distilled 1.1' in t else 0)+(100 if 'distilled' in t else 0)+(30 if '2.3' in t else 0)-(100 if 'dev' in t else 0)
models=sorted(models,key=score,reverse=True)
if not models:raise RuntimeError('No LTX-2 image-to-video model found in this WanGP build.')
MODEL=models[0]['model_type'];print('⚡ Selected:',MODEL,'—',models[0].get('name'))
defaults=session.get_default_settings(MODEL);print('Availability before:',session.get_model_availability(MODEL));print('Native:',defaults.get('resolution'),defaults.get('num_inference_steps'),'steps')
dummy=WARM/'warmup.png';Image.new('RGB',(768,448),(235,235,235)).save(dummy)
s=dict(defaults);s.update({'model_type':MODEL,'prompt':'gentle natural movement, subtle camera motion, coherent realistic video','image_start':str(dummy),'image_prompt_type':'S','duration_seconds':2,'video_length':'2s','resolution':'768x448','num_inference_steps':8,'sample_solver':'distilled_8_steps','batch_size':1,'seed':123})
print('📦 Downloading + verifying ONLY LTX-2.3 Distilled 1.1…');start=time.time();r=session.submit_task(s).result()
if not r.success:
 msg=' • '.join(getattr(e,'message',str(e)) for e in (getattr(r,'errors',[]) or [])) or 'LTX verification failed';raise RuntimeError(msg)
print(f'✅ LTX VERIFIED in {int(time.time()-start)}s');print('🎉 LTX IS READY — run Cell 4.')
try:session.close()
except:pass
