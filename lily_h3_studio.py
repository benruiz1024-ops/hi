from pathlib import Path
import os, sys, shutil, json, time
import gradio as gr
from PIL import Image
ROOT=Path('/kaggle/working/Wan2GP'); DATA=Path('/kaggle/temp/Wan2GP-data'); CACHE=DATA/'cache'; OUTPUTS=Path('/kaggle/working/Wan2GP-outputs')
if not (ROOT/'.git').exists(): raise RuntimeError('WanGP is missing. Kaggle reset; start from Cell 1.')
OUTPUTS.mkdir(parents=True,exist_ok=True)
os.environ['HF_HOME']=str(CACHE/'huggingface'); os.environ['HUGGINGFACE_HUB_CACHE']=str(CACHE/'huggingface'/'hub'); os.environ['TRANSFORMERS_CACHE']=str(CACHE/'huggingface'/'transformers'); os.environ['TORCH_HOME']=str(CACHE/'torch'); os.environ['XDG_CACHE_HOME']=str(CACHE/'.cache'); os.environ['WAN_CACHE_DIR']=str(CACHE); os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
# Reapply the same narrow T4 patch on every launch because a future git pull can restore upstream source.
vdn=ROOT/'models/minimax_h3/vdn_attention.py'; text=vdn.read_text(); old='        volume = F.conv2d(volume, getattr(self, f"{name}_sp").weight, padding=2, groups=channels)'; new='''        # Lily/Kaggle T4: bypass cuDNN for this depthwise 5x5 op only.\n        with torch.backends.cudnn.flags(enabled=False):\n            volume = F.conv2d(volume, getattr(self, f"{name}_sp").weight, padding=2, groups=channels)'''
if old in text: vdn.write_text(text.replace(old,new)); print('🩹 Applied H3 T4 compatibility patch.')
elif 'Lily/Kaggle T4' not in text: raise RuntimeError('WanGP H3 source changed; refusing unsafe patch.')
sys.path.insert(0,str(ROOT)); from shared.api import init
session=init(root=ROOT,output_dir=OUTPUTS,cli_args=['--profile','5','--attention','sdpa'],console_output=True)
def choose_h3():
 c=[]
 for m in session.list_model_defs(main_output='video',inputs='image'):
  blob=json.dumps(m,default=str).lower(); mid=str(m.get('model_type',''))
  if 'minimax' not in blob or 'h3' not in blob: continue
  score=(100 if 'fl2va' in blob else 0)+(50 if 'pruned' in blob else 0)-(80 if 'ref2va' in blob else 0)-(40 if 'pdd' in blob else 0)
  if any(x in blob for x in ('voice','audio_only','refiner','outpaint','viggle')): score-=100
  c.append((score,mid,m))
 if not c: raise RuntimeError('MiniMax H3 I2V is not exposed by this WanGP build.')
 c.sort(key=lambda x:(x[0],x[1]),reverse=True); _,mid,m=c[0]
 if 'fl2va' not in json.dumps(m,default=str).lower(): raise RuntimeError(f'Could not identify H3 FL2VA: {mid}')
 return mid
MODEL=choose_h3(); AVAIL=session.get_model_availability(MODEL); print('🌷 H3:',MODEL,'Availability:',AVAIL)
if isinstance(AVAIL,dict) and AVAIL.get('available') is False: raise RuntimeError('H3 is incomplete. Rerun Cell 3; it preserves existing files and resumes missing ones.')
DEFAULTS=session.get_default_settings(MODEL); NATIVE_RES=str(DEFAULTS.get('resolution','832x480')); ACTIVE_JOB={'job':None}
def free_gib(): return shutil.disk_usage('/kaggle').free/1024**3
def resolution_for(path):
 try:
  a,b=NATIVE_RES.lower().split('x',1); w,h=int(a),int(b)
  with Image.open(path) as im: iw,ih=im.size
  if (ih>iw and w>h) or (iw>=ih and h>w): return f'{h}x{w}'
 except Exception: pass
 return NATIVE_RES
def make_settings(path,prompt,seconds,quality,seed):
 s=session.get_default_settings(MODEL); s.update({'model_type':MODEL,'prompt':prompt.strip(),'image_start':path,'image_prompt_type':'S','video_length':f'{int(seconds)}s','duration_seconds':int(seconds),'resolution':resolution_for(path),'batch_size':1,'seed':int(seed)})
 if quality=='Fast' and 'num_inference_steps' in s:
  try: s['num_inference_steps']=min(int(s['num_inference_steps']),15)
  except Exception: pass
 return s
def generate(path,prompt,seconds,quality,seed):
 if not path: yield None,'📸 Add an image first.'; return
 if not prompt or not prompt.strip(): yield None,'✍️ Tell H3 what should happen.'; return
 disk=free_gib()
 if disk<2: yield None,f'💾 Only {disk:.1f} GiB free; blocked to protect the runtime.'; return
 s=make_settings(path,prompt,seconds,quality,seed); yield None,f"✨ Preparing H3 • {seconds}s • {s.get('resolution')} • {s.get('num_inference_steps','native')} steps"
 try:
  job=session.submit_task(s); ACTIVE_JOB['job']=job; started=time.time()
  while not job.done: yield None,f'🎞️ H3 is cooking… {int(time.time()-started)}s elapsed'; time.sleep(5)
  result=job.result()
  if not result.success:
   msg=' • '.join(getattr(e,'message',str(e)) for e in (getattr(result,'errors',[]) or [])) or 'Generation failed.'; yield None,f'😵 {msg}'; return
  files=[str(p) for p in (getattr(result,'generated_files',[]) or [])]; video=next((p for p in files if p.lower().endswith(('.mp4','.mov','.mkv','.webm'))),files[0] if files else None)
  if not video: yield None,'😵 H3 finished but returned no video.'; return
  yield video,f'💖 DONE in {int(time.time()-started)}s • {free_gib():.1f} GiB free'
 except Exception as e: yield None,f'😵 {type(e).__name__}: {e}'
 finally: ACTIVE_JOB['job']=None
def cancel():
 job=ACTIVE_JOB.get('job')
 if job is None or job.done: return 'Nothing is currently generating.'
 try: job.cancel(); return '🛑 Cancel requested.'
 except Exception as e: return f'Could not cancel: {e}'
CSS='.gradio-container{max-width:760px!important;margin:0 auto!important;padding:10px!important}#hero{text-align:center;padding:12px 4px 2px}#hero h1{font-size:2rem;margin:0 0 .2rem}#make-btn{min-height:62px;font-size:1.18rem;font-weight:800;border-radius:999px!important}#status{text-align:center;font-weight:650}footer{display:none!important}'
with gr.Blocks(css=CSS,theme=gr.themes.Soft(),title='Lily H3 Video Studio') as demo:
 gr.HTML("<div id='hero'><h1>🌷 Lily H3 Video Studio</h1><p>MiniMax H3 • picture → tiny movie + sound</p></div>"); image=gr.Image(type='filepath',label='📸 Your picture',height=320); prompt=gr.Textbox(label='✨ What should happen?',placeholder='she slowly turns toward the camera, natural movement, gentle wind in her hair; soft room ambience…',lines=4); seconds=gr.Dropdown(choices=[5,10,15],value=5,label='⏱️ Seconds')
 with gr.Accordion('tiny settings ⚙️',open=False): quality=gr.Radio(['Fast','Pretty'],value='Fast',label='Quality'); seed=gr.Number(value=-1,precision=0,label='Seed (-1 = random)')
 make=gr.Button('✨ MAKE H3 VIDEO ✨',variant='primary',elem_id='make-btn'); stop=gr.Button('🛑 Cancel current generation'); status=gr.Markdown(f'ready 💫 • H3 `{MODEL}` • `{NATIVE_RES}` • {free_gib():.1f} GiB free',elem_id='status'); output=gr.Video(label='🎞️ Your H3 video',autoplay=True); make.click(fn=generate,inputs=[image,prompt,seconds,quality,seed],outputs=[output,status]); stop.click(fn=cancel,outputs=status,queue=False); gr.Markdown('<small>Runs on Kaggle with WanGP + MiniMax H3.</small>')
print('🌷 Launching Lily H3 Video Studio.'); demo.queue(default_concurrency_limit=1).launch(server_name='0.0.0.0',server_port=7861,share=True,show_error=True)
