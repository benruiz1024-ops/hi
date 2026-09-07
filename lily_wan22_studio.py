from pathlib import Path
import os,sys,time,gradio as gr
from PIL import Image
ROOT=Path('/kaggle/working/Wan2GP'); CACHE=Path('/kaggle/temp/Wan2GP-data/cache'); OUTPUTS=Path('/kaggle/working/Wan2GP-outputs'); OUTPUTS.mkdir(parents=True,exist_ok=True)
os.environ['WAN_CACHE_DIR']=str(CACHE); os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'; sys.path.insert(0,str(ROOT)); from shared.api import init
session=init(root=ROOT,output_dir=OUTPUTS,cli_args=['--profile','5','--attention','sdpa'],console_output=True); MODEL='i2v_2_2'
HIGH='https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/loras_accelerators/Wan2.2_I2V_A14B_HIGH_4steps_lora_rank64_Seko_V1_forKJ.safetensors'; LOW='https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/loras_accelerators/Wan2.2_I2V_A14B_LOW_4steps_lora_rank64_Seko_V1_forKJ.safetensors'
av=session.get_model_availability(MODEL); print('⚡ Wan 2.2 Lightning base:',av)
if isinstance(av,dict) and av.get('available') is False: raise RuntimeError('Wan 2.2 base is incomplete. Run Cell 3 first.')
defaults=session.get_default_settings(MODEL); NATIVE=str(defaults.get('resolution','832x480')); ACTIVE={'job':None}
def res(path):
 try:
  a,b=map(int,NATIVE.lower().split('x')); im=Image.open(path); iw,ih=im.size; im.close(); return f'{b}x{a}' if ((ih>iw and a>b) or (iw>=ih and b>a)) else NATIVE
 except:return NATIVE
def generate(path,prompt,seconds,mode,seed):
 if not path:yield None,'📸 Add an image first.';return
 if not prompt.strip():yield None,'✍️ Tell Wan what should happen.';return
 s=session.get_default_settings(MODEL); s.update({'model_type':MODEL,'prompt':prompt.strip(),'image_start':path,'image_prompt_type':'S','video_length':f'{int(seconds)}s','duration_seconds':int(seconds),'resolution':res(path),'batch_size':1,'seed':int(seed)})
 if mode.startswith('⚡'):
  s.update({'num_inference_steps':4,'guidance_scale':1,'guidance2_scale':1,'switch_threshold':876,'model_switch_phase':1,'guidance_phases':2,'flow_shift':3,'sample_solvers':'euler','activated_loras':[HIGH,LOW],'loras_multipliers':'1;0 0;1'})
 else:
  s['activated_loras']=[]; s['loras_multipliers']=''
 yield None,f"{'⚡ LIGHTNING' if mode.startswith('⚡') else '🌷 NORMAL'} • {seconds}s • {s['resolution']} • {s.get('num_inference_steps','native')} steps"
 try:
  job=session.submit_task(s);ACTIVE['job']=job;start=time.time()
  while not job.done:yield None,f'🎞️ cooking… {int(time.time()-start)}s';time.sleep(5)
  r=job.result()
  if not r.success:
   msg=' • '.join(getattr(e,'message',str(e)) for e in (getattr(r,'errors',[]) or [])) or 'Generation failed';yield None,'😵 '+msg;return
  files=[str(x) for x in (getattr(r,'generated_files',[]) or [])];vid=next((x for x in files if x.lower().endswith(('.mp4','.webm','.mov','.mkv'))),files[0] if files else None);yield vid,f'💖 DONE in {int(time.time()-start)}s'
 except Exception as e:yield None,f'😵 {type(e).__name__}: {e}'
 finally:ACTIVE['job']=None
def cancel():
 j=ACTIVE.get('job')
 if not j or j.done:return 'Nothing is generating.'
 try:j.cancel();return '🛑 Cancel requested.'
 except Exception as e:return str(e)
CSS='.gradio-container{max-width:760px!important;margin:auto!important;padding:10px!important}#make{min-height:62px;font-size:1.2rem;font-weight:800;border-radius:999px!important}footer{display:none!important}'
with gr.Blocks(css=CSS,theme=gr.themes.Soft(),title='Lily Wan 2.2 Lightning') as demo:
 gr.Markdown('# ⚡🌷 Lily Wan 2.2 Lightning\n**4-step turbo image → video**')
 image=gr.Image(type='filepath',label='📸 Your picture',height=320);prompt=gr.Textbox(label='✨ What should happen?',lines=4,placeholder='she turns toward the camera, gentle natural movement, wind in her hair…');seconds=gr.Dropdown([2,4,6,8],value=4,label='⏱️ Seconds')
 with gr.Accordion('tiny settings ⚙️',open=False):mode=gr.Radio(['⚡ Lightning — 4 steps','🌷 Normal Wan — slower'],value='⚡ Lightning — 4 steps',label='Engine');seed=gr.Number(-1,precision=0,label='Seed')
 make=gr.Button('⚡ MAKE VIDEO ⚡',variant='primary',elem_id='make');stop=gr.Button('🛑 Cancel');status=gr.Markdown(f'ready ⚡ • Wan 2.2 Lightning I2V • {NATIVE}');out=gr.Video(label='🎞️ Your video',autoplay=True);make.click(generate,[image,prompt,seconds,mode,seed],[out,status]);stop.click(cancel,outputs=status,queue=False);gr.Markdown('<small>WanGP + Wan 2.2 I2V + LightX2V Lightning accelerator.</small>')
print('⚡ Launching Wan 2.2 Lightning Studio');demo.queue(default_concurrency_limit=1).launch(server_name='0.0.0.0',server_port=7861,share=True,show_error=True)
