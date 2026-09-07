from pathlib import Path
import os,sys,time,gradio as gr
ROOT=Path('/kaggle/working/Wan2GP');CACHE=Path('/kaggle/temp/Wan2GP-data/cache');OUTPUTS=Path('/kaggle/working/Wan2GP-outputs');OUTPUTS.mkdir(parents=True,exist_ok=True)
os.environ['WAN_CACHE_DIR']=str(CACHE);os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True';sys.path.insert(0,str(ROOT));from shared.api import init
session=init(root=ROOT,output_dir=OUTPUTS,cli_args=['--profile','5','--attention','sdpa'],console_output=True)
models=session.list_model_defs(family='ltx2',main_output='video',inputs='image')
def score(m):
 t=(m.get('model_type','')+' '+m.get('name','')).lower();return (200 if 'distilled_1_1' in t or 'distilled 1.1' in t else 0)+(100 if 'distilled' in t else 0)+(30 if '2.3' in t else 0)-(100 if 'dev' in t else 0)
if not models:raise RuntimeError('No LTX image-to-video model found.')
M=sorted(models,key=score,reverse=True)[0];MODEL=M['model_type'];print('⚡',M.get('name'),MODEL);ACTIVE={'job':None}
def gen(image,prompt,seconds,quality,seed):
 if not image:yield None,'📸 Add an image.';return
 if not prompt.strip():yield None,'✍️ Add a prompt.';return
 s=session.get_default_settings(MODEL);resolution='768x448' if quality=='TURBO' else '960x544';s.update({'model_type':MODEL,'prompt':prompt.strip(),'image_start':image,'image_prompt_type':'S','duration_seconds':int(seconds),'video_length':f'{int(seconds)}s','resolution':resolution,'num_inference_steps':8,'sample_solver':'distilled_8_steps','batch_size':1,'seed':int(seed)})
 yield None,f'⚡ LTX cooking • {seconds}s • {resolution} • distilled 8 steps';start=time.time()
 try:
  j=session.submit_task(s);ACTIVE['job']=j
  while not j.done:yield None,f'⚡ generating… {int(time.time()-start)}s';time.sleep(3)
  r=j.result()
  if not r.success:
   msg=' • '.join(getattr(e,'message',str(e)) for e in (getattr(r,'errors',[]) or []));yield None,'😵 '+msg;return
  files=[str(x) for x in (getattr(r,'generated_files',[]) or [])];v=next((x for x in files if x.lower().endswith(('.mp4','.webm','.mov'))),files[0] if files else None);yield v,f'💖 DONE in {int(time.time()-start)}s'
 except Exception as e:yield None,f'😵 {type(e).__name__}: {e}'
 finally:ACTIVE['job']=None
def cancel():
 j=ACTIVE.get('job')
 if not j or j.done:return 'Nothing is generating.'
 try:j.cancel();return '🛑 Cancel requested.'
 except Exception as e:return str(e)
CSS='.gradio-container{max-width:760px!important;margin:auto!important;padding:10px!important}#make{min-height:64px;font-size:1.2rem;font-weight:800;border-radius:999px!important}footer{display:none!important}'
with gr.Blocks(css=CSS,theme=gr.themes.Soft(),title='Lily LTX Turbo') as demo:
 gr.Markdown('# ⚡🌷 Lily LTX Turbo\n**the speedy little freak**')
 image=gr.Image(type='filepath',label='📸 Picture',height=320);prompt=gr.Textbox(label='✨ What happens?',lines=4);seconds=gr.Dropdown([2,4,6,8],value=4,label='⏱️ Seconds')
 with gr.Accordion('tiny settings ⚙️',open=False):quality=gr.Radio(['TURBO','Pretty'],value='TURBO',label='Mode');seed=gr.Number(-1,precision=0,label='Seed')
 make=gr.Button('⚡ MAKE VIDEO ⚡',variant='primary',elem_id='make');stop=gr.Button('🛑 Cancel');status=gr.Markdown('ready ⚡ • LTX-2.3 Distilled 1.1 • 8 steps');out=gr.Video(label='🎞️ Video',autoplay=True);make.click(gen,[image,prompt,seconds,quality,seed],[out,status]);stop.click(cancel,outputs=status,queue=False)
print('⚡ Launching LTX Turbo');demo.queue(default_concurrency_limit=1).launch(server_name='0.0.0.0',server_port=7861,share=True,show_error=True)
