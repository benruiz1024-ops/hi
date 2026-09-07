from pathlib import Path
import os,sys,time
from PIL import Image
ROOT=Path('/kaggle/working/Wan2GP'); DATA=Path('/kaggle/temp/Wan2GP-data'); CACHE=DATA/'cache'; OUTPUTS=Path('/kaggle/working/Wan2GP-outputs'); WARM=Path('/kaggle/working/lily_wan22_warmup')
for p in (CACHE,OUTPUTS,WARM): p.mkdir(parents=True,exist_ok=True)
os.environ['WAN_CACHE_DIR']=str(CACHE); os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
sys.path.insert(0,str(ROOT)); from shared.api import init
session=init(root=ROOT,output_dir=OUTPUTS,cli_args=['--profile','5','--attention','sdpa'],console_output=True)
MODEL='i2v_2_2'
HIGH='https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/loras_accelerators/Wan2.2_I2V_A14B_HIGH_4steps_lora_rank64_Seko_V1_forKJ.safetensors'
LOW='https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/loras_accelerators/Wan2.2_I2V_A14B_LOW_4steps_lora_rank64_Seko_V1_forKJ.safetensors'
print('⚡ Wan 2.2 Lightning availability:',session.get_model_availability(MODEL))
defaults=session.get_default_settings(MODEL); dummy=WARM/'warmup.png'; Image.new('RGB',(832,480),(240,240,240)).save(dummy)
s=dict(defaults); s.update({'model_type':MODEL,'prompt':'very subtle natural movement, gentle breeze, stable composition','image_start':str(dummy),'image_prompt_type':'S','video_length':'2s','duration_seconds':2,'batch_size':1,'seed':123,'num_inference_steps':4,'guidance_scale':1,'guidance2_scale':1,'switch_threshold':876,'model_switch_phase':1,'guidance_phases':2,'flow_shift':3,'sample_solvers':'euler','activated_loras':[HIGH,LOW],'loras_multipliers':'1;0 0;1'})
print('⚡ Downloading the two Lightning LoRAs if needed + verifying 4-step Wan 2.2 I2V.')
print('The giant Wan 2.2 base checkpoint is reused; this only adds the accelerator LoRAs.')
start=time.time(); result=session.submit_task(s).result()
if not result.success:
 msg=' • '.join(getattr(e,'message',str(e)) for e in (getattr(result,'errors',[]) or [])) or 'Wan 2.2 Lightning verification failed'; raise RuntimeError(msg)
print(f'✅ WAN 2.2 LIGHTNING VERIFIED in {int(time.time()-start)}s')
for f in (getattr(result,'generated_files',[]) or []):
 try:
  p=Path(f).resolve()
  if p.is_file() and OUTPUTS.resolve() in p.parents:p.unlink()
 except:pass
try:session.close()
except:pass
print('⚡🎉 LIGHTNING IS READY — run Cell 4.')
