from pathlib import Path
import os, sys, json, time, shutil, urllib.request
from PIL import Image

ROOT = Path('/kaggle/working/Wan2GP')
DATA = Path('/kaggle/temp/Wan2GP-data')
CKPTS = DATA / 'ckpts'
CACHE = DATA / 'cache'
OUTPUTS = Path('/kaggle/working/Wan2GP-outputs')
WARM = Path('/kaggle/working/lily_fastwan5b_warmup')

if not (ROOT / '.git').exists():
    raise RuntimeError('WanGP is missing. Run Cell 1 and Cell 2 first.')

for p in (CKPTS, CACHE, OUTPUTS, WARM):
    p.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(CACHE / 'huggingface')
os.environ['HUGGINGFACE_HUB_CACHE'] = str(CACHE / 'huggingface' / 'hub')
os.environ['TRANSFORMERS_CACHE'] = str(CACHE / 'huggingface' / 'transformers')
os.environ['TORCH_HOME'] = str(CACHE / 'torch')
os.environ['XDG_CACHE_HOME'] = str(CACHE / '.cache')
os.environ['WAN_CACHE_DIR'] = str(CACHE)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def free_gib():
    return shutil.disk_usage('/kaggle').free / 1024**3

print(f'💾 Free disk before prewarm: {free_gib():.1f} GiB')

sys.path.insert(0, str(ROOT))
from shared.api import init

session = init(
    root=ROOT,
    output_dir=OUTPUTS,
    cli_args=['--profile', '5', '--attention', 'sdpa'],
    console_output=True,
)


def choose_model():
    candidates = []
    for m in session.list_model_defs(main_output='video', inputs='image'):
        blob = json.dumps(m, default=str).lower()
        mid = str(m.get('model_type', ''))
        score = 0
        if 'wan' in blob:
            score += 10
        if '2.2' in blob or '2_2' in blob:
            score += 30
        if '5b' in blob:
            score += 50
        if 'ti2v' in blob or 'textimage2video' in blob:
            score += 40
        if 'fastwan' in blob:
            score += 20
        if 'a14b' in blob:
            score -= 100
        if score > 0:
            candidates.append((score, mid, m))
    if not candidates:
        raise RuntimeError('Could not find a Wan 2.2 5B image-capable model in this WanGP build.')
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][1], candidates[0][2]

MODEL, MODEL_DEF = choose_model()
print('⚡ Selected model:', MODEL)
print('Availability before:', session.get_model_availability(MODEL))

profile_url = 'https://raw.githubusercontent.com/deepbeepmeep/Wan2GP/main/profiles/wan_2_2_5B/FastWan%203%20Steps.json'
with urllib.request.urlopen(profile_url) as r:
    FASTWAN_PROFILE = json.loads(r.read().decode('utf-8'))
print('⚡ Loaded official FastWan 3-step profile.')

dummy = WARM / 'warmup.png'
Image.new('RGB', (832, 480), (240, 240, 240)).save(dummy)

defaults = session.get_default_settings(MODEL)
settings = dict(defaults)
settings.update({
    'model_type': MODEL,
    'prompt': 'very subtle natural movement, stable composition, gentle breeze',
    'image_start': str(dummy),
    'image_prompt_type': 'S',
    'video_length': '2s',
    'duration_seconds': 2,
    'resolution': '832x480',
    'batch_size': 1,
    'seed': 123,
})
settings.update(FASTWAN_PROFILE)
print('📦 Downloading + verifying Wan 2.2 5B FastWan 3-Step. Do not interrupt.')
start = time.time()
result = session.submit_task(settings).result()
elapsed = int(time.time() - start)
if not result.success:
    msg = ' • '.join(getattr(e, 'message', str(e)) for e in (getattr(result, 'errors', []) or [])) or 'FastWan verification failed.'
    raise RuntimeError(msg)
print(f'✅ FastWan 5B verified in {elapsed}s')
print('Availability after:', session.get_model_availability(MODEL))
for f in (getattr(result, 'generated_files', []) or []):
    try:
        p = Path(f).resolve()
        if p.is_file() and OUTPUTS.resolve() in p.parents:
            p.unlink()
    except Exception:
        pass
try:
    session.close()
except Exception:
    pass
print(f'💾 Free disk after prewarm: {free_gib():.1f} GiB')
print('🎉 FASTWAN 5B IS READY — run Cell 4.')
