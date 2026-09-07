from pathlib import Path
import os, sys, shutil, json, time
from PIL import Image, ImageDraw

ROOT = Path('/kaggle/working/Wan2GP')
DATA = Path('/kaggle/temp/Wan2GP-data')
CKPTS = DATA / 'ckpts'
CACHE = DATA / 'cache'
OUTPUTS = Path('/kaggle/working/Wan2GP-outputs')
WARM = Path('/kaggle/working/lily_h3_prewarm')

if not (ROOT / '.git').exists():
    raise RuntimeError('WanGP is missing. Run Cells 1 and 2 first; if Kaggle reset, start again from Cell 1.')

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

free = free_gib()
print(f'💾 Free disk before H3: {free:.1f} GiB')
if free < 32:
    raise RuntimeError(
        f'Only {free:.1f} GiB is free. H3 has NOT started downloading. '
        'Use a fresh Kaggle runtime with at least 32 GiB free, then rerun from Cell 1.'
    )
if free < 40:
    print('⚠️ Disk is tighter than ideal, but above the safety cutoff.')

dummy = WARM / 'warmup.png'
im = Image.new('RGB', (832, 480), (242, 242, 242))
d = ImageDraw.Draw(im)
d.rectangle((18, 18, 814, 462), outline=(180, 180, 180), width=3)
d.text((36, 36), 'Lily H3 prewarm', fill=(70, 70, 70))
im.save(dummy)

sys.path.insert(0, str(ROOT))
from shared.api import init

session = init(
    root=ROOT,
    output_dir=OUTPUTS,
    cli_args=['--profile', '5', '--attention', 'sdpa'],
    console_output=True,
)

def choose_h3():
    # Use current WanGP metadata instead of relying on a model ID that may change upstream.
    defs = session.list_model_defs(main_output='video', inputs='image')
    candidates = []
    for m in defs:
        blob = json.dumps(m, default=str).lower()
        mid = str(m.get('model_type', ''))
        if 'minimax' not in blob or 'h3' not in blob:
            continue
        score = 0
        if 'fl2va' in blob: score += 100
        if 'pruned' in blob: score += 50
        if 'ref2va' in blob: score -= 80
        if 'pdd' in blob: score -= 40
        for bad in ('voice', 'audio_only', 'refiner', 'outpaint', 'viggle'):
            if bad in blob: score -= 100
        candidates.append((score, mid, m))

    if not candidates:
        related = [
            str(m.get('model_type', '')) for m in session.list_model_defs(main_output='video')
            if ('h3' in json.dumps(m, default=str).lower() or 'minimax' in json.dumps(m, default=str).lower())
        ]
        raise RuntimeError(f'Current WanGP exposed no usable MiniMax H3 image-to-video model. Related IDs: {related or "none"}')

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print('🔎 H3 candidates:')
    for score, mid, _ in candidates[:10]:
        print(f'  {score:>4}  {mid}')

    _, mid, model_def = candidates[0]
    blob = json.dumps(model_def, default=str).lower()
    if 'fl2va' not in blob:
        raise RuntimeError(f'Could not safely identify H3 FL2VA. Best candidate: {mid}')
    return mid

model = choose_h3()
print('\n🌷 Selected:', model)
try:
    print('Availability before:', session.get_model_availability(model))
except Exception as e:
    print('Availability check skipped:', e)

defaults = session.get_default_settings(model)
print('Default resolution:', defaults.get('resolution'))
print('Default steps:', defaults.get('num_inference_steps'))

settings = dict(defaults)
settings.update({
    'model_type': model,
    'prompt': 'very subtle natural movement, gentle breeze, stable composition; quiet natural ambience, no speech',
    'image_start': str(dummy),
    'image_prompt_type': 'S',
    'video_length': '5s',
    'duration_seconds': 5,
    'batch_size': 1,
    'seed': 123,
})

# Ordinary non-PDD H3 is not an 8-step distilled model. Use 15 only for this verification run.
if 'num_inference_steps' in settings:
    try:
        native = int(settings['num_inference_steps'])
        settings['num_inference_steps'] = min(native, 15)
        print('Warm-up steps:', settings['num_inference_steps'])
    except Exception:
        pass

print('\n📦 Starting H3 download + verification. This is the long cell. Do not interrupt it.')
started = time.time()
try:
    result = session.submit_task(settings).result()
    elapsed = int(time.time() - started)
    if not result.success:
        msg = 'Generation failed.'
        if getattr(result, 'errors', None):
            msg = ' • '.join(getattr(e, 'message', str(e)) for e in result.errors)
        raise RuntimeError(msg)

    files = [str(p) for p in (getattr(result, 'generated_files', []) or [])]
    print(f'✅ H3 downloaded + verified in {elapsed}s.')
    print('Warm-up output:', files[0] if files else 'none')

    after = session.get_model_availability(model)
    print('Availability after:', after)
    if isinstance(after, dict) and after.get('available') is False:
        raise RuntimeError('H3 rendered but WanGP still reports its files unavailable.')

    # Remove only the disposable verification movie; keep every model/checkpoint file.
    output_root = OUTPUTS.resolve()
    for f in files:
        try:
            p = Path(f).resolve()
            if p.is_file() and output_root in p.parents:
                p.unlink()
                print('🧹 Removed disposable warm-up video.')
        except Exception as e:
            print('Warm-up cleanup skipped:', e)
finally:
    try:
        session.close()
    except Exception:
        pass

print(f'💾 Free disk after H3: {free_gib():.1f} GiB')
print('🎉 H3 IS DOWNLOADED AND VERIFIED. Now run Cell 4; do not rerun Cell 3.')
