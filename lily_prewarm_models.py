from pathlib import Path
import os
import time
import traceback
from PIL import Image, ImageDraw

ROOT = Path('/kaggle/working/Wan2GP')
OUTPUTS = Path('/kaggle/working/Wan2GP-outputs')
OUTPUTS.mkdir(parents=True, exist_ok=True)
WARMUP_DIR = Path('/kaggle/working/lily_prewarm')
WARMUP_DIR.mkdir(parents=True, exist_ok=True)
DUMMY_IMAGE = WARMUP_DIR / 'warmup.png'

# A neutral landscape frame works with all three I2V models.
img = Image.new('RGB', (1280, 720), (242, 242, 242))
draw = ImageDraw.Draw(img)
draw.rectangle((24, 24, 1256, 696), outline=(180, 180, 180), width=4)
draw.text((48, 48), 'Lily Video Studio prewarm', fill=(70, 70, 70))
img.save(DUMMY_IMAGE)

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('WAN_CACHE_DIR', '/kaggle/temp/Wan2GP-data/cache')

from shared.api import init

session = init(
    root=ROOT,
    output_dir=OUTPUTS,
    cli_args=['--profile', '5', '--attention', 'sdpa'],
    console_output=True,
)

TARGETS = [
    ('LTX-2.3 Distilled 1.1', 'ltx2_22B_distilled_1_1'),
    ('Wan 2.2 I2V', 'i2v_2_2'),
    ('HunyuanVideo 1.5 I2V', 'hunyuan_1_5_i2v'),
]

available = {m['model_type']: m for m in session.list_model_defs(main_output='video')}
missing = [(label, mid) for label, mid in TARGETS if mid not in available]
if missing:
    text = ', '.join(f'{label} ({mid})' for label, mid in missing)
    raise RuntimeError(f'WanGP did not expose the expected model(s): {text}')

print('\n🌷 Lily prewarm: WanGP found all three target models.')
print('This stage intentionally downloads and runs one tiny 2-second test with each model.')
print('It can take a LONG time on the first fresh Kaggle session.\n')

for index, (label, model_type) in enumerate(TARGETS, start=1):
    print(f'\n===== [{index}/3] {label} =====')
    try:
        before = session.get_model_availability(model_type)
        print('Availability before warm-up:', before)
    except Exception as e:
        print('Availability check skipped:', e)

    try:
        settings = session.get_default_settings(model_type)
        default_resolution = settings.get('resolution')
        default_steps = settings.get('num_inference_steps')
        print('Native resolution:', default_resolution)
        print('Native inference steps:', default_steps)

        settings.update({
            'model_type': model_type,
            'prompt': 'very subtle natural movement, gentle breeze',
            'image_start': str(DUMMY_IMAGE),
            'image_prompt_type': 'S',
            # Seconds-string conversion is handled by WanGP and snapped to each model's valid frame rules.
            'video_length': '2s',
            'duration_seconds': 2,
            'batch_size': 1,
            'seed': 123,
        })

        # Keep native resolution for maximum compatibility; only reduce sampling steps for the warm-up.
        if 'num_inference_steps' in settings:
            try:
                settings['num_inference_steps'] = min(int(settings['num_inference_steps']), 8)
            except Exception:
                pass

        started = time.time()
        job = session.submit_task(settings)
        result = job.result()
        elapsed = int(time.time() - started)

        if not result.success:
            msg = 'Generation failed.'
            if getattr(result, 'errors', None):
                msg = ' • '.join(getattr(e, 'message', str(e)) for e in result.errors)
            raise RuntimeError(msg)

        print(f'✅ {label} downloaded + verified in {elapsed}s.')
        files = list(getattr(result, 'generated_files', []) or [])
        if files:
            print('Warm-up output:', files[0])

        try:
            after = session.get_model_availability(model_type)
            print('Availability after warm-up:', after)
        except Exception:
            pass

    except Exception as exc:
        print(f'\n❌ {label} failed: {type(exc).__name__}: {exc}')
        traceback.print_exc()
        raise

try:
    session.close()
except Exception:
    pass

print('\n🎉 ALL THREE MODELS ARE DOWNLOADED AND VERIFIED FOR THIS KAGGLE SESSION.')
print('Now run the Lily Video Studio launch cell.')
