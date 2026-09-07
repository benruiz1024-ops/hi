from pathlib import Path
import os
import time
import gradio as gr
from PIL import Image

ROOT = Path('/kaggle/working/Wan2GP')
OUTPUTS = Path('/kaggle/working/Wan2GP-outputs')
OUTPUTS.mkdir(parents=True, exist_ok=True)

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('WAN_CACHE_DIR', '/kaggle/temp/Wan2GP-data/cache')

from shared.api import init

session = init(
    root=ROOT,
    output_dir=OUTPUTS,
    cli_args=['--profile', '5', '--attention', 'sdpa'],
    console_output=True,
)

WANTED = [
    ('✨ LTX-2.3 Distilled 1.1', 'ltx2_22B_distilled_1_1'),
    ('🌊 Wan 2.2 I2V', 'i2v_2_2'),
    ('🎬 HunyuanVideo 1.5 I2V', 'hunyuan_1_5_i2v'),
]

available_ids = {m['model_type'] for m in session.list_model_defs(main_output='video')}
MODEL_CHOICES = [(label, mid) for label, mid in WANTED if mid in available_ids]
missing = [(label, mid) for label, mid in WANTED if mid not in available_ids]
if missing:
    print('Warning: unavailable model IDs:', missing)
if not MODEL_CHOICES:
    raise RuntimeError('None of the Lily Video Studio models were found in this WanGP build.')
DEFAULT_MODEL = MODEL_CHOICES[0][1]


def _auto_resolution(image_path, quality, model_type, defaults):
    # Hunyuan 1.5 I2V ships as a 720p model; preserve its native default.
    if model_type == 'hunyuan_1_5_i2v':
        return defaults.get('resolution', '1280x720')

    if not image_path:
        return defaults.get('resolution', '832x480')

    with Image.open(image_path) as im:
        w, h = im.size
    portrait = h > w

    if quality == 'Fast':
        return '480x832' if portrait else '832x480'
    return '720x1280' if portrait else '1280x720'


def _model_label(model_type):
    return next((name for name, mid in MODEL_CHOICES if mid == model_type), model_type)


def _generate(image_path, prompt, model_type, seconds, quality, seed):
    if not image_path:
        yield None, '📸 Add an image first.'
        return
    if not prompt or not prompt.strip():
        yield None, '✍️ Tell me what should happen.'
        return
    if not model_type:
        yield None, '😵 Pick a model.'
        return

    label = _model_label(model_type)
    yield None, f'✨ Preparing {label}…'

    try:
        settings = session.get_default_settings(model_type)
        settings.update({
            'model_type': model_type,
            'prompt': prompt.strip(),
            'image_start': image_path,
            'image_prompt_type': 'S',
            # WanGP officially accepts seconds strings and snaps them to a valid frame count.
            'video_length': f'{int(seconds)}s',
            'duration_seconds': int(seconds),
            'resolution': _auto_resolution(image_path, quality, model_type, settings),
            'batch_size': 1,
            'seed': int(seed),
        })

        if quality == 'Fast' and 'num_inference_steps' in settings:
            try:
                settings['num_inference_steps'] = min(int(settings['num_inference_steps']), 8)
            except Exception:
                pass

        job = session.submit_task(settings)
        started = time.time()
        while not job.done:
            elapsed = int(time.time() - started)
            yield None, f'🎞️ {label} is working… {elapsed}s elapsed'
            time.sleep(5)

        result = job.result()
        if not result.success:
            msg = 'Generation failed.'
            if getattr(result, 'errors', None):
                msg = ' • '.join(getattr(e, 'message', str(e)) for e in result.errors)
            yield None, f'😵 {msg}'
            return

        files = list(getattr(result, 'generated_files', []) or [])
        video = next((str(p) for p in files if str(p).lower().endswith(('.mp4', '.mov', '.mkv', '.webm'))), None)
        if not video and files:
            video = str(files[0])
        if not video:
            yield None, '😵 It finished but I could not find the output video.'
            return

        elapsed = int(time.time() - started)
        yield video, f'💖 DONE in {elapsed}s. The tiny movie has been born.'

    except Exception as e:
        yield None, f'😵 {type(e).__name__}: {e}'


def _cancel():
    try:
        session.cancel()
        return '🛑 Cancel requested.'
    except Exception as e:
        return f'Could not cancel: {e}'


CSS = '''
:root { --radius-lg: 24px; }
.gradio-container { max-width: 760px !important; margin: 0 auto !important; padding: 10px !important; }
#hero { text-align:center; padding: 10px 4px 0; }
#hero h1 { font-size: 2rem; margin-bottom: .15rem; }
#hero p { opacity: .72; margin-top: 0; }
#make-btn { min-height: 60px; font-size: 1.15rem; font-weight: 800; border-radius: 999px !important; }
#status { text-align:center; font-weight:650; }
footer { display:none !important; }
'''

with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title='Lily Video Studio') as demo:
    gr.HTML("<div id='hero'><h1>🌷 Lily Video Studio</h1><p>drop a picture. tell it what happens. make a tiny movie.</p></div>")
    image = gr.Image(type='filepath', label='📸 Your picture', height=320)
    prompt = gr.Textbox(
        label='✨ What should happen?',
        placeholder='she slowly turns toward the camera, natural movement, gentle wind in her hair…',
        lines=3,
    )

    with gr.Row():
        model = gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL, label='🧠 Model')
        seconds = gr.Dropdown(choices=[2, 4, 6, 8], value=2, label='⏱️ Seconds')

    with gr.Accordion('tiny settings ⚙️', open=False):
        quality = gr.Radio(['Fast', 'Pretty'], value='Fast', label='Quality')
        seed = gr.Number(value=-1, precision=0, label='Seed (-1 = random)')

    make = gr.Button('✨ MAKE VIDEO ✨', variant='primary', elem_id='make-btn')
    cancel = gr.Button('🛑 Cancel current generation', variant='secondary')
    status = gr.Markdown('ready when you are 💫', elem_id='status')
    output = gr.Video(label='🎞️ Your video', autoplay=True)

    make.click(fn=_generate, inputs=[image, prompt, model, seconds, quality, seed], outputs=[output, status])
    cancel.click(fn=_cancel, outputs=status, queue=False)

    gr.Markdown('<small>Runs on your Kaggle GPU using WanGP + open video models. The notebook prewarms the three model families before this UI opens.</small>')

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1).launch(
        server_name='0.0.0.0',
        server_port=7861,
        share=True,
        show_error=True,
    )
