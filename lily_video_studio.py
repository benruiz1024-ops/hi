from pathlib import Path
import os
import gradio as gr
from PIL import Image

ROOT = Path('/kaggle/working/Wan2GP')
OUTPUTS = Path('/kaggle/working/Wan2GP-outputs')
OUTPUTS.mkdir(parents=True, exist_ok=True)

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from shared.api import init

session = init(
    root=ROOT,
    output_dir=OUTPUTS,
    cli_args=['--profile', '5', '--attention', 'sdpa'],
    console_output=True,
)

# Three deliberately chosen image-to-video models.
# WanGP downloads the checkpoint the first time that model is actually used.
MODEL_CHOICES = [
    ('✨ LTX-2.3 Distilled 1.1', 'ltx2_22B_distilled_1_1'),
    ('🌊 Wan 2.2 I2V', 'i2v_2_2'),
    ('🎬 HunyuanVideo 1.5', 'hunyuan_1_5_t2v'),
]

# Keep only models that this installed WanGP build actually exposes.
available = {m['model_type'] for m in session.list_model_defs(main_output='video')}
MODEL_CHOICES = [(label, mid) for label, mid in MODEL_CHOICES if mid in available]

if not MODEL_CHOICES:
    raise RuntimeError('None of the three Lily Video Studio models were found in this WanGP build.')

DEFAULT_MODEL = MODEL_CHOICES[0][1]


def _auto_resolution(image_path, quality):
    if not image_path:
        return '832x480' if quality == 'Fast' else '1280x720'
    with Image.open(image_path) as im:
        w, h = im.size
    portrait = h > w
    if quality == 'Fast':
        return '480x832' if portrait else '832x480'
    return '720x1280' if portrait else '1280x720'


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

    label = next((name for name, mid in MODEL_CHOICES if mid == model_type), model_type)
    try:
        availability = session.get_model_availability(model_type)
        downloaded = bool(availability.get('available') or availability.get('is_available') or availability.get('downloaded'))
    except Exception:
        downloaded = False

    if downloaded:
        yield None, f'✨ Waking up {label}…'
    else:
        yield None, f'📦 First use of {label}: downloading its checkpoint, then making your video…'

    try:
        settings = session.get_default_settings(model_type)
        settings.update({
            'model_type': model_type,
            'prompt': prompt.strip(),
            'image_start': image_path,
            'image_prompt_type': 'S',
            'duration_seconds': int(seconds),
            'resolution': _auto_resolution(image_path, quality),
            'batch_size': 1,
            'seed': int(seed),
        })

        # Fast is intentionally conservative for a free T4.
        if quality == 'Fast' and 'num_inference_steps' in settings:
            try:
                settings['num_inference_steps'] = max(4, min(int(settings['num_inference_steps']), 8))
            except Exception:
                pass

        job = session.submit_task(settings)
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

        yield video, '💖 DONE. The tiny movie has been born.'

    except Exception as e:
        yield None, f'😵 {type(e).__name__}: {e}'


CSS = '''
:root { --radius-lg: 24px; }
.gradio-container { max-width: 760px !important; margin: 0 auto !important; }
#hero { text-align:center; padding: 12px 4px 0; }
#hero h1 { font-size: 2.0rem; margin-bottom: 0.15rem; }
#hero p { opacity: .72; margin-top: 0; }
#make-btn { min-height: 58px; font-size: 1.15rem; font-weight: 800; border-radius: 999px !important; }
#status { text-align:center; font-weight:650; }
footer { display:none !important; }
'''

with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title='Lily Video Studio') as demo:
    gr.HTML("<div id='hero'><h1>🌷 Lily Video Studio</h1><p>drop a picture. tell it what happens. make a tiny movie.</p></div>")

    image = gr.Image(type='filepath', label='📸 Your picture', height=320)
    prompt = gr.Textbox(
        label='✨ What should happen?',
        placeholder='she looks toward the camera, smiles softly, wind moves through her hair…',
        lines=3,
    )

    with gr.Row():
        model = gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL, label='🧠 Model')
        seconds = gr.Dropdown(choices=[2, 4, 6, 8], value=4, label='⏱️ Seconds')

    with gr.Accordion('tiny settings ⚙️', open=False):
        quality = gr.Radio(['Fast', 'Pretty'], value='Fast', label='Quality')
        seed = gr.Number(value=-1, precision=0, label='Seed (-1 = random)')

    make = gr.Button('✨ MAKE VIDEO ✨', variant='primary', elem_id='make-btn')
    status = gr.Markdown('ready when you are 💫', elem_id='status')
    output = gr.Video(label='🎞️ Your video', autoplay=True)

    make.click(
        fn=_generate,
        inputs=[image, prompt, model, seconds, quality, seed],
        outputs=[output, status],
    )

    gr.Markdown('<small>Runs on your Kaggle GPU using WanGP + open video models. Each model downloads automatically on first use.</small>')

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1).launch(
        server_name='0.0.0.0',
        server_port=7861,
        share=True,
        show_error=True,
    )
