from pathlib import Path
import os, sys, json, time, urllib.request
import gradio as gr
from PIL import Image

ROOT = Path('/kaggle/working/Wan2GP')
DATA = Path('/kaggle/temp/Wan2GP-data')
CACHE = DATA / 'cache'
OUTPUTS = Path('/kaggle/working/Wan2GP-outputs')
OUTPUTS.mkdir(parents=True, exist_ok=True)

if not (ROOT / '.git').exists():
    raise RuntimeError('WanGP is missing. Run Cell 1 and Cell 2 first.')

os.environ['HF_HOME'] = str(CACHE / 'huggingface')
os.environ['HUGGINGFACE_HUB_CACHE'] = str(CACHE / 'huggingface' / 'hub')
os.environ['TRANSFORMERS_CACHE'] = str(CACHE / 'huggingface' / 'transformers')
os.environ['TORCH_HOME'] = str(CACHE / 'torch')
os.environ['XDG_CACHE_HOME'] = str(CACHE / '.cache')
os.environ['WAN_CACHE_DIR'] = str(CACHE)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
    return candidates[0][1]

MODEL = choose_model()
AVAIL = session.get_model_availability(MODEL)
print('⚡ Wan 2.2 5B:', MODEL, 'Availability:', AVAIL)
if isinstance(AVAIL, dict) and AVAIL.get('available') is False:
    raise RuntimeError('FastWan 5B is incomplete. Run Cell 3 first.')

with urllib.request.urlopen('https://raw.githubusercontent.com/deepbeepmeep/Wan2GP/main/profiles/wan_2_2_5B/FastWan%203%20Steps.json') as r:
    FASTWAN_PROFILE = json.loads(r.read().decode('utf-8'))

DEFAULTS = session.get_default_settings(MODEL)
NATIVE = str(DEFAULTS.get('resolution', '832x480'))
ACTIVE = {'job': None}


def oriented_resolution(path, preset):
    if preset == '⚡ FastWan':
        base = '832x480'
    else:
        base = NATIVE if 'x' in NATIVE else '832x480'
    try:
        w, h = map(int, base.lower().split('x', 1))
        with Image.open(path) as im:
            iw, ih = im.size
        if (ih > iw and w > h) or (iw >= ih and h > w):
            return f'{h}x{w}'
    except Exception:
        pass
    return base


def build_settings(path, prompt, seconds, preset, seed):
    s = session.get_default_settings(MODEL)
    s.update({
        'model_type': MODEL,
        'prompt': prompt.strip(),
        'image_start': path,
        'image_prompt_type': 'S',
        'video_length': f'{int(seconds)}s',
        'duration_seconds': int(seconds),
        'resolution': oriented_resolution(path, preset),
        'batch_size': 1,
        'seed': int(seed),
    })
    if preset == '⚡ FastWan':
        s.update(FASTWAN_PROFILE)
    return s


def generate(path, prompt, seconds, preset, seed):
    if not path:
        yield None, '📸 Add an image first.'
        return
    if not prompt or not prompt.strip():
        yield None, '✍️ Tell it what should happen.'
        return
    s = build_settings(path, prompt, seconds, preset, seed)
    if preset == '⚡ FastWan':
        intro = f"⚡ FastWan 3-Step • {seconds}s • {s['resolution']} • 3 steps"
    else:
        intro = f"🌷 Base Wan 5B • {seconds}s • {s['resolution']} • {s.get('num_inference_steps', 'native')} steps"
    yield None, intro
    try:
        job = session.submit_task(s)
        ACTIVE['job'] = job
        start = time.time()
        while not job.done:
            yield None, f"🎞️ cooking… {int(time.time() - start)}s"
            time.sleep(5)
        r = job.result()
        if not r.success:
            msg = ' • '.join(getattr(e, 'message', str(e)) for e in (getattr(r, 'errors', []) or [])) or 'Generation failed.'
            yield None, '😵 ' + msg
            return
        files = [str(x) for x in (getattr(r, 'generated_files', []) or [])]
        vid = next((x for x in files if x.lower().endswith(('.mp4', '.webm', '.mov', '.mkv'))), files[0] if files else None)
        if not vid:
            yield None, '😵 Finished but no video was returned.'
            return
        yield vid, f'💖 DONE in {int(time.time() - start)}s'
    except Exception as e:
        yield None, f'😵 {type(e).__name__}: {e}'
    finally:
        ACTIVE['job'] = None


def cancel():
    j = ACTIVE.get('job')
    if not j or j.done:
        return 'Nothing is generating.'
    try:
        j.cancel()
        return '🛑 Cancel requested.'
    except Exception as e:
        return str(e)

CSS = '.gradio-container{max-width:760px!important;margin:auto!important;padding:10px!important}#make{min-height:62px;font-size:1.2rem;font-weight:800;border-radius:999px!important}footer{display:none!important}'

with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title='Lily FastWan 5B Studio') as demo:
    gr.Markdown('# ⚡🌷 Lily FastWan 5B Studio\n**Wan 2.2 TI2V 5B • picture → tiny movie**')
    image = gr.Image(type='filepath', label='📸 Your picture', height=320)
    prompt = gr.Textbox(label='✨ What should happen?', lines=4, placeholder='she turns toward the camera, gentle natural movement, wind in her hair…')
    with gr.Row():
        seconds = gr.Dropdown([2, 4, 6, 8], value=4, label='⏱️ Seconds')
        preset = gr.Dropdown(['⚡ FastWan', '🌷 Base Wan'], value='⚡ FastWan', label='Preset')
    with gr.Accordion('tiny settings ⚙️', open=False):
        seed = gr.Number(-1, precision=0, label='Seed')
    make = gr.Button('⚡ MAKE VIDEO ⚡', variant='primary', elem_id='make')
    stop = gr.Button('🛑 Cancel')
    status = gr.Markdown(f'ready ⚡ • model `{MODEL}`', elem_id='status')
    out = gr.Video(label='🎞️ Your video', autoplay=True)
    make.click(generate, [image, prompt, seconds, preset, seed], [out, status])
    stop.click(cancel, outputs=status, queue=False)
    gr.Markdown('<small>FastWan uses the official 3-step accelerator profile. Base Wan is slower but kept as a fallback.</small>')

print('⚡ Launching Lily FastWan 5B Studio')
demo.queue(default_concurrency_limit=1).launch(server_name='0.0.0.0', server_port=7861, share=True, show_error=True)
