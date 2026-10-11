"""RunPod serverless worker for Hob Bridge — ComfyUI backend.

Same job envelope as the A1111 worker (handler.py): `txt2img` / `img2img` /
`ping` / `inventory` / `warm`, `params` in A1111 API field names. The extension
doesn't change: it needs a serverless endpoint running this image instead.

Each job's params are mapped to a ComfyUI API graph, queued on a local ComfyUI
server, and the result is returned in the A1111 shape (`images` + an `info`
JSON with per-image `infotexts`). Failures come back as classified errors.

Supported: txt2img, img2img (one init image), batch_size / n_iter, samplers and
schedulers, <lora:...> tags, embeddings, CLIP skip, VAE override, Hires fix
(latent or model upscalers). Everything else raises an explicit `unsupported`
error instead of being silently dropped: ControlNet and other scripts,
inpainting masks, 'Resize and fill'. Prompt syntax that is A1111-only
([a|b], BREAK, AND, [from:to:when]) is passed through untranslated.

Models are read from the same network-volume layout as the A1111 worker
(HOB_MODEL_DIR/Stable-diffusion, Lora, VAE, ...), mapped with a generated
extra_model_paths config. Cancelling a job (RunPod stop signal) interrupts the
running graph.
"""
import asyncio
import base64
import json
import os
import random
import re
import subprocess
import tempfile
import threading
import time

import requests

import handler as a1111  # same envelope: error classes, S3 output, progress, timeouts

COMFY_DIR = os.environ.get("COMFY_DIR", "/comfyui")
COMFY_URL = "http://127.0.0.1:8188"

# A1111 sampler label (lowercase) → ComfyUI sampler
_SAMPLERS = {
    "euler": "euler", "euler a": "euler_ancestral", "heun": "heun", "dpm2": "dpm_2",
    "dpm2 a": "dpm_2_ancestral", "dpm++ 2s a": "dpmpp_2s_ancestral", "dpm++ 2m": "dpmpp_2m",
    "dpm++ sde": "dpmpp_sde", "dpm++ 2m sde": "dpmpp_2m_sde", "dpm++ 3m sde": "dpmpp_3m_sde",
    "dpm fast": "dpm_fast", "dpm adaptive": "dpm_adaptive", "lms": "lms", "ddim": "ddim",
    "unipc": "uni_pc", "lcm": "lcm",
}
# A1111 scheduler label (lowercase) → ComfyUI scheduler
_SCHEDULERS = {
    "karras": "karras", "exponential": "exponential", "sgm uniform": "sgm_uniform", "simple": "simple",
    "normal": "normal", "ddim": "ddim_uniform", "uniform": "ddim_uniform", "beta": "beta",
    "kl optimal": "kl_optimal",
}
_LORA_TAG = re.compile(r"<(?:lora|lyco):([^:>]+)(?::([-\d.]+))?(?::([-\d.]+))?>")

# the checkpoint a job without an override uses: whatever the last warm-up loaded
_state = {"checkpoint": None}


class Unsupported(Exception):
    """A requested feature this backend doesn't implement."""


class ValidationError(RuntimeError):
    """ComfyUI rejected the graph before running it."""


class _Run:
    """Cancel handle shared between the event loop and the worker thread."""

    def __init__(self):
        self.stop = threading.Event()
        self.prompt_id = None


# ---------------------------------------------------------------------------
# What the volume holds
# ---------------------------------------------------------------------------

def _stem(name):
    return os.path.splitext(os.path.basename(name))[0]


def _match(names, wanted):
    """The volume file for a name as A1111 spells it: path, file name or stem."""
    for name in names:
        if wanted in (name, os.path.basename(name), _stem(name)):
            return name
    return None


def _options(spec):
    """Combo values from /object_info: `[[...], {}]` or `["COMBO", {"options": [...]}]`."""
    return list(spec[1].get("options", [])) if spec[0] == "COMBO" else list(spec[0])


def _catalog():
    def combo(node, field):
        resp = requests.get(f"{COMFY_URL}/object_info/{node}", timeout=30)
        resp.raise_for_status()
        return _options(resp.json()[node]["input"]["required"][field])

    embeddings = requests.get(f"{COMFY_URL}/embeddings", timeout=30)
    embeddings.raise_for_status()
    return {
        "checkpoints": combo("CheckpointLoaderSimple", "ckpt_name"),
        "loras": combo("LoraLoader", "lora_name"),
        "vae": [name for name in combo("VAELoader", "vae_name") if "." in name],  # not 'pixel_space'
        "upscale_models": combo("UpscaleModelLoader", "model_name"),
        "embeddings": embeddings.json(),
    }


def _checkpoint(catalog, wanted):
    names = catalog["checkpoints"]
    if not names:
        raise RuntimeError("no checkpoints found on the volume")
    wanted = wanted or _state["checkpoint"]
    if not wanted:
        return names[0]
    found = _match(names, wanted)
    if not found:
        raise RuntimeError(f"checkpoint '{wanted}' not found on the volume")
    return found


def _inventory():
    catalog = _catalog()
    return {
        "type": "inventory",
        "checkpoints": [{"name": n, "model_name": _stem(n), "filename": n} for n in catalog["checkpoints"]],
        # A1111's /sd-vae names carry the extension; the extension compares against it
        "vae": [{"model_name": os.path.basename(n), "filename": n} for n in catalog["vae"]],
        "loras": [{"name": _stem(n), "alias": None} for n in catalog["loras"]],
        "embeddings": sorted(catalog["embeddings"]),
        "upscalers": ["Latent"] + [_stem(n) for n in catalog["upscale_models"]],
        "controlnet": None,  # not implemented: the extension's setup check flags it if used locally
    }


# ---------------------------------------------------------------------------
# A1111 params → ComfyUI graph
# ---------------------------------------------------------------------------

class _Graph:
    def __init__(self):
        self.nodes = {}

    def add(self, class_type, **inputs):
        """Add a node; returns its id (a link to output i is `[id, i]`)."""
        node_id = str(len(self.nodes) + 1)
        self.nodes[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id


def _text(value):
    if isinstance(value, list):
        if len(value) > 1:
            raise Unsupported("per-image prompt lists are not implemented")
        value = value[0] if value else ""
    return value or ""


def _sampler(name, scheduler):
    low = (name or "Euler").strip().lower()
    sched = str(scheduler or "").strip().lower()
    for suffix in ("karras", "exponential"):  # pre-1.9 labels, e.g. "DPM++ 2M Karras"
        if low.endswith(" " + suffix):
            low = low[: -len(suffix) - 1]
            sched = sched if sched not in ("", "automatic") else suffix
    sampler = _SAMPLERS.get(low)
    if not sampler:
        raise Unsupported(f"sampler '{name}' has no ComfyUI equivalent")
    if sched in ("", "automatic"):
        # ponytail: A1111's per-sampler 'Automatic' approximated; refine if images drift
        return sampler, ("karras" if sampler.startswith("dpmpp") else
                         "ddim_uniform" if sampler == "ddim" else "normal")
    if sched not in _SCHEDULERS:
        raise Unsupported(f"scheduler '{scheduler}' has no ComfyUI equivalent")
    return sampler, _SCHEDULERS[sched]


def _hires_size(params, width, height):
    rx, ry = int(params.get("hr_resize_x") or 0), int(params.get("hr_resize_y") or 0)
    if rx and ry:
        w, h = rx, ry
    elif rx:
        w, h = rx, height * rx / width
    elif ry:
        w, h = width * ry / height, ry
    else:
        scale = float(params.get("hr_scale") or 2)
        w, h = width * scale, height * scale
    return int(round(w / 8) * 8), int(round(h / 8) * 8)


def _workflow(params, seed, catalog, checkpoint, init_image=None):
    """A1111 txt2img/img2img params → ComfyUI API graph (`{node id: node}`)."""
    if params.get("alwayson_scripts"):
        raise Unsupported("ControlNet and other A1111 scripts are not available on the ComfyUI worker")
    if params.get("mask"):
        raise Unsupported("inpainting is not implemented on the ComfyUI worker")
    width, height = int(params.get("width", 512)), int(params.get("height", 512))
    steps, cfg, batch = int(params.get("steps", 20)), float(params.get("cfg_scale", 7)), int(params.get("batch_size", 1))
    sampler, scheduler = _sampler(params.get("sampler_name"), params.get("scheduler"))
    override = params.get("override_settings") or {}

    loras = []

    def take(tag):
        te = float(tag[2] or 1)  # <lora:name:te> or <lora:name:te:unet>
        loras.append((tag[1].strip(), te, float(tag[3]) if tag[3] else te))
        return ""

    prompt = _LORA_TAG.sub(take, _text(params.get("prompt"))).strip()
    negative = _LORA_TAG.sub("", _text(params.get("negative_prompt"))).strip()
    for name in catalog["embeddings"]:  # A1111 writes the bare name, ComfyUI 'embedding:name'
        pattern = rf"(?<![\w:]){re.escape(name)}(?!\w)"
        prompt, negative = re.sub(pattern, f"embedding:{name}", prompt), re.sub(pattern, f"embedding:{name}", negative)

    g = _Graph()
    loader = g.add("CheckpointLoaderSimple", ckpt_name=checkpoint)
    model, clip, vae = [loader, 0], [loader, 1], [loader, 2]
    for name, te, unet in loras:
        file = _match(catalog["loras"], name)
        if not file:
            raise ValueError(f"lora '{name}' not found on the volume")
        node = g.add("LoraLoader", model=model, clip=clip, lora_name=file, strength_model=unet, strength_clip=te)
        model, clip = [node, 0], [node, 1]
    skip = int(override.get("CLIP_stop_at_last_layers") or 1)
    if skip > 1:
        clip = [g.add("CLIPSetLastLayer", clip=clip, stop_at_clip_layer=-skip), 0]
    if override.get("sd_vae") not in (None, "", "Automatic", "None"):
        file = _match(catalog["vae"], override["sd_vae"])
        if not file:
            raise ValueError(f"VAE '{override['sd_vae']}' not found on the volume")
        vae = [g.add("VAELoader", vae_name=file), 0]
    positive = [g.add("CLIPTextEncode", text=prompt, clip=clip), 0]
    negative = [g.add("CLIPTextEncode", text=negative, clip=clip), 0]

    def sample(latent, steps, denoise):
        return g.add("KSampler", model=model, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler,
                     scheduler=scheduler, positive=positive, negative=negative, latent_image=[latent, 0],
                     denoise=denoise)

    if init_image:
        mode = int(params.get("resize_mode", 0))
        if mode == 2:
            raise Unsupported("'Resize and fill' is not implemented on the ComfyUI worker")
        pixels = g.add("LoadImage", image=init_image)
        pixels = g.add("ImageScale", image=[pixels, 0], upscale_method="lanczos", width=width, height=height,
                       crop="center" if mode == 1 else "disabled")
        latent = g.add("VAEEncode", pixels=[pixels, 0], vae=vae)
        if batch > 1:
            latent = g.add("RepeatLatentBatch", samples=[latent, 0], amount=batch)
        samples = sample(latent, steps, float(params.get("denoising_strength", 0.75)))
    else:
        samples = sample(g.add("EmptyLatentImage", width=width, height=height, batch_size=batch), steps, 1.0)

    if params.get("enable_hr"):
        w2, h2 = _hires_size(params, width, height)
        upscaler = str(params.get("hr_upscaler") or "Latent")
        if upscaler.lower().startswith("latent"):
            method = ("bicubic" if "bicubic" in upscaler.lower() else
                      "nearest-exact" if "nearest" in upscaler.lower() else "bilinear")
            samples = g.add("LatentUpscale", samples=[samples, 0], upscale_method=method, width=w2, height=h2,
                            crop="disabled")
        else:
            file = _match(catalog["upscale_models"], upscaler)
            if not file:
                raise ValueError(f"upscaler '{upscaler}' not found on the volume")
            pixels = g.add("VAEDecode", samples=[samples, 0], vae=vae)
            up_model = g.add("UpscaleModelLoader", model_name=file)
            pixels = g.add("ImageUpscaleWithModel", upscale_model=[up_model, 0], image=[pixels, 0])
            pixels = g.add("ImageScale", image=[pixels, 0], upscale_method="lanczos", width=w2, height=h2,
                           crop="disabled")  # the model's fixed factor → the requested size
            samples = g.add("VAEEncode", pixels=[pixels, 0], vae=vae)
        samples = sample(samples, int(params.get("hr_second_pass_steps") or steps),
                         float(params.get("denoising_strength", 0.7)))

    image = g.add("VAEDecode", samples=[samples, 0], vae=vae)
    g.add("PreviewImage", images=[image, 0])
    return g.nodes


def _infotext(params, prompt, negative, seed, checkpoint):
    """The A1111 infotext for one image, so PNG Info / 'send to txt2img' keep working."""
    fields = [f"Steps: {params.get('steps', 20)}", f"Sampler: {params.get('sampler_name') or 'Euler'}"]
    if params.get("scheduler"):
        fields.append(f"Schedule type: {params['scheduler']}")
    fields += [f"CFG scale: {params.get('cfg_scale', 7)}", f"Seed: {seed}",
               f"Size: {params.get('width', 512)}x{params.get('height', 512)}", f"Model: {_stem(checkpoint)}"]
    if params.get("enable_hr"):
        fields += [f"Denoising strength: {params.get('denoising_strength', 0.7)}",
                   f"Hires upscale: {params.get('hr_scale') or 2}",
                   f"Hires upscaler: {params.get('hr_upscaler') or 'Latent'}"]
    return prompt + (f"\nNegative prompt: {negative}" if negative else "") + "\n" + ", ".join(fields)


# ---------------------------------------------------------------------------
# Running a graph on the local ComfyUI
# ---------------------------------------------------------------------------

def _cancel(prompt_id):
    """Drop the prompt from the queue and interrupt it if it is running."""
    if not prompt_id:
        return
    for path, body in (("/queue", {"delete": [prompt_id]}), ("/interrupt", {"prompt_id": prompt_id})):
        try:
            requests.post(f"{COMFY_URL}{path}", json=body, timeout=10)
        except requests.RequestException:
            pass


def _validation_text(body):
    parts = [(body.get("error") or {}).get("message") or "invalid prompt"]
    for node in (body.get("node_errors") or {}).values():
        parts += [f"{err.get('message')}: {err.get('details')}" for err in node.get("errors", [])]
    return "; ".join(parts)[:500]


def _execution_error(status):
    for name, data in status.get("messages", []):
        if name == "execution_error":
            return f"{data.get('exception_type')}: {data.get('exception_message')}"
        if name == "execution_interrupted":
            return "generation interrupted"
    return "ComfyUI execution failed"


def _run_graph(graph, run, deadline, progress=None):
    """Queue a graph and wait for it; returns the output images as PNG bytes."""
    if run.stop.is_set():
        raise RuntimeError("job cancelled")
    resp = requests.post(f"{COMFY_URL}/prompt", json={"prompt": graph}, timeout=30)
    if resp.status_code == 400:
        raise ValidationError(_validation_text(resp.json()))
    resp.raise_for_status()
    run.prompt_id = prompt_id = resp.json()["prompt_id"]
    try:
        while True:
            if run.stop.is_set():
                raise RuntimeError("job cancelled")
            entry = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10).json().get(prompt_id)
            if entry:  # ComfyUI records a prompt in /history only once it has finished
                break
            if time.time() > deadline:
                raise TimeoutError("ComfyUI generation timed out")
            if progress:
                # ponytail: state only; percentages need ComfyUI's websocket (websocket-client)
                running = any(item[1] == prompt_id for item in
                              requests.get(f"{COMFY_URL}/queue", timeout=10).json().get("queue_running", []))
                progress({"state": "generating" if running else "queued — waiting for ComfyUI"})
            time.sleep(a1111.PROGRESS_INTERVAL)
    except BaseException:
        _cancel(prompt_id)
        raise
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        raise RuntimeError(_execution_error(status))
    images = []
    for output in entry.get("outputs", {}).values():
        for image in output.get("images", []):
            resp = requests.get(f"{COMFY_URL}/view", params=image, timeout=60)
            resp.raise_for_status()
            images.append(resp.content)
    return images


def _upload(job, encoded):
    name = re.sub(r"[^\w-]", "_", f"hob_{job.get('id') or 'job'}") + ".png"
    resp = requests.post(f"{COMFY_URL}/upload/image", files={"image": (name, base64.b64decode(encoded), "image/png")},
                         data={"type": "input", "overwrite": "true"}, timeout=60)
    resp.raise_for_status()
    return resp.json()["name"]


def _generate(job, kind, params, run, timeout):
    deadline = time.time() + timeout
    catalog = _catalog()
    checkpoint = _checkpoint(catalog, (params.get("override_settings") or {}).get("sd_model_checkpoint"))
    init = None
    if kind == "img2img":
        sources = params.get("init_images") or []
        if len(sources) != 1:
            raise Unsupported("img2img needs exactly one init image (no batch img2img)")
        init = _upload(job, sources[0])
    seed = params.get("seed")
    seed = random.randrange(2 ** 32) if seed in (None, -1) else int(seed)
    batch = int(params.get("batch_size", 1))

    images, seeds = [], []
    for i in range(int(params.get("n_iter", 1))):
        iter_seed = seed + i * batch
        graph = _workflow(params, iter_seed, catalog, checkpoint, init)
        got = _run_graph(graph, run, deadline, lambda payload: a1111._progress_update(job, payload))
        images += got
        seeds += [iter_seed] * len(got)
    if not images:
        raise RuntimeError("ComfyUI returned no images")

    prompt, negative = _text(params.get("prompt")), _text(params.get("negative_prompt"))
    info = {
        "seed": seed, "subseed": -1, "all_seeds": seeds, "all_subseeds": [-1] * len(seeds),
        "all_prompts": [prompt] * len(seeds), "all_negative_prompts": [negative] * len(seeds),
        "infotexts": [_infotext(params, prompt, negative, s, checkpoint) for s in seeds],
        "sd_model_name": _stem(checkpoint),
    }
    encoded = [base64.b64encode(raw).decode() for raw in images]
    urls = a1111._upload_images(job.get("id") or "job", encoded)
    if urls:
        return {"type": kind, "images": urls, "images_are_urls": True, "info": json.dumps(info)}
    return {"type": kind, "images": encoded, "info": json.dumps(info)}


def _warm(job, wanted, run, timeout):
    """Load a checkpoint now (a 1-step 64x64 run) and make it the default for later jobs."""
    catalog = _catalog()
    checkpoint = _checkpoint(catalog, wanted)
    started = time.time()
    graph = _workflow({"prompt": "warmup", "steps": 1, "width": 64, "height": 64}, 0, catalog, checkpoint)
    _run_graph(graph, run, started + timeout)
    _state["checkpoint"] = checkpoint
    return {"type": "warm", "checkpoint": checkpoint, "seconds": round(time.time() - started, 1)}


def _safe(fn, *args):
    try:
        return fn(*args)
    except Unsupported as exc:
        return a1111._failure({"kind": "unsupported", "message": str(exc),
                               "hint": "drop that option, or run the job on the A1111 worker"})
    except Exception as exc:
        return a1111._failure(a1111._classify_error(str(exc)))


async def handler(job):
    # async so the SDK's stop signal (job cancelled / timed out) reaches us as
    # CancelledError; blocking work runs in a thread to keep the loop free
    job_input = job.get("input") or {}
    kind = job_input.get("type", "txt2img")
    params = job_input.get("params") or {}
    if kind == "ping":
        return {"type": "ping", "ok": True}
    if kind == "inventory":
        return await asyncio.to_thread(_safe, _inventory)
    try:  # the client's total budget caps the run, as in the A1111 worker
        timeout = min(a1111.GENERATION_TIMEOUT, float(job_input.get("timeout") or a1111.GENERATION_TIMEOUT))
    except (TypeError, ValueError):
        timeout = a1111.GENERATION_TIMEOUT
    run = _Run()
    if kind == "warm":
        if not params.get("checkpoint"):
            return a1111._failure(a1111._classify_error("warm: no checkpoint given"))
        work = (_warm, job, params["checkpoint"], run, timeout)
    elif kind in ("txt2img", "img2img"):
        work = (_generate, job, kind, params, run, timeout)
    else:
        return a1111._failure(a1111._classify_error(f"unknown job type: {kind}"))
    try:
        return await asyncio.to_thread(_safe, *work)
    except asyncio.CancelledError:
        run.stop.set()
        _cancel(run.prompt_id)
        raise


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

def _model_paths(base=None):
    """extra_model_paths config mapping ComfyUI's folders onto the volume layout."""
    return {"hob": {
        "base_path": base or a1111.MODEL_DIR, "checkpoints": "Stable-diffusion", "vae": "VAE", "loras": "Lora",
        "embeddings": "embeddings", "upscale_models": "ESRGAN\nSwinIR\nScuNET", "controlnet": "ControlNet",
    }}


def _start_comfy():
    config = os.path.join(tempfile.gettempdir(), "hob_extra_model_paths.yaml")
    with open(config, "w") as fh:
        json.dump(_model_paths(), fh)  # JSON is valid YAML
    args = ["python", "main.py", "--port", "8188", "--disable-auto-launch", "--disable-metadata",
            "--extra-model-paths-config", config] + os.environ.get("HOB_COMFY_ARGS", "").split()
    subprocess.Popen(args, cwd=COMFY_DIR)


if __name__ == "__main__":
    import runpod

    _start_comfy()
    a1111._wait_until_ready(f"{COMFY_URL}/system_stats")
    preload = os.environ.get("HOB_PRELOAD_MODEL", "").strip()
    if preload:  # pay the model load at boot; best effort, never fatal
        result = _safe(_warm, {}, preload, _Run(), a1111.STARTUP_TIMEOUT)
        if "error" in result:
            print(f"[hob-bridge] HOB_PRELOAD_MODEL={preload!r} not loaded: {result['error']}", flush=True)
    runpod.serverless.start({"handler": handler})
