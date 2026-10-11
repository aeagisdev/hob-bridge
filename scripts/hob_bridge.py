"""Hob Bridge — run A1111 generations on a remote GPU (RunPod) from the local UI.

Select "Hob Bridge (run on RunPod)" in the Scripts dropdown to route
generation to the configured endpoint. Leave the dropdown on None to
generate locally as usual.

Transport: serverless jobs are submitted with POST /run and polled on
/status — a cold-starting worker easily exceeds the ~90 s cap of /runsync.
While a job is IN_PROGRESS, /status carries the worker's latest progress
update as `output`; it is mirrored into the UI status line. Clicking
Interrupt cancels the remote job (and the worker interrupts A1111).
"""
import ast
import base64
import importlib.util
import io
import json
import os
import queue
import re
import threading
import time

import requests

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy ships with A1111
    np = None

try:
    from PIL import Image, ImageChops, ImageStat, PngImagePlugin
except ImportError:  # pragma: no cover
    Image = ImageChops = ImageStat = PngImagePlugin = None

# A1111 imports are absent when this module is imported by the test suite.
try:
    import gradio as gr
    from modules import images as webui_images, scripts, sd_models, shared
    from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
    import modules.script_callbacks as script_callbacks
    _IN_WEBUI = True
except ImportError:
    _IN_WEBUI = False

_INVENTORY_TTL = 300
_POLL_INTERVAL = 1.2
_POLL_GRACE = 60  # seconds of failed /status polls tolerated before giving up
_inventory_cache = {"ts": 0.0, "data": None}
_controlnet_module = "unset"


def _load_worker():
    # The extension is installed as the whole repo: direct mode reuses the
    # worker's A1111 client code (inventory, generation, error classification).
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "worker", "handler.py")
    spec = importlib.util.spec_from_file_location("hob_bridge_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_worker = _load_worker()


class JobInterrupted(Exception):
    """Raised when the user clicks Interrupt while a remote job is in flight."""


def _interrupted():
    """Interrupt only counts during a generation: shared.state.interrupted is
    reset by state.begin() and stays stale after a job ends (state.job == "")."""
    return _IN_WEBUI and bool(shared.state.job) and shared.state.interrupted


def _raise_if_interrupted():
    if _interrupted():
        raise JobInterrupted()


def _user_error(message):
    if _IN_WEBUI:
        raise gr.Error(message)
    raise RuntimeError(message)  # test suite / non-UI contexts


def _setting(key, default=""):
    if not _IN_WEBUI:
        return default
    return shared.opts.data.get(key, default)


def _total_timeout():
    try:
        return int(_setting("hob_bridge_timeout", 600) or 600)
    except (TypeError, ValueError):
        return 600


def _base_url(raw):
    """Endpoint base URL; tolerates a full /runsync or /run URL being pasted."""
    url = (raw or "").strip().rstrip("/")
    if url.endswith("/runsync") or url.endswith("/run"):
        url = url.rsplit("/", 1)[0]
    return url


def _set_progress(phase, pct=None, eta=None):
    """Mirror remote progress into the status line shared.state drives."""
    if not _IN_WEBUI:
        return
    bits = [f"Hob Bridge: {phase}"]
    if pct is not None:
        bits.append(f"{min(pct, 100):.0f}%")
        shared.state.sampling_steps = 100
        shared.state.sampling_step = min(pct, 100)
    if eta is not None and eta >= 0:
        bits.append(f"~{eta:.0f}s left")
    shared.state.textinfo = " — ".join(bits)


def _show_remote_progress(payload):
    """Worker progress ({state, percentage, eta_relative} or a string) → status line."""
    if isinstance(payload, dict):
        _set_progress(payload.get("state") or "generating",
                      payload.get("percentage"), payload.get("eta_relative"))
    elif payload:
        _set_progress(str(payload))


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _image_to_b64(value):
    """A UI image value (PIL / numpy / b64 str / list of those) → API b64."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        encoded = [b for b in (_image_to_b64(v) for v in value) if b is not None]
        if not encoded:
            return None
        return encoded if len(encoded) > 1 else encoded[0]
    if np is not None and isinstance(value, np.ndarray):
        value = Image.fromarray(np.asarray(value).astype("uint8"))
    if Image is not None and isinstance(value, Image.Image):
        buf = io.BytesIO()
        value.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    return None


def _parse_info(info):
    """The API's `info` is the JSON of processed.js(); per-image infotexts are in info['infotexts']."""
    if isinstance(info, dict):
        return info
    try:
        parsed = json.loads(info or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _embed_geninfo(raw_images, infotexts):
    """Re-embed each image's remote infotext as the PNG `parameters` chunk.

    Images returned by the A1111 API carry no metadata; without this, a PNG
    dragged out of A1111 loses prompt/seed/settings. Takes raw image bytes,
    returns raw image bytes; on failure the original bytes are kept. Images
    without an infotext (e.g. ControlNet detected maps) are left untouched.
    """
    if Image is None or PngImagePlugin is None:
        return raw_images
    embedded = []
    for index, raw in enumerate(raw_images):
        text = infotexts[index] if index < len(infotexts) else ""
        if not text:
            embedded.append(raw)
            continue
        try:
            img = Image.open(io.BytesIO(raw))
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("parameters", text)
            buf = io.BytesIO()
            img.save(buf, format="PNG", pnginfo=pnginfo)
            embedded.append(buf.getvalue())
        except Exception:
            embedded.append(raw)
    return embedded


def _fetch_output_images(output):
    """Remote output images as raw bytes — downloads URLs, decodes base64."""
    payloads = output.get("images") or []
    if output.get("images_are_urls"):
        raws = []
        for url in payloads:
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
            except requests.RequestException as exc:
                _user_error(f"Hob Bridge: cannot download result image ({exc}).")
            raws.append(resp.content)
        return raws
    return [base64.b64decode(encoded) for encoded in payloads]


def _parse_remote_error(err):
    """RunPod may hand the worker's error dict back serialized (JSON or repr)."""
    if isinstance(err, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(err)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return err


def _format_remote_error(err):
    """Remote errors: plain strings, classified {kind, message, hint} dicts, or
    the SDK's {error_type, error_message} for exceptions raised in the handler."""
    err = _parse_remote_error(err)
    if isinstance(err, dict):
        parts = [err.get("kind") or err.get("error_type") or "error",
                 err.get("message") or err.get("error_message") or ""]
        hint = err.get("hint")
        if hint:
            parts.append(f"hint: {hint}")
        return "Hob Bridge: " + " — ".join(str(part) for part in parts if part)
    return f"Hob Bridge: {err}"


def _checkpoint_name(entry):
    """The checkpoint id to send: A1111's name (relative path), an alias override_settings resolves."""
    return entry.get("name") or entry.get("model_name")


def _checkpoint_names(inventory):
    """Sorted unique checkpoint names for the dropdown."""
    names = {_checkpoint_name(entry) for entry in inventory.get("checkpoints") or []
             if isinstance(entry, dict)}
    return sorted(name for name in names if name)


def _effective_prompts(p, prompt_styles=None):
    """Style-expanded prompt/negative — at script run() time p.prompt is raw.

    p.all_prompts is deliberately not used: it repeats prompts n_iter times,
    which the API would multiply again. Expansion is recomputed from
    p.prompt + p.styles, which is what setup_prompts() does internally.
    """
    if prompt_styles is None and _IN_WEBUI:
        prompt_styles = shared.prompt_styles
    styles = list(getattr(p, "styles", None) or [])
    if not styles:
        prompts = p.prompt if isinstance(p.prompt, list) else [p.prompt]
        negatives = p.negative_prompt if isinstance(p.negative_prompt, list) else [p.negative_prompt]
        return list(prompts), list(negatives)

    def _expand(texts, apply_fn):
        if isinstance(texts, list):
            return [apply_fn(t, styles) for t in texts]
        return [apply_fn(texts, styles)]

    return (_expand(p.prompt, prompt_styles.apply_styles_to_prompt),
            _expand(p.negative_prompt, prompt_styles.apply_negative_styles_to_prompt))


def _build_params(p, prompts, negatives, clip_skip=1, vae_name=None, controlnet_units=None):
    params = {
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "negative_prompt": negatives if len(negatives) > 1 else negatives[0],
        "steps": p.steps,
        "cfg_scale": p.cfg_scale,
        "width": p.width,
        "height": p.height,
        "sampler_name": p.sampler_name,
        "seed": p.seed,
        "subseed": getattr(p, "subseed", -1),
        "subseed_strength": getattr(p, "subseed_strength", 0),
        "batch_size": p.batch_size,
        "n_iter": p.n_iter,
        "send_images": True,
        "save_images": False,
    }
    scheduler = getattr(p, "scheduler", None)
    if scheduler:
        params["scheduler"] = scheduler
    if clip_skip and clip_skip != 1:
        params.setdefault("override_settings", {})["CLIP_stop_at_last_layers"] = clip_skip

    if getattr(p, "enable_hr", False):
        params.update({
            "enable_hr": True,
            "hr_scale": getattr(p, "hr_scale", 2.0),
            "hr_upscaler": getattr(p, "hr_upscaler", "Latent"),
            "hr_second_pass_steps": getattr(p, "hr_second_pass_steps", 0),
            "denoising_strength": getattr(p, "denoising_strength", 0.7),
        })
        for key in ("hr_resize_x", "hr_resize_y"):
            if getattr(p, key, 0):
                params[key] = getattr(p, key)

    override = dict(getattr(p, "override_settings", None) or {})
    if vae_name:
        override.setdefault("sd_vae", vae_name)
    if override:
        params.setdefault("override_settings", {}).update(override)

    init_images = getattr(p, "init_images", None)
    if init_images:
        params["init_images"] = [b for b in (_image_to_b64(img) for img in init_images) if b]
        params["resize_mode"] = getattr(p, "resize_mode", 0)
        params["denoising_strength"] = getattr(p, "denoising_strength", 0.75)
        if getattr(p, "initial_noise_multiplier", None) is not None:
            params["initial_noise_multiplier"] = p.initial_noise_multiplier
        mask = getattr(p, "image_mask", None)
        if isinstance(mask, (list, tuple)):
            mask = mask[0] if mask else None
        mask_b64 = _image_to_b64(mask)
        if mask_b64:
            params["mask"] = mask_b64
            params["mask_blur"] = getattr(p, "mask_blur", 4)
            params["inpainting_fill"] = getattr(p, "inpainting_fill", 1)
            # p.image_mask is not inverted yet at run() time — init() does that
            params["inpainting_mask_invert"] = int(getattr(p, "inpainting_mask_invert", 0) or 0)
            params["inpaint_full_res"] = bool(getattr(p, "inpaint_full_res", False))
            params["inpaint_full_res_padding"] = getattr(p, "inpaint_full_res_padding", 32)

    if controlnet_units:
        params["alwayson_scripts"] = {"controlnet": {"args": controlnet_units}}
    return params


# ---------------------------------------------------------------------------
# ControlNet: local units → standard A1111 alwayson_scripts payload
# ---------------------------------------------------------------------------

def _load_controlnet_external_code():
    global _controlnet_module
    if _controlnet_module != "unset":
        return _controlnet_module
    module = None
    # current controlnet keeps the implementation in internal_controlnet/;
    # older installs expose it through the extension's scripts/ package
    for candidate in ("scripts.external_code", "internal_controlnet.external_code"):
        try:
            mod = importlib.import_module(candidate)
            if hasattr(mod, "get_all_units_in_processing"):
                module = mod
                break
        except Exception:
            continue
    _controlnet_module = module
    return module


def _controlnet_script_present(p):
    runner = getattr(p, "scripts", None)
    for script in getattr(runner, "scripts", None) or []:
        try:
            if "controlnet" in script.title().lower():
                return True
        except Exception:
            continue
    return False


def _enum_value(value):
    return getattr(value, "value", value)


# Forwarded by name, enums by value: the worker's ControlNet rebuilds each unit
# with ControlNetUnit.from_dict (pydantic, extra fields silently ignored), so the
# names must match its model; anything omitted takes the same default as locally.
_CN_FIELDS = ("input_mode", "module", "model", "weight", "resize_mode", "low_vram",
              "processor_res", "threshold_a", "threshold_b", "guidance_start", "guidance_end",
              "pixel_perfect", "control_mode", "inpaint_crop_input_image", "hr_option",
              "save_detected_map", "advanced_weighting", "pulid_mode", "union_control_type")
_CN_IMAGE_FIELDS = ("image", "mask", "effective_region_mask")


def _structured_b64(value):
    """Encode the images inside ControlNet's image shapes ({'image', 'mask'} pairs,
    multi-input lists) without changing the shape: the remote parses it exactly
    as the local unit would be parsed."""
    if isinstance(value, dict):
        return {key: _structured_b64(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_structured_b64(item) for item in value]
    return _image_to_b64(value)


def _controlnet_unit_to_api(unit):
    """A local ControlNetUnit (object or dict) → the dict controlnet's API accepts."""
    if isinstance(unit, dict):
        get = lambda key, default=None: unit.get(key, default)  # noqa: E731
    else:
        get = lambda key, default=None: getattr(unit, key, default)  # noqa: E731
    if not get("enabled", True):
        return None
    if get("model", "None") in (None, "", "None"):
        return None
    if _enum_value(get("input_mode")) == "batch":
        _user_error("Hob Bridge: ControlNet batch mode reads a local folder and cannot run "
                    "remotely — use a single image or multi-inputs.")
    api_unit = {"enabled": True}  # the API default is False
    for key in _CN_FIELDS:
        value = get(key)
        if value is not None:
            api_unit[key] = _enum_value(value)
    for key in _CN_IMAGE_FIELDS:
        value = _structured_b64(get(key))
        if value is not None:
            api_unit[key] = value
    return api_unit


def _get_controlnet_units(p):
    external_code = _load_controlnet_external_code()
    if external_code is None:
        if _controlnet_script_present(p):
            _user_error("Hob Bridge: ControlNet is installed locally but its API module "
                        "could not be imported; refusing to send a job that would "
                        "ignore your ControlNet settings.")
        return []
    try:
        raw_units = external_code.get_all_units_in_processing(p) or []
    except Exception as exc:
        _user_error(f"Hob Bridge: cannot read the local ControlNet units ({exc}).")
    units = []
    for unit in raw_units:
        api_unit = _controlnet_unit_to_api(unit)
        if api_unit:
            units.append(api_unit)
    return units


# ---------------------------------------------------------------------------
# Remote inventory & preflight
# ---------------------------------------------------------------------------

def _lora_names(inventory):
    names = set()
    for lora in inventory.get("loras") or []:
        for key in ("name", "alias"):
            if lora.get(key):
                names.add(lora[key])
    return names


def _extract_lora_names(prompt_text):
    return [name.strip() for name in re.findall(r"<lora:([^:>]+)(?::[^>]*)?>", prompt_text)]


def _missing_loras(inventory, prompt_text):
    if inventory.get("loras") is None:
        return []  # list unavailable (old worker) — let the remote decide
    available = _lora_names(inventory)
    return [name for name in dict.fromkeys(_extract_lora_names(prompt_text)) if name not in available]


def _checkpoint_aliases(entry):
    """What a user may type for a checkpoint: its name, model_name, file name, or stem."""
    name = entry.get("name") or ""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    return {name, entry.get("model_name"), base, os.path.splitext(base)[0]} - {None, ""}


def _resolve_checkpoint(inventory, name):
    entries = inventory.get("checkpoints")
    if entries is None:
        return name  # list unavailable (old worker) — let the remote decide
    for entry in entries:
        if name in _checkpoint_aliases(entry):
            return _checkpoint_name(entry)
    if not entries:
        _user_error("Hob Bridge: no checkpoints found on the remote volume.")
    available = ", ".join(_checkpoint_names(inventory)[:8])
    _user_error(f"Hob Bridge: checkpoint '{name}' not found on the remote volume. "
                f"Available: {available}")


def _vae_missing(inventory, vae):
    vaes = inventory.get("vae")
    if not vae or vae in ("Automatic", "None") or vaes is None:
        return False
    return vae not in {entry.get("model_name") for entry in vaes}


def _preflight(checkpoint, prompt_text, vae=None):
    """Fail fast — before paying for a job — if referenced assets are missing remotely."""
    inventory = _fetch_inventory()
    if checkpoint:
        checkpoint = _resolve_checkpoint(inventory, checkpoint)
    missing = _missing_loras(inventory, prompt_text)
    if missing:
        _user_error("Hob Bridge: LoRA not found on the remote volume: " + ", ".join(missing)
                    + " — sync your models to the network volume, or tick 'Skip remote asset check'.")
    if _vae_missing(inventory, vae):
        _user_error(f"Hob Bridge: VAE '{vae}' not found on the remote volume (models/VAE) — "
                    "sync it, pick another VAE, or tick 'Skip remote asset check'.")
    return checkpoint


def _fetch_inventory(force=False):
    now = time.time()
    if not force and _inventory_cache["data"] and now - _inventory_cache["ts"] < _INVENTORY_TTL:
        return _inventory_cache["data"]
    if _setting("hob_bridge_mode", "serverless") == "serverless":
        inventory = _serverless_request("inventory", {})
    else:
        inventory = _direct_inventory()
    inventory.setdefault("type", "inventory")
    _inventory_cache.update(ts=now, data=inventory)
    return inventory


def _direct_inventory():
    base = _base_url(_setting("hob_bridge_endpoint_url"))
    if not base:
        _user_error("Hob Bridge: endpoint URL is not set (Settings → Hob Bridge).")

    def fetch(path):
        resp = requests.get(f"{base}{path}", timeout=60)
        resp.raise_for_status()
        return resp.json()

    return _worker._collect_inventory(fetch)


# ---------------------------------------------------------------------------
# Setup check (first-run wizard) — items are (passed, label, detail);
# passed None = informational
# ---------------------------------------------------------------------------

def _check_settings():
    """Free checks, no network."""
    mode = _setting("hob_bridge_mode", "serverless")
    base = _base_url(_setting("hob_bridge_endpoint_url"))
    if not base:
        items = [(False, "Endpoint URL", "not set — Settings → Hob Bridge")]
    elif not base.startswith(("http://", "https://")):
        items = [(False, "Endpoint URL", f"'{base}' must start with http:// or https://")]
    elif mode == "serverless" and not re.fullmatch(r"https://api\.runpod\.ai/v2/[\w-]+", base):
        items = [(None, "Endpoint URL", f"{base} — not the usual https://api.runpod.ai/v2/<endpoint-id>")]
    else:
        items = [(True, "Endpoint URL", f"{base} ({mode} mode)")]
    if mode == "serverless":
        key = _setting("hob_bridge_api_key", "").strip()
        items.append((bool(key), "RunPod API key", "set" if key else "not set — Settings → Hob Bridge"))
    return items


def _check_health():
    """RunPod's /health validates endpoint id and API key without starting a worker."""
    base = _base_url(_setting("hob_bridge_endpoint_url"))
    headers = {"Authorization": f"Bearer {_setting('hob_bridge_api_key', '').strip()}"}
    try:
        resp = requests.get(f"{base}/health", headers=headers, timeout=15)
    except requests.RequestException as exc:
        return False, "RunPod endpoint", f"unreachable ({exc})"
    if resp.status_code in (401, 403):
        return False, "RunPod endpoint", "API key rejected"
    if resp.status_code == 404:
        return False, "RunPod endpoint", "endpoint id not found"
    if not resp.ok:
        return False, "RunPod endpoint", f"HTTP {resp.status_code}"
    try:
        data = resp.json()
    except ValueError:
        data = {}
    workers, jobs = data.get("workers") or {}, data.get("jobs") or {}
    return True, "RunPod endpoint", (f"key accepted — {workers.get('idle', 0)} idle / "
                                     f"{workers.get('running', 0)} running workers, "
                                     f"{jobs.get('inQueue', 0)} queued jobs")


def _check_inventory(inventory):
    checkpoints = inventory.get("checkpoints")
    if checkpoints:
        items = [(True, "Volume: checkpoints", f"{len(checkpoints)} found")]
    elif checkpoints is None:
        items = [(False, "Volume: checkpoints", "the worker could not list them")]
    else:
        items = [(False, "Volume: checkpoints", "none — is the network volume mounted at /workspace, "
                                                "with models/Stable-diffusion/ on it?")]
    for key, label in (("loras", "LoRAs"), ("vae", "VAEs"), ("embeddings", "embeddings"),
                       ("upscalers", "upscalers")):
        value = inventory.get(key)
        items.append((True, f"Volume: {label}", str(len(value))) if value
                     else (None, f"Volume: {label}", "none (optional)"))
    vae = _local_vae_name()
    if vae:
        missing = _vae_missing(inventory, vae)
        items.append((not missing, "Local VAE on the volume", vae + (" — missing" if missing else "")))
    controlnet = inventory.get("controlnet")
    if controlnet:
        items.append((True, "ControlNet on the worker", f"{len(controlnet)} models"))
    elif _load_controlnet_external_code() is not None:
        items.append((False, "ControlNet on the worker", "installed locally, but the worker has "
                      + ("no models (models/ControlNet/)" if controlnet is not None else "no ControlNet")))
    else:
        items.append((None, "ControlNet", "not installed locally — skipped"))
    return items


def _render_checks(items, footer=""):
    marks = {True: "🟢", False: "🔴", None: "⚪"}
    lines = [f"{marks[passed]} {label}" + (f" — {detail}" if detail else "")
             for passed, label, detail in items]
    return "\n".join(lines + ([footer] if footer else []))


def _setup_check():
    """UI handler (generator): the first-run checklist, one 🟢/🔴 line per item.

    Free checks come first; the inventory job, which may pay a cold start, runs
    only once they pass. Its checkpoint list also feeds the dropdown.
    """
    items = _check_settings()
    ok = lambda: all(item[0] is not False for item in items)  # noqa: E731 (None = informational)
    if _setting("hob_bridge_mode", "serverless") == "serverless" and ok():
        items.append(_check_health())
    if not ok():
        yield _render_checks(items, "Fix the red items, then run the check again."), gr.update()
        return
    yield _render_checks(items + [(None, "Worker", "⏳ asking for its inventory (may pay a cold start)…")]), gr.update()
    started = time.time()
    try:
        inventory = _fetch_inventory(force=True)
    except Exception as exc:
        yield _render_checks(items + [(False, "Worker", str(exc) or type(exc).__name__)]), gr.update()
        return
    items.append((True, "Worker", f"answered in {time.time() - started:.1f}s (cold start included)"))
    items += _check_inventory(inventory)
    footer = "All set." if ok() else "Fix the red items, then run the check again."
    yield _render_checks(items, footer), gr.update(choices=_checkpoint_names(inventory))


# ---------------------------------------------------------------------------
# Warm-up: load the checkpoint on the worker before a session
# ---------------------------------------------------------------------------

_WARM_LABELS = {"IN_QUEUE": "⏳ pending — waiting for a worker (a cold start can take minutes)",
                "IN_PROGRESS": "🔄 loading the checkpoint on the worker…"}


def _warm_checkpoint(checkpoint):
    """UI handler (generator): pending → loading → ready."""
    if not checkpoint:
        yield "Pick or type a remote checkpoint first."
        return
    statuses, outcome = queue.Queue(), {}

    def work():
        try:
            outcome["name"] = _resolve_checkpoint(_fetch_inventory(), checkpoint)
            outcome["output"] = _send_job("warm", {"checkpoint": outcome["name"]}, on_status=statuses.put)
        except Exception as exc:
            outcome["error"] = exc
        finally:
            statuses.put(None)

    started = time.time()
    threading.Thread(target=work, daemon=True).start()
    yield "⏳ pending — submitting"
    shown = None
    while (status := statuses.get()) is not None:
        if status in _WARM_LABELS and status != shown:
            shown = status
            yield _WARM_LABELS[status]
    if "error" in outcome:
        yield f"🔴 {outcome['error'] or type(outcome['error']).__name__}"
        return
    load = (outcome.get("output") or {}).get("seconds")
    yield (f"🟢 ready — {outcome['name']} loaded" + (f" in {load}s" if load is not None else "")
           + f" ({time.time() - started:.0f}s in total). It stays loaded while the worker is up: "
           "see the endpoint's idle timeout.")


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _cancel_job(base, headers, job_id):
    try:
        requests.post(f"{base}/cancel/{job_id}", headers=headers, timeout=15)
    except requests.RequestException:
        pass


def _serverless_request(kind, params, on_status=None):
    base = _base_url(_setting("hob_bridge_endpoint_url"))
    if not base:
        _user_error("Hob Bridge: endpoint URL is not set (Settings → Hob Bridge).")
    api_key = _setting("hob_bridge_api_key", "").strip()
    if not api_key:
        _user_error("Hob Bridge: RunPod API key is not set (Settings → Hob Bridge).")
    headers = {"Authorization": f"Bearer {api_key}"}

    total = _total_timeout()
    body = {
        "input": {"type": kind, "params": params, "timeout": total},
        # server-side copy of the client's budget: RunPod ends the job (and the
        # worker gets a stop signal) even if this client dies before /cancel
        "policy": {"executionTimeout": max(total, 5) * 1000, "ttl": (max(total, 10) + 60) * 1000},
    }
    try:
        resp = requests.post(f"{base}/run", json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        job_id = (resp.json() or {}).get("id")
    except (requests.RequestException, ValueError) as exc:
        _user_error(f"Hob Bridge: cannot submit job ({exc}).")
    if not job_id:
        _user_error("Hob Bridge: endpoint accepted the request but returned no job id.")

    _set_progress("job submitted — queued")
    deadline = time.time() + total
    last_ok = time.time()
    while True:
        if _interrupted():
            _cancel_job(base, headers, job_id)
            raise JobInterrupted()
        if time.time() > deadline:
            _cancel_job(base, headers, job_id)
            _user_error(f"Hob Bridge: timed out after {total} s — job {job_id} cancelled.")
        try:
            resp = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            # a network blip must not orphan a running (billed) job: retry, and
            # cancel it before giving up
            if time.time() - last_ok > _POLL_GRACE:
                _cancel_job(base, headers, job_id)
                _user_error(f"Hob Bridge: status poll failing for {_POLL_GRACE} s ({exc}) — "
                            f"job {job_id} cancelled.")
            _set_progress(f"status poll failed, retrying ({exc.__class__.__name__})")
            time.sleep(_POLL_INTERVAL)
            continue
        last_ok = time.time()
        status = data.get("status")
        if on_status:
            on_status(status)
        if status == "COMPLETED":
            output = data.get("output") or {}
            if isinstance(output, dict) and output.get("error"):
                _user_error(_format_remote_error(output["error"]))
            return output
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            # the SDK moves a returned {"error": ...} out of `output` into `error`
            error = data.get("error")
            _user_error(_format_remote_error(error) if error
                        else f"Hob Bridge: job {status}: {data.get('output') or 'no details'}")
        if status == "IN_QUEUE":
            _set_progress("queued — waiting for a worker (a cold start can take a couple of minutes)")
        elif status == "IN_PROGRESS":
            _show_remote_progress(data.get("output") or "running on the worker")
        time.sleep(_POLL_INTERVAL)


def _direct_request(kind, params):
    base = _base_url(_setting("hob_bridge_endpoint_url"))
    if not base:
        _user_error("Hob Bridge: endpoint URL is not set (Settings → Hob Bridge).")
    if kind == "warm":
        started = time.time()
        error = _worker._warm(base, params["checkpoint"], _total_timeout())
        if error:
            _user_error(_format_remote_error(_worker._classify_error(error)))
        return {"type": "warm", "checkpoint": params["checkpoint"],
                "seconds": round(time.time() - started, 1)}
    path = "/sdapi/v1/img2img" if kind == "img2img" else "/sdapi/v1/txt2img"
    _set_progress("generating on the remote pod…")
    data, error = _worker._call_a1111(base, path, params, _total_timeout(),
                                      on_progress=_show_remote_progress,
                                      check_stop=_raise_if_interrupted)
    if error:
        _user_error(_format_remote_error(_worker._classify_error(error)))
    return data


def _send_job(kind, params, on_status=None):
    if _setting("hob_bridge_mode", "serverless") == "serverless":
        return _serverless_request(kind, params, on_status)
    if on_status:
        on_status("IN_PROGRESS")  # a pod has no queue
    return _direct_request(kind, params)


# ---------------------------------------------------------------------------
# Local settings read at generation time
# ---------------------------------------------------------------------------

def _clip_skip():
    if not _IN_WEBUI:
        return 1
    try:
        return int(shared.opts.CLIP_stop_at_last_layers)
    except (AttributeError, TypeError, ValueError):
        return 1


def _local_vae_name():
    if not _IN_WEBUI:
        return None
    vae = getattr(shared.opts, "sd_vae", "Automatic")
    return vae if vae not in ("", "Automatic", "None") else None


# run() bypasses process_images() and ControlNet's postprocess(), so the three
# helpers below replay what those would have done with the images locally.

def _save_samples(p, images, infotexts, info):
    """'Always save all generated images'."""
    if not shared.opts.samples_save or getattr(p, "do_not_save_samples", False):
        return
    seeds = info.get("all_seeds") or []
    prompts = info.get("all_prompts") or []
    for index, (image, text) in enumerate(zip(images, infotexts)):
        webui_images.save_image(
            image, p.outpath_samples, "",
            seeds[index] if index < len(seeds) else p.seed,
            prompts[index] if index < len(prompts) else p.prompt,
            shared.opts.samples_format, info=text or None, p=p)


def _batch_grid(p, images, text, info):
    """The batch grid, as process_images_inner() builds it: saved per 'grid_save',
    returned for the gallery per 'return_grid' (else None).

    ponytail: the grid infotext is image 0's; A1111 uses the main prompt, which
    only differs when prompts vary per image (e.g. prompt matrix, wildcards).
    """
    opts = shared.opts
    too_few = len(images) < 2 and opts.grid_only_if_multiple
    if not (opts.return_grid or opts.grid_save) or getattr(p, "do_not_save_grid", False) or too_few:
        return None
    grid = webui_images.image_grid(images, p.batch_size)
    if opts.grid_save:
        webui_images.save_image(
            grid, p.outpath_grids, "grid", (info.get("all_seeds") or [p.seed])[0],
            (info.get("all_prompts") or [p.prompt])[0], opts.grid_format, info=text or None,
            short_filename=not opts.grid_extended_filename, p=p, grid=True)
    if not opts.return_grid:
        return None
    if opts.enable_pnginfo and text:
        grid.info["parameters"] = text
    return grid


def _save_detected_maps(p, maps, units):
    """ControlNet's 'Allow detectmap auto saving' for the maps the worker returned.

    ponytail: the API returns bare images, so a map's module (its folder and file
    name, as ControlNet saves them) is only known when one module produced them
    all; with several, maps go to a shared 'controlnet' folder.
    """
    opts = shared.opts.data
    if not maps or not opts.get("control_net_detectmap_autosaving", False):
        return
    modules = {unit.get("module") for unit in units if unit.get("save_detected_map", True)}
    modules.discard("none")  # ControlNet never autosaves "none" maps
    if not modules:
        return
    module = modules.pop() if len(modules) == 1 else "controlnet"
    directory = os.path.join(opts.get("control_net_detectedmap_dir", ""), module)
    if not os.path.isabs(directory):
        directory = os.path.join(p.outpath_samples, directory)
    os.makedirs(directory, exist_ok=True)
    for image in maps:
        webui_images.save_image(image, directory, module)


def _similarity(a, b):
    """Pixel similarity in % (100 = identical): 100 − mean absolute RGB difference.

    ponytail: a pixel metric, not a perceptual one; SSIM/LPIPS would need new deps.
    """
    a = a.convert("RGB")
    b = b.convert("RGB").resize(a.size)
    mean = ImageStat.Stat(ImageChops.difference(a, b)).mean
    return 100 - sum(mean) / len(mean) / 2.55


def _ab_compare(p, remote, remote_images, info, checkpoint):
    """Dual mode: run the same job locally with the remote's resolved seeds; the
    local images follow the remote ones, with a similarity score per pair."""
    p.seed, p.subseed = info.get("seed", p.seed), info.get("subseed", p.subseed)
    if checkpoint and sd_models.checkpoint_aliases.get(checkpoint):
        p.override_settings = dict(p.override_settings or {}, sd_model_checkpoint=checkpoint)
    try:
        local = process_images(p)
    except Exception as exc:  # keep the (paid) remote result
        remote.comments += f"A/B: the local run failed ({exc})\n"
        return
    local_images = local.images[local.index_of_first_image:][:len(remote_images)]
    scores = ", ".join(f"#{i + 1} {_similarity(a, b):.1f}%"
                       for i, (a, b) in enumerate(zip(remote_images, local_images)))
    remote.images = remote.images + local.images
    remote.infotexts = remote.infotexts + local.infotexts
    remote.comments += (f"A/B: remote ({info.get('sd_model_name', '?')}) vs local "
                        f"({local.sd_model_name}), seed {p.seed} — pixel similarity: {scores or 'n/a'}\n")


# ---------------------------------------------------------------------------
# A1111 integration
# ---------------------------------------------------------------------------

def _on_ui_settings():
    section = ("hob_bridge", "Hob Bridge")
    shared.opts.add_option(
        "hob_bridge_mode",
        shared.OptionInfo("serverless", "Endpoint mode", gr.Radio,
                          {"choices": ["serverless", "direct"]}, section=section),
    )
    shared.opts.add_option(
        "hob_bridge_endpoint_url",
        shared.OptionInfo("", "Endpoint URL", gr.Textbox,
                          {"placeholder": "https://api.runpod.ai/v2/<endpoint-id> or http://<pod>:7860"},
                          section=section),
    )
    shared.opts.add_option(
        "hob_bridge_api_key",
        shared.OptionInfo("", "RunPod API key (serverless mode)", gr.Textbox,
                          {"type": "password"}, section=section),
    )
    shared.opts.add_option(
        "hob_bridge_timeout",
        shared.OptionInfo(600, "Total job timeout (s), cold starts included", gr.Number,
                          {"precision": 0}, section=section),
    )


if _IN_WEBUI:

    class HobBridge(scripts.Script):
        def title(self):
            return "Hob Bridge (run on RunPod)"

        def show(self, is_img2img):
            return True

        def ui(self, is_img2img):
            checkpoint = gr.Dropdown(
                label="Remote checkpoint (optional)",
                info="Populated from the worker's volume by 'Run setup check'; "
                     "you can also type a file name directly. Empty = worker default.",
                choices=[],
                allow_custom_value=True,
            )
            with gr.Row():
                warm_btn = gr.Button("Warm up checkpoint")
                warm_out = gr.Textbox(label="Warm-up", interactive=False, show_label=False,
                                      placeholder="Load it on the worker now, before a session")
            warm_btn.click(_warm_checkpoint, inputs=[checkpoint], outputs=[warm_out])
            skip_check = gr.Checkbox(
                label="Skip remote asset check",
                info="Send the job without verifying checkpoint/LoRA/VAE presence on the volume.",
                value=False,
            )
            dual = gr.Checkbox(
                label="A/B: also generate locally",
                info="Re-runs the same prompt and seed on this machine after the remote job and "
                     "reports a similarity per image. Needs a local model; slow without a GPU.",
                value=False,
            )
            # first run: open with the free settings checks already filled in
            settings = _check_settings()
            configured = all(passed is not False for passed, _, _ in settings)
            with gr.Accordion("Hob Bridge setup & status", open=not configured):
                check_out = gr.Textbox(label="Checklist", lines=10, interactive=False,
                                       value=_render_checks(settings, "Press 'Run setup check' — the worker "
                                                                      "check may pay a cold start."))
                check_btn = gr.Button("Run setup check")
                check_btn.click(_setup_check, inputs=None, outputs=[check_out, checkpoint])
            return [checkpoint, skip_check, dual]

        def run(self, p, checkpoint, skip_check, dual=False):
            kind = "img2img" if isinstance(p, StableDiffusionProcessingImg2Img) else "txt2img"
            prompts, negatives = _effective_prompts(p)
            units = _get_controlnet_units(p)
            params = _build_params(p, prompts, negatives, clip_skip=_clip_skip(),
                                   vae_name=_local_vae_name(), controlnet_units=units)
            try:
                if not skip_check:
                    vae = params.get("override_settings", {}).get("sd_vae")
                    checkpoint = _preflight(checkpoint, "\n".join(prompts), vae)
                if checkpoint:
                    params.setdefault("override_settings", {})["sd_model_checkpoint"] = checkpoint
                output = _send_job(kind, params)
            except JobInterrupted:
                return Processed(p, [], seed=p.seed, info="Hob Bridge: remote job cancelled.")

            info = _parse_info(output.get("info"))
            raw_images = _fetch_output_images(output)
            if not raw_images:
                _user_error("Hob Bridge: the remote endpoint returned no images.")
            # generated images carry an infotext; ControlNet detected maps follow them
            infotexts = list(info.get("infotexts") or [])[:len(raw_images)]
            n_generated = len(infotexts) or len(raw_images)
            texts = (infotexts + [""] * n_generated)[:n_generated]
            images = [Image.open(io.BytesIO(raw))
                      for raw in _embed_geninfo(raw_images, infotexts)]
            generated, maps = images[:n_generated], images[n_generated:]

            _save_samples(p, generated, texts, info)
            _save_detected_maps(p, maps, units)
            if shared.opts.data.get("control_net_no_detectmap", False):
                maps = []
            grid = _batch_grid(p, generated, texts[0], info)
            gallery = ([grid] if grid else []) + generated + maps
            gallery_texts = ([texts[0]] if grid else []) + texts + [""] * len(maps)
            processed = Processed(p, gallery, seed=info.get("seed", p.seed), info=texts[0],
                                  subseed=info.get("subseed", -1),
                                  all_prompts=info.get("all_prompts"),
                                  all_negative_prompts=info.get("all_negative_prompts"),
                                  all_seeds=info.get("all_seeds"),
                                  all_subseeds=info.get("all_subseeds"),
                                  index_of_first_image=1 if grid else 0,
                                  infotexts=gallery_texts)
            if dual:
                _ab_compare(p, processed, generated, info, checkpoint)
            return processed

    script_callbacks.on_ui_settings(_on_ui_settings)
