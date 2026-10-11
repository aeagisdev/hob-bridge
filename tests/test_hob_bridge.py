"""Unit tests for Hob Bridge's pure helpers — no A1111 install required.

Run with pytest, or plain: python3 tests/test_hob_bridge.py
"""
import asyncio
import base64
import contextlib
import http.server
import io
import json
import os
import sys
import threading
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

import hob_bridge  # noqa: E402
import handler as hob_worker  # noqa: E402  (worker module: no side effects at import)
from PIL import Image  # noqa: E402


class FakeStyles:
    def apply_styles_to_prompt(self, prompt, styles):
        return f"[{'/'.join(styles)}] {prompt}"

    def apply_negative_styles_to_prompt(self, negative, styles):
        return f"[N:{'/'.join(styles)}] {negative}"


def _fake_p(**overrides):
    p = types.SimpleNamespace(
        prompt="a cozy room", negative_prompt="ugly",
        styles=[], steps=30, cfg_scale=7, width=1024, height=1024,
        sampler_name="DPM++ 2M", scheduler="karras", seed=-1, subseed=-1,
        subseed_strength=0, batch_size=1, n_iter=1, enable_hr=False,
        override_settings=None, init_images=None,
    )
    for key, value in overrides.items():
        setattr(p, key, value)
    return p


def test_base_url_normalization():
    assert hob_bridge._base_url("https://api.runpod.ai/v2/abc/") == "https://api.runpod.ai/v2/abc"
    assert hob_bridge._base_url("https://api.runpod.ai/v2/abc/runsync") == "https://api.runpod.ai/v2/abc"
    assert hob_bridge._base_url(" https://api.runpod.ai/v2/abc/run ") == "https://api.runpod.ai/v2/abc"
    assert hob_bridge._base_url("") == ""
    assert hob_bridge._base_url(None) == ""


def test_extract_lora_names():
    text = "cozy room <lora:detail:0.7> and <lora: styles :1> and <lyco:other>"
    assert hob_bridge._extract_lora_names(text) == ["detail", "styles"]


def test_missing_loras():
    inv = {"loras": [{"name": "detail", "alias": "d"}]}
    assert hob_bridge._missing_loras(inv, "<lora:detail> text") == []
    assert hob_bridge._missing_loras(inv, "<lora:d> text") == []  # alias counts
    assert hob_bridge._missing_loras(inv, "<lora:absent>") == ["absent"]
    assert hob_bridge._missing_loras({"loras": None}, "<lora:absent>") == []  # unknown
    assert hob_bridge._missing_loras({"loras": []}, "<lora:absent>") == ["absent"]  # empty volume


def test_effective_prompts_applies_styles():
    p = _fake_p(styles=["photo"])
    prompts, negatives = hob_bridge._effective_prompts(p, FakeStyles())
    assert prompts == ["[photo] a cozy room"]
    assert negatives == ["[N:photo] ugly"]


def test_effective_prompts_without_styles():
    p = _fake_p()
    prompts, negatives = hob_bridge._effective_prompts(p, FakeStyles())
    assert prompts == ["a cozy room"]
    assert negatives == ["ugly"]


def test_build_params_basics():
    p = _fake_p(override_settings={"sd_vae": "myvae.safetensors"})
    params = hob_bridge._build_params(p, ["a cozy room"], ["ugly"], clip_skip=2, vae_name="autovae")
    assert params["prompt"] == "a cozy room"
    assert params["scheduler"] == "karras"
    assert params["send_images"] is True
    # clip skip and VAE travel via override_settings; p.override_settings wins
    assert params["override_settings"]["CLIP_stop_at_last_layers"] == 2
    assert params["override_settings"]["sd_vae"] == "myvae.safetensors"


def test_build_params_hr_fix():
    p = _fake_p(enable_hr=True, hr_scale=1.5, hr_upscaler="ESRGAN_4x", hr_resize_x=1536, hr_resize_y=0)
    params = hob_bridge._build_params(p, ["a"], ["b"])
    assert params["enable_hr"] is True
    assert params["hr_upscaler"] == "ESRGAN_4x"
    assert params["hr_resize_x"] == 1536
    assert "hr_resize_y" not in params


def test_build_params_img2img_b64_init_images():
    try:
        from PIL import Image
    except ImportError:
        return
    img = Image.new("RGB", (8, 8), "white")
    p = _fake_p(init_images=[img], denoising_strength=0.6, resize_mode=1, image_mask=None)
    params = hob_bridge._build_params(p, ["a cozy room"], ["ugly"])
    assert params["denoising_strength"] == 0.6
    assert len(params["init_images"]) == 1
    decoded = base64.b64decode(params["init_images"][0])
    assert Image.open(io.BytesIO(decoded)).size == (8, 8)


def test_controlnet_unit_to_api_dict():
    unit = {"enabled": True, "module": "canny", "model": "control_v11p_sd15_canny",
            "weight": 0.8, "image": None}
    api_unit = hob_bridge._controlnet_unit_to_api(unit)
    # unset fields are omitted, so the remote applies the same defaults as locally
    assert api_unit == {"enabled": True, "module": "canny",
                        "model": "control_v11p_sd15_canny", "weight": 0.8}


def test_controlnet_unit_to_api_object_and_enums():
    class Enum:  # stands in for ControlNet's str-valued enums
        def __init__(self, value):
            self.value = value

    unit = types.SimpleNamespace(
        enabled=True, input_mode=Enum("simple"), module="depth", model="control_v11f1p_sd15_depth",
        weight=1.0, image=None, resize_mode=Enum("Resize and Fill"), low_vram=True,
        control_mode=Enum("ControlNet is more important"), save_detected_map=False, hr_option=None,
    )
    api_unit = hob_bridge._controlnet_unit_to_api(unit)
    # field names of the pinned ControlNetUnit model (from_dict ignores unknown keys)
    assert api_unit["resize_mode"] == "Resize and Fill"
    assert api_unit["control_mode"] == "ControlNet is more important"
    assert api_unit["input_mode"] == "simple"
    assert api_unit["low_vram"] is True
    assert api_unit["save_detected_map"] is False
    assert "hr_option" not in api_unit


def test_controlnet_unit_keeps_image_shapes():
    rgb, rgba = Image.new("RGB", (8, 8)), Image.new("RGBA", (8, 8))
    pair = hob_bridge._controlnet_unit_to_api(
        {"model": "m", "image": {"image": rgb, "mask": rgba}})["image"]
    assert set(pair) == {"image", "mask"}
    assert Image.open(io.BytesIO(base64.b64decode(pair["mask"]))).mode == "RGBA"
    merged = hob_bridge._controlnet_unit_to_api(
        {"model": "m", "image": [{"image": rgb}, {"image": rgba}], "mask": None})
    assert [set(item) for item in merged["image"]] == [{"image"}, {"image"}]
    assert "mask" not in merged


def test_controlnet_batch_mode_refused():
    try:
        hob_bridge._controlnet_unit_to_api({"model": "m", "input_mode": "batch"})
        raise AssertionError("expected an error")
    except RuntimeError as exc:
        assert "batch" in str(exc)


def test_build_params_inpaint_invert():
    p = _fake_p(init_images=[Image.new("RGB", (8, 8))], image_mask=Image.new("L", (8, 8), 255),
                inpainting_mask_invert=1, initial_noise_multiplier=0.9)
    params = hob_bridge._build_params(p, ["a"], ["b"])
    assert params["inpainting_mask_invert"] == 1
    assert params["initial_noise_multiplier"] == 0.9


def test_controlnet_unit_skips_disabled_and_none_model():
    assert hob_bridge._controlnet_unit_to_api({"enabled": False, "model": "x"}) is None
    assert hob_bridge._controlnet_unit_to_api({"enabled": True, "model": "None"}) is None
    assert hob_bridge._controlnet_unit_to_api({"enabled": True, "model": None}) is None


def test_build_params_with_controlnet():
    p = _fake_p()
    units = [{"module": "canny", "model": "control_v11p_sd15_canny"}]
    params = hob_bridge._build_params(p, ["a"], ["b"], controlnet_units=units)
    assert params["alwayson_scripts"] == {"controlnet": {"args": units}}


def test_collect_inventory_checkpoint_names():
    def fetch(path):
        if path == "/sdapi/v1/sd-models":
            return [{"title": "sdxl/juggs.safetensors [abcdef1234]", "model_name": "sdxl_juggs",
                     "filename": "/workspace/models/Stable-diffusion/sdxl/juggs.safetensors"},
                    {"title": "plain.ckpt", "model_name": "plain", "filename": "/w/plain.ckpt"}]
        raise IOError("endpoint unavailable")
    inv = hob_worker._collect_inventory(fetch)
    assert [c["name"] for c in inv["checkpoints"]] == ["sdxl/juggs.safetensors", "plain.ckpt"]
    assert inv["loras"] is None


def test_resolve_checkpoint():
    inv = {"checkpoints": [{"name": "sdxl/juggs.safetensors", "model_name": "sdxl_juggs",
                            "filename": "/workspace/models/Stable-diffusion/sdxl/juggs.safetensors"}]}
    # every spelling resolves to A1111's name — a checkpoint_aliases key, unlike the abs filename
    for typed in ("juggs", "juggs.safetensors", "sdxl/juggs.safetensors", "sdxl_juggs"):
        assert hob_bridge._resolve_checkpoint(inv, typed) == "sdxl/juggs.safetensors", typed
    # unknown inventory (old worker) → pass through untouched
    assert hob_bridge._resolve_checkpoint({"checkpoints": None}, "whatever") == "whatever"
    # present inventory but no match → error, not a doomed paid job
    try:
        hob_bridge._resolve_checkpoint(inv, "missing")
        raise AssertionError("expected an error for a missing checkpoint")
    except RuntimeError:
        pass


def test_classify_error_oom():
    err = hob_worker._classify_error(
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.5 GiB")
    assert err["kind"] == "gpu OOM"
    assert "batch" in err["hint"]


def test_classify_error_model_missing():
    err = hob_worker._classify_error("Checkpoint not found: juggs.safetensors")
    assert err["kind"] == "model missing"
    assert "volume" in err["hint"]


def test_classify_error_timeout():
    err = hob_worker._classify_error(
        "HTTPSConnectionPool(host='127.0.0.1'): Read timed out. (read timeout=900)")
    assert err["kind"] == "timeout"


def test_classify_error_unreachable():
    err = hob_worker._classify_error(
        "HTTPConnectionPool(host='127.0.0.1', port=7860): Connection refused")
    assert err["kind"] == "a1111 unreachable"


def test_classify_error_unknown():
    assert hob_worker._classify_error("")["kind"] == "error"


def test_format_remote_error_dict():
    msg = hob_bridge._format_remote_error(
        {"kind": "gpu OOM", "message": "CUDA out of memory", "hint": "lower batch size"})
    assert "gpu OOM" in msg and "CUDA out of memory" in msg and "lower batch size" in msg


def test_format_remote_error_serialized():
    err = {"kind": "gpu OOM", "message": "CUDA out of memory", "hint": "lower batch size"}
    for serialized in (json.dumps(err), repr(err)):
        msg = hob_bridge._format_remote_error(serialized)
        assert "gpu OOM" in msg and "hint: lower batch size" in msg, msg
    sdk = json.dumps({"error_type": "<class 'ValueError'>", "error_message": "boom"})
    assert "boom" in hob_bridge._format_remote_error(sdk)


def test_vae_missing():
    inv = {"vae": [{"model_name": "sdxl_vae.safetensors", "filename": "/w/VAE/sdxl_vae.safetensors"}]}
    assert not hob_bridge._vae_missing(inv, "sdxl_vae.safetensors")
    assert hob_bridge._vae_missing(inv, "other.pt")
    assert not hob_bridge._vae_missing(inv, "Automatic")
    assert not hob_bridge._vae_missing({"vae": None}, "other.pt")


def test_format_remote_error_string():
    assert hob_bridge._format_remote_error("boom") == "Hob Bridge: boom"


def test_checkpoint_names():
    inv = {"checkpoints": [
        {"name": "juggs.safetensors", "model_name": "juggs", "filename": "/w/juggs.safetensors"},
        {"model_name": "a"},
        {},
        None,
    ]}
    assert hob_bridge._checkpoint_names(inv) == ["a", "juggs.safetensors"]
    assert hob_bridge._checkpoint_names({}) == []


def test_embed_geninfo_roundtrip():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, format="PNG")
    original = buf.getvalue()
    # the API's `info` is processed.js() JSON; infotexts are per image
    info = json.dumps({"seed": 1, "infotexts": ["prompt: test\nSteps: 30"]})
    infotexts = hob_bridge._parse_info(info)["infotexts"]

    embedded = hob_bridge._embed_geninfo([original, original], infotexts)
    assert Image.open(io.BytesIO(embedded[0])).text["parameters"] == "prompt: test\nSteps: 30"
    assert embedded[1] == original  # no infotext (e.g. ControlNet map) → untouched


def test_embed_geninfo_without_info_is_noop():
    original = b"not-even-a-png"
    assert hob_bridge._embed_geninfo([original], []) == [original]
    assert hob_bridge._embed_geninfo([original], ["x"]) == [original]  # failure keeps bytes
    assert hob_bridge._parse_info("not json") == {}


# ---------------------------------------------------------------------------
# Transport, against a local fake of the RunPod / A1111 HTTP APIs
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _fake_remote(routes):
    """routes: path -> [(status, body or callable)]; the last response repeats."""
    remote = types.SimpleNamespace(hits=[], bodies={})

    class Handler(http.server.BaseHTTPRequestHandler):
        def _reply(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            path = self.path.split("?")[0]
            remote.hits.append(path)
            try:
                remote.bodies[path] = json.loads(raw) if raw else None
            except ValueError:  # e.g. a multipart upload
                remote.bodies[path] = raw
            queue = routes.get(path, [(404, {})])
            status, body = queue.pop(0) if len(queue) > 1 else queue[0]
            body = body() if callable(body) else body
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "image/png" if isinstance(body, bytes) else "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client gave up first (timeout tests)

        do_GET = do_POST = _reply

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    remote.base = f"http://127.0.0.1:{server.server_port}"
    try:
        yield remote
    finally:
        server.shutdown()


@contextlib.contextmanager
def _patched(module, **values):
    missing = object()
    old = {key: getattr(module, key, missing) for key in values}
    for key, value in values.items():
        setattr(module, key, value)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is missing:
                delattr(module, key)
            else:
                setattr(module, key, value)


def _settings(base, mode="serverless"):
    values = {"hob_bridge_endpoint_url": base, "hob_bridge_api_key": "key",
              "hob_bridge_mode": mode, "hob_bridge_timeout": 30}
    return lambda key, default="": values.get(key, default)


def _blocking_generation():
    """txt2img that runs until /interrupt is hit (or 5 s pass)."""
    interrupted = threading.Event()
    return interrupted, {
        "/sdapi/v1/txt2img": [(200, lambda: {"images": [], "info": "", "interrupted": interrupted.wait(5)})],
        "/sdapi/v1/interrupt": [(200, lambda: interrupted.set() or {})],
    }


_OOM_BODY = {"error": "OutOfMemoryError", "detail": "", "body": "",
             "errors": "CUDA out of memory. Tried to allocate 2.00 GiB"}


def test_call_a1111_keeps_error_body():
    with _fake_remote({"/sdapi/v1/txt2img": [(500, _OOM_BODY)]}) as remote:
        data, error = hob_worker._call_a1111(remote.base, "/sdapi/v1/txt2img", {}, timeout=10)
    assert data is None
    assert hob_worker._classify_error(error)["kind"] == "gpu OOM", error


def test_call_a1111_stop_interrupts_remote():
    def stop():
        raise hob_bridge.JobInterrupted()

    interrupted, routes = _blocking_generation()
    with _patched(hob_worker, PROGRESS_INTERVAL=0.05), _fake_remote(routes) as remote:
        try:
            hob_worker._call_a1111(remote.base, "/sdapi/v1/txt2img", {}, timeout=10, check_stop=stop)
            raise AssertionError("expected JobInterrupted")
        except hob_bridge.JobInterrupted:
            pass
        assert interrupted.is_set()


def test_worker_handler_cancel_interrupts_a1111():
    interrupted, routes = _blocking_generation()

    async def run_and_cancel():
        task = asyncio.create_task(hob_worker.handler({"id": "j", "input": {"type": "txt2img"}}))
        await asyncio.sleep(0.2)
        task.cancel()  # what the SDK does on a RunPod stop signal
        try:
            await task
        except asyncio.CancelledError:
            pass

    with _fake_remote(routes) as remote, _patched(hob_worker, A1111_URL=remote.base):
        asyncio.run(run_and_cancel())
    assert interrupted.is_set()


def test_worker_job_timeout_interrupts_a1111():
    interrupted, routes = _blocking_generation()
    job = {"id": "j", "input": {"type": "txt2img", "timeout": 0.3}}
    with _fake_remote(routes) as remote, _patched(hob_worker, A1111_URL=remote.base):
        result = asyncio.run(hob_worker.handler(job))
    assert json.loads(result["error"])["kind"] == "timeout", result
    assert interrupted.is_set()  # GPU work stopped even without a stop signal


def test_direct_request_classifies_a1111_error():
    with _fake_remote({"/sdapi/v1/txt2img": [(500, _OOM_BODY)]}) as remote, \
            _patched(hob_bridge, _setting=_settings(remote.base, mode="direct")):
        try:
            hob_bridge._send_job("txt2img", {})
            raise AssertionError("expected an error")
        except RuntimeError as exc:
            assert "gpu OOM" in str(exc), exc


def test_serverless_poll_survives_blips_and_reads_progress():
    progress = []
    done = {"images": [], "info": "{}"}
    routes = {
        "/run": [(200, {"id": "j1"})],
        "/status/j1": [(503, {}),
                       (200, {"status": "IN_PROGRESS", "output": {"state": "generating", "percentage": 40}}),
                       (200, {"status": "COMPLETED", "output": done})],
    }
    with _fake_remote(routes) as remote, \
            _patched(hob_bridge, _setting=_settings(remote.base), _POLL_INTERVAL=0.01,
                     _show_remote_progress=progress.append):
        assert hob_bridge._serverless_request("txt2img", {}) == done
    assert progress == [{"state": "generating", "percentage": 40}]
    # the client's budget travels too: RunPod policy + worker-side A1111 timeout
    body = remote.bodies["/run"]
    assert body["input"]["timeout"] == 30
    assert body["policy"] == {"executionTimeout": 30000, "ttl": 90000}


def test_serverless_poll_gives_up_by_cancelling():
    routes = {"/run": [(200, {"id": "j2"})], "/status/j2": [(503, {})], "/cancel/j2": [(200, {})]}
    with _fake_remote(routes) as remote, \
            _patched(hob_bridge, _setting=_settings(remote.base), _POLL_INTERVAL=0.01, _POLL_GRACE=0.05):
        try:
            hob_bridge._serverless_request("txt2img", {})
            raise AssertionError("expected an error")
        except RuntimeError:
            pass
    assert "/cancel/j2" in remote.hits  # never orphan a billed job


def test_serverless_failed_job_formats_worker_error():
    error = json.dumps({"kind": "gpu OOM", "message": "CUDA out of memory", "hint": "lower batch size"})
    routes = {"/run": [(200, {"id": "j3"})], "/status/j3": [(200, {"status": "FAILED", "error": error})]}
    with _fake_remote(routes) as remote, _patched(hob_bridge, _setting=_settings(remote.base)):
        try:
            hob_bridge._serverless_request("txt2img", {})
            raise AssertionError("expected an error")
        except RuntimeError as exc:
            assert "gpu OOM" in str(exc) and "hint: lower batch size" in str(exc), exc


# ---------------------------------------------------------------------------
# Local replay of process_images() / ControlNet postprocess() side effects
# ---------------------------------------------------------------------------

class _FakeWebuiImages:
    def __init__(self):
        self.saved = []

    def image_grid(self, imgs, batch_size=1, rows=None):
        return Image.new("RGB", (8 * len(imgs), 8))

    def save_image(self, image, path, basename, *args, **kwargs):
        self.saved.append((path, basename, kwargs.get("grid", False)))


@contextlib.contextmanager
def _webui(**opts):
    defaults = {"samples_save": True, "samples_format": "png", "return_grid": True,
                "grid_save": True, "grid_only_if_multiple": True, "grid_format": "png",
                "grid_extended_filename": False, "enable_pnginfo": True}
    defaults.update(opts)
    options = types.SimpleNamespace(**defaults)
    options.data = defaults
    fake_images = _FakeWebuiImages()
    hob_bridge.shared = types.SimpleNamespace(opts=options)
    hob_bridge.webui_images = fake_images
    try:
        yield fake_images
    finally:
        del hob_bridge.shared, hob_bridge.webui_images


def _grid_p():
    return _fake_p(batch_size=2, outpath_samples="/out/txt2img", outpath_grids="/out/grids",
                   do_not_save_grid=False, do_not_save_samples=False)


def test_batch_grid_saved_and_returned():
    images = [Image.new("RGB", (8, 8))] * 2
    with _webui() as fake:
        grid = hob_bridge._batch_grid(_grid_p(), images, "Steps: 30", {"all_seeds": [7, 8]})
    assert grid.info["parameters"] == "Steps: 30"
    assert fake.saved == [("/out/grids", "grid", True)]
    with _webui() as fake:  # grid_only_if_multiple
        assert hob_bridge._batch_grid(_grid_p(), images[:1], "x", {}) is None
    assert fake.saved == []
    with _webui(return_grid=False) as fake:
        assert hob_bridge._batch_grid(_grid_p(), images, "x", {}) is None
    assert fake.saved == [("/out/grids", "grid", True)]


def test_detected_maps_autosaved_like_controlnet():
    maps = [Image.new("RGB", (8, 8))]
    with _webui(control_net_detectmap_autosaving=True, control_net_detectedmap_dir="detected_maps") as fake, \
            _patched(hob_bridge.os, makedirs=lambda *a, **k: None):
        hob_bridge._save_detected_maps(_grid_p(), maps, [{"module": "canny"}, {"module": "none"}])
        hob_bridge._save_detected_maps(_grid_p(), maps, [{"module": "canny", "save_detected_map": False}])
    assert fake.saved == [(os.path.join("/out/txt2img", "detected_maps", "canny"), "canny", False)]
    with _webui() as fake:  # autosaving off (ControlNet's default)
        hob_bridge._save_detected_maps(_grid_p(), maps, [{"module": "canny"}])
    assert fake.saved == []


# ---------------------------------------------------------------------------
# Setup check (wizard), warm-up, A/B
# ---------------------------------------------------------------------------

def _fake_gr():
    return types.SimpleNamespace(update=lambda **kwargs: kwargs)


def test_check_settings():
    with _patched(hob_bridge, _setting=_settings("")):
        assert hob_bridge._check_settings()[0][0] is False
    with _patched(hob_bridge, _setting=_settings("http://1.2.3.4:7860")):
        assert hob_bridge._check_settings()[0][0] is None  # unusual for serverless: flagged, not fatal
    with _patched(hob_bridge, _setting=_settings("https://api.runpod.ai/v2/abc123/runsync")):
        assert [item[0] for item in hob_bridge._check_settings()] == [True, True]
    with _patched(hob_bridge, _setting=_settings("http://1.2.3.4:7860", mode="direct")):
        assert [item[0] for item in hob_bridge._check_settings()] == [True]


def test_check_health():
    health = {"workers": {"idle": 1, "running": 0}, "jobs": {"inQueue": 2}}
    with _fake_remote({"/health": [(200, health)]}) as remote, \
            _patched(hob_bridge, _setting=_settings(remote.base)):
        passed, _, detail = hob_bridge._check_health()
    assert passed and "1 idle" in detail and "2 queued" in detail
    with _fake_remote({"/health": [(401, {})]}) as remote, \
            _patched(hob_bridge, _setting=_settings(remote.base)):
        assert hob_bridge._check_health()[:2] == (False, "RunPod endpoint")


def test_check_inventory():
    inventory = {"checkpoints": [{"name": "a.safetensors"}], "loras": [], "vae": None,
                 "embeddings": ["e"], "upscalers": ["Latent"], "controlnet": None}
    with _patched(hob_bridge, _load_controlnet_external_code=lambda: object()):
        items = {label: passed for passed, label, _ in hob_bridge._check_inventory(inventory)}
    assert items["Volume: checkpoints"] is True
    assert items["Volume: LoRAs"] is None  # optional
    assert items["ControlNet on the worker"] is False  # installed locally, absent remotely
    empty = hob_bridge._check_inventory({"checkpoints": []})
    assert empty[0][0] is False


def test_setup_check_runs_free_checks_first():
    with _patched(hob_bridge, gr=_fake_gr(), _setting=_settings("")):
        steps = list(hob_bridge._setup_check())
    assert len(steps) == 1 and "🔴" in steps[0][0]  # no paid job while settings are wrong

    inventory = {"checkpoints": [{"name": "a.safetensors", "model_name": "a"}], "controlnet": []}
    routes = {"/health": [(200, {})], "/run": [(200, {"id": "inv"})],
              "/status/inv": [(200, {"status": "COMPLETED", "output": inventory})]}
    with _fake_remote(routes) as remote, \
            _patched(hob_bridge, gr=_fake_gr(), _setting=_settings(remote.base), _POLL_INTERVAL=0.01,
                     _load_controlnet_external_code=lambda: None):
        steps = list(hob_bridge._setup_check())
    report, dropdown = steps[-1]
    assert "🔴" not in report and "All set." in report, report
    assert dropdown == {"choices": ["a.safetensors"]}


def test_warm_checkpoint_states():
    hob_bridge._inventory_cache.update(
        ts=hob_bridge.time.time(), data={"checkpoints": [{"name": "sdxl/juggs.safetensors", "model_name": "sdxl_juggs"}]})
    routes = {"/run": [(200, {"id": "w1"})],
              "/status/w1": [(200, {"status": "IN_QUEUE"}),
                             (200, {"status": "IN_PROGRESS", "output": {"state": "loading"}}),
                             (200, {"status": "COMPLETED", "output": {"type": "warm", "seconds": 12.5}})]}
    try:
        with _fake_remote(routes) as remote, \
                _patched(hob_bridge, _setting=_settings(remote.base), _POLL_INTERVAL=0.01):
            states = list(hob_bridge._warm_checkpoint("juggs"))
    finally:
        hob_bridge._inventory_cache.update(ts=0.0, data=None)
    assert [s.split()[0] for s in states] == ["⏳", "⏳", "🔄", "🟢"], states
    assert "sdxl/juggs.safetensors loaded in 12.5s" in states[-1]
    assert remote.bodies["/run"]["input"] == {"type": "warm", "params": {"checkpoint": "sdxl/juggs.safetensors"},
                                              "timeout": 30}
    assert list(hob_bridge._warm_checkpoint(""))[0].startswith("Pick")


def test_model_dir_follows_the_runpod_mount():
    assert hob_worker._default_model_dir(isdir=lambda path: path == "/runpod-volume") == "/runpod-volume/models"
    assert hob_worker._default_model_dir(isdir=lambda path: False) == "/workspace/models"


def test_worker_warm_job():
    routes = {"/sdapi/v1/options": [(200, None)], "/sdapi/v1/txt2img": [(200, {"images": []})]}
    job = {"id": "w", "input": {"type": "warm", "params": {"checkpoint": "a.safetensors"}}}
    with _fake_remote(routes) as remote, _patched(hob_worker, A1111_URL=remote.base):
        result = asyncio.run(hob_worker.handler(job))
    assert result["type"] == "warm" and result["checkpoint"] == "a.safetensors"
    assert remote.bodies["/sdapi/v1/options"] == {"sd_model_checkpoint": "a.safetensors"}

    missing = {"error": "RuntimeError", "detail": "", "body": "", "errors": "model 'nope' not found"}
    with _fake_remote({"/sdapi/v1/options": [(500, missing)]}) as remote, \
            _patched(hob_worker, A1111_URL=remote.base):
        result = asyncio.run(hob_worker.handler(job))
    assert json.loads(result["error"])["kind"] == "model missing", result


def test_similarity():
    black, white = Image.new("RGB", (8, 8), "black"), Image.new("RGB", (8, 8), "white")
    assert hob_bridge._similarity(black, black) == 100
    assert round(hob_bridge._similarity(black, white)) == 0
    assert round(hob_bridge._similarity(black, Image.new("RGB", (4, 4), (51, 51, 51)))) == 80  # resized


def test_ab_compare_reuses_remote_seed():
    seen = {}

    def fake_process_images(p):
        seen.update(seed=p.seed, override=p.override_settings)
        return types.SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")], infotexts=["local"],
                                     index_of_first_image=0, sd_model_name="local-model")

    remote = types.SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")], infotexts=["remote"], comments="")
    fake_sd_models = types.SimpleNamespace(checkpoint_aliases={"juggs.safetensors": object()})
    with _patched(hob_bridge, process_images=fake_process_images, sd_models=fake_sd_models):
        hob_bridge._ab_compare(_fake_p(seed=-1, subseed=-1), remote, list(remote.images),
                               {"seed": 1234, "subseed": 99, "sd_model_name": "juggs"}, "juggs.safetensors")
    assert seen == {"seed": 1234, "override": {"sd_model_checkpoint": "juggs.safetensors"}}
    assert remote.infotexts == ["remote", "local"] and len(remote.images) == 2
    assert "#1 100.0%" in remote.comments


def run_all():
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except (AssertionError, Exception) as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc!r}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
