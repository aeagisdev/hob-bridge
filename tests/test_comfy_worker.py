"""Unit tests for worker/comfy_handler.py — the ComfyUI backend.

The graph mapping is tested on plain dicts; job flow runs against a fake
ComfyUI HTTP server (the helper from test_hob_bridge.py). The real server is
exercised in tests/integration/test_comfy_server.py.

Run with pytest, or plain: python3 tests/test_comfy_worker.py
"""
import asyncio
import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

import comfy_handler as comfy  # noqa: E402
from PIL import Image  # noqa: E402
from test_hob_bridge import _fake_remote, _patched  # noqa: E402

CATALOG = {
    "checkpoints": ["plain.ckpt", "sdxl/juggs.safetensors"],
    "loras": ["detail.safetensors", "styles/ink.safetensors"],
    "vae": ["sdxl_vae.safetensors"],
    "upscale_models": ["ESRGAN_4x.pth"],
    "embeddings": ["easynegative"],
}
BASE = {"prompt": "a cat", "negative_prompt": "ugly", "steps": 20, "cfg_scale": 7, "width": 512, "height": 768,
        "sampler_name": "DPM++ 2M", "scheduler": "Karras", "seed": 7, "batch_size": 1}


def _graph(params=None, init=None, checkpoint="sdxl/juggs.safetensors"):
    return comfy._workflow({**BASE, **(params or {})}, 7, CATALOG, checkpoint, init)


def _classes(graph):
    return [node["class_type"] for node in graph.values()]


def _find(graph, class_type):
    return [node["inputs"] for node in graph.values() if node["class_type"] == class_type]


def test_txt2img_graph():
    graph = _graph()
    assert _classes(graph) == ["CheckpointLoaderSimple", "CLIPTextEncode", "CLIPTextEncode", "EmptyLatentImage",
                               "KSampler", "VAEDecode", "PreviewImage"]
    (sampler,) = _find(graph, "KSampler")
    assert (sampler["sampler_name"], sampler["scheduler"], sampler["seed"], sampler["denoise"]) == \
        ("dpmpp_2m", "karras", 7, 1.0)
    assert _find(graph, "EmptyLatentImage") == [{"width": 512, "height": 768, "batch_size": 1}]
    assert [i["text"] for i in _find(graph, "CLIPTextEncode")] == ["a cat", "ugly"]
    # every link points at an existing node
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in graph


def test_sampler_mapping():
    assert comfy._sampler("Euler a", "Automatic") == ("euler_ancestral", "normal")
    assert comfy._sampler("DPM++ 2M", None) == ("dpmpp_2m", "karras")
    assert comfy._sampler("DPM++ 2M Karras", "") == ("dpmpp_2m", "karras")  # pre-1.9 label
    assert comfy._sampler("DPM++ 2M Karras", "Exponential") == ("dpmpp_2m", "exponential")
    assert comfy._sampler("DDIM", "Automatic") == ("ddim", "ddim_uniform")
    assert comfy._sampler("Euler", "SGM Uniform") == ("euler", "sgm_uniform")
    for bad in (("PLMS", None), ("Euler", "Align Your Steps")):
        try:
            comfy._sampler(*bad)
            raise AssertionError(f"{bad} should be unsupported")
        except comfy.Unsupported:
            pass


def test_loras_embeddings_clipskip_vae():
    graph = _graph({"prompt": "a cat <lora:ink:0.6> <lora:detail:0.8:1.2> easynegative",
                    "negative_prompt": "<lora:ink:1> easynegative, bad",
                    "override_settings": {"CLIP_stop_at_last_layers": 2, "sd_vae": "sdxl_vae.safetensors"}})
    loras = _find(graph, "LoraLoader")
    assert [(l["lora_name"], l["strength_clip"], l["strength_model"]) for l in loras] == \
        [("styles/ink.safetensors", 0.6, 0.6), ("detail.safetensors", 0.8, 1.2)]  # <lora:name:te:unet>
    ids = [k for k, n in graph.items() if n["class_type"] == "LoraLoader"]
    assert loras[1]["model"] == [ids[0], 0] and loras[1]["clip"] == [ids[0], 1]  # chained
    (skip,) = _find(graph, "CLIPSetLastLayer")
    assert skip["stop_at_clip_layer"] == -2
    assert _find(graph, "VAELoader") == [{"vae_name": "sdxl_vae.safetensors"}]
    texts = [i["text"] for i in _find(graph, "CLIPTextEncode")]
    assert "<lora" not in texts[0] + texts[1]
    assert texts == ["a cat   embedding:easynegative", "embedding:easynegative, bad"]
    for missing in ({"prompt": "<lora:nope:1>"}, {"override_settings": {"sd_vae": "nope.pt"}}):
        try:
            _graph(missing)
            raise AssertionError("expected a not-found error")
        except ValueError as exc:
            assert "not found" in str(exc)
    assert comfy.a1111._classify_error("lora 'nope' not found on the volume")["kind"] == "model missing"


def test_hires_latent_and_model():
    latent = _graph({"enable_hr": True, "hr_scale": 1.5, "hr_upscaler": "Latent (bicubic)",
                     "denoising_strength": 0.5, "hr_second_pass_steps": 10})
    assert _find(latent, "LatentUpscale") == [{"samples": ["5", 0], "upscale_method": "bicubic", "width": 768,
                                               "height": 1152, "crop": "disabled"}]
    first, second = _find(latent, "KSampler")
    assert (second["denoise"], second["steps"], second["latent_image"]) == (0.5, 10, ["6", 0])

    model = _graph({"enable_hr": True, "hr_resize_x": 1024, "hr_upscaler": "ESRGAN_4x"})
    assert _classes(model)[4:11] == ["KSampler", "VAEDecode", "UpscaleModelLoader", "ImageUpscaleWithModel",
                                     "ImageScale", "VAEEncode", "KSampler"]
    (scale,) = _find(model, "ImageScale")
    assert (scale["width"], scale["height"]) == (1024, 1536)  # one side given → aspect kept
    assert _find(model, "UpscaleModelLoader") == [{"model_name": "ESRGAN_4x.pth"}]
    try:
        _graph({"enable_hr": True, "hr_upscaler": "R-ESRGAN 4x+"})
        raise AssertionError("expected not found")
    except ValueError as exc:
        assert "upscaler" in str(exc)


def test_img2img_graph():
    graph = _graph({"batch_size": 2, "resize_mode": 1, "denoising_strength": 0.6}, init="hob_job.png")
    assert _classes(graph)[3:8] == ["LoadImage", "ImageScale", "VAEEncode", "RepeatLatentBatch", "KSampler"]
    assert _find(graph, "LoadImage") == [{"image": "hob_job.png"}]
    assert _find(graph, "ImageScale")[0]["crop"] == "center"
    assert _find(graph, "RepeatLatentBatch")[0]["amount"] == 2
    assert _find(graph, "KSampler")[0]["denoise"] == 0.6


def test_unsupported_features():
    cases = [{"alwayson_scripts": {"controlnet": {"args": [{}]}}}, {"mask": "b64"}, {"prompt": ["a", "b"]},
             {"sampler_name": "PLMS"}]
    for params in cases:
        try:
            _graph(params)
            raise AssertionError(f"{params} should be unsupported")
        except comfy.Unsupported:
            pass
    try:
        _graph({"resize_mode": 2}, init="x.png")
        raise AssertionError("resize and fill should be unsupported")
    except comfy.Unsupported:
        pass


def test_checkpoint_resolution():
    assert comfy._checkpoint(CATALOG, "sdxl/juggs.safetensors") == "sdxl/juggs.safetensors"
    assert comfy._checkpoint(CATALOG, "juggs") == "sdxl/juggs.safetensors"
    assert comfy._checkpoint(CATALOG, "juggs.safetensors") == "sdxl/juggs.safetensors"
    assert comfy._checkpoint(CATALOG, None) == "plain.ckpt"  # nothing warmed: first on the volume
    for catalog, wanted in ((CATALOG, "nope"), ({"checkpoints": []}, None)):
        try:
            comfy._checkpoint(catalog, wanted)
            raise AssertionError("expected an error")
        except RuntimeError as exc:
            assert comfy.a1111._classify_error(str(exc))["kind"] == "model missing"


# ---------------------------------------------------------------------------
# Jobs against a fake ComfyUI
# ---------------------------------------------------------------------------

PNG = io.BytesIO()
Image.new("RGB", (8, 8), "red").save(PNG, "PNG")
PNG = PNG.getvalue()


def _object_info(field, values, legacy=True):
    spec = [values, {}] if legacy else ["COMBO", {"options": values}]
    return {"input": {"required": {field: spec}}}


def _comfy_routes(history=None, **extra):
    done = {"p1": {"status": {"status_str": "success", "completed": True, "messages": []},
                   "outputs": {"7": {"images": [{"filename": "a.png", "subfolder": "", "type": "temp"}] * 2}}}}
    routes = {
        "/object_info/CheckpointLoaderSimple": [(200, {"CheckpointLoaderSimple": _object_info("ckpt_name", CATALOG["checkpoints"])})],
        "/object_info/LoraLoader": [(200, {"LoraLoader": _object_info("lora_name", CATALOG["loras"])})],
        "/object_info/VAELoader": [(200, {"VAELoader": _object_info("vae_name", CATALOG["vae"] + ["pixel_space"])})],
        "/object_info/UpscaleModelLoader": [(200, {"UpscaleModelLoader": _object_info("model_name", CATALOG["upscale_models"], False)})],
        "/embeddings": [(200, CATALOG["embeddings"])],
        "/prompt": [(200, {"prompt_id": "p1", "node_errors": {}})],
        "/history/p1": history or [(200, {}), (200, done)],
        "/queue": [(200, {"queue_running": [[0, "p1"]], "queue_pending": []})],
        "/view": [(200, PNG)],
        "/upload/image": [(200, {"name": "hob_j1.png", "subfolder": "", "type": "input"})],
        "/interrupt": [(200, {})],
    }
    routes.update(extra)
    return routes


def _run(job_input, routes, job_id="j1"):
    with _fake_remote(routes) as remote, \
            _patched(comfy, COMFY_URL=remote.base), _patched(comfy.a1111, PROGRESS_INTERVAL=0.01):
        comfy._state["checkpoint"] = None
        result = asyncio.run(comfy.handler({"id": job_id, "input": job_input}))
    return result, remote


def test_inventory_shape():
    result, _ = _run({"type": "inventory"}, _comfy_routes())
    assert result["checkpoints"][1] == {"name": "sdxl/juggs.safetensors", "model_name": "juggs",
                                        "filename": "sdxl/juggs.safetensors"}
    assert result["vae"] == [{"model_name": "sdxl_vae.safetensors", "filename": "sdxl_vae.safetensors"}]
    assert [l["name"] for l in result["loras"]] == ["detail", "ink"]
    assert result["upscalers"] == ["Latent", "ESRGAN_4x"] and result["controlnet"] is None


def test_txt2img_job():
    params = {**BASE, "n_iter": 2, "batch_size": 2, "seed": -1,
              "override_settings": {"sd_model_checkpoint": "sdxl/juggs.safetensors"}}
    result, remote = _run({"type": "txt2img", "params": params, "timeout": 30}, _comfy_routes())
    assert result["type"] == "txt2img" and len(result["images"]) == 4  # 2 graphs × 2 images
    assert base64.b64decode(result["images"][0]) == PNG
    info = json.loads(result["info"])
    assert len(info["infotexts"]) == len(info["all_seeds"]) == 4
    assert info["all_seeds"][2] == info["seed"] + 2  # n_iter step = batch_size
    assert "Steps: 20, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7" in info["infotexts"][0]
    assert info["infotexts"][0].startswith("a cat\nNegative prompt: ugly\n")
    assert "Model: juggs" in info["infotexts"][0] and info["sd_model_name"] == "juggs"
    sent = remote.bodies["/prompt"]["prompt"]
    assert sent["1"]["inputs"]["ckpt_name"] == "sdxl/juggs.safetensors"


def test_img2img_job_uploads_the_init_image():
    init = base64.b64encode(PNG).decode()
    params = {**BASE, "init_images": [init], "denoising_strength": 0.5}
    result, remote = _run({"type": "img2img", "params": params}, _comfy_routes())
    assert len(result["images"]) == 2
    assert PNG in remote.bodies["/upload/image"]  # multipart body carries the PNG
    assert any(n["class_type"] == "LoadImage" and n["inputs"]["image"] == "hob_j1.png"
               for n in remote.bodies["/prompt"]["prompt"].values())
    result, _ = _run({"type": "img2img", "params": {**BASE, "init_images": [init, init]}}, _comfy_routes())
    assert json.loads(result["error"])["kind"] == "unsupported"


def test_errors_are_classified():
    result, _ = _run({"type": "txt2img", "params": {**BASE, "alwayson_scripts": {"controlnet": {}}}}, _comfy_routes())
    assert isinstance(result["error"], str)  # the SDK's job `error` is a string
    assert json.loads(result["error"])["kind"] == "unsupported" and "ControlNet" in json.loads(result["error"])["message"]

    invalid = {"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
               "node_errors": {"3": {"errors": [{"message": "Value not in list", "details": "ckpt_name: 'x'"}]}}}
    result, _ = _run({"type": "txt2img", "params": BASE}, _comfy_routes(**{"/prompt": [(400, invalid)]}))
    assert "Value not in list" in json.loads(result["error"])["message"]

    oom = {"p1": {"status": {"status_str": "error", "completed": False, "messages": [
        ["execution_error", {"exception_type": "torch.OutOfMemoryError",
                             "exception_message": "Allocation on device would exceed allowed memory (out of memory)"}]]},
          "outputs": {}}}
    result, _ = _run({"type": "txt2img", "params": BASE}, _comfy_routes(history=[(200, oom)]))
    assert json.loads(result["error"])["kind"] == "gpu OOM", result

    result, _ = _run({"type": "txt2img", "params": {**BASE, "override_settings": {"sd_model_checkpoint": "nope"}}},
                     _comfy_routes())
    assert json.loads(result["error"])["kind"] == "model missing"

    result, _ = _run({"type": "txt2img", "params": BASE, "timeout": 0.05}, _comfy_routes(history=[(200, {})]))
    assert json.loads(result["error"])["kind"] == "timeout"


def test_cancel_interrupts_comfy():
    async def run_and_cancel():
        task = asyncio.create_task(comfy.handler({"id": "j", "input": {"type": "txt2img", "params": BASE}}))
        await asyncio.sleep(0.3)  # queued, never finishes
        task.cancel()  # what the SDK does on a RunPod stop signal
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.1)

    with _fake_remote(_comfy_routes(history=[(200, {})])) as remote, \
            _patched(comfy, COMFY_URL=remote.base), _patched(comfy.a1111, PROGRESS_INTERVAL=0.02):
        asyncio.run(run_and_cancel())
    assert remote.bodies["/interrupt"] == {"prompt_id": "p1"}
    assert remote.bodies["/queue"] == {"delete": ["p1"]}


def test_warm_sets_the_default_checkpoint():
    with _fake_remote(_comfy_routes(history=[(200, _comfy_routes()["/history/p1"][1][1])])) as remote, \
            _patched(comfy, COMFY_URL=remote.base), _patched(comfy.a1111, PROGRESS_INTERVAL=0.01):
        comfy._state["checkpoint"] = None
        warm = asyncio.run(comfy.handler({"id": "w", "input": {"type": "warm", "params": {"checkpoint": "juggs"}}}))
        assert warm["checkpoint"] == "sdxl/juggs.safetensors" and warm["type"] == "warm"
        assert comfy._state["checkpoint"] == "sdxl/juggs.safetensors"
        graph = remote.bodies["/prompt"]["prompt"]
        assert [n["inputs"] for n in graph.values() if n["class_type"] == "KSampler"][0]["steps"] == 1
        asyncio.run(comfy.handler({"id": "j", "input": {"type": "txt2img", "params": BASE}}))  # no override
        assert remote.bodies["/prompt"]["prompt"]["1"]["inputs"]["ckpt_name"] == "sdxl/juggs.safetensors"
        comfy._state["checkpoint"] = None
    result, _ = _run({"type": "warm", "params": {}}, _comfy_routes())
    assert "error" in result


def test_worker_output_feeds_the_extension():
    """The extension needs no ComfyUI-specific code: its parsing and preflight read this worker as they read A1111's."""
    import hob_bridge

    result, _ = _run({"type": "txt2img", "params": BASE}, _comfy_routes())
    info = hob_bridge._parse_info(result["info"])
    embedded = hob_bridge._embed_geninfo(hob_bridge._fetch_output_images(result), info["infotexts"])
    assert [Image.open(io.BytesIO(raw)).text["parameters"] for raw in embedded] == info["infotexts"]
    assert info["seed"] == 7 and info["all_prompts"][0] == "a cat"

    inventory, _ = _run({"type": "inventory"}, _comfy_routes())
    assert hob_bridge._resolve_checkpoint(inventory, "juggs") == "sdxl/juggs.safetensors"
    assert hob_bridge._resolve_checkpoint(inventory, "juggs.safetensors") == "sdxl/juggs.safetensors"
    assert hob_bridge._checkpoint_names(inventory) == ["plain.ckpt", "sdxl/juggs.safetensors"]
    assert hob_bridge._missing_loras(inventory, "<lora:detail:1> <lora:nope:1>") == ["nope"]
    assert not hob_bridge._vae_missing(inventory, "sdxl_vae.safetensors")
    assert hob_bridge._vae_missing(inventory, "other.pt")
    # a failed job's error, as the client receives it, renders with kind and hint
    failed, _ = _run({"type": "txt2img", "params": {**BASE, "override_settings": {"sd_model_checkpoint": "nope"}}},
                     _comfy_routes())
    shown = hob_bridge._format_remote_error(failed["error"])
    assert "model missing" in shown and "hint:" in shown, shown
    # the setup check's verdict on this backend: volume fine, ControlNet absent
    items = {label: passed for passed, label, _ in hob_bridge._check_inventory(inventory)}
    assert items["Volume: checkpoints"] is True and items["Volume: LoRAs"] is True


def test_model_paths_follow_the_volume_layout():
    paths = comfy._model_paths("/vol/models")["hob"]
    assert paths["base_path"] == "/vol/models" and paths["checkpoints"] == "Stable-diffusion"
    assert paths["loras"] == "Lora" and "ESRGAN" in paths["upscale_models"].split("\n")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
