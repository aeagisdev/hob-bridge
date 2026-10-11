"""Integration: worker/comfy_handler.py against a real ComfyUI server (CPU).

Boots the ComfyUI release pinned in worker/Dockerfile.comfy on CPU, with the
worker's own extra_model_paths config over a dummy model folder, and checks:

* the catalog / inventory parsing of /object_info and the volume-layout mapping;
* every graph variant `_workflow` builds is accepted by ComfyUI's validation
  (it then fails only on the empty dummy checkpoint — a validation error would
  mean a wrong node name or input);
* a model-free graph through the real queue / history / view path.

With HOB_COMFY_TEST_CHECKPOINT=/path/to/sd15.safetensors (any small real SD
checkpoint) it also samples for real: txt2img, Hires fix, img2img, warm-up and
cancel. CI skips those; run them by hand when bumping the ComfyUI pin.

    git clone --depth 1 --branch <tag> https://github.com/comfyanonymous/ComfyUI.git
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r ComfyUI/requirements.txt requests pillow pytest
    COMFYUI_DIR=ComfyUI pytest -rs tests/integration/test_comfy_server.py
"""
import asyncio
import base64
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "worker"))

import comfy_handler as comfy  # noqa: E402
import requests  # noqa: E402
from PIL import Image  # noqa: E402


def _pinned_tag():
    with open(os.path.join(ROOT, "worker", "Dockerfile.comfy")) as fh:
        return re.search(r"--branch (v[\d.]+)", fh.read()).group(1)


@pytest.fixture(scope="module")
def comfyui(tmp_path_factory):
    work = tmp_path_factory.mktemp("comfy")
    comfy_dir = os.environ.get("COMFYUI_DIR")
    if not comfy_dir:
        comfy_dir = str(work / "ComfyUI")
        cloned = subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", _pinned_tag(),
                                 "https://github.com/comfyanonymous/ComfyUI.git", comfy_dir])
        if cloned.returncode:
            pytest.skip("cannot clone ComfyUI (set COMFYUI_DIR)")
    if subprocess.run([sys.executable, "-c", "import torch, aiohttp, yaml"], capture_output=True).returncode:
        pytest.skip("ComfyUI's requirements are not installed in this environment")

    models = work / "models"
    for rel in ("Stable-diffusion/sdxl/dummy.safetensors", "Lora/detail.safetensors", "VAE/sdxl_vae.safetensors",
                "ESRGAN/ESRGAN_4x.pth", "embeddings/easynegative.pt", "ControlNet/c.safetensors"):
        (models / rel).parent.mkdir(parents=True, exist_ok=True)
        (models / rel).write_bytes(b"")  # listed by name; never loaded
    real = os.environ.get("HOB_COMFY_TEST_CHECKPOINT")
    if real:
        os.symlink(os.path.abspath(real), models / "Stable-diffusion" / "real.safetensors")
    config = work / "paths.yaml"
    config.write_text(json.dumps(comfy._model_paths(str(models))))

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = subprocess.Popen(
        [sys.executable, "main.py", "--cpu", "--port", str(port), "--disable-auto-launch", "--disable-metadata",
         "--extra-model-paths-config", str(config),
         "--output-directory", str(work / "out"), "--temp-directory", str(work / "tmp"),
         "--input-directory", str(work / "in")],
        cwd=comfy_dir, stdout=open(work / "comfy.log", "w"), stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if urllib.request.urlopen(f"{url}/system_stats", timeout=2).status == 200:
                    break
            except OSError:
                time.sleep(1)
        else:
            pytest.fail("ComfyUI did not start:\n" + (work / "comfy.log").read_text()[-2000:])
        old = comfy.COMFY_URL, comfy.a1111.PROGRESS_INTERVAL
        comfy.COMFY_URL, comfy.a1111.PROGRESS_INTERVAL = url, 0.2
        yield url
    finally:
        comfy.COMFY_URL, comfy.a1111.PROGRESS_INTERVAL = old
        server.terminate()
        server.wait(30)


BASE = {"prompt": "a red apple <lora:detail:0.7> easynegative", "negative_prompt": "blurry", "steps": 2,
        "cfg_scale": 7, "width": 64, "height": 64, "sampler_name": "DPM++ 2M", "scheduler": "Karras", "seed": 3,
        "override_settings": {"CLIP_stop_at_last_layers": 2, "sd_vae": "sdxl_vae.safetensors",
                              "sd_model_checkpoint": "dummy"}}


def test_catalog_and_inventory(comfyui):
    catalog = comfy._catalog()
    assert catalog["checkpoints"][-1] == "sdxl/dummy.safetensors" or "sdxl/dummy.safetensors" in catalog["checkpoints"]
    assert catalog["loras"] == ["detail.safetensors"]
    assert catalog["vae"] == ["sdxl_vae.safetensors"]  # 'pixel_space' (built in, not a file) filtered out
    assert catalog["upscale_models"] == ["ESRGAN_4x.pth"]
    assert catalog["embeddings"] == ["easynegative"]
    inventory = comfy._inventory()
    assert inventory["vae"] == [{"model_name": "sdxl_vae.safetensors", "filename": "sdxl_vae.safetensors"}]
    assert "ESRGAN_4x" in inventory["upscalers"] and "juggs" not in str(inventory)


@pytest.mark.parametrize("name,extra", [
    ("txt2img + lora + embedding + clip skip + vae", {}),
    ("hires, latent upscaler", {"enable_hr": True, "hr_scale": 1.5, "hr_upscaler": "Latent (bicubic)",
                                "denoising_strength": 0.5}),
    ("hires, model upscaler", {"enable_hr": True, "hr_resize_x": 128, "hr_upscaler": "ESRGAN_4x"}),
    ("img2img, batch, crop", {"init_images": ["x"], "batch_size": 2, "resize_mode": 1, "denoising_strength": 0.6}),
])
def test_graph_variants_pass_comfyui_validation(comfyui, name, extra):
    catalog = comfy._catalog()
    init = None
    if "init_images" in extra:
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), "red").save(buf, "PNG")
        init = comfy._upload({"id": "validation"}, base64.b64encode(buf.getvalue()).decode())
    graph = comfy._workflow({**BASE, **extra}, 3, catalog, comfy._checkpoint(catalog, "dummy"), init)
    try:
        comfy._run_graph(graph, comfy._Run(), time.time() + 60)
        pytest.fail("the empty dummy checkpoint cannot produce an image")
    except comfy.ValidationError as exc:
        pytest.fail(f"ComfyUI rejected the graph: {exc}")
    except RuntimeError as exc:  # validated and queued, failed on loading the empty checkpoint
        assert "ComfyUI generation timed out" not in str(exc)


def test_model_free_graph_returns_pngs(comfyui):
    graph = comfy._Graph()
    empty = graph.add("EmptyImage", width=16, height=8, batch_size=2, color=255)
    graph.add("PreviewImage", images=[empty, 0])
    images = comfy._run_graph(graph.nodes, comfy._Run(), time.time() + 60)
    assert [Image.open(io.BytesIO(raw)).size for raw in images] == [(16, 8), (16, 8)]


def test_invalid_graph_is_a_validation_error(comfyui):
    graph = {"1": {"class_type": "KSampler", "inputs": {}},
             "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}}
    with pytest.raises(comfy.ValidationError) as exc:
        comfy._run_graph(graph, comfy._Run(), time.time() + 10)
    assert "validation" in str(exc.value).lower()


def test_execution_error_message_reaches_the_client(comfyui):
    result = asyncio.run(comfy.handler({"id": "bad-ckpt", "input": {"type": "txt2img", "params": BASE}}))
    error = json.loads(result["error"])  # a JSON string, as the SDK's job `error` must be
    assert error["kind"] == "error" and error["message"], result


# ---------------------------------------------------------------------------
# Real sampling — needs HOB_COMFY_TEST_CHECKPOINT
# ---------------------------------------------------------------------------

real = pytest.mark.skipif(not os.environ.get("HOB_COMFY_TEST_CHECKPOINT"),
                          reason="set HOB_COMFY_TEST_CHECKPOINT to a small real SD checkpoint")
REAL = {"prompt": "a red apple", "negative_prompt": "blurry", "steps": 3, "cfg_scale": 7, "width": 128,
        "height": 128, "sampler_name": "Euler a", "scheduler": "Automatic", "seed": 42,
        "override_settings": {"sd_model_checkpoint": "real"}}


def _job(kind, params, **extra):
    return asyncio.run(comfy.handler({"id": f"real-{kind}", "input": {"type": kind, "params": params, **extra}}))


def _sizes(result):
    assert "error" not in result, result
    return [Image.open(io.BytesIO(base64.b64decode(raw))).size for raw in result["images"]]


@real
def test_real_txt2img_hires_img2img(comfyui):
    first = _job("txt2img", {**REAL, "batch_size": 2, "n_iter": 2})
    assert _sizes(first) == [(128, 128)] * 4
    assert json.loads(first["info"])["all_seeds"] == [42, 42, 44, 44]
    again = _job("txt2img", REAL)
    assert again["images"][0] == _job("txt2img", REAL)["images"][0]  # same seed → same image
    assert _sizes(_job("txt2img", {**REAL, "enable_hr": True, "hr_scale": 1.5})) == [(192, 192)]
    assert _sizes(_job("txt2img", {**REAL, "enable_hr": True, "hr_resize_x": 256,
                                   "hr_upscaler": "Latent (bicubic)"})) == [(256, 256)]
    init = first["images"][0]
    img2img = {**REAL, "init_images": [init], "width": 160, "height": 96, "batch_size": 2, "resize_mode": 1,
               "denoising_strength": 0.6}
    assert _sizes(_job("img2img", img2img)) == [(160, 96)] * 2


@real
def test_real_warm_sets_default(comfyui):
    comfy._state["checkpoint"] = None
    warm = _job("warm", {"checkpoint": "real"})
    assert warm["checkpoint"] == "real.safetensors" and comfy._state["checkpoint"] == "real.safetensors"
    no_override = {key: value for key, value in REAL.items() if key != "override_settings"}
    assert _sizes(_job("txt2img", no_override)) == [(128, 128)]
    comfy._state["checkpoint"] = None


@real
def test_real_cancel_stops_sampling(comfyui):
    params = {**REAL, "steps": 150, "width": 256, "height": 256}

    async def main():
        task = asyncio.create_task(comfy.handler({"id": "real-cancel", "input": {"type": "txt2img", "params": params}}))
        for _ in range(60):  # until ComfyUI is sampling
            await asyncio.sleep(0.5)
            if requests.get(comfyui + "/queue").json()["queue_running"]:
                break
        await asyncio.sleep(2)
        started = time.time()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        while requests.get(comfyui + "/queue").json()["queue_running"]:
            assert time.time() - started < 15, "ComfyUI kept sampling after the cancel"
            await asyncio.sleep(0.2)

    asyncio.run(main())
    history = requests.get(comfyui + "/history").json().values()
    latest = [e for e in history if any(n["inputs"].get("steps") == 150 for n in e["prompt"][2].values())]
    assert [m[0] for m in latest[-1]["status"]["messages"]][-1] == "execution_interrupted"
