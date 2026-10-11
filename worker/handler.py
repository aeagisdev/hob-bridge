"""RunPod serverless worker for Hob Bridge.

Boots Automatic1111 (with the ControlNet extension) in API mode once per
container, waits for it to come up, then serves job types:

    txt2img / img2img  → forwarded to A1111, progress reported via progress_update
                         (visible as `output` of /status while IN_PROGRESS)
    ping               → liveness probe
    inventory          → which checkpoints/LoRAs/embeddings the volume holds
    warm               → load a checkpoint now (the extension's warm-up button)

Job params use the standard A1111 API schema; checkpoints and LoRAs are
resolved against the network volume mounted at HOB_MODEL_DIR. Failures are
returned as classified errors ({kind, message, hint}) so the extension can
show a readable message instead of a raw traceback. A cancelled job (RunPod
stop signal) interrupts the running A1111 generation.

The A1111 client helpers (_call_a1111, _collect_inventory, _warm,
_classify_error) are also loaded by the extension for its direct-pod mode.

When S3/R2 output is configured (HOB_S3_*), result images are uploaded to the
bucket and the response carries presigned URLs with `images_are_urls: true`,
keeping the response payload tiny; on any upload failure the worker falls
back to base64 images.

This module has no import-time side effects: the A1111 bootstrap and
runpod.serverless.start() run only when executed directly (`python handler.py`),
which keeps it importable for unit tests.
"""
import asyncio
import base64
import json
import os
import re
import subprocess
import threading
import time

import requests

try:
    import runpod
except ImportError:  # unit tests import this module without the SDK
    runpod = None

A1111_DIR = os.environ.get("A1111_DIR", "/sd-webui")
A1111_URL = "http://127.0.0.1:7860"
MODEL_DIR = os.environ.get("HOB_MODEL_DIR", "/workspace/models")
STARTUP_TIMEOUT = int(os.environ.get("HOB_STARTUP_TIMEOUT", "600"))
GENERATION_TIMEOUT = int(os.environ.get("HOB_GENERATION_TIMEOUT", "900"))
PROGRESS_INTERVAL = float(os.environ.get("HOB_PROGRESS_INTERVAL", "1.5"))


def _classify_error(text):
    """Map an exception / A1111 error payload to a readable, classified error."""
    message = str(text or "").strip()
    low = message.lower()
    if "out of memory" in low:
        return {"kind": "gpu OOM", "message": message,
                "hint": "lower resolution or batch size; consider --medvram via HOB_A1111_ARGS"}
    if (("checkpoint" in low and ("not found" in low or "no such file" in low or "no checkpoints" in low))
            or ("model '" in low and "not found" in low)):  # /options: "model 'x' not found"
        return {"kind": "model missing", "message": message,
                "hint": "sync the model to the network volume (README → Storage)"}
    if "lora" in low and ("not found" in low or "no such file" in low):
        return {"kind": "model missing", "message": message,
                "hint": "sync the LoRA to models/Lora on the network volume"}
    if "timed out" in low or "timeout" in low:
        return {"kind": "timeout", "message": message,
                "hint": "raise HOB_GENERATION_TIMEOUT, or reduce steps/resolution"}
    if "connection" in low or "refused" in low or "connectionreset" in low:
        return {"kind": "a1111 unreachable", "message": message,
                "hint": "check the worker logs: A1111 may still be booting or it crashed"}
    return {"kind": "error", "message": message or "unknown error", "hint": ""}


def _failure(classified):
    """The output of a failed job. The RunPod SDK moves `error` into the job's
    `error` field, which is a string there (its local API rejects anything
    else), so the classified dict is sent JSON-encoded; the extension parses it back."""
    return {"error": json.dumps(classified)}


def _s3_config():
    """S3/R2 output config from HOB_S3_* env vars; None when not configured."""
    endpoint = os.environ.get("HOB_S3_ENDPOINT", "").strip()
    bucket = os.environ.get("HOB_S3_BUCKET", "").strip()
    if not endpoint or not bucket:
        return None
    try:
        ttl = int(os.environ.get("HOB_S3_URL_TTL", "3600") or "3600")
    except ValueError:
        ttl = 3600
    return {
        "endpoint": endpoint.rstrip("/"),
        "bucket": bucket,
        "key_id": os.environ.get("HOB_S3_KEY_ID", ""),
        "secret": os.environ.get("HOB_S3_SECRET", ""),
        "prefix": os.environ.get("HOB_S3_PREFIX", "hob-bridge").strip("/") or "hob-bridge",
        "ttl": ttl,
    }


def _upload_images(job_id, images_b64, client_factory=None):
    """Upload result images to the configured bucket, return presigned URLs.

    Returns None when S3 is not configured or anything fails — the response
    then carries base64 images as usual. `client_factory` exists for tests.
    """
    config = _s3_config()
    if config is None or not images_b64:
        return None
    try:
        if client_factory is not None:
            client = client_factory(config)
        else:
            import boto3
            from botocore.client import Config as BotoConfig
            client = boto3.client(
                "s3",
                endpoint_url=config["endpoint"],
                aws_access_key_id=config["key_id"],
                aws_secret_access_key=config["secret"],
                region_name=os.environ.get("HOB_S3_REGION", "auto"),
                config=BotoConfig(signature_version="s3v4"),
            )
        urls = []
        for index, encoded in enumerate(images_b64):
            key = f"{config['prefix']}/{job_id}/{index}.png"
            client.put_object(Bucket=config["bucket"], Key=key,
                              Body=base64.b64decode(encoded), ContentType="image/png")
            urls.append(client.generate_presigned_url(
                "get_object",
                Params={"Bucket": config["bucket"], "Key": key},
                ExpiresIn=config["ttl"]))
        return urls
    except Exception as exc:
        print(f"[hob-bridge] S3 upload failed, returning base64 images: {exc!r}", flush=True)
        return None


def _start_a1111():
    args = [
        "python", "launch.py",
        "--api", "--listen", "--port", "7860",
        "--skip-install",
        "--ckpt-dir", f"{MODEL_DIR}/Stable-diffusion",
        "--vae-dir", f"{MODEL_DIR}/VAE",
        "--lora-dir", f"{MODEL_DIR}/Lora",
        "--hypernetwork-dir", f"{MODEL_DIR}/hypernetworks",
        "--embeddings-dir", f"{MODEL_DIR}/embeddings",
        "--esrgan-models-path", f"{MODEL_DIR}/ESRGAN",
        "--swinir-models-path", f"{MODEL_DIR}/SwinIR",
        "--ldsr-models-path", f"{MODEL_DIR}/LDSR",
        "--scunet-models-path", f"{MODEL_DIR}/ScuNET",
        # registered by the ControlNet extension baked into the image
        "--controlnet-dir", f"{MODEL_DIR}/ControlNet",
        "--controlnet-annotator-models-path", f"{MODEL_DIR}/Annotators",
    ]
    extra = os.environ.get("HOB_A1111_ARGS", "").split()
    if extra:
        args.extend(extra)
    subprocess.Popen(args, cwd=A1111_DIR)


def _wait_until_ready(url=None):
    url = url or f"{A1111_URL}/sdapi/v1/options"  # comfy_handler passes its own
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"backend not ready after {STARTUP_TIMEOUT}s ({url})")


def _warm(base, checkpoint, timeout):
    """Load `checkpoint` in A1111 now, plus a 1-step generation; returns error text or None.

    Setting sd_model_checkpoint via /options reloads the weights synchronously
    and makes it the default, so later jobs naming it (or none) don't swap.
    """
    try:
        resp = requests.post(f"{base}/sdapi/v1/options",
                             json={"sd_model_checkpoint": checkpoint}, timeout=timeout)
        if not resp.ok:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text[:300]
            return f"HTTP {resp.status_code}: {_a1111_error_text(body)}"
        requests.post(f"{base}/sdapi/v1/txt2img",
                      json={"prompt": "warmup", "steps": 1, "width": 64, "height": 64},
                      timeout=timeout)
    except requests.RequestException as exc:
        return str(exc)
    return None


def _warm_default_model():
    """Optionally load a checkpoint at boot (paid on cold start, not per request).

    Best effort: a failed preload is logged, never fatal — the worker must
    still reach runpod.serverless.start() instead of crash-looping.
    """
    model = os.environ.get("HOB_PRELOAD_MODEL", "").strip()
    if model:
        error = _warm(A1111_URL, model, STARTUP_TIMEOUT)
        if error:
            print(f"[hob-bridge] HOB_PRELOAD_MODEL={model!r} not loaded: {error}", flush=True)


def _progress_update(job, payload):
    if runpod is None:
        return
    try:
        runpod.serverless.progress_update(job, payload)
    except Exception:
        pass


def _a1111_error_text(err):
    """A1111 error bodies are {error, detail, body, errors} — `errors` holds the
    actual message — or FastAPI's {detail}; anything else is stringified."""
    if isinstance(err, dict):
        return " ".join(str(err[key]) for key in ("error", "errors", "detail") if err.get(key))
    return str(err)


def _interrupt(base):
    try:
        requests.post(f"{base}/sdapi/v1/interrupt", timeout=10)
    except requests.RequestException:
        pass


def _call_a1111(base, path, params, timeout, on_progress=None, check_stop=None):
    """POST a generation to A1111 while relaying /progress; returns (data, error_text).

    `check_stop` may raise to abandon the wait. Whatever ends the wait early
    (check_stop, task cancellation, the request timing out) also interrupts
    A1111, so an abandoned generation stops occupying the GPU.
    """
    result = {}
    done = threading.Event()

    def _call():
        try:
            resp = requests.post(f"{base}{path}", json=params, timeout=timeout)
            try:
                data = resp.json()
            except ValueError:
                data = None
            if resp.ok and isinstance(data, dict) and not data.get("error"):
                result["data"] = data
            else:
                # not raise_for_status(): A1111's JSON body carries the real error
                result["error"] = (_a1111_error_text(data) if data
                                   else f"HTTP {resp.status_code}: {resp.text[:500]}")
        except requests.Timeout as exc:
            _interrupt(base)
            result["error"] = str(exc)
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            done.set()

    threading.Thread(target=_call, daemon=True).start()
    try:
        while not done.wait(timeout=PROGRESS_INTERVAL):
            if check_stop:
                check_stop()
            if not on_progress:
                continue
            try:
                progress = requests.get(f"{base}/sdapi/v1/progress?skip_current_image=true",
                                        timeout=5).json()
            except Exception:
                continue
            fraction = float(progress.get("progress") or 0.0)
            on_progress({
                "state": "generating" if fraction > 0 else "preparing (loading model or queued)",
                "percentage": round(fraction * 100, 1),
                "eta_relative": progress.get("eta_relative"),
            })
    except BaseException:
        _interrupt(base)
        raise
    if "error" in result:
        return None, result["error"]
    return result["data"], None


def _generate(job, kind, params, timeout=GENERATION_TIMEOUT):
    path = "/sdapi/v1/img2img" if kind == "img2img" else "/sdapi/v1/txt2img"
    data, error = _call_a1111(A1111_URL, path, params, timeout,
                              on_progress=lambda payload: _progress_update(job, payload))
    if error:
        return _failure(_classify_error(error))

    images_b64 = data.get("images", [])
    urls = _upload_images(job.get("id") or "job", images_b64)
    if urls:
        return {"type": kind, "images": urls, "images_are_urls": True,
                "info": data.get("info", "")}
    return {"type": kind, "images": images_b64, "info": data.get("info", "")}


def _collect_inventory(fetch):
    """What an A1111 API can see; a None category = that endpoint is unavailable.

    Checkpoint `name` is A1111's CheckpointInfo.name (the title minus its
    ' [hash]' suffix): a checkpoint_aliases key that override_settings can
    resolve — unlike `filename`, which is an absolute path.
    """
    def safe(path, mapper):
        try:
            return mapper(fetch(path))
        except Exception:
            return None

    def _checkpoints(items):
        return [{"name": re.sub(r" \[[0-9a-f]+\]$", "", m.get("title") or "") or m.get("model_name"),
                 "model_name": m.get("model_name"), "filename": m.get("filename")} for m in items]

    return {
        "type": "inventory",
        "checkpoints": safe("/sdapi/v1/sd-models", _checkpoints),
        "vae": safe("/sdapi/v1/sd-vae",
                    lambda items: [{"model_name": m.get("model_name"), "filename": m.get("filename")}
                                   for m in items]),
        "loras": safe("/sdapi/v1/loras",
                      lambda items: [{"name": l.get("name"), "alias": l.get("alias")} for l in items]),
        "embeddings": safe("/sdapi/v1/embeddings",
                           lambda data: sorted(list((data or {}).get("loaded") or {})
                                               + list((data or {}).get("skipped") or {}))),
        "upscalers": safe("/sdapi/v1/upscalers", lambda items: [u.get("name") for u in items]),
        "controlnet": safe("/controlnet/model_list",
                           lambda data: list((data or {}).get("model_list") or [])),
    }


def _inventory():
    def fetch(path):
        resp = requests.get(f"{A1111_URL}{path}", timeout=60)
        resp.raise_for_status()
        return resp.json()

    return _collect_inventory(fetch)


async def handler(job):
    # async so the SDK's stop signal (job cancelled / timed out) reaches us as
    # CancelledError; blocking work runs in a thread to keep the loop free
    job_input = job.get("input") or {}
    kind = job_input.get("type", "txt2img")
    if kind == "ping":
        return {"type": "ping", "ok": True}
    if kind == "inventory":
        return await asyncio.to_thread(_inventory)
    if kind == "warm":
        checkpoint = (job_input.get("params") or {}).get("checkpoint")
        if not checkpoint:
            return _failure(_classify_error("warm: no checkpoint given"))
        _progress_update(job, {"state": f"loading {checkpoint}"})
        started = time.time()
        error = await asyncio.to_thread(_warm, A1111_URL, checkpoint, GENERATION_TIMEOUT)
        if error:
            return _failure(_classify_error(error))
        return {"type": "warm", "checkpoint": checkpoint, "seconds": round(time.time() - started, 1)}
    if kind in ("txt2img", "img2img"):
        # the client's total budget also caps the A1111 call: should a stop
        # signal never arrive, the request timeout still interrupts the GPU work
        try:
            timeout = min(GENERATION_TIMEOUT, float(job_input.get("timeout") or GENERATION_TIMEOUT))
        except (TypeError, ValueError):
            timeout = GENERATION_TIMEOUT
        try:
            return await asyncio.to_thread(_generate, job, kind, job_input.get("params") or {}, timeout)
        except asyncio.CancelledError:
            _interrupt(A1111_URL)
            raise
    return _failure(_classify_error(f"unknown job type: {kind}"))


if __name__ == "__main__":
    _start_a1111()
    _wait_until_ready()
    _warm_default_model()
    runpod.serverless.start({"handler": handler})
