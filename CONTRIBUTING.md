# Contributing

Hob Bridge is maintained in spare time. This file is the "how did this work
again?" for whoever picks it up next — very possibly the author, six months on.

## Layout

| Path | Runs where | Notes |
|---|---|---|
| `scripts/hob_bridge.py` | inside the local A1111 (Python 3.10) | Loaded by A1111 as a script: no new dependencies beyond what A1111 ships (`requests`, `PIL`, `numpy`, `gradio`). Every `.py` in `scripts/` is loaded by A1111, so helpers go elsewhere. |
| `worker/handler.py` | in the RunPod worker image (Python 3.11) | Also loaded by the extension for direct-pod mode (`_call_a1111`, `_collect_inventory`, `_warm`, `_classify_error`): keep it free of import-time side effects. |
| `worker/Dockerfile`, `worker/requirements.txt` | image build | Pins: A1111 tag, ControlNet commit, `runpod` version. |
| `worker/comfy_handler.py`, `worker/Dockerfile.comfy` | the ComfyUI image (Python 3.11+) | Imports `handler.py` for the shared bits (error classes, S3 output, progress), so both files ship in both images. Pin: ComfyUI release tag. |
| `tools/hob.py` | your machine | `hob sync`; needs `boto3`. |
| `tests/` | anywhere | Unit tests need neither A1111 nor network. |
| `tests/integration/` | CI, or by hand | Real runpod SDK / real ControlNet model; each needs its own environment. |

## Tests

```bash
python3 -m pip install -r tests/requirements.txt
python3 -m pytest tests            # integration modules skip themselves without their deps
python3 tests/test_comfy_worker.py # the ComfyUI worker, same idea
python3 tests/test_hob_bridge.py   # same unit tests, no pytest needed
```

The unit tests fake the outside world with stdlib only: a local
`http.server` stands in for RunPod and A1111 (see `_fake_remote` in
`tests/test_hob_bridge.py`), and `_patched` swaps module attributes such as
`_setting` or `gr`. Payload mapping is tested against a fake processing object
(`_fake_p`) — extend those when you touch `_build_params` or the ControlNet
serialization.

### Integration checks (run them when you bump a pin)

They need conflicting dependencies (the pinned ControlNet uses pydantic v1;
the runpod SDK doesn't), so each has its own environment — CI runs both:

```bash
# real runpod SDK: a stop signal must reach A1111 as /interrupt
python3 -m venv /tmp/hob-rp && /tmp/hob-rp/bin/pip install -r worker/requirements.txt pytest
/tmp/hob-rp/bin/python -m pytest -rs tests/integration/test_runpod_stop.py

# real ControlNetUnit at the commit pinned in worker/Dockerfile (downloaded by the test)
python3 -m venv /tmp/hob-cn && /tmp/hob-cn/bin/pip install "pydantic<2" numpy pillow opencv-python-headless requests pytest
/tmp/hob-cn/bin/python -m pytest -rs tests/integration/test_controlnet_payload.py

# real ComfyUI at the tag pinned in worker/Dockerfile.comfy, on CPU (a few GB of dependencies)
git clone --depth 1 --branch <tag> https://github.com/comfyanonymous/ComfyUI.git /tmp/ComfyUI
python3 -m venv /tmp/hob-comfy && /tmp/hob-comfy/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
/tmp/hob-comfy/bin/pip install -r /tmp/ComfyUI/requirements.txt requests pillow pytest
COMFYUI_DIR=/tmp/ComfyUI /tmp/hob-comfy/bin/python -m pytest -rs tests/integration/test_comfy_server.py
```

The ComfyUI test boots the server and checks that every graph the worker builds
passes ComfyUI's own validation. CI stops there; to also **sample for real**
(txt2img, Hires fix, img2img, warm-up, cancel), point it at any small SD1.5
checkpoint: `HOB_COMFY_TEST_CHECKPOINT=/path/to/model.safetensors` — about a
minute on CPU. Do that when bumping the ComfyUI pin.

## Bumping pinned versions

| What | Where | Then |
|---|---|---|
| A1111 | `--branch vX.Y.Z` in `worker/Dockerfile` | Re-check what the code relies on: `/sdapi/v1/*` fields (`title`/`model_name`/`filename` of sd-models, the `info` JSON's `infotexts`), `process_images`/`Processed` signatures, the grid logic replayed in `_batch_grid`. |
| ControlNet | commit hash in `worker/Dockerfile` | Run `test_controlnet_payload.py`; if the `ControlNetUnit` fields changed, update `_CN_FIELDS`. Keep the local extension on the same commit. |
| runpod SDK | `worker/requirements.txt` | Run `test_runpod_stop.py` (it drives the SDK's `JobScaler`). |
| ComfyUI | `--branch vX.Y.Z` in `worker/Dockerfile.comfy` | Run `test_comfy_server.py`, with a real checkpoint (see above). It catches renamed nodes/inputs and changed `/object_info`, `/history` or `/prompt` shapes; `_options` already handles both combo formats. |

CI builds the images on every push, so a broken pin shows up there — but
build them yourself first (`podman build`/`docker build`, then the smoke test
below): a green build is not a working worker. A1111's dependency pins and
third-party clones are the usual casualties (see `worker/Dockerfile`).

## Trying it for real

Unit tests can't cover a GPU. Cheapest real loop: a RunPod **pod** running the
worker image (or any A1111 with `--api`), the extension in `direct` mode
pointed at `http://<pod>:7860`, then the same with a serverless endpoint.
The extension's **Run setup check** is the first thing to try.

To smoke-test a worker image without RunPod or a GPU, run it with the SDK's
local API and send jobs to `/runsync` (this is how the ComfyUI image was
checked: boot with the volume mapping, preload, inventory, txt2img, Hires fix,
warm-up, and the `FAILED` error format):

```bash
podman run --rm --security-opt label=disable -v ./models:/workspace/models:ro \
  -e HOB_COMFY_ARGS=--cpu -e HOB_PRELOAD_MODEL=<checkpoint> <image> \
  python -u comfy_handler.py --rp_serve_api --rp_api_host 0.0.0.0 --rp_api_port 8000
# then, inside the container (podman exec ...): POST http://127.0.0.1:8000/runsync
#   {"input": {"type": "inventory"}}   {"input": {"type": "txt2img", "params": {...}}}
```

For the A1111 image use `/handler-venv/bin/python -u handler.py ...` and
`HOB_A1111_ARGS="--use-cpu all --skip-torch-cuda-test --no-half"` instead of
`HOB_COMFY_ARGS`; allow a minute for the boot and mount the volume writable
(A1111 creates missing folders such as `hypernetworks/`). Use `podman exec -i`
to feed a script on stdin. A ControlNet job needs a ControlNet model in
`ControlNet/` (e.g. `control_v11p_sd15_canny_fp16.safetensors`); a working one
returns the generated image **and** the detected map, and the infotext lists
the unit. Note A1111 swallows an exception raised by a script's `process()`
and returns an uncontrolled image, and writes the unit into the infotext
*before* its model is built — so neither "COMPLETED" nor "ControlNet in the
infotext" proves ControlNet ran; the detected map and the worker log do.
Observed while testing: A1111 logs `Failed to create model quickly; will retry
using slow method` at boot with these pins (a longer model load, not an error).

## Conventions

- Match the surrounding code; stdlib first; no new dependency for what a few
  lines can do.
- Deliberate shortcuts carry a `# ponytail:` comment naming their limit and the
  upgrade path — grep for them before assuming something is an oversight.
- Every non-trivial branch gets one small test that would fail if it broke.
- Update `CHANGELOG.md` (`Unreleased`) in the same change.

## Releasing

1. Move the `Unreleased` entries in `CHANGELOG.md` under a new version heading.
2. Tag it: `git tag vX.Y.Z && git push --tags`.
3. CI tests, builds and publishes `ghcr.io/<owner>/hob-bridge-worker:X.Y.Z`
   (plus `:X.Y` and `:latest`); `main` is published as `:edge`.
4. Point the RunPod endpoint at the new tag.
