# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **ComfyUI backend**: `worker/comfy_handler.py` + `worker/Dockerfile.comfy`
  (ComfyUI v0.39.2) serve the same job envelope as the A1111 worker — no change
  to the extension. txt2img/img2img, batches, LoRAs, embeddings, CLIP skip, VAE,
  Hires fix, warm-up, S3/R2 output, cancel; ControlNet, inpainting and other
  unsupported options fail with an explicit `unsupported` error. Published to
  `ghcr.io/<owner>/hob-bridge-worker-comfy`. Tested against a real ComfyUI,
  including sampling (see CONTRIBUTING.md).
- **Setup check**: a first-run checklist with a 🟢/🔴 light per item (settings,
  RunPod `/health`, worker inventory, volume and models, local VAE, ControlNet).
  The panel opens by itself while the settings are incomplete; free checks
  run before the inventory job, which may pay a cold start.
- **Warm-up button** and `warm` job type: load a checkpoint on the worker
  before a session, shown as `pending → loading → ready`.
- **A/B mode**: re-run the same prompt with the remote's resolved seed locally;
  the local images follow the remote ones, with a pixel similarity per pair.
- **`hob sync`** (`tools/hob.py`): diff a local models folder against the
  network volume through RunPod's S3 API; `--dry-run`, `--only-new`; never
  deletes on the volume.
- Grids and ControlNet detected maps are saved locally per A1111/ControlNet
  settings; images are saved to `outputs/` per *Always save all generated images*.
- Cancelling a job stops the generation on the GPU (RunPod stop signal →
  A1111 `/interrupt`), backed by a RunPod `policy` and a worker-side timeout
  derived from the extension's total timeout.
- Direct-pod mode reuses the worker's A1111 client: progress, Interrupt and
  classified errors behave as in serverless mode.
- Preflight also checks the selected VAE.
- CI: unit tests (Python 3.10/3.11), integration tests against the real runpod
  SDK and the pinned ControlNet model, worker image build and publication to
  GHCR. `CONTRIBUTING.md`, this changelog.

### Changed

- Workers return a failed job's classified error as a JSON string in `error`
  (the SDK's contract: the job's `error` is a string; a dict made its local
  API fail). The extension already parsed both forms.
- The worker's `runpod` dependency is pinned to 1.12.0; `boto3` is installed
  for S3/R2 output.
- Remote checkpoints are identified by A1111's checkpoint name (relative path),
  which `override_settings` can resolve; `juggs`, `juggs.safetensors` and
  `sdxl/juggs.safetensors` are all accepted.
- Progress is read from `/status` while a job is `IN_PROGRESS` (no `/stream`).
- ControlNet units are forwarded under the pinned `ControlNetUnit` field names,
  with image shapes (sketch pairs, masks, multi-inputs) preserved; batch mode
  is refused.

### Fixed

- The worker looked for models in `/workspace`, but RunPod mounts a network
  volume at `/runpod-volume` in serverless workers: it would have seen no
  checkpoints. It now picks the mount that exists (`HOB_MODEL_DIR` overrides).
- Checkpoint overrides were silently ignored (an absolute path was sent).
- ControlNet: control images from the UI were dropped; `save_detected_map`,
  `resize_mode`/`control_mode` and `enabled` were not sent in a form the
  worker's ControlNet accepts.
- The A1111 worker image did not build or start (found by actually building and
  booting it): `--prepare-environment` is not an A1111 flag (now `--exit`);
  `Stability-AI/stablediffusion`, which A1111 v1.10.1 clones, is gone from
  GitHub (A1111's `dev` uses the `w-e-w` fork at the same pinned commit);
  ControlNet's loosely pinned requirements pulled numpy 2 and the RunPod SDK
  moved fastapi/websockets/protobuf, breaking scikit-image at import; its
  unpinned `mediapipe` resolved to a release without `mp.solutions`, so the
  ControlNet script did not load at all. ControlNet is now installed after
  A1111's own setup, under A1111's pins plus `mediapipe==0.10.11`, and the
  handler has its own virtualenv. Checked by booting both images on CPU and
  running jobs, including a canny ControlNet job built by the extension's
  converter (the model's infotext echoed every field).
- PNG metadata held the API's JSON instead of each image's infotext.
- A transient status-poll error aborted the run and left the remote job running.
- Worker errors (OOM, missing model) arrived unclassified or as raw dicts.
- A failing `HOB_PRELOAD_MODEL` could crash-loop the worker.
- img2img: "Inpaint not masked" inpainted the masked area instead.
- Interrupt during preflight raised a traceback; a stale interrupt flag could
  cancel the connection test.
- An empty LoRA list on the volume skipped the missing-LoRA check.
