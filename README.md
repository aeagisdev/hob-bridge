# Hob Bridge

> Keep your assets local, offload the heavy lifting.

Hob Bridge lets you drive the **Automatic1111 Stable Diffusion WebUI from your local machine while the model runs on RunPod** — like the house spirit of English folklore that does the heavy chores overnight: everything stays familiar at home, the hob does the work elsewhere.

- **All assets stay local**: checkpoints, LoRAs and embeddings live on your machine, mirrored to a RunPod network volume
- **Models travel by reference**: job payloads name the model files; multi-GB weights never cross the wire
- **Serverless economics**: RunPod workers scale to zero, you pay per second of GPU time
- **Stock A1111 experience**: pick *Hob Bridge* in the Scripts dropdown, press Generate, get images in the local gallery with full infotext
- **Live progress & cancel**: remote progress is mirrored into the UI status line; clicking Interrupt cancels the remote job and stops the generation on the GPU
- **Fail-fast preflight**: missing checkpoints/LoRAs/VAE are caught *before* you pay for a job
- **ControlNet travels too**: your local ControlNet units are forwarded through the standard `alwayson_scripts` API payload
- **Readable remote errors**: OOM, missing models, timeouts and unreachable A1111 come back classified with a hint — not a raw traceback
- **Self-contained PNGs**: the remote infotext is re-embedded into every returned image, so metadata survives even outside A1111
- **Setup check**: a first-run checklist with a 🟢/🔴 light per item — settings, API key, worker, volume, models
- **Warm-up button**: load the checkpoint you're about to use *before* a session (`pending → loading → ready`), so you pay the load when you choose
- **A/B mode**: optionally re-run the same prompt and seed locally and get a per-image similarity score
- **`hob sync`**: diff your local models against the volume and upload only what's missing or changed
- **Two backends**: the default A1111 worker, or a ComfyUI worker that takes the very same jobs — pick it by choosing the endpoint's image; the extension doesn't change

## Architecture

```
[Your machine]                          [RunPod]
A1111 (CPU, UI only)                    Serverless endpoint (FlashBoot)
  └─ Hob Bridge extension               └─ A1111 worker (--api) + ControlNet
      ├─ POST /run ──── job ─────────▶      ├─ mounts Network Volume
      ├─ GET  /status ◀─ poll+progress ▶    │   ├─ models/Stable-diffusion/
      │                                     │   ├─ models/Lora/ VAE/ embeddings/
      └─ POST /cancel (Interrupt)           │   ├─ models/ESRGAN/ SwinIR/ … (upscalers)
                                           │   └─ models/ControlNet/ Annotators/
                                           ◀── images (base64 or S3/R2 URLs) + infotext
[Asset sync]  tools/hob.py sync ───────▶  Network Volume (S3-compatible API)
```

See [docs/architecture.md](docs/architecture.md) for the design constraints
(payload limits, cold starts, why jobs poll instead of using `/runsync`) and
why models go by reference.

## Components

| Path | What it is |
|---|---|
| `scripts/hob_bridge.py` | A1111 extension. A Scripts-dropdown entry that replaces local generation with a remote job: styles/clip-skip expansion, ControlNet serialization, preflight checks, /run + polling with progress and cancel. |
| `worker/handler.py` | RunPod serverless handler. Boots A1111 (+ ControlNet) in `--api` mode, forwards jobs, reports progress, interrupts A1111 on cancel, answers `ping`/`inventory`. |
| `worker/Dockerfile` | Worker image: A1111 + ControlNet + RunPod SDK, deps baked in for fast cold starts. Built by CI and published to `ghcr.io/<owner>/hob-bridge-worker`. |
| `worker/comfy_handler.py`, `worker/Dockerfile.comfy` | The ComfyUI backend: maps the same job envelope to a ComfyUI graph. Image published to `ghcr.io/<owner>/hob-bridge-worker-comfy`. |
| `tools/hob.py` | Command-line helpers: `hob sync` (local models ↔ network volume). |
| `tests/` | Unit tests (no A1111, no network) and `tests/integration/` checks against the real RunPod SDK and ControlNet model — see [CONTRIBUTING.md](CONTRIBUTING.md). |
| `.github/workflows/ci.yml` | Tests on every push/PR; builds the worker image and publishes it to GHCR. |

## Quickstart

### 1. Storage: network volume + model sync

1. Create a RunPod **Network Volume** in the datacenter you'll deploy to, e.g. 50 GB.
2. Lay out the directory structure the worker expects:

   ```
   models/
   ├── Stable-diffusion/   # checkpoints
   ├── Lora/               # LoRAs
   ├── VAE/
   ├── embeddings/
   ├── hypernetworks/
   ├── ESRGAN/  SwinIR/  LDSR/  ScuNET/   # upscaler models (Hires fix)
   ├── ControlNet/         # controlnet models (*.pt / *.safetensors)
   └── Annotators/         # controlnet preprocessor models
   ```

3. Mirror your local collection with `hob sync`, which talks to RunPod's
   [S3-compatible Volume API](https://docs.runpod.io/serverless/storage/s3-api)
   (create an *S3 API key* in the RunPod console → Settings; `pip install boto3`):

   ```bash
   export HOB_VOLUME_ID=<network-volume-id> HOB_VOLUME_DATACENTER=EU-RO-1
   export HOB_VOLUME_KEY_ID=user_... HOB_VOLUME_SECRET=rps_...
   python tools/hob.py sync ./models --dry-run    # what's missing / different / volume-only
   python tools/hob.py sync ./models --only-new   # upload missing files, never overwrite
   python tools/hob.py sync ./models              # upload missing + changed (by size)
   ```

   Your local folder must have the layout above; symlinked folders are
   followed (e.g. `models/embeddings → ../embeddings` for A1111's own layout).
   Nothing on the volume is ever deleted — volume-only files are just
   reported. Any other S3 client works too, e.g.
   `rclone copy ./models runpod-volume:models`.

### 2. Worker: build and deploy the serverless endpoint

CI publishes the image to `ghcr.io/<owner>/hob-bridge-worker` (`:X.Y.Z` for
releases, `:edge` for `main`). To build your own:

```bash
cd worker/
docker build -t ghcr.io/<you>/hob-bridge-worker .
docker push ghcr.io/<you>/hob-bridge-worker
```

Then in RunPod → Serverless → **New Endpoint**:

- Container image: `ghcr.io/<you>/hob-bridge-worker` (pin a release tag)
- Attach your network volume (Edit Endpoint → Advanced → Network Volumes). RunPod mounts it at `/runpod-volume` in serverless workers (`/workspace` in pods); the worker detects which, so `HOB_MODEL_DIR` is normally not needed
- Enable **Flash Boot**
- **Max workers: 1** — A1111 serializes generations anyway, so a second concurrent job would boot a second worker (another cold start + another model load) instead of waiting in queue. RunPod queues jobs while the worker is busy.
- **Idle timeout**: how long a worker stays up after its last job. The default
  (seconds) scales to zero almost at once; for interactive sessions — and for
  the warm-up button to be worth it — raise it to a few minutes. Idle time is
  billed: it's the price of skipping cold starts between prompts.
- Env vars:

  | Var | Meaning |
  |---|---|
  | `HOB_MODEL_DIR` | Model root on the volume (default: `/runpod-volume/models` on serverless, `/workspace/models` in pods) |
  | `HOB_PRELOAD_MODEL` | Checkpoint file name to load at boot — set your most-used one; saves a model swap on every warm request |
  | `HOB_A1111_ARGS` | Extra A1111 flags, e.g. `--xformers` |
  | `HOB_STARTUP_TIMEOUT` | Seconds to wait for A1111 at boot (default 600) |
  | `HOB_GENERATION_TIMEOUT` | Per-request timeout forwarded to A1111 (default 900) |
  | `HOB_PROGRESS_INTERVAL` | Seconds between progress updates, read by the extension from `/status` (default 1.5) |
  | `HOB_S3_ENDPOINT`, `HOB_S3_BUCKET` | Optional S3/R2 output: when both are set, result images are uploaded and returned as presigned URLs (falls back to base64 on failure, see worker logs) |
  | `HOB_S3_KEY_ID`, `HOB_S3_SECRET`, `HOB_S3_REGION`, `HOB_S3_PREFIX`, `HOB_S3_URL_TTL` | Bucket credentials, region (default `auto`), key prefix (default `hob-bridge`), URL lifetime in seconds (default 3600) |

#### Optional: the ComfyUI backend instead

Deploy `ghcr.io/<owner>/hob-bridge-worker-comfy` (or build
`worker/Dockerfile.comfy`) the same way and point the extension at that
endpoint — nothing else changes: same settings, setup check, warm-up button,
preflight, `hob sync`. It reads the same volume layout (`Stable-diffusion/` →
checkpoints, `Lora/`, `VAE/`, `embeddings/`, `ESRGAN/` `SwinIR/` `ScuNET/` →
upscale models) and the same env vars, plus `HOB_COMFY_ARGS` for extra ComfyUI
flags. Serverless only: `direct` mode speaks the A1111 API.

| | ComfyUI worker |
|---|---|
| Supported | txt2img, img2img (one init image), `batch_size` / `n_iter`, samplers and schedulers, `<lora:name:w>` (and `<lora:name:te:unet>`), embeddings, CLIP skip, VAE override, Hires fix (latent upscalers and ESRGAN-type model upscalers), seed, the warm-up job, S3/R2 output |
| Not supported | ControlNet and other A1111 scripts, inpainting masks, *Resize and fill*, batch img2img, per-image prompt lists. These raise an explicit `unsupported` error — nothing is silently dropped. |
| Differs | Progress shows the state (queued / generating), not percentages. A1111-only prompt syntax (`[a\|b]`, `BREAK`, `AND`, `[from:to:when]`) is passed through untranslated. Images of a batch share the graph's seed. "Automatic" scheduler is approximated per sampler. Noise generation differs from A1111, so an A/B comparison shows the same idea, not the same pixels. |

If ControlNet is installed locally, the setup check shows 🔴 *ControlNet on the
worker* for this backend — that is accurate.

### 3. Local: A1111 without a GPU

The local copy never loads a model — it only serves the UI. Launch it on any
machine, GPU not required (A/B mode is the exception: it needs a local model,
and is only practical with a local GPU):

```bash
./webui.sh --skip-torch-cuda-test --use-cpu all
```

### 4. Extension: install and configure

In the local WebUI → Extensions → *Install from URL* → paste this repo URL →
*Apply and restart UI*.

Then Settings → **Hob Bridge**:

| Setting | Value |
|---|---|
| Endpoint mode | `serverless` (RunPod endpoint) or `direct` (a pod running A1111 with `--api`) |
| Endpoint URL | `https://api.runpod.ai/v2/<endpoint-id>` — or `http://<pod-ip>:7860` for direct mode. A full `/runsync` URL is tolerated and normalized. |
| RunPod API key | Required in serverless mode |
| Total job timeout | 600 s is a sane floor: cold starts are slow, warm requests are seconds. Covers the whole submit→poll cycle. |

### 5. Setup check

In txt2img or img2img, open the **Scripts** dropdown and select **Hob Bridge
(run on RunPod)**. Until the settings are complete, the **Hob Bridge setup &
status** panel opens by itself with the settings already checked. Press
**Run setup check** for the full checklist, one 🟢/🔴 light per item:

- endpoint URL and API key (settings);
- RunPod `/health` — endpoint id and key are valid, workers and queue (free,
  starts nothing);
- the worker answers an `inventory` job (may pay a cold start — runs only once
  the free checks are green);
- the volume: checkpoints (red if none — volume not mounted?), LoRAs, VAEs,
  embeddings, upscalers; your local VAE; ControlNet models if you use
  ControlNet locally.

It also fills the **Remote checkpoint** dropdown with the volume's model names
(you can still type one). If you're debugging, try `direct` mode against a pod
first — fewer moving parts.

### 6. Generate

Press Generate. During the run the status line shows queue state and remote
progress (`Hob Bridge: generating — 42% ~8s left`); clicking **Interrupt**
cancels the remote job via `/cancel` and the worker interrupts A1111. Images
land in the local gallery with the remote run's infotext (seed included) and
are saved to your local `outputs/` per A1111's *Always save all generated
images* setting.

Before each job (unless *Skip remote asset check* is ticked) Hob Bridge
verifies against the inventory that:

- the optional **Remote checkpoint** field exists on the volume (name with or
  without extension), and
- every `<lora:name>` referenced in the prompt is present, and
- the selected VAE (if not Automatic/None) is on the volume.

A failed check raises an error *before* the job is submitted — you never pay
for a doomed request. The inventory is cached for 5 minutes.

**Warm-up.** Pick a checkpoint and press **Warm up checkpoint**: the worker
loads it now and the box walks through `pending` (waiting for a worker) →
`loading` → `ready`, with the load time. It stays loaded — and becomes the
worker's default — until the worker scales down (see *Idle timeout*).

**A/B mode.** Tick *A/B: also generate locally* and, after the remote job,
the same prompt runs on this machine with the remote's resolved seed (and the
same checkpoint, if you have it locally). The local images follow the remote
ones in the gallery; the line under the gallery reports a pixel similarity per
pair (100% = identical). Expect close but rarely identical results: GPUs,
attention implementations and precision differ.

Leave the Scripts dropdown on *None* to generate locally as usual.

## Job format

The wire format is deliberately small and backend-agnostic:

```json
{
  "type": "txt2img",
  "params": {
    "prompt": "a photograph of a hob napping by the fireplace <lora:cozy:0.7>",
    "negative_prompt": "",
    "steps": 30,
    "cfg_scale": 7,
    "width": 1024,
    "height": 1024,
    "sampler_name": "DPM++ 2M",
    "scheduler": "karras",
    "seed": -1,
    "batch_size": 1,
    "n_iter": 1,
    "alwayson_scripts": {
      "controlnet": {
        "args": [
          { "enabled": true, "module": "canny", "model": "control_v11p_sd15_canny", "weight": 1.0, "image": "<base64>" }
        ]
      }
    },
    "override_settings": { "sd_model_checkpoint": "juggernautXL_v9.safetensors" }
  }
}
```

- Job types: `txt2img` | `img2img` | `ping` | `inventory` | `warm` (`params: {"checkpoint": ...}`).
- `params` uses the standard A1111 API field names, so the worker forwards it
  nearly 1:1 to `/sdapi/v1/txt2img` or `/sdapi/v1/img2img`.
- Checkpoint, LoRAs and CLIP skip / VAE are **resolved on the worker's
  volume**; LoRA references travel inside the prompt exactly as A1111 expects.
- Styles are applied locally before submission; the remote receives the
  expanded prompt.
- `seed: -1` is rolled remotely; the real seed comes back in `info`.
- ControlNet units configured in the local UI are serialized into
  `alwayson_scripts` (the extension must be installed locally for its UI;
  the worker image carries the remote side). Fields use the names of the
  worker's pinned ControlNet; keep the local extension on the same version.
- The envelope also carries `"timeout"` (the extension's total timeout) and
  the `/run` request a RunPod `policy` (`executionTimeout`, `ttl`) derived
  from it, so an abandoned job ends server-side even if `/cancel` never
  arrives.

## Current limitations

- The refiner / second-pass options of third-party extensions are not forwarded.
- Local-only scripts (XYZ-Plot etc.) run against the local pipeline and don't
  apply to remote jobs.
- Progress percentages come from A1111's sampling progress: model loading
  during a warm checkpoint swap shows as "preparing" without a percentage.
- Interrupt cancels the remote job and stops its generation, but GPU seconds already
  consumed by the worker are billed as usual.
- Without `HOB_S3_*`, results come back as base64 inside the job response:
  the RunPod payload cap (~10 MB) limits batch size × resolution.
- ControlNet *batch* mode (a local input folder) is refused; single image,
  sketch masks, uploaded masks and multi-inputs are forwarded.
- Grids and ControlNet detected maps are saved per your local settings, with
  two approximations: the grid's infotext is the first image's (A1111 uses
  the main prompt — they differ only with per-image prompts), and when units
  use several preprocessors, auto-saved maps share a `controlnet/` folder
  instead of one folder per preprocessor.
- The A1111 worker inherits two A1111 behaviors that fail quietly. An unknown
  checkpoint name is ignored and the default model generates (the extension's
  preflight prevents this unless you tick *Skip remote asset check*; the
  ComfyUI worker errors instead). And if ControlNet raises while preparing a
  unit, A1111 still returns an image, just without ControlNet — it shows only
  in the worker log (`Error running process`). Check the detected-map image
  that comes back with a ControlNet job: no map, no ControlNet.
- Stopping a running generation on cancel relies on RunPod's per-job stop
  signal (SDK 1.10+, the image pins 1.12.0). Without it, the job's `timeout`
  still interrupts A1111 when the client's budget runs out.

## Roadmap

- [x] Upload outputs to S3/R2 and return URLs (`HOB_S3_*`)
- [x] Asset sync helper around the S3-compatible Volume API (`hob sync`)
- [x] Pin a tested A1111 release in the Dockerfile (v1.10.1, built by CI)
- [x] ComfyUI worker adapter (`worker/comfy_handler.py`): same envelope, no change to the extension
- [ ] ComfyUI: ControlNet (needs the preprocessor custom nodes), inpainting, progress percentages (websocket)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to run the tests, bump the
pinned versions and cut a release; changes are tracked in
[CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
