# Architecture notes

Why Hob Bridge is shaped the way it is.

## The core constraint

A1111 cannot split its UI from its compute: the model runs where A1111 runs.
Rather than reimplementing a UI, Hob Bridge keeps the *whole* A1111 experience
local and replaces only the generation step — a Scripts-dropdown entry whose
`run()` serializes the request, sends it to a remote A1111 in API mode, and
feeds the results back through the normal gallery/save/infotext flow.

The local A1111 never loads a model (`--use-cpu all --skip-torch-cuda-test`):
it exists to render the UI and to run the extensions the user likes.

## Models go by reference, not by value

RunPod serverless job payloads are capped at roughly 10 MB, so multi-GB
checkpoints can never travel in a request anyway — but the deeper reason is
ownership: the user's machine holds the canonical asset collection, and a
RunPod **network volume** mirrors it. Jobs reference files by name:

- checkpoint → `params.override_settings["sd_model_checkpoint"]` (A1111 checkpoint name)
- LoRAs → `<lora:name:weight>` inside the prompt (A1111 resolves them)
- CLIP skip / VAE → `params.override_settings` (local settings win)
- control/init images → small, base64 in the payload

New assets reach the volume through RunPod's S3-compatible Volume API — no
helper pod needed. `tools/hob.py sync` diffs the local folder against the
volume by relative path and size and uploads only the difference (`--dry-run`
reports, `--only-new` never overwrites, nothing is ever deleted). It works
around the API's limits: parts of 32 MB, 4 at a time (the documented 500 MB cap
is not the practical one: 256 MB parts got HTTP 524 from RunPod's proxy), a 2 h read timeout for multi-GB completions, and a retry for
the slow first listing of files written outside the S3 API. Any other S3
client (rclone, s5cmd) works too. The worker mounts every model category A1111 knows about
(checkpoints, LoRA, VAE, embeddings, hypernetworks, ESRGAN/SwinIR/LDSR/ScuNET
upscalers, ControlNet models and annotators) so Hires-fix upscalers and
ControlNet work remotely too.

## Job lifecycle: /run + polling, not /runsync

`/runsync` blocks until the job finishes — but RunPod caps it at roughly
90 seconds, and a cold start plus checkpoint load can easily exceed that. So
the extension uses the full asynchronous flow:

1. `POST {base}/run` → job id
2. `GET {base}/status/{id}` every ~1.2 s until `COMPLETED` / `FAILED` /
   `CANCELLED` / `TIMED_OUT`
3. while `IN_PROGRESS`, the `output` field of `/status` holds the worker's
   latest progress update, mirrored into `shared.state` so the stock UI
   status line shows `Hob Bridge: generating — 42% ~8s left`. Failed polls
   (network blips, RunPod 5xx) are retried for up to 60 s; after that the
   job is cancelled rather than left running unwatched
4. clicking Interrupt (or exceeding the configured total timeout) calls
   `POST {base}/cancel/{id}` and the run ends locally with an empty gallery.
   RunPod sends the worker a stop signal; the SDK (1.10+, pinned 1.12.0)
   cancels the (async) handler, which calls A1111's `/sdapi/v1/interrupt` so
   the GPU stops too. Two fallbacks bound an abandoned job when no stop
   signal arrives (client died, older SDK, stop channel unavailable): the
   `/run` request carries a RunPod `policy` (`executionTimeout` = the total
   timeout, `ttl` = total + 60 s), and the job's `timeout` caps the worker's
   A1111 request, whose expiry also interrupts A1111

The worker reports progress by polling A1111's `/sdapi/v1/progress` while
the generation call is in flight in a side thread, and pushing updates
through `runpod.serverless.progress_update`. In direct-pod mode the extension
runs the same A1111 client code (`_call_a1111`, loaded from
`worker/handler.py`), so progress, Interrupt and classified errors behave the
same. The endpoint base URL is what the user configures;
`/runsync` and `/run` suffixes are stripped if pasted.

## Preflight: don't pay for doomed jobs

Before submitting, the extension fetches the worker's **inventory**
(checkpoints, VAEs, LoRAs, embeddings, upscalers, ControlNet models — cached
for 5 minutes) and verifies that the requested checkpoint, the selected VAE
and every `<lora:...>` in the (style-expanded) prompt exist on the volume.
A checkpoint typed as `juggs`, `juggs.safetensors` or `sdxl/juggs.safetensors`
is resolved to A1111's checkpoint name (path relative to the model dir), which
is what `override_settings` can resolve — the API's absolute `filename` is not. Failures raise a UI error
before anything is billed.

The **setup check** in the script panel is the same idea for the whole
installation: free checks first (settings; RunPod's `/health`, which validates
endpoint id and API key without starting a worker), then — only if those are
green — the inventory job, judged item by item (no checkpoints = volume not
mounted; local VAE or ControlNet missing remotely). The panel opens by itself
while the settings are incomplete.

## Styles and settings

At script `run()` time `p.prompt` is still raw (style expansion happens later
in the pipeline, and `p.all_prompts` repeats prompts n_iter times, which the
API would multiply again). The extension therefore expands styles itself via
`shared.prompt_styles` and also forwards `scheduler`, `clip_skip`
(as `override_settings.CLIP_stop_at_last_layers`) and the local VAE choice.

## ControlNet

The local ControlNet extension's units are read through its
`external_code.get_all_units_in_processing(p)` API (imported via
`scripts.external_code` or `internal_controlnet.external_code`, depending on
the installed version) and forwarded inside `params.alwayson_scripts`. The
worker's ControlNet rebuilds each unit with `ControlNetUnit.from_dict` — a
pydantic model that silently ignores unknown keys — so fields are forwarded
under the model's own names, enums as their (string) values, `enabled: true`
explicitly (the API default is false), and unset fields are omitted so both
sides apply the same defaults. Images keep their shape (`{image, mask}`
sketch pairs, uploaded `mask`, multi-input lists, `effective_region_mask`)
with arrays encoded as lossless PNG base64: the worker decodes them to the
same arrays the local unit holds. This round trip was checked against the
real `ControlNetUnit` of the pinned commit. Batch mode reads a local folder
and is refused. Since the worker's detected maps come back after the
generated images, the extension also replays ControlNet's map autosave and
A1111's grid creation locally. If ControlNet
is installed locally but its API can't be imported, the bridge refuses to
send the job rather than silently dropping the user's ControlNet settings.
The worker image ships the ControlNet extension and mounts
`ControlNet/` + `Annotators/` from the volume.

## Cold starts and checkpoint swapping

With Flash Boot, container boot is sub-second; the real cold-start cost is
loading the checkpoint from the network volume (tens of seconds for an SDXL
checkpoint, billed as GPU time). Two mitigations:

1. `HOB_PRELOAD_MODEL` loads the most-used checkpoint during boot, so it's
   paid once per cold start instead of on the first request.
2. Warm workers switch checkpoints via per-request `override_settings`, but a
   swap costs another model load — if you batch-work with one checkpoint,
   leave the override empty and keep the preloaded default.
3. The **warm-up button** sends a `warm` job: the worker sets
   `sd_model_checkpoint` through `/sdapi/v1/options` — A1111 reloads the
   weights synchronously and makes it the default — plus a 1-step generation.
   The UI follows the job's RunPod status (`IN_QUEUE` → pending,
   `IN_PROGRESS` → loading, `COMPLETED` → ready). The load pays off only if
   the worker stays up until you generate: set the endpoint's idle timeout
   accordingly (idle time is billed).

Deploy the endpoint with **max workers = 1**: A1111 processes one generation
at a time, so a second concurrent job gains nothing — it would boot another
worker (cold start + model load) instead of queueing. RunPod queues jobs
while the worker is busy.

Order of magnitude: a warm SDXL generation on a 4090-class worker costs
fractions of a cent; even amortizing cold starts, per-image cost stays in the
cents range.

## Job envelope

```json
{ "type": "txt2img" | "img2img" | "ping" | "inventory" | "warm",
  "params": { ...A1111 API fields... },
  "timeout": 600 }
```

`params` reuses the standard A1111 API schema on purpose: the worker forwards
it 1:1, direct-pod mode posts it verbatim, and the extension builds it from
the processing object without translation layers. The ComfyUI backend is
therefore a second worker that accepts the same envelope and maps `params` to
a graph internally — the extension doesn't change.

## ComfyUI backend

`worker/comfy_handler.py` boots a local ComfyUI next to the RunPod handler and
serves the same job types, so the extension, preflight, warm-up button, setup
check and `hob sync` work unchanged.

- **Models.** An `extra_model_paths` config (generated at boot) maps ComfyUI's
  folders onto the volume layout the A1111 worker uses, so one volume serves
  both backends. The inventory is built from `/object_info`, shaped like the
  A1111 one (a checkpoint's `name` is its path under `Stable-diffusion/`; VAE
  `model_name` keeps the extension, as A1111 reports it), so the extension's
  checks work as they are. `/object_info` has two combo formats
  (`[[...], {}]` and `["COMBO", {"options": [...]}]`); both are read.
- **Graph.** `_workflow` turns A1111 params into an API graph:
  checkpoint → `<lora:...>` chain (`te`/`unet` weights) → CLIP skip → text
  encode (embedding names rewritten to `embedding:name`) → `KSampler`, with
  `LoadImage → ImageScale → VAEEncode` for img2img and, for Hires fix, either
  `LatentUpscale` or decode → `ImageUpscaleWithModel` → `ImageScale` to the
  exact target → encode, then a second `KSampler`. A1111 sampler/scheduler
  labels (including pre-1.9 "DPM++ 2M Karras") are mapped by table.
  `n_iter` runs one graph per iteration (seed + `batch_size` each), keeping
  VRAM bounded as in A1111.
- **Running it.** `POST /prompt`, then poll `/history/{id}` (ComfyUI records a
  prompt there only when it has finished) and `/queue` for the
  queued/generating state; images are fetched from `/view`. Validation
  failures (HTTP 400) and execution errors (`execution_error` messages) are
  turned into text and classified like A1111's (`gpu OOM`, `model missing`,
  `timeout`…).
- **Cancel.** The handler is async, as in the A1111 worker: a RunPod stop
  signal becomes `CancelledError`, which removes the prompt from the queue and
  interrupts it (`POST /interrupt` with the prompt id, so only that prompt).
  Checked against a real sampling run: ComfyUI is idle again in well under a
  second and records `execution_interrupted`.
- **Warm-up.** ComfyUI loads models lazily and has no "current checkpoint", so
  `warm` runs a 1-step 64×64 graph with it — which loads and caches the model —
  and remembers it as the default for jobs without an override.
- **Output.** The `info` JSON mirrors A1111's (`seed`, `all_seeds`,
  `infotexts`, …) with infotexts written in A1111's format, so PNG Info and
  "send to txt2img" work on the images.

What it refuses rather than approximates: ControlNet and other scripts
(ComfyUI core has no preprocessors), inpainting masks, *Resize and fill*,
batch img2img, per-image prompt lists. Each returns an `unsupported` error.
Percentage progress would need ComfyUI's websocket (`websocket-client`); the
state-only version avoids the dependency.

## A/B mode

With *A/B: also generate locally*, after the remote result `run()` calls
A1111's own `process_images(p)` on the same processing object, with
`p.seed`/`p.subseed` set to the remote's resolved seeds and the remote
checkpoint if it exists locally. The local run goes through the normal
pipeline (saving, grids, local ControlNet); its images are appended after the
remote ones and the comments line reports `100 − mean absolute RGB
difference` per pair — a pixel metric, not a perceptual one. A failed local
run is reported there too, never at the expense of the paid remote result.

## Output path

Responses carry images (base64, or presigned S3/R2 URLs when `HOB_S3_*` is
set — the escape hatch for the ~10 MB response cap) plus A1111's `info`:
the JSON of `processed.js()`, whose `infotexts` list gives each image its
local infotext (seed, model, sampler — everything the PNG metadata would
have had). Because the script's `run()` bypasses `process_images()`, the
extension saves the images to `outputs/` itself when *Always save all
generated images* is on.

The extension also re-encodes every returned PNG with the remote infotext
embedded as the standard `parameters` chunk — API-returned images carry no
metadata by themselves, and without this a PNG dragged out of A1111 would
lose prompt/seed/settings. Re-encoding never replaces an image: on failure
the original bytes are kept.

## Classified remote errors

The worker maps failures to `{kind, message, hint}` via `_classify_error`
(`gpu OOM`, `model missing`, `timeout`, `a1111 unreachable`, `error`),
covering both transport exceptions and A1111's `{error, detail}` payloads.
The extension renders them with `_format_remote_error` as a single readable
message — kind, message and hint — instead of a raw traceback. Client-side
preflight (missing checkpoint/LoRA/VAE) catches the most common "model missing"
case before a job is even submitted.

On the wire the classified error travels as a **JSON string** in the job's
`error` (`_failure`): the RunPod SDK moves a returned `error` into the job's
`error` field, which is a string there — the SDK's own local API rejects a
dict, and its exception path uses `json.dumps`. The job then ends `FAILED`
with that string, and the extension parses it back (it also accepts dicts and
Python-repr strings). The ComfyUI worker adds one kind, `unsupported`, for
features it refuses (ControlNet, inpainting…).
