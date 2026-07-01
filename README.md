# gameplay-autopost

Automated pipeline that turns a long gameplay recording into short, vertical,
ready-to-post highlight clips (Shorts / Reels / TikTok). It finds the best
moments, cuts them, and reframes them into polished vertical cards — driven by
an **n8n** workflow that calls a small **Python (FastAPI) helper** for all the
heavy video/AI work.

---

## What it does

Drop a raw gameplay `.mp4` into `media/inbox/` and the pipeline will:

1. **Claim** the clip and set up a working folder for the job.
2. **Probe** it for duration / resolution.
3. **Find the best moments** — a multi-signal highlight detector combining
   scene-cut detection, motion analysis, YOLO object detection, OCR of reward
   text (KILL / HEADSHOT / VICTORY / ACE …), smart audio-loudness spikes, and a
   **vision model** (LLaVA / Qwen2.5-VL) that judges whether each candidate is
   real, highlight-worthy gameplay.
4. **Render** the top clips at near-lossless quality (original resolution kept).
5. **Edit** each clip into two vertical variants of the same rounded 1:1 cutout:
   - **card** — the cutout on a solid black background.
   - **blurred** — the same cutout over a blurred zoom of the clip.
6. **Finalize** — archive the source, keep the renders + edited shorts, and move
   intermediates aside for cleanup.

The clip selection uses variable-length clips that snap to natural scene
boundaries, asymmetric intro/outro fades, and a motion-continuity guard that
rejects explosion/flash false cuts.

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │                 Docker host                  │
                    │                                              │
  media/inbox  ───► │  ┌─────────┐   ┌─────────┐   ┌────────────┐  │
  (raw .mp4)        │  │  n8n    │──►│ helper  │──►│ postgres   │  │
                    │  │ (flow)  │   │(FastAPI)│   │ (n8n DB)   │  │
                    │  └─────────┘   └────┬────┘   └────────────┘  │
                    │                     │                        │
                    └─────────────────────┼────────────────────────┘
                                          │ host.docker.internal
                                          ▼
                              ┌───────────────────────┐
                              │  Vision backend (HOST) │
                              │  Ollama  OR  LM Studio │  ◄── uses your GPU
                              │  (LLaVA / Qwen2.5-VL)  │
                              └───────────────────────┘
```

- **n8n** — orchestrates the workflow (the "brain"), stores state in Postgres.
- **helper** (`helper/app.py`) — FastAPI service doing ffmpeg, YOLO, OCR, motion
  math and clip editing. Runs on port `8000`.
- **postgres** — n8n's database (workflow definitions, execution history).
- **Vision backend** — runs on the **host machine** (not in a container) so it
  can use your GPU. The helper calls it over `host.docker.internal`.

Everything except the vision backend runs from `docker-compose.yml`.

---

## Prerequisites

On the new device you need:

- **Docker Desktop** (Windows/macOS) or **Docker Engine + Compose** (Linux).
- **Git** (to clone this repo).
- A **vision backend** for the highlight judgement — pick one:
  - **Ollama** — easiest, great with **NVIDIA** GPUs (and CPU-only fallback).
  - **LM Studio** — best for **AMD** GPUs (ROCm/Vulkan) on Windows.
- ~10 GB free disk for Docker images + model weights, plus room for your media.

> The helper's own AI (YOLO `yolov8n`, RapidOCR) runs **CPU-only inside the
> container** and is light. The GPU that matters is the one running the vision
> model on the host.

---

## Setup on a new device

### 1. Clone the repo

```powershell
git clone <your-repo-url> gameplay-autopost
cd gameplay-autopost
```

### 2. Create the `.env` file

`docker-compose.yml` reads secrets from a `.env` file (git-ignored). Create one
in the project root:

```dotenv
# .env
POSTGRES_PASSWORD=change-me-to-a-strong-password
N8N_ENCRYPTION_KEY=generate-a-long-random-string
GENERIC_TIMEZONE=Asia/Kolkata
```

- `POSTGRES_PASSWORD` — any strong password (used only inside Docker).
- `N8N_ENCRYPTION_KEY` — a long random string. **Keep it stable** — n8n uses it
  to encrypt stored credentials; changing it later locks you out of them.
  Generate one with: `python -c "import secrets; print(secrets.token_hex(32))"`.
- `GENERIC_TIMEZONE` — your IANA timezone (e.g. `America/New_York`).

> Do **not** copy `postgres-data/` or `n8n-data/` from another machine — they are
> git-ignored runtime state and are recreated fresh on first start.

### 3. Ensure the media folders exist

The pipeline reads/writes under `media/` (mounted into the containers as
`/data/media`). These should already exist in the repo; if not, create them:

```powershell
mkdir media\inbox, media\work, media\output, media\archive
```

### 4. Set up the vision backend (on the host)

Choose the option that matches your hardware, then set the matching values in
`docker-compose.yml` under the `helper` service (see
[Vision backend configuration](#vision-backend-configuration)).

#### Option A — Ollama (recommended; NVIDIA GPU or CPU)

1. Install Ollama from <https://ollama.com/download>.
2. Pull a vision model:
   ```powershell
   ollama pull qwen2.5vl:7b      # good default
   # or a heavier one:
   ollama pull llava:13b
   ```
3. Ollama serves on `http://localhost:11434` automatically. With an NVIDIA GPU
   it uses CUDA out of the box; with no GPU it falls back to CPU (slower).
4. In `docker-compose.yml`, keep the Ollama block active:
   ```yaml
   - VISION_BACKEND=ollama
   - VISION_MODEL=qwen2.5vl:7b
   - OLLAMA_URL=http://host.docker.internal:11434
   ```

#### Option B — LM Studio (recommended for AMD GPUs on Windows)

1. Install LM Studio from <https://lmstudio.ai>.
2. Download `unsloth/Qwen2.5-VL-7B-Instruct-GGUF` (quant `Q4_K_S`).
3. Start its local server (default port `1234`) with the model loaded. Full
   recommended settings are in [flow/AMD LM Studio configs.md](flow/AMD%20LM%20Studio%20configs.md)
   (context 8192, GPU offload maximum, keep model in memory, temp 0.1).
4. In `docker-compose.yml`, switch to the LM Studio block:
   ```yaml
   # - VISION_BACKEND=ollama
   # - VISION_MODEL=qwen2.5vl:7b
   - VISION_BACKEND=lmstudio
   - VISION_MODEL=qwen2.5-vl-7b-instruct
   - LMSTUDIO_URL=http://host.docker.internal:1234/v1/chat/completions
   ```
   The `VISION_MODEL` value must match the model ID LM Studio exposes.

### 5. Start the stack

```powershell
docker compose up -d --build
```

This builds the helper image and starts `postgres`, `n8n`, and `helper`. First
build installs Python deps and downloads the YOLO weights (a few minutes).

Check it's healthy:

```powershell
docker compose ps
curl http://localhost:8000/health      # -> {"status":"ok"}
```

### 6. Import the n8n workflow

1. Open n8n at <http://localhost:5678> and create the owner account.
2. Import your workflow (or rebuild the nodes using
   [flow/understanding n8n flow.md](flow/understanding%20n8n%20flow.md) and
   [flow/n8n-node-changes.md](flow/n8n-node-changes.md) as the reference).
3. Inside n8n, the helper is reachable at `http://helper:8000` (container DNS),
   **not** `localhost`. All HTTP nodes already use `http://helper:8000/...`.

### 7. Run it

1. Copy a gameplay `.mp4` into `media/inbox/`.
2. Trigger the n8n workflow (Manual start).
3. Outputs land in `media/work/<jobId>/`:
   - `renders/` — full-resolution cut clips.
   - `edited/card/` and `edited/blurred/` — the two vertical variants.

---

## Vision backend configuration

Set via environment variables on the `helper` service in `docker-compose.yml`
(read in [helper/app.py](helper/app.py)):

| Variable          | Default                                             | Purpose                                                        |
|-------------------|-----------------------------------------------------|----------------------------------------------------------------|
| `VISION_BACKEND`  | `ollama`                                             | `ollama` or `lmstudio`.                                        |
| `VISION_MODEL`    | `llava:13b`                                          | Model name/ID as known to the chosen backend.                 |
| `OLLAMA_URL`      | `http://host.docker.internal:11434`                 | Ollama endpoint on the host.                                   |
| `LMSTUDIO_URL`    | `http://host.docker.internal:1234/v1/chat/completions` | LM Studio OpenAI-compatible endpoint on the host.          |
| `MAX_COARSE_FRAMES` | `160`                                             | Cap on frames scanned per segment. Lower = faster/less memory. |

`host.docker.internal` lets the container reach services on the host; it's wired
up via `extra_hosts` in the compose file (works on Docker Desktop and, via
`host-gateway`, on Linux).

---

## GPU vs CPU — what runs where

| Stage                         | Where it runs        | Uses GPU?                                  |
|-------------------------------|----------------------|--------------------------------------------|
| Vision judgement (LLaVA/Qwen) | Host (Ollama/LMStudio) | **Yes** — this is the main GPU workload. |
| YOLO object detection         | helper container     | CPU (`yolov8n`, lightweight).              |
| OCR (RapidOCR / EasyOCR)      | helper container     | CPU.                                       |
| ffmpeg encode/decode          | helper container     | CPU (libx264) by default.                  |

**To take advantage of a powerful new machine:**

- **Biggest win:** run the vision backend with full GPU offload and, if you have
  the VRAM, use a larger/better model (e.g. `llava:13b` or a higher-quant
  Qwen2.5-VL). This is the slowest per-clip step, so GPU here speeds the whole
  pipeline the most.
- Raise `MAX_COARSE_FRAMES` (e.g. `240`–`320`) on a strong CPU/lots of RAM to
  scan long videos more thoroughly; lower it on a weak box to go faster.
- **Optional (code change):** the container's ffmpeg uses CPU `libx264`. On an
  NVIDIA host you could switch encoding to NVENC for faster renders — this
  requires a GPU-enabled ffmpeg in the helper image and changing the `-c:v`
  flags in `render_clip` / `edit_clip`. Not needed for correctness.
- **Optional (code change):** YOLO and RapidOCR can be moved to GPU
  (`ultralytics` with CUDA, RapidOCR with `onnxruntime-directml`/`gpu`), but they
  are already fast on CPU and rarely the bottleneck.

---

## Media folder layout

`media/` is mounted into the containers as `/data/media`.

```
media/
  inbox/              # drop raw gameplay .mp4 here (input)
  work/<jobId>/       # per-job working area
    source.mp4        #   the claimed clip
    renders/          #   full-res cut clips (clip_1.mp4 …)
    edited/
      card/           #   rounded 1:1 cutout on black
      blurred/        #   same cutout over a blurred background
  archive/            # source videos after /finalize (named <jobId>.mp4)
  output/             # (reserved for final posted outputs)
  temp/<jobId>/       # intermediates moved aside by /finalize
```

Folder contents are git-ignored (the folders themselves are kept via
`.gitkeep`), so nothing large gets committed.

---

## Helper API reference

All endpoints are POST (except `/health`) and are called by n8n at
`http://helper:8000`:

| Endpoint       | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| `GET /health`  | Liveness check.                                                         |
| `/probe`       | ffprobe duration / resolution / fps for a media file.                  |
| `/claim`       | Move the next `inbox/*.mp4` into a new `work/<jobId>/` and return the job. |
| `/candidates`  | Full highlight detection → ranked candidate clips.                     |
| `/render`      | Cut the chosen clips to full-res files under `renders/`.               |
| `/edit`        | Reframe each render into `card` + `blurred` vertical variants.         |
| `/finalize`    | Archive the source, keep renders/edited, move intermediates to `temp/`. |

Detailed request/response bodies and the n8n node bodies live in
[flow/n8n-node-changes.md](flow/n8n-node-changes.md).

### `/edit` options (vertical card styling)

```json
{
  "jobId": "…",
  "aspect": "9:16",
  "fade": 0.5,
  "fadeOut": 1.0,
  "cardAspect": "1:1",
  "radius": 48,
  "borderWidth": 0,
  "sideMargin": 48
}
```

- `aspect` — output frame (`9:16` default, `4:5`, `1:1`, `16:9`, `source`).
- `fade` / `fadeOut` — intro fade-in / outro fade-out seconds.
- `cardAspect` — crop aspect of the cutout (`1:1` = tall square, more zoom).
- `radius` — corner radius in px.
- `borderWidth` — white border ring on the black card (`0` = none).
- `sideMargin` — horizontal inset each side so the background shows around the
  card (px).

---

## Common commands

```powershell
# Start / rebuild everything
docker compose up -d --build

# Rebuild only the helper after editing helper/app.py
docker compose up -d --build helper

# Follow logs
docker compose logs -f helper
docker compose logs -f n8n

# Stop
docker compose down
```

Validate the helper Python locally before rebuilding:

```powershell
python -m py_compile helper/app.py
```

---

## Troubleshooting

- **`/edit` produces a 0 KB / errored file** — check the helper logs for the
  `EDIT: … rc=<code>` line and the ffmpeg stderr just above it.
- **Vision step always approves/rejects, or times out** — confirm the backend is
  running on the host and reachable: from the helper container,
  `curl http://host.docker.internal:11434/api/tags` (Ollama). Check
  `VISION_BACKEND`/`VISION_MODEL` match the loaded model.
- **n8n can't reach the helper** — inside n8n use `http://helper:8000`, not
  `localhost`.
- **Out-of-memory on long videos** — lower `MAX_COARSE_FRAMES` in compose.
- **n8n credentials broke after moving machines** — the `N8N_ENCRYPTION_KEY` in
  `.env` must be identical to the one used when the credentials were created.

---

## Repo layout

```
docker-compose.yml     # the whole stack (postgres + n8n + helper)
helper/
  app.py               # FastAPI service: detection, render, edit
  motion.py            # motion / frame-quality helpers
  Dockerfile           # python:3.11-slim + ffmpeg
  requirements.txt     # Python deps
flow/                  # n8n flow docs, node bodies, backend config notes
media/                 # inbox / work / output / archive (git-ignored contents)
config/                # helper config mount
```
