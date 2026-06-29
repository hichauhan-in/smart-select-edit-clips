# n8n Node Changes

This documents what (if anything) needs to change in each n8n node after the
helper API (`helper/app.py`) was aligned to this flow.

> Summary: The API was changed to match your existing flow, so **no node is
> broken**. Sections marked **REQUIRED** must be verified; sections marked
> **OPTIONAL** are quality improvements you can skip.

---

## 1. FIND BEST MOMENTS  (HTTP Request -> POST /candidates)

**Status: NO CHANGE REQUIRED.**

The API still accepts your existing body. To get **variable-length clips that
cut on natural scene changes**, add `clipMin` and `clipMax` instead of relying
on a fixed `clipLen`:

```json
{
  "path": "{{ $json.path }}",
  "jobId": "{{ $json.jobId }}",
  "clipMin": 10,
  "clipMax": 20,
  "sampleInterval": 2,
  "topMotion": 20,
  "finalCandidates": 4
}
```

What each field now does in the API:
- `clipLen`         -> fixed clip length (seconds). Used only when `clipMin`/`clipMax` are 0.
- `clipMin`         -> shortest allowed clip (seconds). 0 = use fixed `clipLen`.
- `clipMax`         -> longest allowed clip (seconds). 0 = use fixed `clipLen`.
- `sampleInterval`  -> fallback frame sampling interval when scene detection finds too few scenes
- `topMotion`       -> how many top-motion candidates to keep before scoring
- `ocrTop`          -> how many candidates to OCR + vision-score. 0 = auto (half the refined candidates). Lower = much faster; never below `finalCandidates`
- `finalCandidates` -> how many final clips to return

**How the range works:** the clip is centred on the busiest moment, then its
start and end snap to the nearest scene cuts so it begins and ends on a real
visual change — but the total length is always kept between `clipMin` and
`clipMax`. So `clipMin:10, clipMax:20` yields clips somewhere in 10–20s, each
ending where the action actually changes. Leave both at 0 (or omit them) to
keep the old exact `clipLen` behaviour.

> Because clips are now variable length, the API also keeps them from
> overlapping by their real start/end times. `minGap` (optional) adds extra
> spacing between kept clips.

---

## 2. PICK BEST  (Code node)

**Status: NO CHANGE REQUIRED.** (the fields `vision_score` and `frame_rel`
now exist in the API response). Below is the verified code. Replace your
node body with this only if you want the extra safety comments / guards.

```js
const res = $input.first().json;

const cands = res.candidates || [];

if (!cands.length) {
    throw new Error("No candidates found");
}

// API already filters out rejected clips and sorts by final_score,
// so the first candidate is the best one.
const best = cands[0];
const job = $('Job Info').first().json;

return [{
    json: {
        jobId: job.jobId,
        name: job.name,
        path: job.path,

        start: best.start,
        end: best.end,

        motion: best.motion,
        visionScore: best.vision_score,   // now provided by the API
        finalScore: best.final_score,
        confidence: best.confidence,
        audioScore: best.audio_norm,      // smart-loudness (impact sounds) 0..1

        frame: best.frame_rel,            // now provided by the API (relative path)

        reason: best.reason,

        allCandidates: cands
    }
}];
```

Field reference (each candidate object returned by `/candidates`):
- `rank`               -> 1-based posting order (1 = best); list is pre-sorted best -> worst
- `segment`            -> which segment of a long video the clip came from (1,2,3...; 0 = single segment)
- `segment_rank`       -> 1-based rank WITHIN its segment (so each segment renders clip_1, clip_2, clip_3)
- `start`, `end`        -> clip time range in seconds (absolute, in the source video)
- `time`               -> the peak moment inside the clip
- `motion`             -> raw motion value
- `motion_norm`, `yolo_norm`, `ocr_norm`, `audio_norm` -> normalized 0..1 feature scores
- `quality_norm`       -> 0..1 sharpness/brightness (drops blurry, frozen, black/loading clips)
- `cuts_norm`          -> 0..1 scene-cut density (rewards one clean kill/replay cut, penalises menus)
- `audio`              -> raw smart-loudness value (largest short-term loudness jump in the clip)
- `audio_norm`         -> normalized 0..1 smart-loudness (impact sounds, not loud talking)
- `final_score`        -> combined score (already multiplied by vision confidence)
- `confidence` / `vision_score` -> vision model confidence, boosted by engagement (hook/multikill/clutch/funny/vertical)
- `gameplay`, `approve` -> vision model booleans
- `reason`             -> short text reason from the vision model
- `frame`              -> absolute path inside the container (`/data/media/...`)
- `frame_rel`          -> path relative to the media root (use this in n8n)

---

## 3. RENDER CLIPS  (HTTP Request -> POST /render)

**Status: NO CHANGE REQUIRED.**

The API now:
- returns each candidate with an explicit `rank` (1 = best), already sorted
  best -> worst;
- renders the **full original frame by default** (`aspect:"source"`, no crop,
  no fade) at near-lossless quality. Cropping to vertical Shorts is available
  but off until we start editing.

### Goal: edit the TOP 3 clips (or fewer) into postable Shorts

In the **Pick Best** Code node, instead of returning only `cands[0]`, output
the top 3 so the render step produces 3 Shorts:

```js
const res = $input.first().json;
const cands = res.candidates || [];

if (!cands.length) {
    throw new Error("No candidates found");
}

const job = $('Job Info').first().json;

// Keep the best 3 of EACH segment (a 40-min video = ~3 segments = up to 9
// clips). cands is pre-sorted best->worst, so each group is too. For a single
// segment this is just the global top 3.
const bySeg = {};
for (const c of cands) (bySeg[c.segment || 0] ||= []).push(c);
const top = Object.values(bySeg).flatMap(g => g.slice(0, 3));

return [{
    json: {
        jobId: job.jobId,
        name: job.name,
        path: job.path,
        topClips: top,        // <- up to 3 per segment, in posting order
        allCandidates: cands
    }
}];
```

### Render node body (full clips for now)

Editing (vertical crop, fades) comes later. For now render the **full,
original frame** - the API default `aspect:"source"` does exactly that, so you
don't need to send `aspect`/`cropMode`/`fade` at all:

```json
{
    "path": "{{ $json.path }}",
    "jobId": "{{ $json.jobId }}",
    "clips": {{ JSON.stringify(
        ($json.topClips || $json.allCandidates || ($json.best ? [$json.best] : []))
            .map(c => ({ start: c.start, end: c.end, rank: c.segment_rank || c.rank || 0, segment: c.segment || 0 }))
    ) }}
}
```

This still gives you the **top 3 ranked clips** at near-lossless quality with
the source resolution preserved (4K stays 4K). When we start editing, add
`"aspect": "9:16"`, `"cropMode": "center"`, `"fade": 0.5` to switch to Shorts.

> Pass `segment` through so long videos render into per-segment folders:
> `work/<jobId>/renders/segment_1/clip_1.mp4`. Single-segment videos keep the
> flat `work/<jobId>/renders/clip_1.mp4` layout (segment 0). Drop `top.slice(0,3)`
> to render every candidate from every segment instead of only the global top 3.

> Make sure the Render node's input item still has `path` and `jobId` (they
> come from the **Job Info** node). If Pick Best drops them, add them back
> there.

The response is `{ "clips": [ "work/<jobId>/renders/clip_1.mp4", ... ] }` where
`clip_1` is the best. For long videos clips are grouped per segment, e.g.
`work/<jobId>/renders/segment_2/clip_3.mp4`. Post them in numeric rank order.

**Render options you can change in the body:**
- `aspect`   -> `"9:16"` (Shorts/Reels/TikTok, default), `"4:5"`, `"1:1"`,
               `"16:9"`, or `"source"` to keep the original frame.
- `cropMode` -> `"center"` (fills the screen, best for gameplay; loses the
               left/right edges), `"blur"` (keeps the whole frame + HUD over a
               blurred background, no bars), `"fit"` (letterbox on black).
- `fade`     -> fade in/out seconds, `0` to disable.

> Tip: if your HUD (health/ammo/score) sits near the left or right edge and a
> center crop hides it, switch `cropMode` to `"blur"`.

---

## 4. EDIT  (HTTP Request -> POST /edit)

**Status: BODY CHANGED.** `cropMode` removed - both variants are always made.

Render keeps the **full resolution**. Edit reframes EVERY rendered clip into
TWO vertical variants. Add this node AFTER Render, BEFORE Cleanup.

```json
{
    "jobId": "{{ $('Job Info').item.json.jobId }}",
    "aspect": "9:16",
    "fade": 0.5
}
```

Method: POST · URL: `http://helper:8000/edit`.

Processes all clips in `renders/` (flat + every `segment_*`). For each it makes
a smart-cropped (player-tracked) and a blurred-background version, mirroring the
renders layout:

```
work/<jobId>/edited/cropped/clip_1.mp4   blurred/clip_1.mp4
                    cropped/segment_2/clip_3.mp4  ...
```

Response: `{ "cropped": [...], "blurred": [...] }`.

- `aspect` -> `9:16` | `4:5` | `1:1` | `16:9` | `source`
- `fade`   -> fade in/out seconds
- crop = smart YOLO pan (player tracked) · blur = full frame, no bars/HUD loss

> `jobId` must come from **Job Info** - Render's output only has `clips`.

---

## 5. CLEANUP  (HTTP Request -> POST /finalize)  **REQUIRED for cleanup**

**Status: NO CHANGE REQUIRED.** Node already added with current config.

Add ONE node after Edit. Without it nothing is moved and every job
keeps `source.mp4 / frames / refine / clips / renders` under `work/<jobId>`.

```json
{
    "jobId": "{{ $('Job Info').item.json.jobId }}",
    "name": "{{ $('Job Info').item.json.name }}"
}
```

Method: POST · URL: `http://helper:8000/finalize`.

After it runs, the job folder is cleaned:
- `work/<jobId>/renders/` + `work/<jobId>/edited/` -> kept
- `source.mp4` -> moved to `media/archive/<jobId>.mp4`
- `frames/`, `refine/`, `clips/`, `segment_*` -> moved to `media/temp/<jobId>/`
- nothing deleted; clear `media/temp` manually when you want.

> `jobId` and `name` come from **Job Info**. Render's output only has `clips`,
> so reference the earlier node, not `$json`.

---


## How the helper ranks clips (internal, no n8n change)

The funnel is cheap-first, expensive-last. Counts scale with video length —
nothing is hardcoded except a vision cap of 6.

1. **Coarse scan** — sample frames, score motion + YOLO. Keep top `topMotion` (≤20, capped by `duration / clipLen`).
2. **Refine** — dense 0.5s motion sweep per candidate to find the exact peak + snap to scene cuts.
3. **Pre-OCR gate** — keep half on cheap signals: motion 60% + audio 20% + YOLO 20%.
4. **OCR + tech quality** — RapidOCR (CPU, ONNX) reads reward text; sharpness/brightness + scene-cut density scored from the same 5 stills.
5. **Vision** — half of OCR survivors (≤6) judged by llava: gameplay/approve/confidence + hook/multikill/clutch/funny/vertical engagement boost.
6. **Render** — `finalCandidates` per segment, deduped, ranked, capped by `maxCandidates`.

- **OCR engine:** RapidOCR (`rapidocr-onnxruntime`), EasyOCR fallback. CPU-only; AMD GPU not usable in Docker-on-Windows.
- **Combined score:** motion .35, OCR .30, audio .13, quality .10, YOLO .07, cuts .05 — then ×vision confidence.
- **Per segment:** a 40-min video splits into ~3 segments, each with its own `clips/`, `refine/`, and `renders/segment_N/` folders, ~3 clips each (fewer if short).

---

## Nothing else changes

JOB INFO, CLAIM, PROBE, and GOT A CLIP? are unaffected by the API edits.

---

## Node order

```
Claim -> Probe -> Find Best Moments (/candidates) -> Pick Best
   -> Render (/render, source) -> Edit (/edit, 9:16) -> Cleanup (/finalize)
```