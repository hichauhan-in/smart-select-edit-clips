# n8n Node Changes

This documents what (if anything) needs to change in each n8n node after the
helper API (`helper/app.py`) was aligned to this flow.

> Summary: The API was changed to match your existing flow, so **no node is
> broken**. Sections marked **REQUIRED** must be verified; sections marked
> **OPTIONAL** are quality improvements you can skip.

---

## 1. FIND BEST MOMENTS  (HTTP Request -> POST /candidates)

**Status: NO CHANGE REQUIRED.**

> **What changed under the hood (no node edits needed):**
> - **Whole-video dense scan (revamped core).** Each segment is now decoded ONCE
>   into a dense timeline instead of sampling a few sparse frames. Good moments
>   can no longer be missed because the shortlist skipped them.
> - **Genre-agnostic "activity" curve, not raw motion.** Moments are ranked by
>   motion *novelty* (a sudden burst above the local baseline) + audio *onset*
>   (sudden loud events: hits, goals, crashes, explosions) — NOT raw motion.
>   This is the fix for "person walking / looting / panning" clips: steady
>   movement no longer scores high, so it stops dominating the picks. Works for
>   any genre (open-world, racing, sports, platformers, 2D, fighting, shooters).
> - **The vision model is the judge.** Each shortlisted moment gets an explicit
>   0–10 "how entertaining is this" rating from the vision model, and that
>   rating drives the ranking. It's told high motion does NOT mean exciting and
>   to score idle walking / looting / menus low. Many moments per segment are
>   sent to the model (the user accepted longer processing for better picks),
>   and each clip is sampled with 8 frames so brief highlights aren't missed.
> - **Coherent clip placement.** Clips are positioned over the real *burst* of
>   action from the dense timeline, then snapped to nearby scene cuts, so they
>   no longer start/stop at random points.
> - **Balanced splitting of long videos.** A video over 15 min is split into the
>   fewest EQUAL parts that each stay ≤ 15 min (20 min → 2×10 min, not 15+5;
>   40 min → ~3×13.3 min). Each part is processed independently and its clips
>   land in `renders/segment_N/`.

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
    "fade": 0.5,
    "fadeOut": 1.0,
    "headroom": 0.42,
    "cardAspect": "1:1",
    "radius": 48,
    "borderWidth": 5
}
```

Method: POST · URL: `http://helper:8000/edit`.

Processes all clips in `renders/` (flat + every `segment_*`). For each it makes
two variants of the SAME rounded 1:1 cutout (full width, pushed down so the
empty space splits 60% above / 40% below): a **card** version on a solid black
background with a thin white border ring, and a **blurred** version where that
same cutout sits over a blurred zoom of the clip (no border). Mirrors the
renders layout:

```
work/<jobId>/edited/card/clip_1.mp4   blurred/clip_1.mp4
                    card/segment_2/clip_3.mp4  ...
```

Response: `{ "card": [...], "blurred": [...] }`.

- `aspect` -> `9:16` | `4:5` | `1:1` | `16:9` | `source`
- `fade`    -> intro fade-IN seconds from black (0.5 = quick, snappy open)
- `fadeOut` -> outro fade-OUT seconds to black (1.0 = slower, softer ending).
               Both only apply when the clip is longer than fade + fadeOut.
- `cardAspect` -> crop aspect of the cutout (`1:1` default = tall square card,
                  centre-cropped so the left/right is trimmed and the action
                  zooms in; `4:3` for a shorter/wider card)
- `radius` -> corner radius in px (48 default; 0 = square corners)
- `borderWidth` -> black card only: white border-ring thickness in px (5 = thin,
                   0 = no border)
- **card** = 1:1 rounded cutout on solid black, thin white border, 60/40 bars
- **blurred** = the same 1:1 rounded cutout over a blurred zoom of the clip

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

The pipeline decodes each segment ONCE, builds an activity timeline, then lets
the vision model judge a generous shortlist. Counts scale with video length.

1. **Dense scan** — decode the whole segment once into small grayscale stills;
   compute a dense motion series (one pass, no per-candidate re-decodes).
2. **Activity curve** — blend motion *novelty* (burst above local baseline) +
   audio *onset* (sudden loud events) + a little absolute motion. This is what
   ignores steady walking / panning and surfaces real events, any genre.
3. **Candidate moments** — take the top activity peaks (spaced apart), place
   each clip's start/end over the action burst and snap to nearby scene cuts.
4. **Vision judging** — send the top `max(finalCandidates×4, 12)` moments (8
   frames each) to the vision model, which returns a 0–10 rating + gameplay /
   approve + engagement flags. The rating drives the ranking.
5. **Select** — keep approved moments, rank by rating, pick `finalCandidates`
   non-overlapping clips per segment (deduped, capped by `maxCandidates`).
6. **Fill fields** — OCR / quality / YOLO run only on the final clips to
   populate the API response fields.

- **Final score:** `rating ×0.85 + audio_norm ×0.08 + motion_norm ×0.07` — the
  vision rating dominates; audio / motion are only tie-breakers.
- **OCR engine:** RapidOCR (`rapidocr-onnxruntime`), EasyOCR fallback. CPU-only.
- **Tuning env vars:** `SCAN_MAX_FRAMES` (700), `SCAN_WIDTH` (480) control the
  dense scan cost; larger = finer but slower.
- **Per segment:** a 40-min video splits into ~3 segments, each with its own
  `scan/`, `clips/`, and `renders/segment_N/` folders, ~3 clips each.

---

## Nothing else changes

JOB INFO, CLAIM, PROBE, and GOT A CLIP? are unaffected by the API edits.

---

## Node order

```
Claim -> Probe -> Find Best Moments (/candidates) -> Pick Best
   -> Render (/render, source) -> Edit (/edit, 9:16) -> Cleanup (/finalize)
```