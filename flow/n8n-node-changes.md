# n8n Node Changes

This documents what (if anything) needs to change in each n8n node after the
helper API (`helper/app.py`) was aligned to this flow.

> Summary: The API was changed to match your existing flow, so **no node is
> broken**. Sections marked **REQUIRED** must be verified; sections marked
> **OPTIONAL** are quality improvements you can skip.

---

## 1. FIND BEST MOMENTS  (HTTP Request -> POST /candidates)

**Status: NO CHANGE REQUIRED.**

The API now accepts exactly these fields. Confirm your JSON body still reads:

```json
{
  "path": "{{ $json.path }}",
  "jobId": "{{ $json.jobId }}",
  "clipLen": 15,
  "sampleInterval": 2,
  "topMotion": 20,
  "finalCandidates": 4
}
```

What each field now does in the API:
- `clipLen`         -> length of each highlight clip (seconds)
- `sampleInterval`  -> fallback frame sampling interval when scene detection finds too few scenes
- `topMotion`       -> how many top-motion candidates to keep before scoring
- `finalCandidates` -> how many final clips to return

---

## 2. PICK BEST  (Code node)

**Status: NO CHANGE REQUIRED** (the fields `vision_score` and `frame_rel`
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
- `start`, `end`        -> clip time range in seconds (absolute, in the source video)
- `time`               -> the peak moment inside the clip
- `motion`             -> raw motion value
- `motion_norm`, `yolo_norm`, `ocr_norm`, `audio_norm` -> normalized 0..1 feature scores
- `audio`              -> raw smart-loudness value (largest short-term loudness jump in the clip)
- `audio_norm`         -> normalized 0..1 smart-loudness (impact sounds, not loud talking)
- `final_score`        -> combined score (already multiplied by vision confidence)
- `confidence` / `vision_score` -> vision model confidence (same value)
- `gameplay`, `approve` -> vision model booleans
- `reason`             -> short text reason from the vision model
- `frame`              -> absolute path inside the container (`/data/media/...`)
- `frame_rel`          -> path relative to the media root (use this in n8n)

---

## 3. RENDER CLIPS  (HTTP Request -> POST /render)

**Status: UPDATED for vertical Shorts + best-to-worst ranking.**

The API now:
- returns each candidate with an explicit `rank` (1 = best), already sorted
  best -> worst;
- renders **vertical 9:16 Shorts by default**, with fade in/out, at near-
  lossless quality (the source resolution is preserved for the cropped frame).

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

// API already sorts best -> worst and stamps `rank`. Take up to 3.
const top = cands.slice(0, 3);

return [{
    json: {
        jobId: job.jobId,
        name: job.name,
        path: job.path,
        topClips: top,        // <- array of up to 3, in posting order
        allCandidates: cands
    }
}];
```

### Render node body (vertical Shorts)

```json
{
    "path": "{{ $json.path }}",
    "jobId": "{{ $json.jobId }}",
    "aspect": "9:16",
    "cropMode": "center",
    "fade": 0.5,
    "clips": {{ JSON.stringify($json.topClips.map(c => ({
        start: c.start,
        end: c.end,
        rank: c.rank
    }))) }}
}
```

The response is `{ "clips": [ "work/<jobId>/renders/clip_1.mp4", ... ] }` where
`clip_1` is the best. Post them in numeric order.

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

## Nothing else changes

JOB INFO, CLAIM, PROBE, and GOT A CLIP? are unaffected by the API edits.
