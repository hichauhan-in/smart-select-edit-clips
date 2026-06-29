import os, json, subprocess
import shutil, glob, time
import re, base64, requests
import math
import numpy as np

from motion import motion_score
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import easyocr

app = FastAPI()
MEDIA = "/data/media"

@app.get("/health")
def health():
    return {"status": "ok"}

class ProbeIn(BaseModel):
    path: str  # relative to media, e.g. "inbox/clip.mp4"

@app.post("/probe")
def probe(inp: ProbeIn):
    full = os.path.join(MEDIA, inp.path)
    if not os.path.exists(full):
        return {"error": "file not found", "path": full}
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", full]
    out = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(out.stdout or "{}")
    fmt = data.get("format", {})
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    return {
        "duration": float(fmt.get("duration", 0) or 0),
        "size": int(fmt.get("size", 0) or 0),
        "width": v.get("width"),
        "height": v.get("height"),
        "fps": v.get("r_frame_rate"),
    }


@app.post("/claim")
def claim():
    files = sorted(glob.glob(os.path.join(MEDIA, "inbox", "*.mp4")))
    if not files:
        return {"empty": True}
    src = files[0]
    job = time.strftime("%Y%m%d-%H%M%S")
    name = os.path.splitext(os.path.basename(src))[0]
    workdir = os.path.join(MEDIA, "work", job)
    os.makedirs(workdir, exist_ok=True)
    dst = os.path.join(workdir, "source.mp4")
    shutil.move(src, dst)
    return {"empty": False, "jobId": job, "name": name, "path": f"work/{job}/source.mp4"}


VISION_BACKEND = os.environ.get(
    "VISION_BACKEND",
    "ollama"
).lower()

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://host.docker.internal:11434"
)

LMSTUDIO_URL = os.environ.get(
    "LMSTUDIO_URL",
    "http://host.docker.internal:1234/v1/chat/completions"
)

VISION_MODEL = os.environ.get(
    "VISION_MODEL",
    "llava:13b"
)

# Hard cap on the coarse whole-segment scan. When scene detection is weak we
# fall back to fixed-interval sampling; a 15-minute segment at 2s would be
# ~450 frames, and every frame costs an ffmpeg extract + a YOLO inference.
# That stage is the heaviest in the pipeline and can exhaust container memory
# on long segments. We widen the interval so the coarse scan never exceeds
# this many frames (still far more than topMotion needs to pick from).
MAX_COARSE_FRAMES = int(os.environ.get("MAX_COARSE_FRAMES", "160"))

print("Loading YOLO...", flush=True)
YOLO_MODEL = YOLO("yolov8n.pt")
print("YOLO loaded.", flush=True)

YOLO_WEIGHTS = {

    "person": 2.0,

    "car": 1.5,
    "truck": 1.5,
    "bus": 1.5,
    "motorcycle": 1.5,
    "bicycle": 1.2,

    "airplane": 4.0,
    "train": 4.0,

    "boat": 2.0,

    "dog": 0.5,
    "cat": 0.5
}

OCR_WEIGHTS = {

    "ACE": 10,

    "VICTORY": 9,
    "CHAMPION": 9,

    "WIN": 8,

    "QUAD": 8,

    "TRIPLE": 7,

    "HEADSHOT": 6,

    "DOUBLE": 5,

    "ELIMINATION": 4,

    "KILL": 3,
    "DOWNED": 3,

    "+500": 5,
    "+250": 3,
    "+100": 1,

    "XP": 0.5
}

print("Loading EasyOCR...", flush=True)

ocr = easyocr.Reader(
    ['en'],
    gpu=False
)

print("EasyOCR loaded.", flush=True)

def _loudness_curve(full, hop=0.25):
    """
    Return (times[], loudness_dbfs[]) for the whole file.

    We decode the audio to mono 16 kHz PCM and compute RMS loudness (in dBFS)
    over short hops. This is far more reliable than scraping ffmpeg's ebur128
    stderr output, whose line format differs between ffmpeg builds and was
    silently producing zero samples here.
    """
    cmd = [
        "ffmpeg", "-nostats", "-v", "error",
        "-i", full,
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "s16le", "-"
    ]
    p = subprocess.run(cmd, capture_output=True)

    if not p.stdout:
        return [], []

    samples = np.frombuffer(p.stdout, dtype=np.int16)

    if samples.size == 0:
        return [], []

    audio = samples.astype(np.float32) / 32768.0

    sr = 16000
    win = max(1, int(hop * sr))

    times, vals = [], []
    for i in range(0, audio.size - win + 1, win):
        chunk = audio[i:i + win]
        rms = float(np.sqrt(np.mean(chunk * chunk)) + 1e-9)
        db = 20.0 * math.log10(rms)
        times.append(i / sr)
        vals.append(db)

    return times, vals

def _pick_peaks(times, vals, k, min_gap):
    """Greedily pick k loudest moments at least min_gap seconds apart."""
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
    chosen = []
    for i in order:
        t = times[i]
        if all(abs(t - c[0]) > min_gap for c in chosen):
            chosen.append((t, vals[i]))
        if len(chosen) >= k:
            break
    return chosen

def clip_audio_score(times, vals, start, end):
    """
    Smart loudness for a single clip window.

    Raw loudness is a bad highlight signal: a person speaking loudly into the
    mic stays loud the whole time and would always score high. Instead we score
    the largest short-term LOUDNESS JUMP inside the window.

    Sudden transients (gunfire, explosions, kill / reward stingers) cause big
    jumps; sustained speech or background music stays flat and scores low.
    """
    window = [
        (t, v)
        for t, v in zip(times, vals)
        if start <= t <= end
    ]

    if len(window) < 2:
        return 0.0

    best_jump = 0.0
    for i in range(1, len(window)):
        jump = window[i][1] - window[i - 1][1]
        if jump > best_jump:
            best_jump = jump

    return best_jump

def _extract_frame(full, t, out_path):
    subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", full,
                    "-frames:v", "1", "-q:v", "3", out_path], capture_output=True)

def yolo_score(frame_path):

    results = YOLO_MODEL(frame_path, verbose=False)

    result = results[0]

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return 0, []

    score = 0
    detected = []

    names = result.names

    for cls, conf in zip(
        boxes.cls.tolist(),
        boxes.conf.tolist()
    ):

        label = names[int(cls)]

        detected.append(label)

        weight = YOLO_WEIGHTS.get(label, 1.0)

        score += conf * weight

    return score, detected

def _extract_clip_frames(full, start, end, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    duration = end - start

    offsets = [
        0,
        duration * 0.25,
        duration * 0.50,
        duration * 0.75,
        duration * 0.95
    ]

    frames = []

    for i, offset in enumerate(offsets):

        t = start + offset

        out = os.path.join(out_dir, f"frame_{i}.jpg")

        _extract_frame(full, t, out)

        frames.append(out)

    return frames

def build_cv_summary(
    motion,
    yolo,
    ocr,
    audio
):

    summary = []

    if motion >= 0.80:
        summary.append("Very high motion detected.")
    elif motion >= 0.60:
        summary.append("Moderate motion detected.")
    elif motion >= 0.30:
        summary.append("Low motion detected.")
    else:
        summary.append("Almost no motion detected.")

    if yolo >= 0.80:
        summary.append("Many gameplay objects detected.")
    elif yolo >= 0.60:
        summary.append("Several gameplay objects detected.")
    elif yolo >= 0.30:
        summary.append("Few gameplay objects detected.")
    else:
        summary.append("Almost no gameplay objects detected.")

    if ocr >= 0.80:
        summary.append("Strong reward text detected.")
    elif ocr >= 0.50:
        summary.append("Some reward text detected.")
    elif ocr >= 0.20:
        summary.append("Very little reward text detected.")
    else:
        summary.append("No reward text detected.")

    if audio >= 0.80:
        summary.append("Strong impact sounds detected.")
    elif audio >= 0.50:
        summary.append("Some impact sounds detected.")
    elif audio >= 0.20:
        summary.append("Faint impact sounds detected.")
    else:
        summary.append("No notable impact sounds detected.")

    return summary

def build_prompt(
    motion,
    yolo,
    ocr,
    audio,
    clip_len
):

    cv_summary = build_cv_summary(
        motion,
        yolo,
        ocr,
        audio
    )

    return f"""
You are an expert esports highlight curator.

You are given 5 frames sampled from ONE single {clip_len:.0f}-second gameplay clip.
The frames are in chronological order:

Frame 1  ~  0%  (start of the clip)
Frame 2  ~ 25%
Frame 3  ~ 50%  (middle)
Frame 4  ~ 75%
Frame 5  ~ 95%  (end of the clip)

Judge the clip as ONE continuous moment. Read the progression of the
action across the 5 frames instead of rating each image on its own.

These frames were already pre-selected by an automated highlight
detection pipeline, so they are likely (but not guaranteed) interesting.

--- Computer Vision Analysis ---
These scores are RELATIVE to the other candidate clips detected in THIS
video (1.00 = strongest among the candidates, 0.00 = weakest). They are
NOT absolute quality measures, so treat them as ranking hints, not truth.

Motion Score : {motion:.2f}   (relative amount of on-screen movement)
YOLO Score   : {yolo:.2f}   (relative object count - weak signal, low weight)
OCR Score    : {ocr:.2f}   (relative reward text such as HEADSHOT, KILL, VICTORY, ACE)
Audio Score  : {audio:.2f}   (relative impact sounds: gunfire, explosions, reward stingers - NOT loud talking)

Summary:
{chr(10).join(cv_summary)}

--- Your task ---
1. gameplay : true ONLY if these frames clearly show real, live video-game
   gameplay. Set gameplay = false for anything that is not live gameplay,
   including: advertisements, sponsor / promotional / brand screens or
   logos, intros, outros, menus, loading screens, scoreboards, desktop,
   webcam / face-cam, black screens, OR frozen / static frames where almost
   nothing changes across the 5 frames.
2. approve  : true only if this is genuinely highlight-worthy. Reject
   (approve = false) advertisements, promos, intros / outros, and static
   or idle moments even if a game image is technically visible.
3. confidence : how strong this highlight is (see guide below).
4. Use the CV scores as supporting evidence:
   - If the visuals agree with the scores, raise your confidence.
   - If the scores look misleading (e.g. high motion but nothing happens),
     lower your confidence.

Consistency rules:
- If gameplay is false, approve MUST be false and confidence MUST be 0.00.
- If approve is false, confidence MUST be <= 0.30.
- If the 5 frames look almost identical (no real change), treat it as
  static: gameplay = false and confidence = 0.00.

Confidence guide:
1.00 = Exceptional highlight
0.90 = Excellent gameplay
0.80 = Strong highlight
0.70 = Good gameplay
0.60 = Average gameplay
0.40 = Weak highlight
0.20 = Probably not a highlight
0.00 = Not gameplay / definitely reject

Return ONLY this JSON object, nothing else:
{{
    "gameplay": true,
    "approve": true,
    "confidence": 0.0,
    "reason": "short reason under 12 words"
}}

Never return markdown. Never explain. Return JSON only.
"""

def parse_vision_response(data):

    gameplay = bool(
        data.get("gameplay", True)
    )

    approve = bool(
        data.get("approve", True)
    )

    confidence = float(
        data.get("confidence", 0.5)
    )

    reason = str(
        data.get("reason", "")
    )

    return (
        gameplay,
        approve,
        confidence,
        reason
    )

def vision_failed(e):

    return (
        True,
        True,
        0.5,
        f"vision-failed: {e}"
    )

def normalize_feature(
    frames,
    source_key,
    target_key
):

    values = [
        f[source_key]
        for f in frames
    ]

    min_value = min(values)
    max_value = max(values)

    for frame in frames:

        if max_value == min_value:
            frame[target_key] = 0.5
        else:
            frame[target_key] = (
                frame[source_key] - min_value
            ) / (max_value - min_value)

    print(
        f"\n=== {target_key.upper()} ===",
        flush=True
    )

    for frame in frames:
        print(
            frame[source_key],
            "->",
            round(frame[target_key], 3),
            flush=True
        )

def compute_fast_score(frame):

    return (
        frame["motion_norm"] * 0.40
        + frame["ocr_norm"] * 0.35
        + frame["audio_norm"] * 0.15
        + frame["yolo_norm"] * 0.10
    )

def split_video_jobs(
    duration_seconds,
    max_minutes=15,
    overlap_seconds=15
):

    max_duration = max_minutes * 60

    if duration_seconds <= max_duration:
        return [
            {
                "job": 1,
                "start": 0.0,
                "end": duration_seconds
            }
        ]

    jobs = []

    parts = math.ceil(
        duration_seconds / max_duration
    )

    for i in range(parts):

        start = i * max_duration

        if i > 0:
            start -= overlap_seconds

        end = min(
            (i + 1) * max_duration,
            duration_seconds
        )

        jobs.append({
            "job": i + 1,
            "start": start,
            "end": end
        })

    return jobs

def _vision_score_ollama(
    frame_paths,
    motion,
    yolo,
    ocr,
    audio,
    clip_len
):
    try:
        images = []

        for frame in frame_paths:
            with open(frame, "rb") as f:
                images.append(
                    base64.b64encode(f.read()).decode()
                )

        prompt = build_prompt(motion, yolo, ocr, audio, clip_len)

        #r = requests.post(f"{OLLAMA}/api/generate", json={
        #    "model": VISION_MODEL, "prompt": prompt, "images": [b64],
        #    "stream": False, "format": "json"}, timeout=180)

        #data = json.loads(r.json().get("response", "{}"))

        #print("Before requests.post()", flush=True)
        #print("OLLAMA =", OLLAMA_URL, flush=True)
        #print("MODEL =", VISION_MODEL, flush=True)
        r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0}
        },
        timeout=180
        )
        #print("After requests.post()", flush=True)
        #print("STATUS:", r.status_code, flush=True)
        #print("RAW:", r.text, flush=True)

        resp = r.json()
        print("PARSED:", resp, flush=True)

        data = json.loads(resp.get("response", "{}"))

        return parse_vision_response(data)

    except Exception as e:
        return vision_failed(e)

def _vision_score_lmstudio(
    frame_paths,
    motion,
    yolo,
    ocr,
    audio,
    clip_len
):

    try:

        images = []

        for frame in frame_paths:
            with open(frame, "rb") as f:
                b64 = base64.b64encode(
                    f.read()
                ).decode()

            images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            })


        prompt = build_prompt(motion, yolo, ocr, audio, clip_len)

        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]

        content.extend(images)

        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }

        r = requests.post(
            LMSTUDIO_URL,
            json=payload,
            timeout=300
        )

        print("LM Studio Status:", r.status_code, flush=True)

        resp = r.json()

        answer = resp["choices"][0]["message"]["content"]

        answer = (
            answer
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        print(answer, flush=True)

        data = json.loads(answer)

        return parse_vision_response(data)

    except Exception as e:
        return vision_failed(e)

def _vision_score(
    frame_paths,
    motion,
    yolo,
    ocr,
    audio,
    clip_len
):

    print("\n" + "=" * 70, flush=True)
    print("VISION INFERENCE", flush=True)
    print(f"Backend : {VISION_BACKEND}", flush=True)
    print(f"Model   : {VISION_MODEL}", flush=True)
    print(f"Images  : {len(frame_paths)}", flush=True)
    print("=" * 70, flush=True)

    if VISION_BACKEND == "ollama":
        return _vision_score_ollama(
            frame_paths,
            motion,
            yolo,
            ocr,
            audio,
            clip_len
        )

    elif VISION_BACKEND == "lmstudio":
        return _vision_score_lmstudio(
            frame_paths,
            motion,
            yolo,
            ocr,
            audio,
            clip_len
        )

    raise ValueError(
        f"Unknown vision backend: {VISION_BACKEND}"
    )

class CandIn(BaseModel):
    path: str
    jobId: str
    clipLen: float = 15.0
    # Variable clip length. Leave both 0 to keep fixed clipLen behaviour.
    # Set clipMin and/or clipMax to let the system choose a NATURAL length in
    # that range, ending the clip where the scene actually changes.
    #   e.g. clipMin=10, clipMax=20  -> clips run 10-20s, cut on scene boundaries
    clipMin: float = 0.0
    clipMax: float = 0.0
    sampleInterval: float = 2.0
    topMotion: int = 20
    finalCandidates: int = 4
    # 0 == "auto": fall back to one clip length so highlights can sit
    # back-to-back instead of being forced 30 s apart.
    minGap: float = 0.0
    # 0 == no cap. When > 0, the merged result (across all segments of a long
    # video) is trimmed to at most this many clips, keeping the highest scored.
    maxCandidates: int = 0

class RenderClip(BaseModel):
    start: float
    end: float
    # Best-to-worst position (1 = best). Used only to name the output file so
    # the editor / uploader can post them in order. 0 == fall back to index.
    rank: int = 0


class RenderIn(BaseModel):
    path: str
    jobId: str
    clips: list[RenderClip]
    # Output framing. Default "source" keeps the ORIGINAL full frame untouched
    # (no cropping) - editing comes later. Set to "9:16" for Shorts, or "4:5",
    # "1:1", "16:9".
    aspect: str = "source"
    # How to reach the target aspect (only used when aspect != "source"):
    #   "center" - crop the centre strip and fill the frame (most engaging
    #              for gameplay, but loses the left/right edges)
    #   "blur"   - whole frame fitted over a zoomed, blurred copy of itself
    #              (keeps all gameplay + HUD, no hard black bars)
    #   "fit"    - whole frame letterboxed on black
    cropMode: str = "center"
    # Fade-in / fade-out duration in seconds (0 = none, the default for now).
    fade: float = 0.0

def detect_scenes(full, min_threshold=0.12):

    # One decode pass at a low threshold surfaces every candidate cut together
    # with its scene_score, so the caller can pick an effective threshold in
    # Python (adaptive) instead of re-decoding the video several times.
    cmd = [
        "ffmpeg",
        "-i", full,
        "-filter:v",
        f"select='gt(scene,{min_threshold})',metadata=print",
        "-vsync", "0",
        "-f", "null",
        "-"
    ]

    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    pairs = []
    cur_time = None

    for line in p.stderr.splitlines():

        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            cur_time = float(m.group(1))
            continue

        ms = re.search(r"scene_score=([0-9.]+)", line)
        if ms and cur_time is not None:
            pairs.append((cur_time, float(ms.group(1))))
            cur_time = None

    # Older/edge ffmpeg builds may not surface scene_score lines. Fall back to
    # treating every selected frame as a cut at the minimum threshold.
    if not pairs:
        for line in p.stderr.splitlines():
            m = re.search(r"pts_time:([0-9.]+)", line)
            if m:
                pairs.append((float(m.group(1)), min_threshold))

    return pairs

def ocr_score(image_path):

    results = ocr.readtext(image_path)

    text = " ".join(
        r[1]
        for r in results
    ).upper()

    score = 0

    matched = []

    for word, weight in OCR_WEIGHTS.items():

        if word in text:

            score += weight
            matched.append(word)

    return score, text, matched

def ocr_score_frames(frame_paths):
    """
    Run OCR over several frames of one clip and keep the strongest hit.

    Reward text (e.g. "DOUBLE KILL") often flashes for ~1 second and is gone by
    the middle frame, so scanning only the 50% frame misses it. We score every
    frame and take the max, merging the matched words and the text of the
    best-scoring frame.
    """
    best_score = 0
    best_text = ""
    all_hits = []

    for fp in frame_paths:
        score, text, hits = ocr_score(fp)

        if score > best_score:
            best_score = score
            best_text = text

        for h in hits:
            if h not in all_hits:
                all_hits.append(h)

    return best_score, best_text, all_hits

def sample_video(duration, interval=2.0):
    times = []

    t = interval

    while t < duration:
        times.append(round(t, 2))
        t += interval

    return times

def audio_peak_times(times, vals, min_gap, max_peaks):
    """
    Pick the timestamps where loudness JUMPS the most.

    A sudden rise in loudness is the onset of a loud event - a gunshot,
    explosion, hit or shout - which is exactly where highlights live. We rank
    by the size of the rise from the previous sample and keep the strongest,
    spaced at least ``min_gap`` seconds apart so one loud burst doesn't claim
    every slot.
    """
    if not times or len(times) < 3:
        return []

    rises = []
    for i in range(1, len(vals)):
        rise = vals[i] - vals[i - 1]
        if rise > 0:
            rises.append((rise, times[i]))

    rises.sort(reverse=True)

    kept = []
    for _, t in rises:
        if all(abs(t - k) >= min_gap for k in kept):
            kept.append(t)
        if len(kept) >= max_peaks:
            break

    return sorted(kept)

def merge_seed_times(*lists, min_gap, max_count):
    """
    Merge candidate timestamps from several sources into one clean list.

    Times closer than ``min_gap`` collapse to one (a scene cut and an audio
    peak a fraction of a second apart are the same moment), and the result is
    thinned uniformly if it still exceeds ``max_count``.
    """
    times = sorted({
        round(t, 2)
        for lst in lists
        for t in lst
    })

    kept = []
    for t in times:
        if not kept or t - kept[-1] >= min_gap:
            kept.append(t)

    if len(kept) > max_count:
        step = len(kept) / max_count
        kept = [kept[int(i * step)] for i in range(max_count)]

    return kept

def choose_clip_bounds(peak, scene_cuts, dur, clip_min, clip_max):
    """
    Pick a natural ``[start, end]`` for one highlight.

    The length stays within ``[clip_min, clip_max]`` but, instead of a fixed
    stopwatch length, the start and end are snapped to nearby scene cuts so the
    clip BEGINS and ENDS on a real visual change (a new room, the kill cam
    ending, etc). The peak (busiest) moment always stays inside the window.

    When ``clip_min`` and ``clip_max`` are (almost) equal we keep the old
    behaviour: a fixed-length clip centred on the peak.
    """
    clip_min = max(1.0, clip_min)
    clip_max = max(clip_min, clip_max)
    target = (clip_min + clip_max) / 2.0

    # Fixed length requested -> centre on the peak, no snapping.
    if clip_max - clip_min < 1.0:
        start = max(0.0, peak - target / 2.0)
        end = min(dur, start + target)
        if end - start < target:
            start = max(0.0, end - target)
        return round(start, 2), round(end, 2)

    cuts = sorted(scene_cuts)

    # START: the latest scene cut shortly before the peak (the action's start).
    # Must leave >= 0.5s of lead-in and sit no further back than clip_max.
    start = None
    for s in cuts:
        if peak - clip_max <= s <= peak - 0.5:
            start = s
    if start is None:
        start = peak - target / 2.0

    # END: the earliest scene cut after the peak that yields a valid length.
    # Cuts are sorted, so once a cut would exceed clip_max we can stop looking.
    end = None
    for e in cuts:
        if e <= peak:
            continue
        length = e - start
        if length < clip_min:
            continue
        if length > clip_max:
            break
        end = e
        break
    if end is None:
        end = start + target

    # Enforce the length range.
    length = end - start
    if length < clip_min:
        end = start + clip_min
    elif length > clip_max:
        end = start + clip_max

    # Clamp to the video and guarantee the peak stays inside the window.
    start = max(0.0, start)
    end = min(dur, end)
    if end - start < clip_min:
        start = max(0.0, end - clip_max)
    if peak < start:
        start = max(0.0, peak - 0.5)
    if peak > end:
        end = min(dur, peak + 0.5)
        if end - start > clip_max:
            start = max(0.0, end - clip_max)

    return round(start, 2), round(end, 2)

# Target pixel size for each supported aspect ratio. Shorts/Reels/TikTok all
# use 1080x1920; the rest follow the same 1080-wide convention. "source"
# (None) keeps the original frame size.
ASPECT_DIMS = {
    "9:16": (1080, 1920),
    "4:5": (1080, 1350),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
    "source": None,
}


def build_video_filter(aspect, crop_mode, fade_dur, duration):
    """
    Build the ffmpeg filter for reframing a clip to ``aspect`` with the chosen
    ``crop_mode`` plus optional fade in/out.

    Returns ``(flag, filter_string, extra_maps)`` where ``flag`` is either
    ``"-vf"`` (single chain) or ``"-filter_complex"`` (blur needs a split), and
    ``extra_maps`` are any explicit ``-map`` args the complex graph requires.
    """
    dims = ASPECT_DIMS.get(aspect, ASPECT_DIMS["9:16"])

    fades = ""
    if fade_dur and fade_dur > 0 and duration > 2 * fade_dur:
        out_start = max(0.0, duration - fade_dur)
        fades = (
            f",fade=t=in:st=0:d={fade_dur}"
            f",fade=t=out:st={out_start:.3f}:d={fade_dur}"
        )

    # Keep the original frame size, just (optionally) fade.
    if dims is None:
        return "-vf", "null" + fades, []

    W, H = dims

    if crop_mode == "blur":
        # Whole frame fitted onto a zoomed, blurred copy of itself - nothing is
        # cropped away and there are no hard black bars.
        fc = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},gblur=sigma=25[bg2];"
            f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease[fg2];"
            f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2"
            f"{fades}[v]"
        )
        return "-filter_complex", fc, ["-map", "[v]", "-map", "0:a?"]

    if crop_mode == "fit":
        chain = (
            f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black"
            f"{fades}"
        )
        return "-vf", chain, []

    # Default "center": crop the centre strip to the target aspect, then scale.
    chain = (
        f"crop='min(iw,ih*{W}/{H})':'min(ih,iw*{H}/{W})',"
        f"scale={W}:{H}"
        f"{fades}"
    )
    return "-vf", chain, []


def render_clip(
    source,
    start,
    end,
    output,
    aspect="source",
    crop_mode="center",
    fade=0.0
):

    duration = end - start

    flag, filter_string, extra_maps = build_video_filter(
        aspect,
        crop_mode,
        fade,
        duration
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", source,
        "-t", str(duration),
        flag, filter_string,
    ]

    cmd += extra_maps

    cmd += [
        # crf 18 + faststart keeps the clip visually near-lossless and ready
        # for instant web playback; the source resolution is preserved unless
        # an aspect crop/scale was requested above.
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    print("=" * 80, flush=True)
    print("FFMPEG RETURN:", result.returncode, flush=True)
    print("OUTPUT FILE:", output, flush=True)
    print("ASPECT:", aspect, "CROP:", crop_mode, flush=True)
    print("STDERR:", result.stderr, flush=True)
    print("=" * 80, flush=True)


def split_video(
    source,
    start,
    end,
    output
):

    duration = end - start

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", source,
            "-t", str(duration),
            "-c", "copy",
            output
        ],
        check=True,
        capture_output=True
    )

    return output

def refine_candidate(
    video_path,
    approx_time,
    duration,
    job_dir,
    window=7.0,
    step=0.5
):

    work = os.path.join(
        job_dir,
        "refine",
        f"candidate_{int(approx_time)}"
    )
    os.makedirs(work, exist_ok=True)

    start = max(0.0, approx_time - window)
    end = min(duration, approx_time + window)
    span = end - start

    if span <= 0:
        return approx_time, 0.0, 0.0

    fps = 1.0 / step

    # One ffmpeg pass dumps the whole window as frames, instead of spawning
    # ~(2*window/step) separate ffmpeg processes.
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(span),
            "-vf", f"fps={fps}",
            "-q:v", "3",
            os.path.join(work, "f_%04d.jpg")
        ],
        capture_output=True
    )

    frames = sorted(glob.glob(os.path.join(work, "f_*.jpg")))

    best_time = approx_time
    best_motion = -1.0
    motion_sum = 0.0
    motion_count = 0
    prev_frame = None

    for i, img in enumerate(frames):

        if prev_frame is not None:

            m = motion_score(prev_frame, img)

            motion_sum += m
            motion_count += 1

            if m > best_motion:
                best_motion = m
                # frame i was sampled at time start + i / fps
                best_time = start + i / fps

        prev_frame = img

    # Sustained motion = average frame-to-frame change across the whole window.
    # It represents how action-packed the clip really is, and (unlike the single
    # peak) is not faked by one scene cut inside the window.
    mean_motion = motion_sum / motion_count if motion_count else 0.0

    print(
        f"Refined {approx_time:.2f}s -> {best_time:.2f}s "
        f"(peak={best_motion:.2f}, mean={mean_motion:.2f})",
        flush=True
    )

    return best_time, best_motion, mean_motion


def print_final_results(selected):

    print("\n===== FINAL RANKING =====", flush=True)

    for i, frame in enumerate(selected, 1):

        print(
            f"{i}. "
            f"Time={frame['time']:.1f}s | "
            f"Motion={frame['motion_norm']:.2f} | "
            f"YOLO={frame['yolo_norm']:.2f} | "
            f"OCR={frame['ocr_norm']:.2f} | "
            f"Audio={frame['audio_norm']:.2f} | "
            f"Gameplay={frame['gameplay']} "
            f"Approve={frame['approve']} "
            f"Confidence={frame['confidence']:.2f} "
            f"Final={frame['final_score']:.2f} "
            f"Reason={frame['reason']}",
            flush=True
        )

def load_clip_frames(
    video_path,
    frame,
    job_dir
):

    # OCR and vision both ask for the same 5 clip frames. Extract once,
    # then reuse the cached paths (frame start/end no longer change here).
    cached = frame.get("clip_frames")
    if cached and all(os.path.exists(p) for p in cached):
        return cached

    clip_dir = os.path.join(
        job_dir,
        "clips",
        f"clip_{frame['idx']}"
    )

    clip_frames = _extract_clip_frames(
        video_path,
        frame["start"],
        frame["end"],
        clip_dir
    )

    frame["clip_frames"] = clip_frames
    return clip_frames

def process_job(
    full,
    dur,
    inp,
    job_dir,
    job_start=0.0,
    job_end=None
):

    if job_end is None:
        job_end = dur

    # ---- Clip length range -------------------------------------------------
    # clipLen alone => fixed-length clips (back-compat). Provide clipMin and/or
    # clipMax to let the system pick a NATURAL length in that range, snapping
    # the start/end to nearby scene cuts. clip_target drives seeding, counts
    # and the refine window so everything still scales sensibly.
    clip_min = inp.clipMin if inp.clipMin and inp.clipMin > 0 else inp.clipLen
    clip_max = inp.clipMax if inp.clipMax and inp.clipMax > 0 else inp.clipLen
    if clip_max < clip_min:
        clip_max = clip_min
    clip_target = (clip_min + clip_max) / 2.0

    # Smart-loudness curve for the whole segment (computed once).
    # Used later to score each clip by its loudest sudden impact sound.
    audio_times, audio_vals = _loudness_curve(full)
    if audio_vals:
        print(
            f"Audio loudness samples: {len(audio_times)} "
            f"(min={min(audio_vals):.1f} dBFS, max={max(audio_vals):.1f} dBFS)",
            flush=True
        )
    else:
        print(
            "Audio loudness samples: 0 (no audio decoded)",
            flush=True
        )

    # ---- Candidate seeding -------------------------------------------------
    # Where should we take a first cheap look? Three sources, best first, so we
    # look where something actually happens instead of sampling blindly:
    #   1. Scene cuts  - the picture changed a lot (new room, kill cam, etc).
    #   2. Audio peaks - loudness jumped (gunshot, explosion, hit, shout).
    #   3. Uniform grid - last-resort backstop so we never come up empty.

    # One decode pass gives every cut + its scene_score; we then relax the
    # threshold in Python until we have enough cuts. Quiet gameplay rarely
    # trips the strict 0.30, so the adaptive step keeps us off the dumb grid.
    scene_pairs = detect_scenes(full)

    # Every detected cut (at the low base threshold) is a potential clip
    # boundary; variable-length clips are later snapped to these.
    boundary_cuts = sorted(t for (t, _s) in scene_pairs)

    scene_threshold = 0.30
    scene_times = []
    for scene_threshold in (0.30, 0.20, 0.12):
        scene_times = [
            t for (t, s) in scene_pairs
            if s >= scene_threshold
        ]
        if len(scene_times) >= 20:
            break

    print(
        f"Detected {len(scene_times)} scene cuts "
        f"(threshold {scene_threshold:.2f})",
        flush=True
    )

    # Audio peaks come for free from the loudness curve computed above.
    peak_times = audio_peak_times(
        audio_times,
        audio_vals,
        min_gap=max(2.0, clip_target / 2),
        max_peaks=MAX_COARSE_FRAMES
    )

    print(
        f"Found {len(peak_times)} audio peaks",
        flush=True
    )

    # Merge the two smart signals, collapsing near-duplicates and capping count.
    sample_times = merge_seed_times(
        scene_times,
        peak_times,
        min_gap=max(1.0, clip_target / 4),
        max_count=MAX_COARSE_FRAMES
    )

    # Last-resort backstop: if scene + audio were too sparse (flat, silent
    # footage), add a capped uniform grid so the segment is still scanned.
    if len(sample_times) < 20:
        interval = inp.sampleInterval
        if dur / interval > MAX_COARSE_FRAMES:
            interval = dur / MAX_COARSE_FRAMES

        print(
            f"Sparse seeds. Adding {interval:.1f}-second grid backstop.",
            flush=True
        )

        sample_times = merge_seed_times(
            sample_times,
            sample_video(dur, interval=interval),
            min_gap=max(1.0, clip_target / 4),
            max_count=MAX_COARSE_FRAMES
        )

    print(
        f"Sampling {len(sample_times)} frames",
        flush=True
    )

    frames_dir = os.path.join(
        job_dir,
        "frames"
    )

    os.makedirs(frames_dir, exist_ok=True)

    frames = []
    for i, t in enumerate(sample_times):
        start = max(0.0, t - clip_target / 2)
        end = min(dur, start + clip_target)
        fp = os.path.join(frames_dir, f"cand_{i}.jpg")
        _extract_frame(full, t, fp)
        frames.append({
            "idx": i, "time": t, "start": start, "end": end, "frame":fp
        })
    print(f"Extracted {len(frames)} frames", flush=True)
    
    motion_frames = []
    for i in range(1, len(frames)):
        prev = frames[i - 1]
        curr = frames[i]

        motion = motion_score(
        prev["frame"],
        curr["frame"]
        )
        
        yolo, yolo_hits = yolo_score(curr["frame"])

        motion_frames.append({
            "idx": curr["idx"], "time": curr["time"], 
            "start": curr["start"], "end": curr["end"],
            "motion": motion, "yolo": yolo, "yolo_hits": yolo_hits
        })
    
    motion_frames.sort(
    key=lambda x: x["motion"],
    reverse=True
    )

    max_possible = int(dur / clip_target)

    candidate_count = max(
        2,
        min(inp.topMotion, max_possible)
    )

    print(
        f"Selecting top {candidate_count} candidates",
        flush=True
    )

    interesting = motion_frames[:candidate_count]

    # Refine search window scales with the clip (clip_target / 3), clamped so it
    # is never less than 7 s or more than 15 s on each side of the motion peak.
    refine_window = max(7.0, min(clip_target / 3.0, 15.0))

    for frame in interesting:

        refined_time, refined_peak, refined_mean = refine_candidate(
            full,
            frame["time"],
            dur,
            job_dir,
            window=refine_window
        )

        frame["time"] = refined_time
        # Score on sustained (mean) intra-window motion, not the single peak,
        # so a lone scene cut inside the window can't fake a high-action clip.
        # The peak still decides where to center the clip (refined_time).
        frame["motion"] = refined_mean

        # Snap the clip to natural scene boundaries within the length range
        # (or centre it on the peak when a fixed clipLen was requested).
        frame["start"], frame["end"] = choose_clip_bounds(
            refined_time,
            boundary_cuts,
            dur,
            clip_min,
            clip_max
        )

    print("\n===== AFTER REFINEMENT =====", flush=True)

    for frame in interesting:
        print(
            f"Time={frame['time']:.2f}s "
            f"Motion={frame['motion']:.2f} "
            f"YOLO={frame['yolo']:.2f}",
            f"Objects={frame['yolo_hits']}",
            flush=True
        )

    print(f"Scoring {len(interesting)} frames with OCR only", flush=True)

    scored = []
    for frame in interesting:

        clip_frames = load_clip_frames(
            full,
            frame,
            job_dir
        )

        ocr_points, ocr_text, ocr_hits = ocr_score_frames(clip_frames)

        frame["ocr"] = ocr_points
        frame["ocr_text"] = ocr_text
        frame["ocr_hits"] = ocr_hits

        frame["audio"] = clip_audio_score(
            audio_times,
            audio_vals,
            frame["start"],
            frame["end"]
        )

        print(
            f"OCR={frame['ocr']:.1f}",
            f"Hits={ocr_hits}",
            f"Audio={frame['audio']:.1f}",
            f"Text={frame['ocr_text'][:80]}",
            flush=True
        )
        frame["reason"] = ""

        scored.append(frame)

    normalize_feature(scored, "motion", "motion_norm") 

    normalize_feature(scored, "yolo", "yolo_norm")

    normalize_feature(scored, "ocr", "ocr_norm")

    normalize_feature(scored, "audio", "audio_norm")

    for frame in scored:

        frame["final_score"] = compute_fast_score(
            frame
        )

    scored.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    llava_candidates = min(6, len(scored))

    interesting = scored[:llava_candidates]

    print(
        f"Running {VISION_BACKEND} ({VISION_MODEL}) on {len(interesting)} clips",
        flush=True
    )

    for frame in interesting:

        clip_frames = load_clip_frames(full, frame, job_dir)

        gameplay, approve, confidence, reason = _vision_score(
            clip_frames,
            frame["motion_norm"],
            frame["yolo_norm"],
            frame["ocr_norm"],
            frame["audio_norm"],
            frame["end"] - frame["start"]
        )

        frame["gameplay"] = gameplay
        frame["approve"] = approve
        frame["confidence"] = confidence
        frame["reason"] = reason
        if not gameplay or not approve:
            print(
                f"Rejected by Vision: {reason}",
                flush=True
            )
            # Drop the score so a rejected clip can never win
            frame["final_score"] = 0.0
            continue

        # Blend the vision model's confidence into the fast score.
        # llava is unreliable at the confidence NUMBER: it frequently returns
        # 0.00 for clips it simultaneously approves and calls "highlight-worthy".
        # A raw multiply would wrongly zero those good clips, so instead map
        # confidence onto a 0.5 - 1.0 multiplier. An approved clip keeps at
        # least half of its CV score, and higher confidence is rewarded on top.
        frame["final_score"] *= (0.5 + 0.5 * confidence)

    # Keep only clips the vision model actually approved. We rely on the
    # binary gameplay / approve flags, which llava sets reliably, rather than
    # its confidence number, which it does not (it often returns 0.00 for clips
    # it just approved). Ads, menus and static frames are already rejected via
    # gameplay = false in the prompt, so no extra confidence floor is needed.
    approved = [
        f for f in interesting
        if f.get("gameplay") and f.get("approve")
    ]

    approved.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    # Clips now have variable lengths, so we reject by ACTUAL time overlap of
    # the [start, end] windows rather than a fixed centre-to-centre gap.
    # minGap (when > 0) adds extra breathing room between kept clips.
    extra_gap = inp.minGap if inp.minGap and inp.minGap > 0 else 0.0

    selected = []

    for frame in approved:
        keep = all(
            frame["end"] + extra_gap <= chosen["start"]
            or frame["start"] >= chosen["end"] + extra_gap
            for chosen in selected
        )
        if keep:
            selected.append(frame)
        if len(selected) == inp.finalCandidates:
            break

    # Fields the n8n flow expects on each candidate.
    # motion_frames don't carry the original candidate frame path, so use the
    # cached middle clip frame as a representative thumbnail.
    for frame in selected:
        clip_frames = frame.get("clip_frames") or []
        thumb = clip_frames[2] if len(clip_frames) > 2 else (
            clip_frames[0] if clip_frames else None
        )
        frame["frame_rel"] = (
            os.path.relpath(thumb, MEDIA) if thumb else None
        )
        frame["vision_score"] = frame.get("confidence", 0.0)

    print_final_results(selected)

    return selected

@app.post("/candidates")
def candidates(inp: CandIn):
    full = os.path.join(MEDIA, inp.path)
    dur = probe(ProbeIn(path=inp.path))["duration"]
    jobs = split_video_jobs(dur)

    print(
        "\n========== VIDEO JOBS ==========",
        flush=True
    )

    for job in jobs:
        print(
            job,
            flush=True
        )

    print(
        "================================\n",
        flush=True
    )
    
    all_selected = []

    for job in jobs:

        print(
            f"\nProcessing Job {job['job']}",
            flush=True
        )

        if len(jobs) == 1:
            job_dir = os.path.join(
                MEDIA,
                "work",
                inp.jobId
            )
        else:
            job_dir = os.path.join(
                MEDIA,
                "work",
                inp.jobId,
                f"segment_{job['job']}"
            )

        os.makedirs(
            job_dir,
            exist_ok=True
        )

        if len(jobs) == 1:

            segment = full

        else:

            segment = os.path.join(
                job_dir,
                "source.mp4"
            )

            split_video(
                full,
                job["start"],
                job["end"],
                segment
            )

        segment_duration = (
            job["end"] -
            job["start"]
        )

        selected = process_job(
            segment,
            segment_duration,
            inp,
            job_dir
        )

        for frame in selected:

            frame["time"] += job["start"]
            frame["start"] += job["start"]
            frame["end"] += job["start"]

        all_selected.extend(selected)

    #print("INPUT PATH:", repr(inp.path))
    #info = probe(ProbeIn(path=inp.path))
    #import sys

    #print("=" * 80, flush=True)
    #print(f"INPUT PATH: {repr(inp.path)}", flush=True)
    #print(f"PROBE RESULT: {repr(info)}", flush=True)
    #print("=" * 80, flush=True)
    #sys.stdout.flush()
    #dur = info["duration"]

    all_selected.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    # Same non-overlapping rule as inside process_job, by actual [start, end]
    # overlap (clips are variable length), applied when merging clips from
    # multiple segments of a long video.
    extra_gap = inp.minGap if inp.minGap and inp.minGap > 0 else 0.0

    final = []

    for frame in all_selected:

        if all(
            frame["end"] + extra_gap <= f["start"]
            or frame["start"] >= f["end"] + extra_gap
            for f in final
        ):
            final.append(frame)

    # Optional overall cap across the whole video (all segments combined).
    # final is already sorted by final_score, so we keep the highest scored.
    if inp.maxCandidates and inp.maxCandidates > 0:
        final = final[:inp.maxCandidates]

    # final is already sorted best -> worst by final_score. Stamp an explicit
    # 1-based rank so the editor / uploader can post them in order.
    for idx, frame in enumerate(final):
        frame["rank"] = idx + 1

    return {
        "dur": dur,
        "candidates": final
    }

    

@app.post("/render")
def render(inp: RenderIn):

    source = os.path.join(
        MEDIA,
        inp.path
    )

    out_dir = os.path.join(
        MEDIA,
        "work",
        inp.jobId,
        "renders"
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    rendered = []

    print("SOURCE:", source, flush=True)
    print("EXISTS:", os.path.exists(source), flush=True)
    print("OUTDIR:", out_dir, flush=True)

    for i, clip in enumerate(inp.clips):

        rank = clip.rank if clip.rank else i + 1

        out_file = os.path.join(
            out_dir,
            f"clip_{rank}.mp4"
        )

        render_clip(
            source,
            clip.start,
            clip.end,
            out_file,
            aspect=inp.aspect,
            crop_mode=inp.cropMode,
            fade=inp.fade
        )

        rendered.append(
            f"work/{inp.jobId}/renders/clip_{rank}.mp4"
        )

    return {
        "clips": rendered
    }