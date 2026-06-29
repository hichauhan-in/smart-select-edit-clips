import os, json, subprocess
import shutil, glob, time
import re, base64, requests
import math
import numpy as np

from motion import motion_score, frame_quality
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO

try:
    from rapidocr_onnxruntime import RapidOCR
    HAVE_RAPIDOCR = True
except Exception:
    HAVE_RAPIDOCR = False

try:
    import easyocr
    HAVE_EASYOCR = True
except Exception:
    HAVE_EASYOCR = False

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

# OCR backend: RapidOCR (ONNXRuntime, CPU) is the default — much faster than
# EasyOCR with comparable accuracy on reward text. EasyOCR is kept as a
# fallback. To move OCR onto an AMD/Windows GPU later, run RapidOCR on the
# host with onnxruntime-directml; the call sites here stay identical.
OCR_ENGINE = ""
rapid_ocr = None
ocr = None

if HAVE_RAPIDOCR:
    print("Loading RapidOCR...", flush=True)
    rapid_ocr = RapidOCR()
    OCR_ENGINE = "rapidocr"
    print("RapidOCR loaded.", flush=True)
elif HAVE_EASYOCR:
    print("Loading EasyOCR...", flush=True)
    ocr = easyocr.Reader(
        ['en'],
        gpu=False
    )
    OCR_ENGINE = "easyocr"
    print("EasyOCR loaded.", flush=True)
else:
    print("WARNING: no OCR backend available; OCR scores will be 0.", flush=True)

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

def clip_quality_score(frame_paths):
    """
    Visual quality of a clip from its 5 stills, model-free.

    Frozen/loading/blurred frames have low Laplacian variance; black/loading
    screens have near-zero brightness. We average sharpness across frames and
    zero the score for clips that are too dark, so static or loading clips
    rank below real action even when motion looks high from a single cut.
    """
    sharps = []
    brights = []
    for fp in frame_paths:
        s, b = frame_quality(fp)
        sharps.append(s)
        brights.append(b)

    if not sharps:
        return 0.0

    mean_sharp = sum(sharps) / len(sharps)
    mean_bright = sum(brights) / len(brights)

    # Kill clips that are basically black (loading / fade) regardless of sharp.
    if mean_bright < 18.0:
        return 0.0

    return mean_sharp

def cut_density_score(boundary_cuts, start, end):
    """
    Reward a clip with ~one clean scene change (a kill/replay transition) and
    penalise both zero cuts (static) and many cuts (menus/montage churn).
    Returns a 0..1 score; uses cuts we already detected, so it is free.
    """
    cuts = sum(1 for c in boundary_cuts if start < c < end)
    if cuts == 0:
        return 0.3
    if cuts <= 2:
        return 1.0
    if cuts <= 4:
        return 0.6
    return 0.2

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


def subject_center_x(frames):
    """Average horizontal centre (0..1) of detected people across frames, used
    to pan a vertical crop so the player stays in shot. Returns 0.5 (frame
    centre) when no person is found."""
    xs = []
    for fp in frames:
        res = YOLO_MODEL(fp, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        names = res.names
        for cls, xc, w in zip(
            boxes.cls.tolist(),
            boxes.xywhn[:, 0].tolist(),
            boxes.xywhn[:, 2].tolist(),
        ):
            if names[int(cls)] == "person":
                xs.append(xc)
    return sum(xs) / len(xs) if xs else 0.5

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
2. approve  : true only if this is genuinely highlight-worthy ACTION. Reject
   (approve = false): advertisements, promos, intros / outros, and ALSO low-
   action filler even if it is real gameplay - inventory / menu / map / shop /
   loadout screens, lock-picking or safe-cracking, idle standing, slow walking
   or running with nothing happening, and crafting / looting. A highlight needs
   a real moment: a fight, kill, chase, explosion, clutch or escape. If the 5
   frames are mostly menu, inventory, or just walking, approve = false.
3. confidence : how strong this highlight is (see guide below).
4. Use the CV scores as supporting evidence:
   - If the visuals agree with the scores, raise your confidence.
   - If the scores look misleading (e.g. high motion but nothing happens),
     lower your confidence.
5. Engagement signals (these decide how well the clip performs on Shorts /
   Reels / TikTok). Be honest; most clips are average:
   - hook       : 0.0-1.0, does the FIRST frame instantly grab attention?
   - multikill  : true if a kill / multi-kill / big play is clearly visible.
   - clutch     : true if it looks like a comeback or high-stakes moment.
   - funny      : true if it is funny / surprising / meme-worthy.
   - vertical   : true if the main action is centered and survives a 9:16 crop.

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
    "hook": 0.0,
    "multikill": false,
    "clutch": false,
    "funny": false,
    "vertical": true,
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

    # Engagement boost: clips with a strong hook or a clear money moment win on
    # short-form platforms. Only nudge approved gameplay; rejects stay rejected.
    if gameplay and approve:
        hook = float(data.get("hook", 0.0) or 0.0)
        bonus = hook * 0.10
        if data.get("multikill"):
            bonus += 0.10
        if data.get("clutch"):
            bonus += 0.07
        if data.get("funny"):
            bonus += 0.05
        if not data.get("vertical", True):
            bonus -= 0.05
        confidence = max(0.0, min(1.0, confidence + bonus))

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
        frame["motion_norm"] * 0.42
        + frame["ocr_norm"] * 0.23
        + frame["audio_norm"] * 0.13
        + frame["yolo_norm"] * 0.07
        + frame.get("quality_norm", 0.5) * 0.10
        + frame.get("cuts_norm", 0.5) * 0.05
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
    # OCR + vision are the slow stages. We refine all topMotion candidates
    # (cheap motion math) but only run OCR/vision on the strongest ones.
    # 0 == auto: half of the refined candidates (never below finalCandidates).
    # Set a positive number to force an exact cap. Lower = faster.
    ocrTop: int = 0
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
    # Which segment of a long video this clip came from (1-based). 0 == single
    # segment. Used to render into that segment's own folder.
    segment: int = 0


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

    if OCR_ENGINE == "rapidocr":
        result, _ = rapid_ocr(image_path)
        text = " ".join(
            r[1]
            for r in (result or [])
        ).upper()
    elif OCR_ENGINE == "easyocr":
        results = ocr.readtext(image_path)
        text = " ".join(
            r[1]
            for r in results
        ).upper()
    else:
        text = ""

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

def find_motion_lull(video_path, lo, hi, job_dir, step=0.5):
    """Return the time of lowest motion inside [lo, hi]. Used to end a clip on a
    calm moment when there is no scene cut to snap to, so a continuous scene
    stops where the action actually subsides instead of at a fixed stopwatch."""
    if hi - lo < step or not video_path:
        return None
    work = os.path.join(job_dir, "refine", f"lull_{int(lo)}")
    os.makedirs(work, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(lo), "-i", video_path, "-t", str(hi - lo),
         "-vf", f"fps={1.0/step}", "-q:v", "5", os.path.join(work, "l_%04d.jpg")],
        capture_output=True
    )
    frames = sorted(glob.glob(os.path.join(work, "l_*.jpg")))
    best_t, best_m, prev = None, None, None
    for i, img in enumerate(frames):
        if prev is not None:
            m = motion_score(prev, img)
            if best_m is None or m < best_m:
                best_m, best_t = m, lo + i * step
        prev = img
    return best_t


def choose_clip_bounds(peak, scene_cuts, dur, clip_min, clip_max,
                       video_path=None, job_dir=None):
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

    # END: snap to the scene cut whose length is closest to the target, so the
    # clip ends on a real visual change rather than a stopwatch. Only cuts that
    # land after the peak and keep length within [clip_min, clip_max] qualify;
    # among those pick the one nearest the ideal target length.
    valid_ends = [
        e for e in cuts
        if e > peak and clip_min <= (e - start) <= clip_max
    ]
    if valid_ends:
        end = min(valid_ends, key=lambda e: abs((e - start) - target))
    else:
        # No scene cut in range: end at the calmest moment between clip_min and
        # clip_max so a continuous scene stops where action drops, not on a
        # fixed timer. Falls back to target if motion can't be sampled.
        lull = find_motion_lull(
            video_path,
            min(dur, start + clip_min),
            min(dur, start + clip_max),
            job_dir,
        )
        end = lull if lull else start + target

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


def build_video_filter(aspect, crop_mode, fade_dur, duration, cx=0.5):
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
    # "smart": same crop width but panned toward the subject (cx 0..1) so the
    # player isn't cut off; cx=0.5 is identical to center.
    cx = min(0.85, max(0.15, cx))
    chain = (
        f"crop='min(iw,ih*{W}/{H})':'min(ih,iw*{H}/{W})':"
        f"'(iw-min(iw,ih*{W}/{H}))*{cx:.3f}':0,"
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
            clip_max,
            video_path=full,
            job_dir=job_dir
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

    # OCR (EasyOCR on CPU) is the single slowest stage - 5 frames per clip,
    # extracted + read. Only the strongest candidates are worth that cost.
    # ocrTop 0 == auto: keep about half the refined candidates; otherwise use
    # the explicit cap. Never fewer than finalCandidates so enough survive.
    auto_top = max(inp.finalCandidates, (len(interesting) + 1) // 2)
    ocr_top = inp.ocrTop if inp.ocrTop and inp.ocrTop > 0 else auto_top
    ocr_top = max(inp.finalCandidates, min(ocr_top, len(interesting)))

    # Pre-OCR gate. OCR is the slow stage, so we only run it on the strongest
    # half. Rank on every cheap signal we already have (motion, YOLO objects,
    # audio) instead of motion alone, so a clip that is loud or object-rich but
    # only moderately fast still survives. OCR is the only feature missing here,
    # so it cannot be used yet. Audio is scored now and reused in the OCR loop.
    for frame in interesting:
        frame["audio"] = clip_audio_score(
            audio_times,
            audio_vals,
            frame["start"],
            frame["end"]
        )
    normalize_feature(interesting, "motion", "motion_norm")
    normalize_feature(interesting, "yolo", "yolo_norm")
    normalize_feature(interesting, "audio", "audio_norm")
    interesting.sort(
        key=lambda x: (
            x["motion_norm"] * 0.60
            + x["audio_norm"] * 0.20
            + x["yolo_norm"] * 0.20
        ),
        reverse=True
    )
    interesting = interesting[:ocr_top]

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

        # Cheap technical-quality signals reusing the 5 frames we just loaded.
        frame["quality"] = clip_quality_score(clip_frames)
        frame["cuts"] = cut_density_score(
            boundary_cuts,
            frame["start"],
            frame["end"]
        )

        print(
            f"OCR={frame['ocr']:.1f}",
            f"Hits={ocr_hits}",
            f"Audio={frame['audio']:.1f}",
            f"Quality={frame['quality']:.0f}",
            f"Cuts={frame['cuts']:.1f}",
            f"Text={frame['ocr_text'][:80]}",
            flush=True
        )
        frame["reason"] = ""

        scored.append(frame)

    normalize_feature(scored, "motion", "motion_norm") 

    normalize_feature(scored, "yolo", "yolo_norm")

    normalize_feature(scored, "ocr", "ocr_norm")

    normalize_feature(scored, "audio", "audio_norm")

    normalize_feature(scored, "quality", "quality_norm")

    normalize_feature(scored, "cuts", "cuts_norm")

    for frame in scored:

        frame["final_score"] = compute_fast_score(
            frame
        )

    scored.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    # Vision is the most expensive stage, so feed it half the OCR survivors
    # (rounded up), never fewer than finalCandidates, never more than 6. Short
    # videos with few OCR clips shrink naturally; long ones stay capped.
    llava_candidates = max(
        inp.finalCandidates,
        min(6, (len(scored) + 1) // 2)
    )

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
            frame["segment"] = job["job"] if len(jobs) > 1 else 0

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

    # final is already sorted best -> worst by final_score. Stamp a global rank
    # for overall posting order, plus a per-segment rank so each segment's
    # render folder gets clip_1, clip_2, ... independently.
    seg_counts = {}
    for idx, frame in enumerate(final):
        frame["rank"] = idx + 1
        seg = frame.get("segment", 0)
        seg_counts[seg] = seg_counts.get(seg, 0) + 1
        frame["segment_rank"] = seg_counts[seg]

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

        # Long videos are split into segments; each segment renders into its
        # own folder so clips stay grouped by where they came from. Single
        # segment (segment == 0) keeps the flat renders/ layout.
        if clip.segment and clip.segment > 0:
            clip_dir = os.path.join(out_dir, f"segment_{clip.segment}")
        else:
            clip_dir = out_dir
        os.makedirs(clip_dir, exist_ok=True)

        out_file = os.path.join(
            clip_dir,
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

        if clip.segment and clip.segment > 0:
            rendered.append(
                f"work/{inp.jobId}/renders/segment_{clip.segment}/clip_{rank}.mp4"
            )
        else:
            rendered.append(
                f"work/{inp.jobId}/renders/clip_{rank}.mp4"
            )

    return {
        "clips": rendered
    }


class FinalizeIn(BaseModel):
    jobId: str
    name: str = ""


@app.post("/finalize")
def finalize(inp: FinalizeIn):
    """End-of-flow cleanup. Keeps only work/<jobId>/renders. Moves source.mp4
    to media/archive and every other intermediate (frames, refine, clips and
    any segment_* dirs) to media/temp/<jobId>. Nothing is deleted - the user
    can clear media/temp manually."""

    work_dir = os.path.join(MEDIA, "work", inp.jobId)
    if not os.path.isdir(work_dir):
        return {"ok": False, "reason": f"no work dir for {inp.jobId}"}

    archive_dir = os.path.join(MEDIA, "archive")
    temp_dir = os.path.join(MEDIA, "temp", inp.jobId)
    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    archived = None
    moved = []

    for entry in os.listdir(work_dir):
        src = os.path.join(work_dir, entry)

        # Keep the final renders + edited shorts in work.
        if entry in ("renders", "edited"):
            continue
        # Source video -> archive, named after the job.
        if entry == "source.mp4":
            dst = os.path.join(archive_dir, f"{inp.jobId}.mp4")
            shutil.move(src, dst)
            archived = os.path.relpath(dst, MEDIA)
            continue

        # Everything else (frames, refine, clips, segment_* ...) -> temp.
        dst = os.path.join(temp_dir, entry)
        if os.path.exists(dst):
            shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
        shutil.move(src, dst)
        moved.append(entry)

    print(f"FINALIZE {inp.jobId}: archived={archived} moved={moved}", flush=True)

    return {
        "ok": True,
        "archived": archived,
        "moved": moved,
        "temp": f"temp/{inp.jobId}",
    }


def edit_clip(source, output, aspect, crop_mode, fade):
    """Reframe an already-rendered full-res clip to a short-form aspect.
    Unlike render_clip this takes a finished file (no -ss/-t) and just applies
    the crop/scale/fade filter. crop_mode "smart" pans the vertical window to
    keep the player in shot; "center" keeps the middle."""

    dur = probe(ProbeIn(path=os.path.relpath(source, MEDIA)))["duration"]

    # Smart crop: detect where the player sits and pan the crop toward them.
    cx = 0.5
    if crop_mode == "smart":
        tmp = os.path.join(os.path.dirname(output), f".sub_{os.path.basename(output)}")
        sub = _extract_clip_frames(source, 0.0, dur, tmp)
        cx = subject_center_x(sub)
        print(f"SMART CROP cx={cx:.3f} for {os.path.basename(source)}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)

    flag, filter_string, extra_maps = build_video_filter(
        aspect, crop_mode, fade, dur, cx=cx
    )
    cmd = ["ffmpeg", "-y", "-i", source, flag, filter_string] + extra_maps + [
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", output,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("EDIT:", output, "rc=", r.returncode, flush=True)


class EditIn(BaseModel):
    jobId: str
    aspect: str = "9:16"
    fade: float = 0.5


@app.post("/edit")
def edit(inp: EditIn):
    """Reframe EVERY rendered clip into two variants: a smart-cropped vertical
    (player tracked) and a blurred-background vertical. Mirrors the renders
    layout (flat or segment_*) under work/<jobId>/edited/cropped and /blurred."""

    renders = os.path.join(MEDIA, "work", inp.jobId, "renders")
    if not os.path.isdir(renders):
        return {"ok": False, "reason": f"no renders for {inp.jobId}"}

    # All rendered clips, flat and per-segment, with their path relative to
    # renders/ so each variant folder keeps the same structure.
    sources = sorted(glob.glob(os.path.join(renders, "**", "clip_*.mp4"), recursive=True))
    if not sources:
        return {"ok": False, "reason": "no clips rendered"}

    out_dir = os.path.join(MEDIA, "work", inp.jobId, "edited")
    cropped, blurred = [], []

    for src in sources:
        rel = os.path.relpath(src, renders)
        for mode, dest in (("smart", "cropped"), ("blur", "blurred")):
            out_file = os.path.join(out_dir, dest, rel)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            edit_clip(src, out_file, inp.aspect, mode, inp.fade)
            relout = f"work/{inp.jobId}/edited/{dest}/{rel.replace(os.sep, '/')}"
            (cropped if mode == "smart" else blurred).append(relout)

    return {"ok": True, "cropped": cropped, "blurred": blurred}