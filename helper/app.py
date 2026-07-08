import os, json, subprocess
import shutil, glob, time
import re, base64, requests
import math
import numpy as np
import cv2

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

    # --- Universal win / completion / achievement (any genre) ---
    "VICTORY": 9,
    "CHAMPION": 9,
    "WINNER": 8,
    "WIN": 8,
    "COMPLETE": 6,
    "COMPLETED": 6,
    "FLAWLESS": 7,
    "PERFECT": 6,
    "SUCCESS": 5,
    "NEW RECORD": 7,
    "RECORD": 4,
    "LEVEL UP": 6,
    "LEVELUP": 6,
    "RANK UP": 6,
    "ACHIEVEMENT": 5,
    "UNLOCKED": 4,
    "COMBO": 5,
    "FINISH": 5,
    "GOAL": 5,
    "BOSS": 4,
    "CHECKPOINT": 2,
    "SCORE": 2,

    # --- Combat / shooter cues (still supported, no longer dominant) ---
    "ACE": 9,
    "QUAD": 8,
    "FATALITY": 8,
    "TRIPLE": 7,
    "HEADSHOT": 6,
    "FIRST BLOOD": 6,
    "DOUBLE": 5,
    "KO": 5,
    "ELIMINATION": 4,
    "ELIMINATED": 4,
    "KILL": 3,
    "KNOCKED": 3,
    "DOWNED": 3,

    # --- Point / XP pop-ups ---
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


def track_subject_x(video_path, dur, job_dir, step=1.0):
    """Sample the player's horizontal centre once per `step` seconds and return
    a list of (time, cx) used to pan the crop window dynamically. Empty frames
    fall back to the previous point so the window holds instead of jumping."""
    work = os.path.join(job_dir, f".track_{int(time.time()*1000)}")
    os.makedirs(work, exist_ok=True)
    n = max(2, int(dur / step))
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps=1/{step}",
         "-q:v", "5", os.path.join(work, "t_%04d.jpg")],
        capture_output=True
    )
    frames = sorted(glob.glob(os.path.join(work, "t_*.jpg")))
    pts, last = [], 0.5
    for i, fp in enumerate(frames):
        cx = subject_center_x([fp])
        if cx == 0.5:
            cx = last
        # FPS kills land on the centre crosshair while the player sits off to
        # one side, so only lean ~40% toward the player and keep the centre.
        cx = 0.5 + (cx - 0.5) * 0.4
        last = cx
        pts.append((round(i * step, 2), round(cx, 3)))
    shutil.rmtree(work, ignore_errors=True)
    # Smooth so the pan glides rather than snaps.
    sm = []
    for i, (t, c) in enumerate(pts):
        lo = max(0, i - 1)
        hi = min(len(pts), i + 2)
        sm.append((t, sum(p[1] for p in pts[lo:hi]) / (hi - lo)))
    return sm or [(0.0, 0.5)]


def pan_x_expr(points):
    """Build a piecewise-linear ffmpeg x expression (fraction 0..1) over time
    from tracked (time, cx) points, clamped 0.30..0.70 to keep the centre
    crosshair/action in frame even when leaning toward the player."""
    pts = [(t, min(0.70, max(0.30, c))) for t, c in points]
    expr = f"{pts[-1][1]:.3f}"
    for i in range(len(pts) - 2, -1, -1):
        t0, c0 = pts[i]
        t1, c1 = pts[i + 1]
        slope = (c1 - c0) / (t1 - t0) if t1 > t0 else 0.0
        expr = f"if(lt(t,{t1:.2f}),{c0:.3f}+({slope:.5f})*(t-{t0:.2f}),{expr})"
    return expr


def find_motion_peak(video_path, dur, job_dir, step=0.5):
    """Return the time of HIGHEST motion in a clip - the moment to punch in on
    for an auto-zoom. Returns dur/2 when motion can't be sampled."""
    work = os.path.join(job_dir, f".peak_{int(time.time()*1000)}")
    os.makedirs(work, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={1.0/step}",
         "-q:v", "5", os.path.join(work, "p_%04d.jpg")],
        capture_output=True
    )
    frames = sorted(glob.glob(os.path.join(work, "p_*.jpg")))
    best_t, best_m, prev = dur / 2.0, -1.0, None
    for i, img in enumerate(frames):
        if prev is not None:
            m = motion_score(prev, img)
            if m > best_m:
                best_m, best_t = m, i * step
        prev = img
    shutil.rmtree(work, ignore_errors=True)
    return best_t


def zoom_expr(peak_t, amp=0.18, sigma=2.0):
    """Crop-size multiplier over time: 1.0 normally, shrinking (punch-in) to
    1-amp around peak_t with a smooth gaussian, so motion drives a subtle zoom."""
    return f"(1-{amp}*exp(-((t-{peak_t:.2f})/{sigma})^2))"

def _extract_clip_frames(full, start, end, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    duration = end - start

    # 8 frames evenly spaced across the clip (start -> end). More frames than
    # before give the vision model finer temporal resolution so a brief highlight
    # (a kill, a goal, a crash that lasts ~1s) is far less likely to fall
    # between samples and be missed.
    fractions = [0.0, 0.14, 0.28, 0.42, 0.56, 0.70, 0.84, 0.97]
    offsets = [duration * f for f in fractions]

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
        summary.append("Strong on-screen reward/achievement text.")
    elif ocr >= 0.50:
        summary.append("Some on-screen reward/achievement text.")
    elif ocr >= 0.20:
        summary.append("Very little on-screen reward text.")
    else:
        summary.append("No on-screen reward text.")

    if audio >= 0.80:
        summary.append("Strong impact/event sounds.")
    elif audio >= 0.50:
        summary.append("Some impact/event sounds.")
    elif audio >= 0.20:
        summary.append("Faint impact/event sounds.")
    else:
        summary.append("No notable impact/event sounds.")

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
You are an expert gaming highlights editor who curates short clips for social
media (Shorts / Reels / TikTok) across EVERY kind of video game - shooters,
battle royales, MOBAs, fighting games, racing, sports, open-world, RPGs,
survival, platformers, roguelikes, horror, puzzle and 2D indie games.

Do NOT assume this is a shooter. Judge what is actually happening in the
frames for THIS game, whatever genre it is.

You are given several frames sampled at EQUAL time steps from ONE single
{clip_len:.0f}-second gameplay clip, in chronological order: the FIRST frame is
the start of the clip and the LAST frame is the end.

Judge the clip as ONE continuous moment. Follow how the action PROGRESSES from
the first frame to the last, instead of rating each image on its own.

These frames were already pre-selected by an automated highlight
detection pipeline, so they are likely (but not guaranteed) interesting.

--- What makes a HIGHLIGHT (any genre) ---
STRONG (rate high, approve = true): an actual fight / firefight with visible
enemies, a kill or multi-kill, getting killed in a dramatic way, a clutch or
1-vs-many, a comeback, an explosion, a vehicle chase or crash, a race overtake
or finish, a goal or big score, a boss fight, a clean skill shot or trick, a
jump-scare, a win, or a genuinely funny / surprising moment.

NOT a highlight (reject, low rating) - these are the exact traps to avoid:
dropping in / parachuting / gliding / landing, running or driving around with
no enemies, looting crates or bodies, picking up ammo / weapons / cash / armor,
crafting, opening doors, reading the map, inventory, menus, shops, loadouts,
buy / respawn screens, idle standing, or slow "walking between fights". These
are boring EVEN WHEN the camera moves fast or the scenery flies by. When in
doubt, REJECT.

--- Computer Vision Analysis ---
These scores are RELATIVE to the other candidate clips detected in THIS
video (1.00 = strongest among the candidates, 0.00 = weakest). They are
NOT absolute quality measures, so treat them as ranking hints, not truth.

Motion Score : {motion:.2f}   (relative on-screen movement - WARNING: high motion ALSO comes from camera panning, walking, driving and looting, so a high value does NOT by itself mean the clip is exciting)
YOLO Score   : {yolo:.2f}   (relative object count - weak signal, low weight)
OCR Score    : {ocr:.2f}   (relative on-screen reward/score text: e.g. WIN, VICTORY, LEVEL UP, COMBO, KILL, +points)
Audio Score  : {audio:.2f}   (relative impact/event sounds. SUSTAINED loud, rapid-fire audio almost always means a real gunfight / firefight / combat; a quiet clip is almost always looting, walking, parachuting or menus - reject those)

Summary:
{chr(10).join(cv_summary)}

--- Your task ---
1. gameplay : true ONLY if these frames clearly show real, live video-game
   gameplay (of ANY genre). Set gameplay = false for anything that is not live
   gameplay, including: advertisements, sponsor / promotional / brand screens or
   logos, intros, outros, menus, loading screens, pure scoreboards, desktop,
   webcam / face-cam, black screens, OR frozen / static frames where almost
   nothing changes across the 5 frames.
2. approve  : true only if this is a genuinely ENTERTAINING or exciting moment
   worth posting. What counts depends on the genre - any of: a fight / kill /
   multi-kill, a clutch or comeback, a chase or escape, a crash or wipeout, a
   race finish or overtake, a goal / score / big play, a boss fight, a stunt or
   skillful trick, a jump-scare, a win, or a funny / surprising / satisfying
   moment. Reject (approve = false): ads and promos, intros / outros, and
   low-action filler even if it IS real gameplay - menus, inventory, map, shop,
   loadout, dialogue / cutscene with no action, idle standing, aimless slow
   walking / driving with nothing happening, plain looting or crafting. If the
   frames are mostly menu, inventory, or just wandering with nothing happening,
   approve = false.
3. rating : an INTEGER from 0 to 10 for how entertaining THIS clip is as a
   standalone highlight to a stranger scrolling their feed. Judge ONLY what is
   actually happening in the scene, NOT how much the screen moves:
     0-2  = boring filler: dropping in / parachuting / landing, idle standing,
            aimless walking / running / driving with no enemies, looting crates
            or bodies, picking up ammo / weapons / cash, crafting, menus,
            inventory, map, shop, buy / respawn screens, or just wandering with
            nothing happening - EVEN IF the camera moves a lot.
     3-4  = minor activity but no real payoff.
     5-6  = a decent moment (a small fight, a nice bit of play).
     7-8  = a strong highlight (a clean kill / multi-kill, a clutch, a great
            overtake, a goal, a boss phase, a cool stunt, a scary jump-scare).
     9-10 = an exceptional, must-watch moment.
   Be strict and honest: most clips are 2-5. NEVER inflate a clip just because
   its Motion Score is high - a spinning camera or someone running around is
   still boring.
4. confidence : how strong this highlight is (see guide below).
5. Use the CV scores as supporting evidence:
   - If the visuals agree with the scores, raise your confidence.
   - If the scores look misleading (e.g. high motion but nothing happens, just a
     spinning camera), lower your confidence.
6. Engagement signals (these decide how well the clip performs on Shorts /
   Reels / TikTok). Be honest; most clips are average:
   - hook       : 0.0-1.0, does the FIRST frame instantly grab attention?
   - bigplay    : true if a clearly impressive or skillful moment / big play is
                  visible (a kill, a clutch shot, a perfect combo, a great
                  overtake, a trick, a goal - whatever is impressive for this game).
   - clutch     : true if it looks like a comeback or high-stakes moment.
   - funny      : true if it is funny / surprising / meme-worthy.
   - vertical   : true if the main action is centered and survives a 9:16 crop.

Consistency rules:
- If gameplay is false, approve MUST be false, rating MUST be 0 and confidence MUST be 0.00.
- If approve is false, rating MUST be <= 3 and confidence MUST be <= 0.30.
- If the frames look almost identical (no real change), treat it as
  static: gameplay = false, rating = 0 and confidence = 0.00.

Confidence guide:
1.00 = Exceptional highlight
0.90 = Excellent gameplay moment
0.80 = Strong highlight
0.70 = Good gameplay moment
0.60 = Average gameplay
0.40 = Weak highlight
0.20 = Probably not a highlight
0.00 = Not gameplay / definitely reject

Return ONLY this JSON object, nothing else:
{{
    "gameplay": true,
    "approve": true,
    "rating": 0,
    "confidence": 0.0,
    "hook": 0.0,
    "bigplay": false,
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

    # Vision rating 0-10 -> 0..1. This is the PRIMARY highlight signal in the
    # final ranking: the model is the only stage that actually understands the
    # scene, so it separates real highlights from mere camera movement far
    # better than motion. Fall back to confidence if the model omitted it.
    try:
        rating = max(0.0, min(1.0, float(data.get("rating")) / 10.0))
    except (TypeError, ValueError):
        rating = confidence

    if not (gameplay and approve):
        rating = min(rating, 0.30)

    # Engagement boost: clips with a strong hook or a clear money moment win on
    # short-form platforms. Only nudge approved gameplay; rejects stay rejected.
    if gameplay and approve:
        hook = float(data.get("hook", 0.0) or 0.0)
        bonus = hook * 0.10
        # `bigplay` is the genre-agnostic "impressive moment" flag; older
        # prompts called it `multikill`, so accept either.
        if data.get("bigplay") or data.get("multikill"):
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
        rating,
        reason
    )

def vision_failed(e):

    return (
        True,
        True,
        0.5,
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
    # Genre-agnostic highlight score. Leans on the two signals every game
    # shares - sustained on-screen action (motion) and event/impact audio -
    # plus technical quality. Reward text (OCR) and object count (YOLO) are
    # kept only as small bonuses so shooter-specific cues no longer dominate
    # and non-shooter games (open-world, racing, platformers, 2D, etc.) are
    # scored fairly.
    return (
        frame["motion_norm"] * 0.40
        + frame["audio_norm"] * 0.28
        + frame.get("quality_norm", 0.5) * 0.14
        + frame.get("cuts_norm", 0.5) * 0.08
        + frame["ocr_norm"] * 0.06
        + frame["yolo_norm"] * 0.04
    )

def split_video_jobs(
    duration_seconds,
    max_minutes=15,
    overlap_seconds=15
):

    # Split a long video into the FEWEST EQUAL-length parts that each stay at or
    # under max_minutes, instead of greedily filling max-length chunks and
    # leaving a tiny remainder. A 20-min video becomes 2x10 min (not 15+5); a
    # 40-min video becomes ~3x13.3 min; every part is balanced and never
    # exceeds max_minutes, so the workload is spread evenly.
    max_duration = max_minutes * 60

    if duration_seconds <= max_duration:
        return [
            {
                "job": 1,
                "start": 0.0,
                "end": duration_seconds
            }
        ]

    parts = math.ceil(
        duration_seconds / max_duration
    )

    # Balanced length per part; guaranteed <= max_duration because
    # parts = ceil(duration / max_duration).
    base = duration_seconds / parts

    jobs = []

    for i in range(parts):

        start = i * base
        end = (i + 1) * base

        # Overlap the start of every part after the first so an action that
        # straddles a boundary is still fully captured in one part.
        if i > 0:
            start -= overlap_seconds

        if i == parts - 1:
            end = duration_seconds

        jobs.append({
            "job": i + 1,
            "start": round(start, 3),
            "end": round(end, 3)
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
    finalCandidates: int = 3
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


def cut_ends_scene(video_path, cut_t, dur, job_dir, gap=0.6, step=0.4):
    """Tell a real scene change from an in-action flash / explosion.

    ``motion_score`` is the mean absolute luma difference between two frames
    (0-255). At a genuine hard cut the picture a moment BEFORE the cut and a
    moment AFTER it are unrelated, so their difference is far larger than the
    scene's own frame-to-frame motion. An explosion or muzzle flash spikes the
    scene_score for a single frame but the SAME scene continues on both sides,
    so the before/after pictures stay similar.

    We compare the frame ~``gap``s before the cut to the frame ~``gap``s after
    it (skipping the bright flash frame itself) against the in-scene motion
    baseline just before the cut. Returns True only when the two sides differ
    clearly more than normal in-scene motion -> a cut that truly ENDS the
    action. Returns True when it can't sample (no video / too near an edge) so
    the scene_score decision stands.
    """
    if not video_path or cut_t <= gap:
        return True
    lo = max(0.0, cut_t - gap - 2 * step)
    hi = min(dur, cut_t + gap + step)
    if hi - lo < 3 * step:
        return True
    work = os.path.join(job_dir, "refine", f"cut_{int(cut_t * 1000)}")
    os.makedirs(work, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(lo), "-i", video_path, "-t", str(hi - lo),
         "-vf", f"fps={1.0/step}", "-q:v", "5", os.path.join(work, "c_%04d.jpg")],
        capture_output=True
    )
    frames = sorted(glob.glob(os.path.join(work, "c_*.jpg")))
    if len(frames) < 4:
        shutil.rmtree(work, ignore_errors=True)
        return True
    times = [lo + i * step for i in range(len(frames))]

    def nearest(target):
        return min(range(len(times)), key=lambda i: abs(times[i] - target))

    bi = nearest(cut_t - gap)
    ai = nearest(cut_t + gap)
    across = motion_score(frames[bi], frames[ai]) if bi != ai else 0.0

    # In-scene baseline: consecutive motion among frames strictly before the cut.
    base = [
        motion_score(frames[i - 1], frames[i])
        for i in range(1, len(frames))
        if times[i] <= cut_t - step
    ]
    shutil.rmtree(work, ignore_errors=True)
    baseline = (sum(base) / len(base)) if base else 0.0

    # Real cut: the two sides differ much more than normal in-scene motion, and
    # by a meaningful absolute amount so a low-motion scene can't trip on noise.
    settled = across >= max(12.0, baseline * 1.6)
    print(
        f"  cut@{cut_t:.2f}s across={across:.1f} baseline={baseline:.1f} "
        f"-> {'scene-end' if settled else 'flash/keep-going'}",
        flush=True
    )
    return settled


def _audio_rise_series(times, audio):
    """Aligned, normalized (0..1) audio-onset series for ``times``.

    For each sample time we take the RISE in loudness over the previous ~0.5s
    (``loud(t) - loud(t-0.5)``); positive rises are the ONSET of a loud event
    (a hit, pickup, explosion, sting, goal, crash) - genre-agnostic. Sustained
    loud talking or background music stays flat and scores ~0. Returns zeros
    when no audio was decoded."""
    n = len(times)
    if not audio or not audio[0]:
        return [0.0] * n
    at, av = audio
    if len(at) < 2 or not av:
        return [0.0] * n
    ahop = at[1] - at[0]
    if ahop <= 0:
        return [0.0] * n

    def loud(t):
        i = int(round((t - at[0]) / ahop))
        i = max(0, min(len(av) - 1, i))
        return av[i]

    rises = [max(0.0, loud(t) - loud(t - 0.5)) for t in times]
    lo, hi = min(rises), max(rises)
    if hi <= lo:
        return [0.0] * n
    return [(r - lo) / (hi - lo) for r in rises]


def _intensity_series(times, mvals, audio):
    """Blend normalized motion and audio-onset into ONE smoothed 0..1
    'how much is happening right now' curve, sampled at ``times``.

    Genre-agnostic: motion covers on-screen action for any game (a firefight, a
    car chase, a platforming run, a 2D boss), audio covers event stingers. A
    3-tap smoothing stops a single noisy frame (a scene cut, a muzzle flash)
    from defining the burst on its own."""
    n = len(times)
    if n == 0:
        return []
    lo, hi = min(mvals), max(mvals)
    m01 = [((v - lo) / (hi - lo)) if hi > lo else 0.5 for v in mvals]
    a01 = _audio_rise_series(times, audio)
    raw = [0.65 * m01[i] + 0.35 * a01[i] for i in range(n)]
    sm = []
    for i in range(n):
        a = max(0, i - 1)
        b = min(n, i + 2)
        sm.append(sum(raw[a:b]) / (b - a))
    return sm


def _best_fixed_window(times, ivals, wlen, dur):
    """Slide a ``wlen``-second window over the intensity curve and return the
    ``[start, end]`` whose AVERAGE intensity is highest - the densest patch of
    action of that fixed length. This replaces guessing a centre from one spiky
    motion frame, which is what made clips start/stop at random points."""
    if not times:
        return None
    best_avg, best = -1.0, None
    for s in times:
        e = s + wlen
        seg = [ivals[j] for j, t in enumerate(times) if s <= t <= e]
        if not seg:
            continue
        avg = sum(seg) / len(seg)
        if avg > best_avg:
            best_avg, best = avg, s
    if best is None:
        return None
    s = max(0.0, min(best, dur - wlen if dur > wlen else 0.0))
    return s, min(dur, s + wlen)


def _burst_window(times, ivals, clip_min, clip_max, dur):
    """Grow a window outward from the intensity peak while activity stays above
    a threshold, capturing the WHOLE burst of action, then clamp its length
    into ``[clip_min, clip_max]``. Genre-agnostic natural clip length."""
    if not times:
        return None
    n = len(times)
    pi = max(range(n), key=lambda i: ivals[i])
    mean = sum(ivals) / n
    std = (sum((v - mean) ** 2 for v in ivals) / n) ** 0.5
    thr = max(ivals[pi] * 0.5, mean + 0.4 * std)

    a = b = pi
    while a - 1 >= 0 and ivals[a - 1] >= thr:
        a -= 1
    while b + 1 < n and ivals[b + 1] >= thr:
        b += 1

    ta = max(0.0, times[a] - 1.0)   # a beat of lead-in
    tb = min(dur, times[b] + 0.6)   # a beat of trail-out
    length = tb - ta
    peak_t = times[pi]

    if length < clip_min:
        c = (ta + tb) / 2.0
        ta, tb = c - clip_min / 2.0, c + clip_min / 2.0
    elif length > clip_max:
        # Keep only the densest clip_max seconds of a long burst.
        win = _best_fixed_window(
            [t for t in times if ta <= t <= tb],
            [ivals[i] for i, t in enumerate(times) if ta <= t <= tb],
            clip_max, dur,
        )
        ta, tb = win if win else (peak_t - clip_max / 2.0, peak_t + clip_max / 2.0)

    return max(0.0, ta), min(dur, tb)


def choose_clip_bounds(center, scene_pairs, dur, clip_min, clip_max,
                       video_path=None, job_dir=None,
                       intensity=None, audio=None):
    """Pick a coherent ``[start, end]`` for one highlight.

    The clip is placed to cover the actual BURST of action around ``center``,
    measured genre-agnostically from a blended motion + audio intensity curve
    (``intensity`` = ``(times, motion_values)`` already sampled during
    refinement, ``audio`` = ``(times, loudness)``). This replaces the old
    "centre a fixed window on one motion spike", which made clips begin and end
    at seemingly random points and only really worked for busy shooters.

    After the action window is found, its edges are snapped to nearby scene
    cuts (only when a real cut is close) so the clip also begins/ends on a clean
    visual change. The length is always kept within ``[clip_min, clip_max]`` and
    ``center`` stays inside the window.
    """
    clip_min = max(1.0, clip_min)
    clip_max = max(clip_min, clip_max)
    target = (clip_min + clip_max) / 2.0
    fixed = (clip_max - clip_min) < 1.0

    # ---- 1) Action window from the intensity curve -------------------------
    times = list(intensity[0]) if (intensity and intensity[0]) else None
    ivals = _intensity_series(times, intensity[1], audio) if times else None

    win = None
    if times and ivals:
        win = (_best_fixed_window(times, ivals, target, dur) if fixed
               else _burst_window(times, ivals, clip_min, clip_max, dur))
    if win is None:
        s = max(0.0, center - target / 2.0)
        win = (s, min(dur, s + target))
    start, end = win

    # ---- 2) Snap edges to nearby scene cuts (variable-length only) ---------
    # For fixed length the sliding window already found the best placement, and
    # snapping would only fight the hard length constraint, so we skip it there.
    if not fixed:
        pairs = []
        for p in (scene_pairs or []):
            if isinstance(p, (tuple, list)):
                pairs.append((float(p[0]), float(p[1])))
            else:
                pairs.append((float(p), 0.0))
        pairs.sort()
        cuts = [t for t, _ in pairs]
        STRONG = 0.20
        SNAP = 1.6   # only snap when a cut is within this many seconds of edge

        # Start: snap to the nearest cut around `start`, preferring one just
        # before it so we never clip into the middle of the action.
        near_start = [c for c in cuts if start - SNAP <= c <= start + SNAP]
        if near_start:
            before = [c for c in near_start if c <= start]
            start = max(before) if before else min(near_start)

        # End: snap to a strong cut near `end` that a motion-continuity check
        # confirms actually ENDS the scene (not just an explosion/flash).
        near_end = sorted(
            t for t, s in pairs
            if s >= STRONG and end - SNAP <= t <= end + SNAP
        )
        for c in near_end:
            if cut_ends_scene(video_path, c, dur, job_dir):
                end = c - 0.12   # stop a hair before the next scene's 1st frame
                break

    # ---- 3) Enforce length range, clamp, keep centre inside ----------------
    if end - start < clip_min:
        end = start + clip_min
    elif end - start > clip_max:
        if fixed:
            mid = (start + end) / 2.0
            start, end = mid - target / 2.0, mid + target / 2.0
        else:
            end = start + clip_max

    start = max(0.0, start)
    end = min(dur, end)
    if end - start < clip_min:
        start = max(0.0, end - clip_max)
    if center < start:
        start = max(0.0, center - 0.5)
    if center > end:
        end = min(dur, center + 0.5)
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


def build_video_filter(aspect, crop_mode, fade_dur, duration, cx=0.5,
                       cx_expr=None, zoom=None, hud_safe=False, headroom=0.42,
                       card_aspect="1:1", radius=48, fade_out=None, border=5,
                       side_margin=48):
    """
    Build the ffmpeg filter for reframing a clip to ``aspect`` with the chosen
    ``crop_mode`` plus optional fade in/out.

    ``fade_dur`` is the intro fade-IN length; ``fade_out`` is the outro
    fade-OUT length (defaults to ``fade_dur`` when not given, so a single value
    still gives a symmetric fade).

    Returns ``(flag, filter_string, extra_maps)`` where ``flag`` is either
    ``"-vf"`` (single chain) or ``"-filter_complex"`` (blur needs a split), and
    ``extra_maps`` are any explicit ``-map`` args the complex graph requires.
    """
    dims = ASPECT_DIMS.get(aspect, ASPECT_DIMS["9:16"])

    fin = fade_dur if fade_dur and fade_dur > 0 else 0.0
    fout = fade_out if fade_out is not None else fin
    fout = fout if fout and fout > 0 else 0.0
    fades = ""
    # Only fade when the clip is long enough to hold both an intro and outro.
    if (fin > 0 or fout > 0) and duration > (fin + fout):
        out_start = max(0.0, duration - fout)
        parts = []
        if fin > 0:
            parts.append(f",fade=t=in:st=0:d={fin}")
        if fout > 0:
            parts.append(f",fade=t=out:st={out_start:.3f}:d={fout}")
        fades = "".join(parts)

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

    if crop_mode == "template":
        # The whole gameplay frame (NO cropping) sits as a band over a blurred
        # copy of itself, leaving deliberate empty room above and below for a
        # title / logo / handles. The video is full-width (as big as possible
        # without cutting). ``headroom`` 0..1 = how much of the leftover space
        # goes ABOVE the video (0.42 = slightly high, more room at the bottom
        # for captions).
        head = min(0.9, max(0.1, headroom))
        fc = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},gblur=sigma=25[bg2];"
            f"[fg]scale={W}:-2[fg2];"
            f"[bg2][fg2]overlay=(W-w)/2:(H-h)*{head:.3f}"
            f"{fades}[v]"
        )
        return "-filter_complex", fc, ["-map", "[v]", "-map", "0:a?"]

    if crop_mode in ("card", "card_blur"):
        # Screenshot-style card: crop the gameplay to a taller (default 1:1)
        # window, round the corners and place it inset from the left/right
        # (``side_margin`` px each side) so the background shows around it, and
        # pushed down so the empty space splits 60% above / 40% below. The
        # background is either a solid BLACK canvas ("card") or a BLURRED zoom
        # of the frame itself ("card_blur"). The black card also gets a thin
        # white border ring tracing the rounded cutout.
        try:
            caw, cah = (float(x) for x in card_aspect.split(":"))
            car = caw / cah
        except Exception:
            car = 1.0
        sm = max(0, int(side_margin))
        Wc = W - 2 * sm
        Wc -= Wc % 2
        Hc = int(round(Wc / car))
        Hc -= Hc % 2
        r = max(0, int(radius))

        def _round_alpha(rad):
            # Opaque everywhere except outside the four corner quarter-circles
            # of a rounded rectangle the size of the layer being filtered.
            rad = max(0, int(rad))
            return (
                f"if(gt(abs(X-(W-1)/2),(W-1)/2-{rad})*gt(abs(Y-(H-1)/2),(H-1)/2-{rad}),"
                f"if(lte(hypot(abs(X-(W-1)/2)-((W-1)/2-{rad}),abs(Y-(H-1)/2)-((H-1)/2-{rad})),{rad}),255,0),255)"
            )

        # Card placement (top-left of the cutout): centred horizontally with a
        # side_margin gap each side, 60/40 vertically.
        vx = (W - Wc) // 2
        oy = int(round((H - Hc) * 0.6))
        dur_s = f"{max(0.1, duration):.3f}"

        crop_scale = (
            f"crop='min(iw,ih*{car:.5f})':'min(ih,iw/{car:.5f})',scale={Wc}:{Hc}"
        )
        fg = (
            f"{crop_scale},format=yuva420p,"
            f"geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a='{_round_alpha(r)}'[fg]"
        )

        if crop_mode == "card_blur":
            # Blurred zoom of the same clip behind the rounded cutout (no border).
            fc = (
                f"[0:v]split=2[src][bgsrc];"
                f"[bgsrc]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},gblur=sigma=25[bg];"
                f"[src]{fg};"
                f"[bg][fg]overlay={vx}:{oy}{fades}[v]"
            )
            return "-filter_complex", fc, ["-map", "[v]", "-map", "0:a?"]

        # Black card, optionally with a thin white border ring. The border is a
        # white rounded card (radius r+b) sitting b px larger behind the cutout,
        # so a uniform b-px ring of white shows around the rounded video.
        b = max(0, int(border))
        if b > 0:
            Wbc, Hbc = Wc + 2 * b, Hc + 2 * b
            fc = (
                f"[0:v]{fg};"
                f"color=c=black:s={W}x{H}:d={dur_s}[bg];"
                f"color=c=white:s={Wbc}x{Hbc}:d={dur_s},format=yuva420p,"
                f"geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a='{_round_alpha(r + b)}'[bd];"
                f"[bg][bd]overlay={vx - b}:{oy - b}[bg2];"
                f"[bg2][fg]overlay={vx}:{oy}{fades}[v]"
            )
        else:
            fc = (
                f"[0:v]{fg};"
                f"color=c=black:s={W}x{H}:d={dur_s}[bg];"
                f"[bg][fg]overlay={vx}:{oy}{fades}[v]"
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
    # "smart": same crop width but panned toward the subject. A time-based
    # cx_expr makes the pan follow the player frame-by-frame; a constant cx is
    # a single static offset. zoom (multiplier expr) punches in on the action.
    # hud_safe keeps the pan near centre so edge HUD isn't lost. cx=0.5 + no
    # zoom is identical to center.
    cx = min(0.85, max(0.15, cx))
    pan_lo, pan_hi = (0.35, 0.65) if hud_safe else (0.15, 0.85)
    cw = f"min(iw,ih*{W}/{H})"
    ch = f"min(ih,iw*{H}/{W})"
    frac = cx_expr if cx_expr else f"{cx:.3f}"
    frac = f"clip({frac},{pan_lo},{pan_hi})"
    x = f"(iw-({cw}))*({frac})"
    y = f"(ih-({ch}))/2"
    chain = f"crop='{cw}':'{ch}':'{x}':'{y}',scale={W}:{H}"
    chain += fades
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
        # crf 18 keeps the clip visually near-lossless; preset veryfast trades a
        # little file size for a big drop in encode time (the source resolution
        # is preserved - no downscaling). faststart makes it web-ready.
        "-c:v", "libx264",
        "-preset", "veryfast",
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
        return approx_time, 0.0, 0.0, [], []

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
    series_t = []
    series_v = []

    for i, img in enumerate(frames):

        if prev_frame is not None:

            m = motion_score(prev_frame, img)

            motion_sum += m
            motion_count += 1
            # frame i was sampled at time start + i / fps
            series_t.append(start + i / fps)
            series_v.append(m)

            if m > best_motion:
                best_motion = m
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

    # The motion time series (times, values) is returned so the bounds picker
    # can localise the real action burst without re-decoding the video.
    return best_time, best_motion, mean_motion, series_t, series_v


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


# =======================================================================
# Highlight detection core: dense whole-segment scan + activity curve.
# =======================================================================
# Cap on the dense motion scan. The whole segment is decoded ONCE into small
# downscaled grayscale stills; motion is the mean absolute pixel difference
# between consecutive stills. Downscaling keeps even a 4K / 15-min segment fast.
SCAN_MAX_FRAMES = int(os.environ.get("SCAN_MAX_FRAMES", "700"))
SCAN_WIDTH = int(os.environ.get("SCAN_WIDTH", "480"))


def _minmax(values, flat=0.5):
    """Normalize a list to 0..1. Returns ``flat`` for every element when the
    list is empty or perfectly constant."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [flat] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _scan_motion(full, seg_dur, job_dir):
    """Decode the WHOLE segment ONCE into small grayscale stills and return a
    dense motion time-series ``(times, values, step)``.

    ``values[i]`` is the mean absolute pixel difference between still ``i-1``
    and ``i`` (0..255) - i.e. how much the picture changed. This single pass
    replaces the old per-candidate ffmpeg re-decodes, so scanning the ENTIRE
    segment is actually cheaper than the previous sparse sampling while giving a
    complete picture of where the action is."""
    step = (
        min(2.0, max(0.5, seg_dur / SCAN_MAX_FRAMES))
        if seg_dur > 0 else 1.0
    )
    work = os.path.join(job_dir, "scan")
    os.makedirs(work, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", full,
         "-vf", f"fps={1.0 / step},scale={SCAN_WIDTH}:-2",
         "-q:v", "5", os.path.join(work, "s_%05d.jpg")],
        capture_output=True
    )
    files = sorted(glob.glob(os.path.join(work, "s_*.jpg")))
    times, vals, prev = [], [], None
    for i, fp in enumerate(files):
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            prev = None
            continue
        if prev is not None and prev.shape == img.shape:
            vals.append(float(cv2.absdiff(prev, img).mean()))
            times.append(round(i * step, 3))
        prev = img
    shutil.rmtree(work, ignore_errors=True)
    return times, vals, step


def _audio_window_features(scan_times, audio, half=3.0):
    """For every scan time compute two windowed audio features over +/- ``half``
    seconds, both normalized 0..1 across the whole segment:

      * sustained loudness - the MEAN loudness in the window. A real firefight
        stays loud for seconds; a single looting 'ding' does not.
      * transient energy   - the SUM of positive loudness JUMPS in the window.
        Gunfire is many rapid transients back to back; walking, looting and
        menus have almost none.

    Together these are the single most reliable 'this is actual combat, not
    someone wandering around' cue, and they are what the visual signals get
    gated by. Returns ``([sus_loud], [transient])`` (zeros when no audio)."""
    n = len(scan_times)
    if not audio or not audio[0]:
        return [0.0] * n, [0.0] * n
    at, av = audio
    if len(at) < 2 or not av:
        return [0.0] * n, [0.0] * n
    ahop = at[1] - at[0]
    if ahop <= 0:
        return [0.0] * n, [0.0] * n

    # Positive loudness jumps between consecutive audio samples (dB rises).
    jumps = [0.0] + [max(0.0, av[k] - av[k - 1]) for k in range(1, len(av))]
    w = max(1, int(round(half / ahop)))

    def idx(t):
        return max(0, min(len(av) - 1, int(round((t - at[0]) / ahop))))

    sus_raw, tr_raw = [], []
    for t in scan_times:
        c = idx(t)
        lo = max(0, c - w)
        hi = min(len(av), c + w + 1)
        seg = av[lo:hi]
        sus_raw.append(sum(seg) / len(seg) if seg else -120.0)
        # Only count meaningful transients (>= 2 dB) so background hiss is ignored.
        tr_raw.append(sum(j for j in jumps[lo:hi] if j >= 2.0))

    return _minmax(sus_raw, flat=0.0), _minmax(tr_raw, flat=0.0)


def _scan_scene_cuts(times, mvals):
    """Approximate scene-cut timestamps derived FROM the dense motion scan, so
    we do not pay for a second full-resolution decode just to detect scenes.

    A genuine hard cut makes two ~1s-apart scan frames almost unrelated, so its
    motion value is a strong outlier versus the local median. We flag those as
    cuts and give each a rough 0..1 pseudo scene-score. Precision is ~1 scan
    step, which is plenty for edge-snapping and the cut-density tie-breaker."""
    n = len(mvals)
    if n < 3:
        return []
    smed = sorted(mvals)[n // 2]
    pairs = []
    for i in range(1, n):
        lo = max(0, i - 4)
        hi = min(n, i + 5)
        local = sorted(mvals[lo:hi])
        lmed = local[len(local) // 2]
        if mvals[i] >= max(35.0, lmed * 2.2, smed * 1.8):
            pairs.append((times[i], min(1.0, mvals[i] / 255.0 * 3.0)))
    return pairs


def _build_activity(times, mvals, audio, step):
    """Turn the dense motion + audio series into a genre-agnostic 'this is a
    real highlight, not someone wandering around' curve (0..1 per sample).

    This is the fix for 'it keeps picking looting / landing / walking': raw
    motion is maximised by parachuting, driving and camera panning, so it can
    NEVER separate combat from traversal on its own. Instead we require BOTH:

      audio_energy = combat audio  (sustained loudness + rapid-fire transients)
      visual       = motion novelty (burst above ~6s baseline) + sustained motion

    and combine them so audio GATES the score:

      action = (0.30 + 0.70*audio_energy) * (0.35 + 0.65*visual)

    A moment with lots of motion but no combat audio (running, parachuting,
    looting, panning) is capped low; a moment with sustained gunfire AND action
    scores high. A 3-tap smoothing stops a single spike defining a peak."""
    n = len(times)
    if n == 0:
        return []
    mv = np.asarray(mvals, dtype=np.float32)

    # Motion novelty: how far above its own ~6s local baseline each sample sits.
    half = max(1, int(round(6.0 / step)))
    baseline = np.empty(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        baseline[i] = np.median(mv[lo:hi])
    nov = _minmax(np.clip(mv - baseline, 0.0, None).tolist(), flat=0.0)

    # Sustained motion over ~3s (a firefight keeps moving; a single cut spikes
    # once then settles).
    smw = max(1, int(round(3.0 / step)))
    sus_m_raw = [float(mv[max(0, i - smw):min(n, i + smw + 1)].mean()) for i in range(n)]
    sus_motion = _minmax(sus_m_raw, flat=0.0)

    # Combat audio (sustained loud + transient density) over ~3s.
    sus_loud, trans = _audio_window_features(times, audio, half=3.0)

    raw = []
    for i in range(n):
        audio_energy = 0.60 * trans[i] + 0.40 * sus_loud[i]
        visual = 0.60 * nov[i] + 0.40 * sus_motion[i]
        raw.append(float((0.30 + 0.70 * audio_energy) * (0.35 + 0.65 * visual)))

    out = []
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n, i + 2)
        out.append(float(sum(raw[lo:hi]) / (hi - lo)))
    return out


def _pick_activity_peaks(times, activity, min_gap, k):
    """Greedily pick up to ``k`` highest-activity moments at least ``min_gap``
    seconds apart - the shortlist of places worth showing the vision model."""
    order = sorted(
        range(len(activity)),
        key=lambda i: activity[i],
        reverse=True
    )
    chosen = []
    for i in order:
        t = times[i]
        if all(abs(t - c) >= min_gap for c in chosen):
            chosen.append(t)
        if len(chosen) >= k:
            break
    return sorted(chosen)


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
    # the start/end to nearby scene cuts.
    clip_min = inp.clipMin if inp.clipMin and inp.clipMin > 0 else inp.clipLen
    clip_max = inp.clipMax if inp.clipMax and inp.clipMax > 0 else inp.clipLen
    if clip_max < clip_min:
        clip_max = clip_min
    clip_target = (clip_min + clip_max) / 2.0

    # ---- Cheap whole-segment signals (computed once) -----------------------
    # Audio loudness curve (0.25s hops) - used for the audio-onset activity
    # signal and to score each final clip's loudest impact sound.
    audio_times, audio_vals = _loudness_curve(full)
    print(
        f"Audio loudness samples: {len(audio_times)}"
        + (f" (min={min(audio_vals):.1f} max={max(audio_vals):.1f} dBFS)"
           if audio_vals else " (no audio decoded)"),
        flush=True
    )

    # ---- 1) Dense activity timeline (ONE decode of the whole segment) ------
    # This is the heart of the revamp. Instead of sampling a few sparse frames
    # and ranking them by RAW motion (which is maximised by camera panning,
    # walking and looting - exactly the boring clips), we decode the entire
    # segment once into small grayscale stills and build a combat-aware
    # "is this a real highlight" curve (see _build_activity). One pass replaces
    # all the old per-candidate re-decodes AND doubles as scene detection, so we
    # never pay for a second full-resolution decode.
    scan_t, scan_v, scan_step = _scan_motion(full, dur, job_dir)
    if not scan_t:
        print("Motion scan produced no frames; nothing to do.", flush=True)
        return []

    # Scene cuts derived from the SAME scan (no extra decode) - used only to
    # snap clip edges / score cut density, NOT to pick interesting moments.
    boundary_pairs = _scan_scene_cuts(scan_t, scan_v)
    boundary_cuts = [t for (t, _s) in boundary_pairs]
    print(f"Derived {len(boundary_cuts)} scene cuts from the scan", flush=True)

    activity = _build_activity(
        scan_t, scan_v, (audio_times, audio_vals), scan_step
    )
    print(
        f"Dense scan: {len(scan_t)} samples @ {scan_step:.2f}s "
        f"(motion max={max(scan_v):.1f})",
        flush=True
    )

    # ---- 2) Candidate moments = peaks of the activity curve ----------------
    # Generous shortlist so the vision model - the real judge of quality - gets
    # to see plenty of moments. The user is fine with longer processing for
    # better picks, so we err on the side of MORE candidates.
    n_scan = min(28, max(8, int(dur / max(1.0, clip_target * 0.6))))
    peak_times = _pick_activity_peaks(
        scan_t, activity,
        min_gap=max(3.0, clip_target * 0.75),
        k=n_scan
    )
    print(f"Picked {len(peak_times)} candidate moments", flush=True)

    # Local window (seconds each side) used to place the clip bounds. We reuse
    # the dense scan series here - NO extra video decode per candidate.
    refine_window = max(8.0, min(clip_max, 18.0))

    cands = []
    for j, ct in enumerate(peak_times):
        lo, hi = ct - refine_window, ct + refine_window
        loc_t, loc_v = [], []
        for t, v in zip(scan_t, scan_v):
            if lo <= t <= hi:
                loc_t.append(t)
                loc_v.append(v)

        start, end = choose_clip_bounds(
            ct, boundary_pairs, dur, clip_min, clip_max,
            video_path=full, job_dir=job_dir,
            intensity=(loc_t, loc_v),
            audio=(audio_times, audio_vals)
        )

        seg_v = [v for t, v in zip(scan_t, scan_v) if start <= t <= end]
        seg_a = [activity[i] for i, t in enumerate(scan_t) if start <= t <= end]
        cands.append({
            "idx": j,
            "time": round(ct, 2),
            "start": start,
            "end": end,
            "motion": (sum(seg_v) / len(seg_v)) if seg_v else 0.0,
            "activity": (sum(seg_a) / len(seg_a)) if seg_a else 0.0,
            "audio": clip_audio_score(audio_times, audio_vals, start, end),
        })

    if not cands:
        return []

    # ---- 3) Cheap gate: choose how many get the (slow) vision judgement -----
    normalize_feature(cands, "activity", "activity_norm")
    normalize_feature(cands, "audio", "audio_norm")
    normalize_feature(cands, "motion", "motion_norm")
    cands.sort(
        key=lambda x: (
            x["activity_norm"] * 0.55
            + x["audio_norm"] * 0.30
            + x["motion_norm"] * 0.15
        ),
        reverse=True
    )
    vision_n = min(len(cands), max(inp.finalCandidates * 4, 12))
    vis = cands[:vision_n]

    # ---- 4) VISION MODEL judges each candidate (the real intelligence) -----
    # Each candidate gets an explicit 0-10 excitement rating. This - not raw
    # motion - decides the ranking, which is what stops boring walking/looting
    # clips being chosen: the model can SEE that nothing is happening.
    print(
        f"Running {VISION_BACKEND} ({VISION_MODEL}) on {len(vis)} moments",
        flush=True
    )
    for frame in vis:
        clip_frames = load_clip_frames(full, frame, job_dir)
        gameplay, approve, confidence, rating, reason = _vision_score(
            clip_frames,
            frame["motion_norm"],
            0.5,   # yolo hint not computed at this stage
            0.5,   # ocr hint not computed at this stage
            frame["audio_norm"],
            frame["end"] - frame["start"]
        )
        frame["gameplay"] = gameplay
        frame["approve"] = approve
        frame["confidence"] = confidence
        frame["rating"] = rating
        frame["reason"] = reason
        print(
            f"  moment {frame['time']:.1f}s "
            f"[{frame['start']:.1f}-{frame['end']:.1f}] "
            f"rating={rating:.2f} approve={approve} gameplay={gameplay} "
            f":: {reason}",
            flush=True
        )

    # Keep only moments the model both approved AND rated as a real highlight.
    # A hard quality floor (>= 4/10) is what stops the pipeline padding the
    # result with filler - a dull segment now returns FEWER, better clips
    # instead of map/menu/looting shots just to reach finalCandidates.
    RATING_FLOOR = 0.40   # 4 out of 10
    approved = [
        f for f in vis
        if f.get("gameplay") and f.get("approve")
        and f.get("rating", 0.0) >= RATING_FLOOR
    ]
    if not approved:
        # Whole segment was genuinely dull: keep ONLY the single best-rated
        # moment so the flow still gets something, rather than filler.
        approved = sorted(
            vis, key=lambda x: x.get("rating", 0.0), reverse=True
        )[:1]
        print(
            "No moment cleared the quality floor; keeping best-rated only.",
            flush=True
        )
    approved.sort(
        key=lambda x: (x.get("rating", 0.0), x.get("activity_norm", 0.0)),
        reverse=True
    )

    # ---- 5) Pick non-overlapping finals ------------------------------------
    # Variable-length clips, so reject by ACTUAL [start, end] overlap. minGap
    # (when > 0) adds extra breathing room between kept clips.
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

    # ---- 6) Fill the API-contract fields (OCR / quality / YOLO) on finals ---
    # The vision rating already decided the ranking, so the slow signals run on
    # only the handful of clips we actually return. They populate the response
    # fields the n8n flow reads and act as light tie-breakers.
    for frame in selected:
        clip_frames = load_clip_frames(full, frame, job_dir)
        ocr_points, ocr_text, ocr_hits = ocr_score_frames(clip_frames)
        frame["ocr"] = ocr_points
        frame["ocr_text"] = ocr_text
        frame["ocr_hits"] = ocr_hits
        frame["quality"] = clip_quality_score(clip_frames)
        frame["cuts"] = cut_density_score(
            boundary_cuts, frame["start"], frame["end"]
        )
        mid = clip_frames[len(clip_frames) // 2] if clip_frames else None
        frame["yolo"], frame["yolo_hits"] = (
            yolo_score(mid) if mid else (0.0, [])
        )

    if selected:
        normalize_feature(selected, "motion", "motion_norm")
        normalize_feature(selected, "yolo", "yolo_norm")
        normalize_feature(selected, "ocr", "ocr_norm")
        normalize_feature(selected, "audio", "audio_norm")
        normalize_feature(selected, "quality", "quality_norm")
        normalize_feature(selected, "cuts", "cuts_norm")
        normalize_feature(selected, "activity", "activity_norm")

    for frame in selected:
        # Vision rating leads (it is the only stage that understands the scene),
        # but the measured combat/activity signal now co-decides the order so a
        # clip with real sustained gunfire ranks above a quieter one the model
        # rated similarly. audio/motion stay as light tie-breakers.
        frame["final_score"] = round(
            frame.get("rating", 0.0) * 0.72
            + frame["activity_norm"] * 0.16
            + frame["audio_norm"] * 0.08
            + frame["motion_norm"] * 0.04,
            4
        )
        frame["vision_score"] = frame.get("rating", 0.0)
        clip_frames = frame.get("clip_frames") or []
        thumb = clip_frames[len(clip_frames) // 2] if clip_frames else None
        frame["frame"] = thumb
        frame["frame_rel"] = (
            os.path.relpath(thumb, MEDIA) if thumb else None
        )

    selected.sort(key=lambda x: x["final_score"], reverse=True)

    print_final_results(selected)

    return selected


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays (and nested containers) into
    plain JSON-serializable Python types. The dense scan uses numpy, so some
    candidate fields can end up as ``numpy.float32``, which FastAPI's encoder
    cannot serialize - this guards the whole response against that."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


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
        "dur": float(dur),
        "candidates": [_json_safe(f) for f in final]
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


def edit_clip(source, output, aspect, crop_mode, fade, slowmo=False, headroom=0.42,
              card_aspect="1:1", radius=48, fade_out=None, border=5, side_margin=48):
    """Reframe an already-rendered full-res clip to a short-form aspect.
    Unlike render_clip this takes a finished file (no -ss/-t) and just applies
    the crop/scale/fade filter. ``fade`` is the intro fade-IN and ``fade_out``
    the outro fade-OUT (defaults to ``fade`` when omitted). crop_mode "smart"
    pans the vertical window to keep the player in shot; "center" keeps the
    middle; "template" puts the full uncropped frame in a middle band with room
    above/below for branding; "card" is a rounded-corner 1:1 cutout on black
    (with a thin white border); "card_blur" is the same cutout over a blurred
    zoom of the clip. slowmo slows ~1s around the peak for a dramatic kill
    moment."""

    dur = probe(ProbeIn(path=os.path.relpath(source, MEDIA)))["duration"]

    # Smart crop: track the player over time and pan to follow (biased to
    # centre so the crosshair/action stays in frame).
    cx_expr = None
    if crop_mode == "smart":
        pts = track_subject_x(source, dur, os.path.dirname(output))
        cx_expr = pan_x_expr(pts)
        print(f"SMART TRACK {len(pts)} pts for {os.path.basename(source)}", flush=True)

    flag, filter_string, extra_maps = build_video_filter(
        aspect, crop_mode, fade, dur, cx_expr=cx_expr, headroom=headroom,
        card_aspect=card_aspect, radius=radius, fade_out=fade_out, border=border,
        side_margin=side_margin
    )
    if slowmo and flag == "-vf":
        # Dramatic slow: split the clip into pre / peak / post, slow the ~1.5s
        # peak window to 0.5x, concat. Audio dropped (slow-mo desyncs it).
        peak = find_motion_peak(source, dur, os.path.dirname(output))
        a = max(0.0, peak - 0.75); b = min(dur, peak + 0.75)
        filter_string = (
            f"{filter_string},split=3[p0][p1][p2];"
            f"[p0]trim=0:{a:.2f},setpts=PTS-STARTPTS[s0];"
            f"[p1]trim={a:.2f}:{b:.2f},setpts=2.0*(PTS-STARTPTS)[s1];"
            f"[p2]trim={b:.2f},setpts=PTS-STARTPTS[s2];"
            f"[s0][s1][s2]concat=n=3:v=1:a=0[v]"
        )
        flag = "-filter_complex"
        extra_maps = ["-map", "[v]"]
    cmd = ["ffmpeg", "-y", "-i", source, flag, filter_string] + extra_maps + [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", output,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("EDIT:", output, "rc=", r.returncode, flush=True)


class EditIn(BaseModel):
    jobId: str
    aspect: str = "9:16"
    fade: float = 0.5         # intro fade-IN seconds (quick)
    fadeOut: float = 1.0      # outro fade-OUT seconds (slower, softer ending)
    headroom: float = 0.42    # template: fraction of empty space above the video (rest below)
    cardAspect: str = "1:1"   # card: crop aspect of the rounded-corner gameplay window (taller = more zoom)
    radius: int = 48          # card: corner radius in px
    borderWidth: int = 0      # card (black) only: white border ring thickness in px (0 = none)
    sideMargin: int = 48      # card: horizontal inset each side so background shows around the card (px)


@app.post("/edit")
def edit(inp: EditIn):
    """Reframe EVERY rendered clip into two variants of the SAME rounded 1:1
    cutout (pushed down 60/40): a CARD on a solid black background with a thin
    white border, and a BLURRED version of that same cutout over a blurred zoom
    of the clip. Mirrors the renders layout (flat or segment_*) under
    work/<jobId>/edited/card and /blurred."""

    renders = os.path.join(MEDIA, "work", inp.jobId, "renders")
    if not os.path.isdir(renders):
        return {"ok": False, "reason": f"no renders for {inp.jobId}"}

    # All rendered clips, flat and per-segment, with their path relative to
    # renders/ so each variant folder keeps the same structure.
    sources = sorted(glob.glob(os.path.join(renders, "**", "clip_*.mp4"), recursive=True))
    if not sources:
        return {"ok": False, "reason": "no clips rendered"}

    out_dir = os.path.join(MEDIA, "work", inp.jobId, "edited")
    card, blurred = [], []

    for src in sources:
        rel = os.path.relpath(src, renders)
        for mode, dest in (("card", "card"), ("card_blur", "blurred")):
            out_file = os.path.join(out_dir, dest, rel)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            edit_clip(src, out_file, inp.aspect, mode, inp.fade,
                      headroom=inp.headroom, card_aspect=inp.cardAspect,
                      radius=inp.radius, fade_out=inp.fadeOut,
                      border=inp.borderWidth, side_margin=inp.sideMargin)
            relout = f"work/{inp.jobId}/edited/{dest}/{rel.replace(os.sep, '/')}"
            (card if mode == "card" else blurred).append(relout)

    return {"ok": True, "card": card, "blurred": blurred}
