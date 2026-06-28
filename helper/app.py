import os, json, subprocess
import shutil, glob, time
import re, base64, requests
import math

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

def _loudness_curve(full):
    """Return (times[], momentary_LUFS[]) using ffmpeg's ebur128 meter."""
    cmd = ["ffmpeg", "-nostats", "-i", full, "-af", "ebur128=metadata=1", "-f", "null", "-"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    times, vals = [], []
    for line in p.stderr.splitlines():
        m = re.search(r"t:\s*([0-9.]+).*?M:\s*(-?[0-9.]+)", line)
        if m:
            times.append(float(m.group(1)))
            vals.append(float(m.group(2)))
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
    ocr
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

    return summary

def build_prompt(
    motion,
    yolo,
    ocr
):

    cv_summary = build_cv_summary(
        motion,
        yolo,
        ocr
    )

    return f"""
The frames are ordered chronologically.

Frame 1 is the beginning of the clip.
Frame 5 is the end of the clip.

Judge the progression of the action across all frames instead of treating each image independently.

These are 5 frames taken from the SAME 15-second gameplay clip.

These frames have ALREADY been selected by an automated highlight detection pipeline.

Computer Vision Analysis

Motion Score : {motion:.2f}
YOLO Score   : {yolo:.2f}
OCR Score    : {ocr:.2f}

Summary:
{chr(10).join(cv_summary)}

Interpretation:

• Motion measures how dynamic the scene is.
• YOLO estimates how many important gameplay objects are visible.
• OCR detects reward text such as HEADSHOT, KILL, VICTORY, ACE, etc.

Your task is NOT to score this clip from scratch.

Instead:

1. Decide whether these frames show genuine gameplay.
2. Decide whether this is actually highlight-worthy.
3. Use the Computer Vision scores as evidence.
4. If the Computer Vision scores appear misleading, reduce your confidence.
5. If they agree with what you see, increase your confidence.

Return ONLY valid JSON.

{{
    "gameplay": true,
    "approve": true,
    "confidence": 0.0,
    "reason": "short reason under 12 words"
}}

Confidence guide:

1.00 = Exceptional highlight
0.90 = Excellent gameplay
0.80 = Strong highlight
0.70 = Good gameplay
0.60 = Average gameplay
0.40 = Weak highlight
0.20 = Probably not a highlight
0.00 = Not gameplay or definitely reject

Never return markdown.
Never explain your answer.
Return JSON only.
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
        + frame["yolo_norm"] * 0.30
        + frame["ocr_norm"] * 0.30
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
    ocr
):
    try:
        images = []

        for frame in frame_paths:
            with open(frame, "rb") as f:
                images.append(
                    base64.b64encode(f.read()).decode()
                )

        prompt = build_prompt(motion, yolo, ocr)

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
            "format": "json"
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
    ocr
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


        prompt = build_prompt(motion, yolo, ocr)

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
    ocr
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
            ocr
        )

    elif VISION_BACKEND == "lmstudio":
        return _vision_score_lmstudio(
            frame_paths,
            motion,
            yolo,
            ocr
        )

    raise ValueError(
        f"Unknown vision backend: {VISION_BACKEND}"
    )

class CandIn(BaseModel):
    path: str
    jobId: str
    clipLen: float = 15.0
    count: int = 4
    minGap: float = 8.0

class RenderClip(BaseModel):
    start: float
    end: float


class RenderIn(BaseModel):
    path: str
    jobId: str
    clips: list[RenderClip]

def detect_scenes(full):

    cmd = [
        "ffmpeg",
        "-i", full,
        "-filter:v",
        "select='gt(scene,0.30)',showinfo",
        "-vsync", "0",
        "-f", "null",
        "-"
    ]

    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    times = []

    for line in p.stderr.splitlines():

        m = re.search(r"pts_time:([0-9.]+)", line)

        if m:
            times.append(float(m.group(1)))

    return times

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
    
def sample_video(duration, interval=2.0):
    times = []

    t = interval

    while t < duration:
        times.append(round(t, 2))
        t += interval

    return times

def render_clip(source, start, end, output):

    duration = end - start

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", source,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-c:a", "aac",
            output
        ],
        capture_output=True,
        text=True
    )

    print("=" * 80, flush=True)
    print("FFMPEG RETURN:", result.returncode, flush=True)
    print("OUTPUT FILE:", output, flush=True)
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

    best_time = approx_time
    best_motion = -1

    prev_frame = None

    t = max(0.0, approx_time - window)

    while t <= min(duration, approx_time + window):

        img = os.path.join(
            work,
            f"{int(t*1000)}.jpg"
        )

        _extract_frame(video_path, t, img)

        if prev_frame is not None:

            m = motion_score(
                prev_frame,
                img
            )

            if m > best_motion:
                best_motion = m
                best_time = t

        prev_frame = img

        t += step

    print(
        f"Refined {approx_time:.2f}s -> {best_time:.2f}s "
        f"(motion={best_motion:.2f})",
        flush=True
    )

    return best_time, best_motion


def print_final_results(selected):

    print("\n===== FINAL RANKING =====", flush=True)

    for i, frame in enumerate(selected, 1):

        print(
            f"{i}. "
            f"Time={frame['time']:.1f}s | "
            f"Motion={frame['motion_norm']:.2f} | "
            f"YOLO={frame['yolo_norm']:.2f} | "
            f"OCR={frame['ocr_norm']:.2f} | "
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

    clip_dir = os.path.join(
        job_dir,
        f"clip_{frame['idx']}"
    )

    return _extract_clip_frames(
        video_path,
        frame["start"],
        frame["end"],
        clip_dir
    )

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

    sample_times = detect_scenes(full)

    print(
        f"Detected {len(sample_times)} scene changes",
        flush=True
    )

    # Fallback if scene detection found too few scenes
    if len(sample_times) < 20:
        print(
            "Too few scene changes. Falling back to 2-second sampling.",
            flush=True
        )

        sample_times = sample_video(
            dur,
            interval=2.0
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
        start = max(0.0, t - inp.clipLen / 2)
        end = min(dur, start + inp.clipLen)
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

    max_possible = int(dur / inp.clipLen)

    candidate_count = max(
        2,
        min(12, max_possible // 2)
    )

    print(
        f"Selecting top {candidate_count} candidates",
        flush=True
    )

    interesting = motion_frames[:candidate_count]

    for frame in interesting:

        refined_time, refined_motion = refine_candidate(
            full,
            frame["time"],
            dur,
            job_dir
        )

        frame["time"] = refined_time
        frame["motion"] = refined_motion

        frame["start"] = max(
            0,
            refined_time - inp.clipLen / 2
        )
        frame["end"] = min(
            dur,
            frame["start"] + inp.clipLen
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

        ocr_points, ocr_text, ocr_hits = ocr_score(clip_frames[2])

        frame["ocr"] = ocr_points
        frame["ocr_text"] = ocr_text
        frame["ocr_hits"] = ocr_hits

        print(
            f"OCR={frame['ocr']:.1f}",
            f"Hits={ocr_hits}",
            f"Text={frame['ocr_text'][:80]}",
            flush=True
        )
        frame["reason"] = ""

        scored.append(frame)

    normalize_feature(scored, "motion", "motion_norm") 

    normalize_feature(scored, "yolo", "yolo_norm")

    normalize_feature(scored, "ocr", "ocr_norm")

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
            frame["ocr_norm"]
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
            continue

        # Add LLaVA's opinion to the existing fast score
        frame["final_score"] *= confidence

    interesting.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    selected = []
    MIN_DISTANCE = 30

    for frame in interesting:
        keep = True
        for chosen in selected:
            if abs(frame["time"] - chosen["time"]) < MIN_DISTANCE:
                keep = False
                break
        if keep:
            selected.append(frame)
        if len(selected) == inp.count:
            break

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

    final = []

    for frame in all_selected:

        if all(
            abs(frame["time"] - f["time"]) >= 30
            for f in final
        ):
                final.append(frame)

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

        out_file = os.path.join(
            out_dir,
            f"clip_{i+1}.mp4"
        )

        render_clip(
            source,
            clip.start,
            clip.end,
            out_file
        )

        rendered.append(
            f"work/{inp.jobId}/renders/clip_{i+1}.mp4"
        )

    return {
        "clips": rendered
    }