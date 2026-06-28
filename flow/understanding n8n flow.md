MANUAL START
    │
    ▼
CLAIM CLIP
POST /claim
──────────────────────────────────────────
• http://helper:8000/claim
──────────────────────────────────────────
    │
    ▼
GOT A CLIP?
──────────────────────────────────────────
If no video is available
    → Stop workflow

If video exists
    → Continue
──────────────────────────────────────────
    │
    ▼
PROBE
──────────────────────────────────────────
POST /probe http://helper:8000/probe
──────────────────────────────────────────
    │
    ▼
JOB INFO
──────────────────────────────────────────
Stores values needed later.

Example:

"jobId": 
"20260628-090658",
  
"name": 
"test40",
  
"path": 
"work/20260628-090658/source.mp4",
  
"duration": 
2482.329252,
  
"width": 
1280,
  
"height": 
720

──────────────────────────────────────────
    │
    ▼
FIND BEST MOMENTS
──────────────────────────────────────────
POST /candidates http://helper:8000/candidates

{
  "path": "{{ $json.path }}",
  "jobId": "{{ $json.jobId }}",
  "clipLen": 15,
  "sampleInterval": 2,
  "topMotion": 20,
  "finalCandidates": 4
}
──────────────────────────────────────────
    │
    ▼
PICK BEST
──────────────────────────────────────────
const res = $input.first().json;

const cands = res.candidates || [];

if (!cands.length) {
    throw new Error("No candidates found");
}

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
        visionScore: best.vision_score,
        finalScore: best.final_score,

        frame: best.frame_rel,

        reason: best.reason,

        allCandidates: cands
    }
}];
──────────────────────────────────────────
    │
    ▼
RENDER CLIPS
──────────────────────────────────────────
POST /render http://helper:8000/render

{
    "path": "{{ $json.path }}",
    "jobId": "{{ $json.jobId }}",
    "clips": {{ JSON.stringify($json.allCandidates.map(c => ({
        start: c.start,
        end: c.end
    }))) }}
}
──────────────────────────────────────────