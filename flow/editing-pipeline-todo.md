# Editing Pipeline - Roadmap & TODO

The end goal: turn a long gameplay video into **postable short-form content**
(YouTube Shorts / Instagram Reels / TikTok), fully automatic.

Pipeline stages:

```
Video -> Highlight Detection -> Editing -> Branding -> Captions -> Upload
```

We work in **phases**. Each phase produces something you can actually post,
then we layer polish on top. Boxes are checked as they land in the helper API
(`helper/app.py`) or n8n / ComfyUI.

Legend: `[x]` done & in code  `[~]` partial / configurable  `[ ]` not started

Where each thing runs:
- **helper (ffmpeg/Python)** - cuts, crops, fades, overlays, audio, encode.
- **n8n** - orchestration: which clips, in what order, calls + uploads.
- **ComfyUI** - heavy AI only (thumbnails, AI VFX, smart subject tracking).
  Not needed for Phase 1-2; see `comfyui-nodes.md`.

---

## PHASE 1 - Postable Shorts (CURRENT)

Get 3 ranked, vertical, watchable clips out the door.

- [x] Highlight detection (motion + audio peaks + scene cuts + OCR + vision)
- [x] **Best-to-worst ranking** - `/candidates` returns `rank` (1 = best)
- [x] **Edit top 3** - n8n sends top 3 to `/render` (or fewer if <3 found)
- [x] **Auto 9:16 Shorts** - `aspect:"9:16"`, center-crop / blur / fit modes
- [x] **Fade In / Fade Out** - `fade` seconds on every rendered clip
- [x] **Quality preserved** - crf 18, source resolution kept, +faststart
- [x] Auto 1:1 Square - `aspect:"1:1"`
- [x] Auto 4:5 Instagram - `aspect:"4:5"`
- [ ] Loudness Normalization (ffmpeg `loudnorm` -> consistent volume)
- [ ] Pick the single best 15-30s window length per clip (hook-aware)

**Try now:** rebuild helper, run the flow, set the Render body to
`aspect:"9:16"`. See `n8n-node-changes.md` section 3.

---

## PHASE 2 - Branding & Captions (NEXT)

Make clips look like *yours* and readable on mute.

### Branding (helper: overlay/drawtext)
- [ ] Logo Overlay (PNG corner overlay via ffmpeg `overlay`)
- [ ] Watermark (semi-transparent, anti-reupload)
- [ ] Social Handles / Channel Name (drawtext lower corner)
- [ ] Intro Screen / Outro / End Card (concat a branded card)
- [ ] Brand Colors / Custom Fonts (drawtext font + color config)

### Captions (Whisper -> burned-in subtitles)
- [ ] Auto transcription (faster-whisper in the helper container)
- [ ] Burn-in captions (ffmpeg `subtitles`/`ass`, word or line timing)
- [ ] AI Caption Highlighting (keyword pop / color the punchy word)

### Audio polish (helper: ffmpeg filters)
- [ ] Audio Ducking (lower game audio under voice/music)
- [ ] Compressor / Limiter (consistent punch)
- [ ] Background Music + auto-sync to the action beat

---

## PHASE 3 - Smart Framing & Effects

Make the crop and motion feel hand-edited.

### Cropping & Layout
- [x] Smart Subject Tracking (follow the player / crosshair) - YOLO pan over time
- [ ] Auto Zoom on Action (punch in on the kill / peak motion) - needs zoompan, deferred (crop size can't vary)
- [~] HUD-safe Cropping - `hud_safe` clamps pan to centre; no HUD detection yet
- [x] Dynamic Crop (move the crop window over time) - smart cropMode

### Visual Effects (helper ffmpeg, simple ones first)
- [x] Speed Ramp / Slow Motion on the peak moment - /edit slowmo (cropped variant)
- [ ] Freeze Frame / Replay Effect
- [ ] Cinematic Bars / Vignette / Color Grading (LUT)
- [ ] Screen Shake / Flash / Hit Marker on impact (driven by audio peak time)

> Smart tracking + AI zoom are the first jobs that may go to **ComfyUI**
> (subject detection). Everything else stays ffmpeg.

---

## PHASE 4 - Thumbnails, Metadata, Upload

- [ ] Best Frame Selection (reuse the highest-scoring frame we already have)
- [ ] AI Thumbnail (ComfyUI: upscale + text + background blur)
- [ ] AI Title / Description / Hashtag generation (vision/LLM we already run)
- [ ] AI Viral Score (rank clips for which to post first)
- [ ] Upload to YouTube Shorts / Instagram Reels / TikTok (n8n + platform APIs)
- [ ] Schedule Upload / Auto Publish

---

## PHASE 5 - Templates, Analytics, Future

- [ ] Per-game templates (FPS / RPG / Reaction presets bundling the above)
- [ ] Analytics: processing time, vision confidence, clip ranking dashboard
- [ ] Upload history + success rate
- [ ] (Future) ComfyUI advanced VFX, multi-GPU, cloud rendering

---

## Done so far (API surface)

`POST /candidates` -> `{ dur, candidates: [ { rank, start, end, time,
final_score, confidence, ... } ] }` sorted best -> worst.

`POST /render` body:
```json
{
  "path": "...", "jobId": "...",
  "aspect": "9:16",        // 9:16 | 4:5 | 1:1 | 16:9 | source
  "cropMode": "center",    // center | blur | fit
  "fade": 0.5,
  "clips": [ { "start": 0, "end": 20, "rank": 1 } ]
}
```
-> `{ "clips": ["work/<jobId>/renders/clip_1.mp4", ...] }` (clip_1 = best).
