# ComfyUI Nodes

This file holds the ComfyUI graph instructions for the parts of the pipeline
that need **heavy AI**. It mirrors `n8n-node-changes.md`: whenever a stage is
better done in ComfyUI than in ffmpeg, the node setup goes here.

> Current status: **ComfyUI is NOT needed yet.** Phase 1 (vertical Shorts,
> ranking, fades) and most of Phase 2 (logo/captions/audio) are all done with
> ffmpeg inside the helper container. ComfyUI comes in for thumbnails, AI VFX,
> and smart subject tracking. This doc is the staging area for those.

---

## When we will use ComfyUI

| Task                          | Phase | Why ComfyUI (vs ffmpeg) |
|-------------------------------|-------|--------------------------|
| AI Thumbnail generation       | 4     | upscale + relight + text composition |
| Smart subject / player tracking | 3   | detection/segmentation models |
| AI zoom / camera move decisions | 3   | model picks the focal point |
| Background removal / relight  | 3-5   | matting models |
| Style transfer / AI effects   | 5     | diffusion-based VFX |

Everything else (crop, fade, overlay, captions, audio, encode) stays in
ffmpeg - it is faster and deterministic.

---

## How n8n will call ComfyUI

ComfyUI exposes an HTTP API (`POST /prompt` with a workflow JSON, then poll
`GET /history/{id}`). The plan when we reach Phase 3-4:

1. Export the ComfyUI graph as **API format** (Settings -> "Save (API format)").
2. In n8n add an **HTTP Request** node -> `POST http://host.docker.internal:8188/prompt`
   with the exported JSON, injecting the input image/clip path.
3. Add a second HTTP Request (or Wait + poll) on `/history/{prompt_id}` to get
   the output file path.
4. Feed that output back into the helper `/render` (or a new `/compose` step).

> When we build the first graph, the exact node list, model files, and the
> input/output wiring will be filled in below.

---

## Graph 1: AI Thumbnail  (Phase 4 - NOT BUILT YET)

Planned nodes (to be confirmed when we build it):
- [ ] Load Image (best frame from `/candidates`)
- [ ] Upscale (e.g. 4x-UltraSharp) to 1280x720
- [ ] Text / title overlay
- [ ] Background blur / subject pop
- [ ] Save Image -> `media/output/thumbnails/<jobId>.png`

Models needed: _TBD_.

---

## Graph 2: Smart Subject Tracking  (Phase 3 - NOT BUILT YET)

Planned approach: per-frame subject mask -> crop window follows the subject ->
hand the crop keyframes back to ffmpeg in the helper.

Models needed: _TBD_ (segmentation / detection).

---

When you want to start on one of these, say which graph and I'll give you the
exact nodes to add, what to connect to what, and the model downloads.
