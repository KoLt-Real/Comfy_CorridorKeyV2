# AGENTS.md — Comfy CorridorKey

Working notes for AI coding agents. Read this before touching the code.

## What this is

ComfyUI nodes wrapping the CorridorKey keyer. The repository is a **derivative work under
CC BY-NC-SA 4.0 with Corridor Digital's additional terms** — non-commercial, attribution
required. See `NOTICE` before adding, moving or relicensing anything.

## Layout

```
__init__.py          node mappings + registers models/corridorkey and models/birefnet
nodes/               one node per file, thin wrappers — no math here
pipeline/
  prepare.py         checkpoint/snapshot download + completeness checks (CKPTS registry)
  loader.py          GreenFormer load, device/precision, refiner mode, model cache
  keyer.py           the inference pass (pre/post-processing re-expressed for ComfyUI)
  birefnet.py        alpha-hint model (VARIANTS registry), its own cache
  post.py            ComfyUI-shaped wrappers over the vendored math, chunked on GPU
  vendor/            VENDORED upstream code — see below
example_workflows/   ready-to-drag graph + the sample clip it keys
docs/                README media (before/after loop, graph screenshot)
tests/               torch-free guard tests
```

## Invariants — do not break these

1. **`pipeline/vendor/` is vendored at a pinned upstream commit** (see `NOTICE`). Do not
   refactor, reformat or "improve" it. If behaviour must change, wrap it in `pipeline/`.
2. **Nodes import their heavy dependencies lazily**, inside the method that needs them — never
   at module level. ComfyUI imports every node file at startup; a top-level `import torch`
   would slow every launch and break registration if a dependency is missing. This discipline
   is also what makes `tests/` runnable without torch.
3. **Registries are the single source of truth.** `prepare.CKPTS` owns screen colours (repo,
   filename, despill channel) and `birefnet.VARIANTS` owns the hint models. The node dropdowns
   derive from them (`list(_CKPTS)`, `list(_VARIANTS)`) so a dropdown can never offer something
   the loader rejects. Add a model in the registry, nowhere else.
4. **A snapshot is only complete when the weights are there.** `_birefnet_complete()` checks for
   an actual weight file, not just `config.json` — `snapshot_download` fetches in parallel and
   the small config lands long before the ~1 GB of weights, so an interrupted download would
   otherwise look complete forever.
5. **`min_island_size` is floored to 20** (`post._MIN_AREA_THRESHOLD`). The vendored
   `clean_matte_torch` derives its flood-fill iteration count from `area_threshold // 20`; below
   20 it runs zero merge passes, every pixel keeps a unique label, and the **entire matte is
   wiped black**. Never pass a raw user value through.
6. **fp16 is downgraded to fp32 off CUDA/MPS.** Both loaders do this; keep them in step.
7. **Batches are processed in chunks with progress + interruption.** `comfy.utils.ProgressBar`
   and `throw_exception_if_processing_interrupted()` — a long video must stay cancellable.

## Conventions

- Comments: English in this repo (it mirrors upstream's language). Keep it consistent.
- Tensors follow ComfyUI conventions: `IMAGE` = `[B,H,W,3]` float32 CPU, `MASK` = `[B,H,W]`.
  Conversion to `[B,C,H,W]` and to the GPU happens inside `pipeline/`, never in `nodes/`.
- Everything downstream of the model forward is fp32; fp16 stops at the forward.

## Tests

```bash
python tests/test_pipeline_guards.py
```

Covers the two guards that silently destroy output when wrong (despeckle threshold, snapshot
completeness). Deliberately torch-free and network-free, so it runs anywhere — which is only
possible because of invariant 2.

Full-pipeline verification needs a GPU and is not part of this suite; use the example workflow.

## Traps found the hard way

- The keyer's `mask` input is **optional**: with nothing wired it generates a BiRefNet hint
  itself. Any change to hint handling must keep that path working.
- `refiner_scale` scales only the alpha delta; the foreground keeps the full correction.
- The tiled refiner must stay output-identical to `full` (measured PSNR 46.8 dB, deltas confined
  to the soft edge) — it is a VRAM option, not a quality option.
- A `MASK` with batch 1 is broadcast over the whole image batch; other mismatches must raise
  early, before the model runs.
