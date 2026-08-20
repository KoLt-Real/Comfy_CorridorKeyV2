# Comfy CorridorKey

ComfyUI nodes for [CorridorKey](https://github.com/nikopueringer/CorridorKey) (Corridor
Digital) — a neural green/blue screen keyer that produces a physically correct alpha **and** a
true foreground colour from an RGB frame plus a coarse alpha hint.

This port bundles the community VRAM work that other ComfyUI ports don't have: fp16 weights,
fused-SDPA Hiera attention (timm ≥ 1.0.27), optional `torch.compile`, and EZ-CorridorKey's
tiled refiner — **2.98 GB** peak instead of ~23 GB for the reference port, on the same task.

![Source on a green screen, and the same frame keyed and comped — hair and soft edges
kept](docs/sidebyside.webp)

*Left: source. Right: keyed and comped. Full resolution:
[sidebyside.mp4](docs/sidebyside.mp4).*

---

## Nodes

All under the **`CorridorKeyerV2`** category:

| Node | In → Out | Role |
|---|---|---|
| **Load Model** | → `CORRIDORKEY_MODEL` | screen colour, fp16/fp32, refiner full/tiled/off, optional compile |
| **Keyer** | `IMAGE` (+ optional `MASK`) → `alpha`, `fg_srgb` | the keyer itself — **the hint is optional**, one is generated with BiRefNet if you wire nothing |
| **Alpha Hint (BiRefNet)** | `IMAGE` → `MASK` | explicit hint when you want control (variant, precision, binarize, dilate/erode) |
| **Despeckle Matte** | `MASK` → `MASK` | removes small islands (tracking markers, noise) |
| **Despill** | `IMAGE` (+ optional model) → `IMAGE` | luminance-preserving spill removal; takes the screen colour from the model if connected |
| **Premultiply & Comp** | `IMAGE` + `MASK` (+ optional BG) → `premult_linear`, `comp_srgb` | linear premultiply + preview comp over a checkerboard or your own background |
| **Colorspace Convert** | `IMAGE` → `IMAGE` | exact piecewise sRGB ↔ linear |

Typical graph:

```
LoadImage/LoadVideo ─┬─────────────► Keyer ──┬──► Despeckle ─┐
                     └─► Alpha Hint ─┘  ▲    │               ├─► Premultiply & Comp ─► SaveImage
                                   Load Model└──► Despill ───┘
```

A ready-made graph is in
[`example_workflows/corridorkey_example.json`](example_workflows/corridorkey_example.json) —
drag it onto the ComfyUI canvas. It keys the sample clip
[`DemoCorridorKeyerV2.mp4`](example_workflows/DemoCorridorKeyerV2.mp4): copy that file into
`ComfyUI/input/` first, or point the *Load Video* node at your own footage. It needs
[VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for the video
load/save nodes; every keying node comes from this pack.

![The example graph in ComfyUI: green-screen source on the left, the alpha hint and the keyed
matte in the middle, the comp on the right](docs/workflow.png)

---

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/KoLt-Real/ComfyUI-CorridorKeyV2.git
pip install -r ComfyUI-CorridorKeyV2/requirements.txt
```

Use the **same Python environment as ComfyUI** for the `pip install` (activate its venv, or use
`ComfyUI/python_embeded/python.exe -m pip` on the portable Windows build). Then restart ComfyUI.

Only `timm` is likely to be missing from a stock install — everything else (torch, transformers,
huggingface-hub, opencv, numpy, Pillow) already ships with ComfyUI. The floor is deliberately
low (`timm>=1.0.3`) so this node never force-upgrades a package another node depends on;
**`timm>=1.0.27` is recommended** for the low-VRAM fused-attention path, and a one-line warning
at load time tells you if you're below it.

### Weights

Nothing to download by hand. On first use:

- the keyer checkpoint (~300 MB per screen colour) lands in `models/corridorkey/`;
- the BiRefNet snapshot for the hint lands in `models/birefnet/`.

Both folders are registered with ComfyUI at import. Set `HF_TOKEN` if you want higher
HuggingFace rate limits — the repos are public, so it is optional.

---

## VRAM

Measured on an RTX 4090 Laptop (16 GB), torch 2.12+cu130, 1024² input, 2048² processing window:

| Loader configuration | Peak VRAM | Notes |
|---|---|---|
| fp16 + refiner `full` *(default)* | **2.98 GB** | 1.17 s/frame steady state |
| fp16 + refiner `tiled` (512/128) | **1.46 GB** | output ≡ full (PSNR 46.8 dB, deltas confined to the soft edge) |
| fp16 + full + `torch_compile` | **1.86 GB** | first compile ~8 min (max-autotune), inductor cache persists |
| fp32 + refiner `full` (autocast) | 4.90 GB | quality reference |
| + BiRefNet General fp16 resident | +~1.6 GB | hint and keyer together peak at 3.41 GB |

---

## Notes

- Native processing resolution is 2048². Any input size is accepted (resized in and back out);
  extreme aspect ratios are squeezed into the square window, exactly like upstream — no
  letterboxing.
- Upstream's "Processed RGBA" pass = the `premult_linear` output **plus** the `alpha` wire
  (ComfyUI IMAGEs are 3-channel, so alpha travels on its own MASK wire).
- Video batches are processed frame by frame, with a progress bar and clean interruption. A
  MASK with a batch of 1 is broadcast over the whole image batch.
- `refiner_scale` only scales the alpha delta (EZ-CorridorKey behaviour) — the foreground keeps
  the full anti-macroblocking correction.
- **Despeckle**: `min_island_size` values below 20 are treated as 20. The underlying component
  labelling derives its iteration count from that threshold and does nothing at all below 20,
  which would wipe the matte entirely.

---

## Tests

```bash
python tests/test_pipeline_guards.py
```

Guards the two settings that silently destroy output when wrong: the despeckle threshold (which
used to wipe the matte to black) and BiRefNet snapshot completeness (a half-finished download
that would otherwise look complete forever). Deliberately torch-free and network-free, so it
runs anywhere. See [AGENTS.md](AGENTS.md) for the invariants behind them.

---

## Licence

CC BY-NC-SA 4.0 plus Corridor Digital's additional terms — see [LICENSE](LICENSE) and
[NOTICE](NOTICE). **Non-commercial use only**, attribution to "CorridorKey" required,
ShareAlike. The vendored files and model weights carry the same licence.
