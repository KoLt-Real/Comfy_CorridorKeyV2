"""Locate / auto-download the CorridorKey and BiRefNet weights.

CorridorKey ships two ~300 MB fp32 safetensors on HuggingFace (public, not
gated — a token is therefore optional, but $HF_TOKEN is still picked up to
raise rate limits):

    green screen : nikopueringer/CorridorKey_v1.0   / CorridorKey_v1.0.safetensors
    blue screen  : nikopueringer/CorridorKeyBlue_1.0 / CorridorKeyBlue_1.0.safetensors

Checkpoints land in ``ComfyUI/models/corridorkey/`` and BiRefNet snapshots in
``ComfyUI/models/birefnet/`` (both registered in the package ``__init__``).
fp16 is a cast at load time — no separate disk artifact is needed at this size.
"""

from __future__ import annotations

import os

# screen_color -> (HF repo, filename, despill channel index: 0=R 1=G 2=B)
CKPTS: dict[str, tuple[str, str, int]] = {
    "green": ("nikopueringer/CorridorKey_v1.0", "CorridorKey_v1.0.safetensors", 1),
    "blue": ("nikopueringer/CorridorKeyBlue_1.0", "CorridorKeyBlue_1.0.safetensors", 2),
}

# A complete checkpoint is ~300 MB; anything under this is a partial download.
_MIN_CKPT_BYTES = 100 * 2**20


def _repo_root() -> str:
    # custom_nodes/<this-repo>/pipeline/prepare.py -> repo root is 4 up.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def hf_token() -> str | None:
    """Return the HuggingFace token from the environment, or None.

    The CorridorKey repos are public, so this is only a nicety (rate limits).
    """
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _models_subdir(category: str, fallback: str) -> str:
    """Resolve a ComfyUI models subdir, working with or without folder_paths."""
    try:
        import folder_paths

        return folder_paths.get_folder_paths(category)[0]
    except Exception:
        # Headless diag scripts run with cwd=ComfyUI; fall back to the standard layout.
        return os.path.join(_repo_root(), "ComfyUI", "models", fallback)


def corridorkey_dir() -> str:
    return _models_subdir("corridorkey", "corridorkey")


def birefnet_dir() -> str:
    return _models_subdir("birefnet", "birefnet")


def ensure_corridorkey(screen_color: str) -> str:
    """Return the local checkpoint path for the given screen color, downloading if missing."""
    if screen_color not in CKPTS:
        raise ValueError(f"Unknown screen_color '{screen_color}'. Valid: {sorted(CKPTS)}")
    repo_id, filename, _ = CKPTS[screen_color]

    dest_dir = corridorkey_dir()
    local = os.path.join(dest_dir, filename)
    if os.path.isfile(local) and os.path.getsize(local) > _MIN_CKPT_BYTES:
        return local

    os.makedirs(dest_dir, exist_ok=True)
    from huggingface_hub import hf_hub_download

    print(f"[CorridorKey] Downloading {repo_id}/{filename} (~300MB, resumable)...", flush=True)
    return hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token(), local_dir=dest_dir)


# A BiRefNet snapshot is config + code + ~1 GB of weights; anything smaller is a shard or
# an auxiliary file, not the real thing.
_MIN_WEIGHT_BYTES = 50 * 2**20


def _birefnet_complete(dest: str) -> bool:
    """True only if the snapshot has its *weights*, not just its metadata.

    snapshot_download fetches files concurrently, so the tiny config.json lands long before
    the weights: treating it as the completeness marker makes an interrupted download look
    finished forever, and from_pretrained then fails on every later run.
    """
    if not os.path.isfile(os.path.join(dest, "config.json")):
        return False
    try:
        names = os.listdir(dest)
    except OSError:
        return False
    for name in names:
        if not name.endswith((".safetensors", ".bin")):
            continue
        try:
            if os.path.getsize(os.path.join(dest, name)) > _MIN_WEIGHT_BYTES:
                return True
        except OSError:
            continue
    return False


def ensure_birefnet(repo_id: str) -> str:
    """Return a local snapshot dir for a BiRefNet HF repo, downloading if missing."""
    dest = os.path.join(birefnet_dir(), repo_id.split("/")[-1])
    if _birefnet_complete(dest):
        return dest

    os.makedirs(dest, exist_ok=True)
    from huggingface_hub import snapshot_download

    # Re-running is cheap when files are already there (hub only revalidates metadata).
    print(f"[CorridorKey] Downloading BiRefNet snapshot {repo_id} ...", flush=True)
    return snapshot_download(repo_id=repo_id, token=hf_token(), local_dir=dest)
