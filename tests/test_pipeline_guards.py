"""Pipeline guards: the despeckle threshold and BiRefNet snapshot completeness.

Both are settings that silently destroy output when wrong — a black matte, or a half
downloaded model that looks complete forever. Neither torch nor network is needed here,
because nodes and pipeline modules only import their heavy dependencies lazily.

    python tests/test_pipeline_guards.py
"""
import importlib
import os
import sys
import tempfile

# The package is imported under whatever name its folder has (free to rename after a clone).
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(REPO))
_pkg = os.path.basename(REPO)
_post = importlib.import_module(f"{_pkg}.pipeline.post")
_prep = importlib.import_module(f"{_pkg}.pipeline.prepare")
_MIN_AREA_THRESHOLD = _post._MIN_AREA_THRESHOLD
_birefnet_complete, _MIN_WEIGHT_BYTES = _prep._birefnet_complete, _prep._MIN_WEIGHT_BYTES

fails = []


def ck(name, cond, extra=""):
    print(("OK   " if cond else "FAIL ") + name + ("" if cond else "  -> " + str(extra)))
    if not cond:
        fails.append(name)


# ---------- despeckle: never zero merge iterations --------------------------
# clean_matte_torch computes `max_iterations = area_threshold // 20`; zero wipes the matte.
def iterations(user_value):
    if user_value <= 0:
        return None                     # despeckle disabled, mask returned untouched
    return max(user_value, _MIN_AREA_THRESHOLD) // 20


ck("0 disables despeckle (mask untouched)", iterations(0) is None)
bad = [v for v in range(1, 4097) if iterations(v) is not None and iterations(v) < 1]
ck("no value in the node's range yields zero iterations", not bad, bad[:10])
ck("the once-fatal range (2-19) now merges",
   all(iterations(v) >= 1 for v in range(2, 20)))
ck("the default of 400 is unchanged", iterations(400) == 20)
ck("large values are untouched", iterations(4096) == 204)


# ---------- snapshot: complete only once the weights are there --------------
def snap(**files):
    d = tempfile.mkdtemp()
    for name, size in files.items():
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"\0" * size)
    return d


big = _MIN_WEIGHT_BYTES + 1
ck("empty folder -> incomplete", not _birefnet_complete(snap()))
ck("config.json alone -> INCOMPLETE (this was the bug)",
   not _birefnet_complete(snap(**{"config.json": 500})))
ck("config + weights -> complete",
   _birefnet_complete(snap(**{"config.json": 500, "model.safetensors": big})))
ck("config + .bin weights -> complete",
   _birefnet_complete(snap(**{"config.json": 500, "pytorch_model.bin": big})))
ck("truncated weights -> incomplete",
   not _birefnet_complete(snap(**{"config.json": 500, "model.safetensors": 1024})))
ck("weights without config -> incomplete",
   not _birefnet_complete(snap(**{"model.safetensors": big})))
ck("sharded weights -> complete",
   _birefnet_complete(snap(**{"config.json": 500,
                              "model-00001-of-00002.safetensors": big,
                              "model-00002-of-00002.safetensors": big})))
ck("missing folder -> incomplete", not _birefnet_complete("/does/not/exist"))

print(f"\n{'FAILURES: ' + ', '.join(fails) if fails else 'pipeline guards OK'}")
sys.exit(1 if fails else 0)
