"""Aggregate the node mappings (one node per file, house pattern)."""

from . import alpha_hint, colorspace, composite, despeckle, despill, keyer, loader

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _mod in (loader, keyer, despeckle, despill, composite, colorspace, alpha_hint):
    NODE_CLASS_MAPPINGS.update(_mod.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)
