"""Dispatch helpers for a LiteRT Torch LocateAnything model extension."""

from __future__ import annotations

from typing import Any


def is_locateanything_config(model_config: Any) -> bool:
    """Return true for the HF custom LocateAnything config."""

    return getattr(model_config, "model_type", None) == "locateanything"


def get_prefill_decode_exportables(model_config: Any, export_config: Any):
    """Return text exportables for LocateAnything when the config matches."""

    if not is_locateanything_config(model_config):
        return None
    from . import exportable_module

    if getattr(export_config, "split_cache", False):
        return (
            exportable_module.LiteRTExportableModuleForLocateAnythingPrefillSplitCache,
            exportable_module.LiteRTExportableModuleForLocateAnythingGenerateSplitCache,
        )
    return (
        exportable_module.LiteRTExportableModuleForLocateAnythingPrefillExternalEmbedder,
        exportable_module.LiteRTExportableModuleForLocateAnythingGenerateExternalEmbedder,
    )


def get_vision_exportables(model_config: Any):
    """Return MoonViT encoder and mlp1 adapter exportables for LocateAnything."""

    if not is_locateanything_config(model_config):
        return None
    from . import vision_exportable

    return (
        vision_exportable.LiteRTExportableModuleForLocateAnythingVisionEncoder,
        vision_exportable.LiteRTExportableModuleForLocateAnythingVisionAdapter,
    )


def get_metadata_builder(model_config: Any):
    """Return the LocateAnything metadata hook when the config matches."""

    if not is_locateanything_config(model_config):
        return None
    from . import metadata_builder

    return metadata_builder.build_llm_metadata


def required_loader_auto_model(model_config: Any) -> str | None:
    """LocateAnything needs `transformers.AutoModel`, not image-text AutoModel classes."""

    if not is_locateanything_config(model_config):
        return None
    return "AutoModel"
