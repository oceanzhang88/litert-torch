"""LocateAnything metadata builder hook for LiteRT-LM packaging."""

from __future__ import annotations

from typing import Any

from litert_lm_builder.runtime.proto import llm_model_type_pb2


LOCATEANYTHING_METADATA_SIDECAR_KEYS = {
    "image_token": "<IMG_CONTEXT>",
    "image_start": "<img>",
    "image_end": "</img>",
    "box_start": "<box>",
    "box_end": "</box>",
    "coord_range": "<0>..<1000>",
    "image_grid_hws": "MoonViT patch grid per image",
    "vision_adapter": "mlp1(MoonViT features) -> Qwen2 hidden size",
}

LOCATEANYTHING_LITERT_RUNTIME_IMAGE_TRIGGER = "<start_of_image>"
LOCATEANYTHING_LITERT_FASTVLM_IMAGE_TRIGGER = "<image_soft_token>"
LOCATEANYTHING_LITERT_IMAGE_TOKEN = "<img>"
LOCATEANYTHING_LITERT_END_IMAGE_TOKEN = "</img>"
LOCATEANYTHING_LITERT_IMAGE_SIZE = 448

# LiteRT-LM's FastVlmDataProcessor owns the high-level NHWC image preprocessing
# path we need for the current LocateAnything export. Its regex scans
# `<image_soft_token>` and replaces that delimiter in-place, without the
# Gemma3DataProcessor's hardcoded newlines before and after the image span.
# LocateAnything was trained on prompts shaped like:
#
#   <image 1><img>{256 image-context embeddings}</img>{user text}
#
# The template therefore preserves the HF-visible `<image 1>` prefix, emits the
# FastVLM runtime trigger between LocateAnything's `<img>` and `</img>` tokens,
# and leaves the end token in the remaining prompt text. Do not emit literal
# `<IMG_CONTEXT>` tokens here: in this W1 runtime path the image span is owned
# by the external vision embeddings, not by text-token placeholders.
LOCATEANYTHING_QWEN_MULTIMODAL_CHAT_TEMPLATE = r"""{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
{%- else %}
    {{- '<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message.role == 'system' and loop.first %}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\n' }}
        {%- if message.content is string %}
            {{- message.content }}
        {%- else %}
            {%- for part in message.content %}
                {%- if part.type == 'image' %}
                    {{- '<image 1><img><image_soft_token></img>' }}
                {%- elif part.type == 'text' %}
                    {{- part.text }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""


def build_llm_metadata(
    source_model_artifacts: Any,
    export_config: Any,
    exported_model_artifacts: Any,
    llm_metadata: Any,
) -> Any:
    """Build LocateAnything's first LiteRT-LM multimodal runtime contract."""

    del source_model_artifacts, export_config, exported_model_artifacts

    llm_metadata.llm_model_type.CopyFrom(
        llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm())
    )
    llm_metadata.llm_model_type.fast_vlm.image_tensor_height = (
        LOCATEANYTHING_LITERT_IMAGE_SIZE
    )
    llm_metadata.llm_model_type.fast_vlm.image_tensor_width = (
        LOCATEANYTHING_LITERT_IMAGE_SIZE
    )
    llm_metadata.jinja_prompt_template = LOCATEANYTHING_QWEN_MULTIMODAL_CHAT_TEMPLATE
    return llm_metadata
