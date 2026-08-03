# Copyright 2025 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exportable modules."""

import dataclasses
import enum
import pprint
from typing import Any

from litert_torch.generative.export_hf.core import utils
import torch


class ExportTask(str, enum.Enum):
  TEXT_GENERATION = "text_generation"
  IMAGE_TEXT_TO_TEXT = "image_text_to_text"
  AUTOMATIC_SPEECH_RECOGNITION = "automatic_speech_recognition"


class TokenizerFormat(str, enum.Enum):
  """Tokenizer representation written into a LiteRT-LM bundle."""

  AUTO = "auto"
  HUGGING_FACE = "hugging_face"
  SENTENCEPIECE = "sentencepiece"


@dataclasses.dataclass
class ExportableModuleConfig:
  """Config for exportable modules."""

  model: str
  output_dir: str | None = None
  task: ExportTask | str = ExportTask.TEXT_GENERATION
  keep_temporary_files: bool = False
  trust_remote_code: bool = False
  prefill_lengths: list[int] = dataclasses.field(default_factory=lambda: [128])
  cache_length: int = 4096
  # For quantization
  quantization_recipe: str | None = "dynamic_wi8_afp32"
  # For dynamic shape
  enable_dynamic_shape: bool = False
  # If True, prefill lengths are adjusted to magic numbers for GPU execution.
  enable_gpu_dynamic_prefill: bool = False
  # If True, cache length is adjusted to a magic number for GPU execution.
  enable_gpu_dynamic_cache: bool = False
  # Export configs
  externalize_embedder: bool = False
  single_token_embedder: bool = False
  k_ts_idx: int = 2
  v_ts_idx: int = 3
  split_cache: bool = False
  # Additive value used for masked positions in the split-cache auxiliary
  # graph. This is checkpoint-owned: a value deep enough for one checkpoint can
  # unnecessarily coarsen the int16 mask scale of another. None preserves the
  # attention module's compatibility default; production recipes should set it.
  split_cache_mask_fill: float | None = None
  cache_implementation: str | None = None
  auto_model_override: str | None = None
  use_jinja_template: bool = True
  bundle_litert_lm: bool = True
  # Experimental configs
  experimental_use_mixed_precision: bool = False
  experimental_use_fp16: bool = False
  export_vision_encoder: bool = True
  export_audio_encoder: bool = True
  fuse_gate_up: bool = False
  fuse_qkv: bool = False
  use_rope_composite: bool = False
  # Whether attention softmax is outlined as an odml.softmax StableHLO
  # composite. None preserves each attention implementation's existing default:
  # post-dcbe166 non-split attention outlines it, while split-cache attention
  # retains a builtin TFLite SOFTMAX. Compatibility recipes can set False
  # explicitly to require the builtin form across both paths.
  use_softmax_composite: bool | None = None
  input_sec: float = 1.0
  # If >= 0, the model runs in stateful mode after this many tokens.
  stateful_after: int = -1
  # TODO(weiyiw): Update when b/481323182 is fixed.
  # For now, for vision encoder, if there's conv op, set weight_only_wi8_afp32
  # if you intend to run on CPU, and set dynamic_wi8_afp32 if you intend to run
  # on GPU.
  vision_encoder_quantization_recipe: str | None = "dynamic_wi8_afp32"
  # Also export a block-signature embedder for the calibration executor. Only
  # meaningful when `single_token_embedder` is on (the vision-language shape),
  # where the shipped embedder cannot serve `prefill_embedder_{length}`.
  export_calibration_embedder: bool = False
  # When set, the block-signature calibration embedder is quantized into a
  # separate file with this recipe and becomes the embedder shipped in the
  # bundle. The original float graph remains available for calibration.
  block_embedder_quantization_recipe: str | None = None
  litert_lm_model_type_override: str | None = None
  litert_lm_llm_metadata_override: str | None = None
  tokenizer_path_override: str | None = None
  tokenizer_format: TokenizerFormat | str = TokenizerFormat.AUTO
  llm_metadata_max_num_tokens_override: int | None = None
  sampler_top_p: float | None = None
  sampler_temperature: float | None = None
  sampler_top_k: int | None = None
  jinja_chat_template_override: str | None = None
  use_random_weights: bool = False

  experimental_lightweight_conversion: bool = False
  experimental_transpile_chat_template_for_minijinja: bool = False

  assistant_model: str | None = None
  mtp_verifier_step: int = 5

  moe_exports_implementation: str = "litert_moe"

  # AOT Compilation.
  aot_backend: str | None = None
  aot_soc_model: str | None = None
  aot_compilation_config_dict: dict[str, Any] | None = None

  extra_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  # Internal configs
  work_dir: str | None = None
  batch_size: int = 1
  cache_length_dim: torch.export.Dim | None = None
  prefill_length_dim: torch.export.Dim | None = None
  externalize_rope: bool = False

  def __post_init__(self):
    """Refines configuration based on task-specific rules."""
    try:
      self.tokenizer_format = TokenizerFormat(self.tokenizer_format)
    except ValueError as e:
      supported = ", ".join(x.value for x in TokenizerFormat)
      raise ValueError(
          f"Unsupported tokenizer_format {self.tokenizer_format!r}; expected"
          f" one of: {supported}."
      ) from e

    if isinstance(self.prefill_lengths, int):
      self.prefill_lengths = [self.prefill_lengths]
    elif isinstance(self.prefill_lengths, str):
      self.prefill_lengths = [
          int(x) for x in self.prefill_lengths.split(",") if x
      ]

    # pylint: disable=g-bool-id-comparison
    match self.task:
      case ExportTask.IMAGE_TEXT_TO_TEXT:
        if self.export_vision_encoder:
          self.externalize_embedder = True
          self.single_token_embedder = True
        self.export_audio_encoder = False
      case ExportTask.AUTOMATIC_SPEECH_RECOGNITION:
        self.export_vision_encoder = False
        self.split_cache = False
        self.externalize_embedder = False
        self.externalize_rope = False
        self.bundle_litert_lm = False
      case _:
        self.export_vision_encoder = False
        self.export_audio_encoder = False

    if self.split_cache:
      self.externalize_embedder = True
      self.externalize_rope = True
      if self.cache_implementation is None:
        self.cache_implementation = "LiteRTLMSplitCache"
      self.moe_exports_implementation = "litert_moe_sequential"

    if self.block_embedder_quantization_recipe:
      if not self.split_cache:
        raise ValueError(
            "block_embedder_quantization_recipe requires split_cache=True; the"
            " portable executor consumes the single-signature embedder."
        )
      # The block graph is exported once in float for calibration, then copied
      # through the requested dynamic weight quantizer for packaging.
      self.export_calibration_embedder = True

    if self.enable_gpu_dynamic_prefill or self.enable_gpu_dynamic_cache:
      if self.enable_dynamic_shape:
        raise ValueError(
            "enable_dynamic_shape and enable_gpu_dynamic_prefill/cache"
            " cannot be both True."
        )
    if self.enable_gpu_dynamic_prefill:
      self.prefill_lengths = [
          utils.get_magic_number_for(l) for l in self.prefill_lengths
      ]
    if self.enable_gpu_dynamic_cache:
      self.cache_length = utils.get_magic_number_for(self.cache_length)

    if self.enable_dynamic_shape:
      self.prefill_length_dim = torch.export.Dim(
          "prefill_length", min=1, max=self.cache_length
      )
      self.cache_length_dim = torch.export.Dim("cache_length")

    if self.cache_implementation is None:
      self.cache_implementation = "LiteRTLMCache"
    # pylint: enable=g-bool-id-comparison

  def __repr__(self):
    """Returns a pretty-printed string representation of the config."""

    data = dataclasses.asdict(self)
    lines = [f"{' Export Configuration ':=^50}"]
    for key, value in sorted(data.items()):
      val_str = pprint.pformat(value, width=60, compact=True)
      if "\n" in val_str:
        val_str = val_str.replace("\n", "\n" + " " * 25)
      lines.append(f"{key:<22} : {val_str}")
    lines.append("=" * 50)

    return "\n".join(lines)

  def print_summary(self):
    """Directly prints the formatted configuration."""
    print(self.__repr__())
