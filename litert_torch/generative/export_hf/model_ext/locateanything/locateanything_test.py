# Copyright 2026 The LiteRT Torch Authors.
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
"""Tests for the LocateAnything model extension.

These tests load the LocateAnything extension files by path with small dependency
stubs. That keeps source-level extension coverage useful in lightweight test
environments where importing top-level `litert_torch` requires optional LiteRT
runtime wheels such as `ai_edge_litert`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

from absl.testing import absltest
import torch


_EXTENSION_ROOT = Path(__file__).resolve().parent


def _load_source_module(module_name: str, path: Path):
  spec = importlib.util.spec_from_file_location(module_name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)  # pytype: disable=attribute-error
  return module


class _Config:

  def __init__(self, model_type: str):
    self.model_type = model_type


class _ExportConfig:

  def __init__(self, *, split_cache: bool = False):
    self.split_cache = split_cache


def _install_registry_test_package():
  package_name = "_locateanything_registry_test_pkg"
  package = types.ModuleType(package_name)
  package.__path__ = [str(_EXTENSION_ROOT)]
  sys.modules[package_name] = package

  exportable_module = types.ModuleType(f"{package_name}.exportable_module")
  for class_name in (
      "LiteRTExportableModuleForLocateAnythingPrefillExternalEmbedder",
      "LiteRTExportableModuleForLocateAnythingGenerateExternalEmbedder",
      "LiteRTExportableModuleForLocateAnythingPrefillSplitCache",
      "LiteRTExportableModuleForLocateAnythingGenerateSplitCache",
  ):
    setattr(exportable_module, class_name, type(class_name, (), {}))
  sys.modules[exportable_module.__name__] = exportable_module

  vision_exportable = types.ModuleType(f"{package_name}.vision_exportable")
  for class_name in (
      "LiteRTExportableModuleForLocateAnythingVisionEncoder",
      "LiteRTExportableModuleForLocateAnythingVisionAdapter",
  ):
    setattr(vision_exportable, class_name, type(class_name, (), {}))
  sys.modules[vision_exportable.__name__] = vision_exportable

  metadata_builder = types.ModuleType(f"{package_name}.metadata_builder")
  metadata_builder.build_llm_metadata = object()
  sys.modules[metadata_builder.__name__] = metadata_builder

  registry = _load_source_module(
      f"{package_name}.registry", _EXTENSION_ROOT / "registry.py"
  )
  return registry, exportable_module, vision_exportable, metadata_builder


def _install_litert_torch_export_stubs():
  for module_name in (
      "litert_torch",
      "litert_torch.backend",
      "litert_torch.generative",
      "litert_torch.generative.export_hf",
      "litert_torch.generative.export_hf.core",
      "litert_torch.generative.export_hf.core.external_emb",
      "litert_torch.generative.export_hf.core.split_cache",
  ):
    module = types.ModuleType(module_name)
    module.__path__ = []
    sys.modules[module_name] = module

  optimization_barrier = types.ModuleType(
      "litert_torch.backend.optimization_barrier"
  )
  optimization_barrier.optimization_barrier = lambda value: value
  sys.modules[optimization_barrier.__name__] = optimization_barrier

  attention_module = types.ModuleType(
      "litert_torch.generative.export_hf.core.attention"
  )
  attention_module.transposed_attention = lambda *args, **kwargs: (
      args[1],
      None,
  )
  sys.modules[attention_module.__name__] = attention_module

  split_cache_attention = types.ModuleType(
      "litert_torch.generative.export_hf.core.split_cache.attention"
  )
  split_cache_attention.split_cache_attention = lambda *args, **kwargs: (
      args[1],
      None,
  )
  sys.modules[split_cache_attention.__name__] = split_cache_attention

  class _ExportableModuleBase(torch.nn.Module):

    def __init__(self, export_config=None):
      super().__init__()
      self.export_config = export_config

  core_exportable = types.ModuleType(
      "litert_torch.generative.export_hf.core.exportable_module"
  )
  core_exportable.ExportableModuleBase = _ExportableModuleBase
  sys.modules[core_exportable.__name__] = core_exportable

  class _ExternalPrefill(torch.nn.Module):
    pass

  class _ExternalGenerate(torch.nn.Module):
    pass

  external_emb_exportable = types.ModuleType(
      "litert_torch.generative.export_hf.core.external_emb.exportable_module"
  )
  external_emb_exportable.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder = (
      _ExternalPrefill
  )
  external_emb_exportable.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder = (
      _ExternalGenerate
  )
  sys.modules[external_emb_exportable.__name__] = external_emb_exportable

  class _SplitCacheBase(torch.nn.Module):

    @staticmethod
    def adapt_inputs(exportable, embeddings, pos_emb, mask, kv_cache):
      del exportable
      return {
          "inputs_embeds": embeddings,
          "position_ids": pos_emb,
          "attention_mask": mask,
          "past_key_values": kv_cache,
      }

  class _SplitCachePrefill(_SplitCacheBase):
    pass

  class _SplitCacheGenerate(_SplitCacheBase):
    pass

  split_cache_exportable = types.ModuleType(
      "litert_torch.generative.export_hf.core.split_cache.exportable_module"
  )
  split_cache_exportable.LiteRTSplitCacheExportableModuleForDecoderOnlyLM = (
      _SplitCacheBase
  )
  split_cache_exportable.LiteRTSplitCacheExportableModuleForDecoderOnlyLMPrefill = (
      _SplitCachePrefill
  )
  split_cache_exportable.LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate = (
      _SplitCacheGenerate
  )
  sys.modules[split_cache_exportable.__name__] = split_cache_exportable


class _LlmModelTypeHolder:

  def CopyFrom(self, value):
    self.value = value
    self.fast_vlm = value.fast_vlm


class _Metadata:

  def __init__(self):
    self.llm_model_type = _LlmModelTypeHolder()
    self.jinja_prompt_template = ""


def _install_metadata_proto_stubs():
  for module_name in (
      "litert_lm_builder",
      "litert_lm_builder.runtime",
      "litert_lm_builder.runtime.proto",
  ):
    module = types.ModuleType(module_name)
    module.__path__ = []
    sys.modules[module_name] = module

  proto_module = types.ModuleType(
      "litert_lm_builder.runtime.proto.llm_model_type_pb2"
  )

  class FastVlm:

    def __init__(self):
      self.image_tensor_height = 0
      self.image_tensor_width = 0

  class LlmModelType:

    def __init__(self, fast_vlm=None):
      self.fast_vlm = fast_vlm

  proto_module.FastVlm = FastVlm
  proto_module.LlmModelType = LlmModelType
  sys.modules[proto_module.__name__] = proto_module


class _QwenRotaryModule:

  @staticmethod
  def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _QwenRotaryEmbedding(torch.nn.Module):

  def __init__(self, dim=4, base=10000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  def forward(self, seq_len, dtype=torch.float32):
    t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
    freqs = torch.outer(t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _reference_apply_rotary_pos_emb(
    q, k, cos, sin, position_ids, unsqueeze_dim=1
):
  cos = cos[position_ids.to(torch.long)].unsqueeze(unsqueeze_dim)
  sin = sin[position_ids.to(torch.long)].unsqueeze(unsqueeze_dim)
  q_embed = (q * cos) + (_QwenRotaryModule.rotate_half(q) * sin)
  k_embed = (k * cos) + (_QwenRotaryModule.rotate_half(k) * sin)
  return q_embed, k_embed


def _reference_nhwc_image_to_locateanything_patches(
    vision_exportable, images: torch.Tensor
) -> torch.Tensor:
  if images.dim() == 3:
    images = images.unsqueeze(0)
  image = images[0, :, :, :3].permute(2, 0, 1)
  mean = torch.tensor(
      vision_exportable.DEFAULT_IMAGE_MEAN,
      dtype=image.dtype,
      device=image.device,
  ).view(3, 1, 1)
  std = torch.tensor(
      vision_exportable.DEFAULT_IMAGE_STD,
      dtype=image.dtype,
      device=image.device,
  ).view(3, 1, 1)
  image = (image - mean) / std
  patch_size = 14
  channels = 3
  height = vision_exportable.DEFAULT_EXPORT_IMAGE_SIZE
  width = vision_exportable.DEFAULT_EXPORT_IMAGE_SIZE
  patches = image.reshape(
      channels,
      height // patch_size,
      patch_size,
      width // patch_size,
      patch_size,
  )
  patches = patches.permute(1, 3, 0, 2, 4)
  return patches.contiguous().view(-1, channels, patch_size, patch_size)


class LocateAnythingRegistryTest(absltest.TestCase):

  def test_registry_dispatches_external_and_split_cache_exportables(self):
    registry, exportable_module, _, _ = _install_registry_test_package()

    external = registry.get_prefill_decode_exportables(
        _Config("locateanything"), _ExportConfig(split_cache=False)
    )
    split_cache = registry.get_prefill_decode_exportables(
        _Config("locateanything"), _ExportConfig(split_cache=True)
    )

    self.assertEqual(
        external,
        (
            exportable_module.LiteRTExportableModuleForLocateAnythingPrefillExternalEmbedder,
            exportable_module.LiteRTExportableModuleForLocateAnythingGenerateExternalEmbedder,
        ),
    )
    self.assertEqual(
        split_cache,
        (
            exportable_module.LiteRTExportableModuleForLocateAnythingPrefillSplitCache,
            exportable_module.LiteRTExportableModuleForLocateAnythingGenerateSplitCache,
        ),
    )

  def test_registry_dispatches_vision_metadata_and_loader(self):
    registry, _, vision_exportable, metadata_builder = (
        _install_registry_test_package()
    )

    self.assertEqual(
        registry.get_vision_exportables(_Config("locateanything")),
        (
            vision_exportable.LiteRTExportableModuleForLocateAnythingVisionEncoder,
            vision_exportable.LiteRTExportableModuleForLocateAnythingVisionAdapter,
        ),
    )
    self.assertIs(
        registry.get_metadata_builder(_Config("locateanything")),
        metadata_builder.build_llm_metadata,
    )
    self.assertEqual(
        registry.required_loader_auto_model(_Config("locateanything")),
        "AutoModel",
    )
    self.assertIsNone(registry.get_vision_exportables(_Config("gemma4")))
    self.assertIsNone(registry.required_loader_auto_model(_Config("gemma4")))


class LocateAnythingMetadataBuilderTest(absltest.TestCase):

  def test_metadata_builder_sets_fast_vlm_envelope_and_template(self):
    _install_metadata_proto_stubs()
    metadata_builder = _load_source_module(
        "_locateanything_metadata_builder_test",
        _EXTENSION_ROOT / "metadata_builder.py",
    )
    metadata = _Metadata()

    result = metadata_builder.build_llm_metadata(None, None, None, metadata)

    self.assertIs(result, metadata)
    self.assertEqual(metadata.llm_model_type.fast_vlm.image_tensor_height, 448)
    self.assertEqual(metadata.llm_model_type.fast_vlm.image_tensor_width, 448)
    self.assertIn("<image 1><img><image_soft_token></img>", result.jinja_prompt_template)
    self.assertIn("You are a helpful assistant.", result.jinja_prompt_template)


class LocateAnythingTextExportableHelpersTest(absltest.TestCase):

  def test_input_sanitizer_drops_valid_mask_for_vendored_qwen(self):
    _install_litert_torch_export_stubs()
    exportable_module = _load_source_module(
        "_locateanything_text_exportable_test_sanitize",
        _EXTENSION_ROOT / "exportable_module.py",
    )
    inputs = {
        "inputs_embeds": torch.zeros((1, 2, 4)),
        "position_ids": torch.zeros((1, 2), dtype=torch.int32),
        "valid_mask": torch.ones((1, 2), dtype=torch.bool),
        "cache_position": torch.arange(2),
        "k_ts_idx": torch.tensor(0),
        "v_ts_idx": torch.tensor(1),
    }

    sanitized = exportable_module._drop_locateanything_unsupported_inputs(inputs)

    self.assertNotIn("valid_mask", sanitized)
    self.assertNotIn("cache_position", sanitized)
    self.assertNotIn("k_ts_idx", sanitized)
    self.assertNotIn("v_ts_idx", sanitized)
    self.assertIn("inputs_embeds", sanitized)
    self.assertIn("position_ids", sanitized)

  def test_rope_formula_matches_vendored_qwen_table_rows(self):
    _install_litert_torch_export_stubs()
    exportable_module = _load_source_module(
        "_locateanything_text_exportable_test",
        _EXTENSION_ROOT / "exportable_module.py",
    )
    q = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4) / 13.0
    k = torch.flip(q, dims=[-1])
    positions = torch.tensor([[0, 2, 5]], dtype=torch.int32)
    rotary = _QwenRotaryEmbedding(dim=4)
    full_cos, full_sin = rotary.forward(6)

    cos, sin = exportable_module._export_safe_rope_from_position_ids(
        rotary, positions, dtype=q.dtype, device=q.device
    )
    expected_q, expected_k = _reference_apply_rotary_pos_emb(
        q, k, full_cos, full_sin, positions
    )
    actual_q, actual_k = exportable_module._export_safe_apply_position_aligned_rotary(
        _QwenRotaryModule, q, k, cos, sin, positions
    )

    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)

  def test_cpu_gpu_attention_uses_converter_proven_formula_rope_shape(self):
    _install_litert_torch_export_stubs()
    exportable_module = _load_source_module(
        "_locateanything_text_exportable_test_cpu_gpu_rope",
        _EXTENSION_ROOT / "exportable_module.py",
    )

    class _RecorderModule(types.ModuleType):

      def __init__(self, name):
        super().__init__(name)
        self.position_ids = None

      @staticmethod
      def rotate_half(x):
        return _QwenRotaryModule.rotate_half(x)

      def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        raise AssertionError("CPU/GPU export must not use cached-table RoPE.")

    recorder_module = _RecorderModule("_locateanything_fake_qwen_module")
    sys.modules[recorder_module.__name__] = recorder_module
    class _RecordingRotary(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.register_buffer(
            "inv_freq", torch.tensor([1.0, 0.5], dtype=torch.float32)
        )

      def forward(self, value_states, seq_len):
        raise AssertionError("CPU/GPU export must not materialize full RoPE table.")

    class _FakeAttention(torch.nn.Module):
      __module__ = recorder_module.__name__

      def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Identity()
        self.k_proj = torch.nn.Identity()
        self.v_proj = torch.nn.Identity()
        self.o_proj = torch.nn.Identity()
        self.rotary_emb = _RecordingRotary()
        self.num_heads = 1
        self.num_key_value_heads = 1
        self.head_dim = 4
        self.hidden_size = 4
        self.max_position_embeddings = 16
        self.layer_idx = 0
        self.export_k_ts_idx = 2
        self.export_v_ts_idx = 3

    attention = _FakeAttention()
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 11.0
    position_ids = torch.tensor([[0, 2, 5]], dtype=torch.long)

    output, _, cache = exportable_module._export_safe_attention_forward(
        attention,
        hidden_states,
        position_ids=position_ids,
        past_key_value=None,
    )

    self.assertIsNone(cache)
    self.assertEqual(tuple(output.shape), (1, 3, 4))
    positions = torch.arange(6, dtype=torch.float32)
    freqs = positions.unsqueeze(-1) * attention.rotary_emb.inv_freq.reshape(1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    full_cos = emb.cos()
    full_sin = emb.sin()
    expected_query, _ = _reference_apply_rotary_pos_emb(
        hidden_states.reshape(1, 1, 3, 4),
        hidden_states.reshape(1, 1, 3, 4),
        full_cos,
        full_sin,
        position_ids,
    )
    torch.testing.assert_close(output, expected_query.reshape(1, 3, 4))

  def test_split_cache_preselected_rope_rows_match_absolute_rows(self):
    _install_litert_torch_export_stubs()
    exportable_module = _load_source_module(
        "_locateanything_text_exportable_test_split",
        _EXTENSION_ROOT / "exportable_module.py",
    )
    q = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4) / 7.0
    k = q + 0.25
    absolute_positions = torch.tensor([[2, 4, 5]], dtype=torch.int32)
    local_positions = torch.arange(3, dtype=torch.int32).unsqueeze(0)
    freqs = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4) / 5.0
    full_cos = freqs.cos()
    full_sin = freqs.sin()
    selected_cos = torch.index_select(full_cos, 0, absolute_positions.reshape(-1))
    selected_sin = torch.index_select(full_sin, 0, absolute_positions.reshape(-1))

    expected_q, expected_k = _reference_apply_rotary_pos_emb(
        q, k, full_cos, full_sin, absolute_positions
    )
    actual_q, actual_k = exportable_module._export_safe_apply_position_aligned_rotary(
        _QwenRotaryModule, q, k, selected_cos, selected_sin, local_positions
    )

    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)


class LocateAnythingVisionExportableHelpersTest(absltest.TestCase):

  def test_conv2d_patchifier_matches_legacy_fixed_grid_layout(self):
    _install_litert_torch_export_stubs()
    vision_exportable = _load_source_module(
        "_locateanything_vision_exportable_test",
        _EXTENSION_ROOT / "vision_exportable.py",
    )
    height = vision_exportable.DEFAULT_EXPORT_IMAGE_SIZE
    width = vision_exportable.DEFAULT_EXPORT_IMAGE_SIZE
    images = torch.linspace(
        0.0, 1.0, steps=height * width * 3, dtype=torch.float32
    ).reshape(1, height, width, 3)

    expected = _reference_nhwc_image_to_locateanything_patches(
        vision_exportable, images
    )
    actual = vision_exportable._nhwc_image_to_locateanything_patches(images)

    self.assertEqual(actual.shape, (1024, 3, 14, 14))
    torch.testing.assert_close(actual, expected)

  def test_fixed_patch_merger_matches_legacy_fixed_grid_layout(self):
    _install_litert_torch_export_stubs()
    vision_exportable = _load_source_module(
        "_locateanything_vision_exportable_test_merge",
        _EXTENSION_ROOT / "vision_exportable.py",
    )
    height, width = vision_exportable.DEFAULT_EXPORT_GRID_HW
    hidden_size = 8
    features = torch.arange(
        height * width * hidden_size, dtype=torch.float32
    ).reshape(height * width, hidden_size)

    grid = features.reshape(height, width, hidden_size)
    expected = torch.cat(
        (
            grid[0::2, 0::2, :],
            grid[0::2, 1::2, :],
            grid[1::2, 0::2, :],
            grid[1::2, 1::2, :],
        ),
        dim=-1,
    ).reshape(256, hidden_size * 4)
    actual = vision_exportable._fixed_patch_merger(features, (2, 2))[0]

    self.assertEqual(actual.shape, (256, hidden_size * 4))
    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
  absltest.main()
