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
"""Tests for externalized rotary embedding."""

from absl.testing import absltest as googletest
import torch

from litert_torch.generative.export_hf.core.external_rope import (
    exportable_module as external_rope_module,
)


class _MockConfig:
  rope_parameters = {}


class _LegacySeqLenRotaryEmbedding(torch.nn.Module):
  """Qwen2-style rotary module that accepts seq_len but not position_ids."""

  def __init__(self):
    super().__init__()
    self.register_buffer(
        "cos_cached",
        torch.arange(32, dtype=torch.float32).reshape(8, 4),
        persistent=False,
    )
    self.register_buffer(
        "sin_cached",
        (100 + torch.arange(32, dtype=torch.float32)).reshape(8, 4),
        persistent=False,
    )

  def forward(self, x, seq_len=None):
    return (
        self.cos_cached[:seq_len].to(dtype=x.dtype),
        self.sin_cached[:seq_len].to(dtype=x.dtype),
    )


class _MockDecoder(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.original_rotary_emb = _LegacySeqLenRotaryEmbedding()


class _MockModel(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.config = _MockConfig()
    self.model = _MockDecoder()


class _TopLevelLanguageModel(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.model = _MockDecoder()


class _TopLevelLanguageModelWrapper(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.config = _MockConfig()
    self.language_model = _TopLevelLanguageModel()


class ExternalRoPEExportableModuleTest(googletest.TestCase):

  def test_seq_len_rotary_gathers_absolute_positions(self):
    model = _MockModel()
    rope_module = external_rope_module.RoPEEmbedder(model)

    output = rope_module(torch.tensor([0, 3, 5], dtype=torch.int32))

    expected_cos = model.model.original_rotary_emb.cos_cached[
        torch.tensor([0, 3, 5], dtype=torch.int64)
    ].unsqueeze(0).unsqueeze(2)
    expected_sin = model.model.original_rotary_emb.sin_cached[
        torch.tensor([0, 3, 5], dtype=torch.int64)
    ].unsqueeze(0).unsqueeze(2)
    self.assertTrue(torch.equal(output["pos_emb_cos"], expected_cos))
    self.assertTrue(torch.equal(output["pos_emb_sin"], expected_sin))
    self.assertFalse(
        torch.equal(
            output["pos_emb_cos"].squeeze(0).squeeze(1),
            model.model.original_rotary_emb.cos_cached[:3],
        )
    )

  def test_seq_len_rotary_decode_position_is_not_zero_reused(self):
    model = _MockModel()
    rope_module = external_rope_module.RoPEEmbedder(model)

    output = rope_module(torch.tensor([7], dtype=torch.int32))

    self.assertTrue(
        torch.equal(
            output["pos_emb_cos"].squeeze(0).squeeze(1),
            model.model.original_rotary_emb.cos_cached[7:8],
        )
    )
    self.assertFalse(
        torch.equal(
            output["pos_emb_cos"].squeeze(0).squeeze(1),
            model.model.original_rotary_emb.cos_cached[:1],
        )
    )

  def test_top_level_language_model_wrapper_finds_decoder(self):
    model = _TopLevelLanguageModelWrapper()
    rope_module = external_rope_module.RoPEEmbedder(model)

    output = rope_module(torch.tensor([4], dtype=torch.int32))

    self.assertTrue(
        torch.equal(
            output["pos_emb_cos"].squeeze(0).squeeze(1),
            model.language_model.model.original_rotary_emb.cos_cached[4:5],
        )
    )


if __name__ == "__main__":
  googletest.main()
