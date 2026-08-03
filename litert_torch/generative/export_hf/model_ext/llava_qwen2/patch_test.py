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
"""Tests for FastVLM (llava_qwen2) export patches."""

from absl.testing import absltest
from ai_edge_litert import schema_py_generated as schema_fb
import litert_torch
from litert_torch.generative.export_hf.core import patches as core_patches
from litert_torch.generative.export_hf.model_ext.llava_qwen2 import patch
import torch


class Qwen2RMSNorm(torch.nn.Module):
  """Import-order regression stand-in for transformers' inline Qwen2 norm."""

  def __init__(self, hidden_size: int, eps: float = 1e-6):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.randn(hidden_size))
    self.variance_epsilon = eps

  def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(
        variance + self.variance_epsilon
    )
    return self.weight * hidden_states.to(input_dtype)


class _DecoderLayer(torch.nn.Module):

  def __init__(self, hidden_size: int):
    super().__init__()
    self.input_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-5)
    self.post_attention_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-5)


class _TextModel(torch.nn.Module):

  def __init__(self, hidden_size: int, num_layers: int):
    super().__init__()
    self.layers = torch.nn.ModuleList(
        [_DecoderLayer(hidden_size) for _ in range(num_layers)]
    )
    self.norm = Qwen2RMSNorm(hidden_size, eps=1e-5)


class _FastVlm(torch.nn.Module):

  def __init__(self, hidden_size: int = 8, num_layers: int = 2):
    super().__init__()
    self.model = _TextModel(hidden_size, num_layers)

  def forward(self, inputs_embeds=None, **kwargs):
    del kwargs
    return inputs_embeds


def _top_graph_operator_codes(lrt_model):
  model = schema_fb.Model.GetRootAsModel(lrt_model.model_content(), 0)
  subgraph = model.Subgraphs(0)
  return [
      model.OperatorCodes(subgraph.Operators(i).OpcodeIndex()).BuiltinCode()
      for i in range(subgraph.OperatorsLength())
  ]


class FastVlmPatchTest(absltest.TestCase):

  def test_loaded_qwen2_norms_are_adapted_and_restored(self):
    model = _FastVlm()
    original_norms = [
        module
        for module in model.modules()
        if type(module).__name__ == "Qwen2RMSNorm"
    ]
    original_weights = [module.weight for module in original_norms]
    sample = torch.randn(2, 3, 8)
    expected = [module(sample) for module in original_norms]

    with patch.llava_qwen2_litert_patch(model, object()):
      adapted_norms = [
          module
          for module in model.modules()
          if isinstance(module, patch.FastVlmRMSNorm)
      ]
      self.assertLen(adapted_norms, 5)
      for index, adapted in enumerate(adapted_norms):
        self.assertIs(adapted.weight, original_weights[index])
        self.assertEqual(adapted.variance_epsilon, 1e-5)
        torch.testing.assert_close(adapted(sample), expected[index])

    restored_norms = [
        module
        for module in model.modules()
        if type(module).__name__ == "Qwen2RMSNorm"
    ]
    self.assertEqual(restored_norms, original_norms)
    for restored, weight in zip(restored_norms, original_weights):
      self.assertIs(restored.weight, weight)

  def test_existing_exporter_norm_is_not_double_wrapped(self):
    model = _FastVlm(num_layers=0)
    existing = core_patches.RMSNorm(8, eps=1e-5)
    model.model.norm = existing

    with patch.llava_qwen2_litert_patch(model, object()):
      self.assertIs(model.model.norm, existing)

    self.assertIs(model.model.norm, existing)

  def test_adapted_norm_is_outlined_in_signature_graph(self):
    adapted = patch.FastVlmRMSNorm(Qwen2RMSNorm(8, eps=1e-5)).eval()
    converted = litert_torch.convert(
        adapted,
        (torch.randn(1, 4, 8),),
        strict_export=False,
    )

    top_graph_codes = _top_graph_operator_codes(converted)
    self.assertIn(
        schema_fb.BuiltinOperator.STABLEHLO_COMPOSITE, top_graph_codes
    )
    self.assertNotIn(schema_fb.BuiltinOperator.RSQRT, top_graph_codes)
    self.assertNotIn(schema_fb.BuiltinOperator.SUM, top_graph_codes)


if __name__ == "__main__":
  absltest.main()
