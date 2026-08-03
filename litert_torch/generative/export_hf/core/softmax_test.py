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
"""Tests for attention softmax graph policy."""

from absl.testing import absltest
from ai_edge_litert import schema_py_generated as schema_fb
import litert_torch
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import softmax as softmax_lib
import torch


def _operator_codes(lrt_model, *, subgraph_index: int | None = None):
  model = schema_fb.Model.GetRootAsModel(lrt_model.model_content(), 0)
  codes = []
  subgraph_indices = (
      range(model.SubgraphsLength())
      if subgraph_index is None
      else (subgraph_index,)
  )
  for index in subgraph_indices:
    subgraph = model.Subgraphs(index)
    for op_index in range(subgraph.OperatorsLength()):
      operator = subgraph.Operators(op_index)
      code = model.OperatorCodes(operator.OpcodeIndex())
      codes.append(code.BuiltinCode())
  return codes


class _SoftmaxPolicyModule(torch.nn.Module):

  def __init__(self, use_composite: bool):
    super().__init__()
    self.use_composite = use_composite

  def forward(self, x):
    return softmax_lib.apply_softmax(
        x, dim=-1, use_composite=self.use_composite
    )


class _DummyExportable(exportable_module.ExportableModuleBase):

  def get_sample_inputs(self, model_config, **kwargs):
    del model_config, kwargs
    return {}


class SoftmaxTest(absltest.TestCase):

  def test_builtin_policy_keeps_softmax_in_signature_graph(self):
    converted = litert_torch.convert(
        _SoftmaxPolicyModule(use_composite=False).eval(),
        (torch.randn(1, 4),),
        strict_export=False,
    )

    top_graph_codes = _operator_codes(converted, subgraph_index=0)
    self.assertIn(schema_fb.BuiltinOperator.SOFTMAX, top_graph_codes)
    self.assertNotIn(
        schema_fb.BuiltinOperator.STABLEHLO_COMPOSITE, top_graph_codes
    )

  def test_composite_policy_outlines_softmax_from_signature_graph(self):
    converted = litert_torch.convert(
        _SoftmaxPolicyModule(use_composite=True).eval(),
        (torch.randn(1, 4),),
        strict_export=False,
    )

    top_graph_codes = _operator_codes(converted, subgraph_index=0)
    all_codes = _operator_codes(converted)
    self.assertIn(
        schema_fb.BuiltinOperator.STABLEHLO_COMPOSITE, top_graph_codes
    )
    self.assertNotIn(schema_fb.BuiltinOperator.SOFTMAX, top_graph_codes)
    # The composite's decomposition still owns the builtin implementation; it
    # is merely absent from the delegate-facing signature graph.
    self.assertIn(schema_fb.BuiltinOperator.SOFTMAX, all_codes)

  def test_exportable_only_passes_an_explicit_policy(self):
    automatic = _DummyExportable(
        exportable_module_config.ExportableModuleConfig(model="unused")
    )
    builtin = _DummyExportable(
        exportable_module_config.ExportableModuleConfig(
            model="unused", use_softmax_composite=False
        )
    )
    composite = _DummyExportable(
        exportable_module_config.ExportableModuleConfig(
            model="unused", use_softmax_composite=True
        )
    )

    self.assertNotIn("use_softmax_composite", automatic.attention_kwargs())
    self.assertIs(builtin.attention_kwargs()["use_softmax_composite"], False)
    self.assertIs(composite.attention_kwargs()["use_softmax_composite"], True)


if __name__ == "__main__":
  absltest.main()
