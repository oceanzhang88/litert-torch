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
"""Tests for FastVLM vision export adaptations."""

from absl.testing import absltest
from litert_torch.generative.export_hf.model_ext.llava_qwen2 import vision_exportable
import torch


class MHSA(torch.nn.Module):
  """Checkpoint-shaped reference implementation with a rank-five QKV view."""

  def __init__(self, channels: int = 8, head_dim: int = 2):
    super().__init__()
    self.head_dim = head_dim
    self.num_heads = channels // head_dim
    self.scale = head_dim**-0.5
    self.qkv = torch.nn.Linear(channels, channels * 3, bias=True)
    self.attn_drop = torch.nn.Dropout(0.0)
    self.proj = torch.nn.Linear(channels, channels, bias=True)
    self.proj_drop = torch.nn.Dropout(0.0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    tokens = height * width
    x = torch.flatten(x, start_dim=2).transpose(-2, -1)
    qkv = (
        self.qkv(x)
        .reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    query, key, value = qkv.unbind(0)
    attention = (query * self.scale) @ key.transpose(-2, -1)
    attention = self.attn_drop(attention.softmax(dim=-1))
    x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
    x = self.proj_drop(self.proj(x))
    return x.transpose(-2, -1).reshape(batch, channels, height, width)


class _Tower(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.stage = torch.nn.Sequential(MHSA(), MHSA())
    self.lookalike = torch.nn.Linear(8, 8)


class _DepthwiseTower(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.transition = torch.nn.Conv2d(
        4, 8, kernel_size=3, stride=2, padding=1, groups=4, bias=True
    )
    self.ordinary = torch.nn.Conv2d(8, 8, kernel_size=1, bias=True)


class FastVlmVisionExportableTest(absltest.TestCase):

  def test_batch_one_adaptation_is_numerically_equivalent(self):
    torch.manual_seed(7)
    original = MHSA().eval()
    sample = torch.randn(1, 8, 3, 2)
    expected = original(sample)

    adapted = vision_exportable._BatchOneFastVlmMhsa(original).eval()
    actual = adapted(sample)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    self.assertIs(adapted.qkv, original.qkv)
    self.assertIs(adapted.proj, original.proj)
    self.assertIs(adapted.qkv.weight, original.qkv.weight)
    self.assertIs(adapted.proj.weight, original.proj.weight)

  def test_tree_adaptation_preserves_state_dict_paths(self):
    tower = _Tower().eval()
    before_keys = tuple(tower.state_dict())
    before_parameters = tuple(tower.parameters())

    count = vision_exportable.adapt_fastvit_mhsa_for_litert_gpu(tower)

    self.assertEqual(count, 2)
    self.assertIsInstance(
        tower.stage[0], vision_exportable._BatchOneFastVlmMhsa
    )
    self.assertIsInstance(
        tower.stage[1], vision_exportable._BatchOneFastVlmMhsa
    )
    self.assertIsInstance(tower.lookalike, torch.nn.Linear)
    self.assertEqual(tuple(tower.state_dict()), before_keys)
    after_parameters = tuple(tower.parameters())
    self.assertLen(after_parameters, len(before_parameters))
    self.assertTrue(
        all(after is before for after, before in zip(
            after_parameters, before_parameters
        ))
    )

  def test_batch_larger_than_one_is_rejected(self):
    adapted = vision_exportable._BatchOneFastVlmMhsa(MHSA()).eval()
    with self.assertRaisesRegex(ValueError, 'batch size one'):
      adapted(torch.randn(2, 8, 3, 2))

  def test_split_depthwise_multiplier_is_numerically_equivalent(self):
    torch.manual_seed(19)
    original = torch.nn.Conv2d(
        4, 12, kernel_size=3, stride=2, padding=1, groups=4, bias=True
    ).eval()
    sample = torch.randn(1, 4, 9, 7)
    expected = original(sample)

    adapted = vision_exportable._BatchOneSplitDepthwiseMultiplierConv(
        original
    ).eval()
    actual = adapted(sample)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    self.assertIs(adapted.weight, original.weight)
    self.assertIs(adapted.bias, original.bias)
    self.assertEqual(adapted.depth_multiplier, 3)

  def test_split_depthwise_tree_preserves_state_dict_paths(self):
    tower = _DepthwiseTower().eval()
    before_keys = tuple(tower.state_dict())
    before_parameters = tuple(tower.parameters())

    count = (
        vision_exportable.adapt_fastvit_depthwise_multipliers_for_litert(
            tower
        )
    )

    self.assertEqual(count, 1)
    self.assertIsInstance(
        tower.transition,
        vision_exportable._BatchOneSplitDepthwiseMultiplierConv,
    )
    self.assertIsInstance(tower.ordinary, torch.nn.Conv2d)
    self.assertEqual(tuple(tower.state_dict()), before_keys)
    after_parameters = tuple(tower.parameters())
    self.assertLen(after_parameters, len(before_parameters))
    self.assertTrue(
        all(
            after is before
            for after, before in zip(after_parameters, before_parameters)
        )
    )

  def test_split_depthwise_rejects_batch_larger_than_one(self):
    original = torch.nn.Conv2d(4, 8, 3, groups=4).eval()
    adapted = vision_exportable._BatchOneSplitDepthwiseMultiplierConv(original)
    with self.assertRaisesRegex(ValueError, 'batch size one'):
      adapted(torch.randn(2, 4, 7, 7))


if __name__ == '__main__':
  absltest.main()
