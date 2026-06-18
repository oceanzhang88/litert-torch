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
"""Exportable module for externalized rotary embedding."""

import inspect

from litert_torch.generative.export_hf.core import exportable_module as base_exportable_module
from litert_torch.generative.export_hf.core import utils
import torch


class RoPEEmbedder(torch.nn.Module):
  """Exportable module for decoder only LM running on NPU."""

  def __init__(self, model):
    super().__init__()
    self.model = model

    self.rotary_emb = utils.get_decoder_module(
        model, "original_rotary_emb"
    ).original_rotary_emb

    assert self.rotary_emb is not None
    rotary_parameters = inspect.signature(self.rotary_emb.forward).parameters
    self.rotary_uses_seq_len = (
        "seq_len" in rotary_parameters and "position_ids" not in rotary_parameters
    )
    self.has_local_rope = utils.has_local_rope(model)

  def _seq_len_rotary_emb_for_positions(
      self,
      dummy: torch.Tensor,
      input_pos: torch.Tensor,
  ):
    """Returns legacy seq_len-style RoPE tables gathered by absolute position.

    Older Qwen-style rotary modules accept only ``seq_len`` and return cached
    tables starting at position zero. LiteRT-LM split-cache runtime supplies
    absolute positions for prefill/decode through ``input_pos``; using only the
    input length would make decode position N reuse the position-0 RoPE row.
    When the original module exposes cached tables, gather from them by
    ``input_pos`` so the auxiliary RoPE signature preserves runtime position
    semantics. Modules without cache buffers keep the historical length-only
    fallback.
    """

    if hasattr(self.rotary_emb, "cos_cached") and hasattr(
        self.rotary_emb, "sin_cached"
    ):
      positions = input_pos.to(torch.int64)
      cos_cached = self.rotary_emb.cos_cached.to(
          device=input_pos.device, dtype=dummy.dtype
      )
      sin_cached = self.rotary_emb.sin_cached.to(
          device=input_pos.device, dtype=dummy.dtype
      )
      return (
          torch.index_select(cos_cached, 0, positions),
          torch.index_select(sin_cached, 0, positions),
      )
    return self.rotary_emb(dummy, seq_len=input_pos.shape[0])

  def forward(
      self,
      input_pos: torch.Tensor,  # [T]
  ):
    dummy = torch.ones((1, 1, 1), dtype=torch.float32)
    position_ids = input_pos.unsqueeze(0)

    if self.has_local_rope:
      pos_emb = self.rotary_emb(dummy, position_ids, 'full_attention')
      if pos_emb[0].ndim == 3:
        pos_emb = [x.unsqueeze(-2) for x in pos_emb]
      ret = {
          'pos_emb_cos': pos_emb[0],
          'pos_emb_sin': pos_emb[1],
      }
      pos_emb_local = self.rotary_emb(dummy, position_ids, 'sliding_attention')
      if pos_emb_local[0].ndim == 3:
        pos_emb_local = [x.unsqueeze(-2) for x in pos_emb_local]
      ret.update({
          'pos_emb_local_cos': pos_emb_local[0],
          'pos_emb_local_sin': pos_emb_local[1],
      })
    else:
      if self.rotary_uses_seq_len:
        pos_emb = self._seq_len_rotary_emb_for_positions(dummy, input_pos)
      else:
        pos_emb = self.rotary_emb(dummy, position_ids)
      if pos_emb[0].ndim == 2:
        pos_emb = [x.unsqueeze(0).unsqueeze(2) for x in pos_emb]
      elif pos_emb[0].ndim == 3:
        pos_emb = [x.unsqueeze(-2) for x in pos_emb]
      ret = {
          'pos_emb_cos': pos_emb[0],
          'pos_emb_sin': pos_emb[1],
      }
    return ret

  @classmethod
  def get_sample_inputs(
      cls,
      model_config,
      export_config: base_exportable_module.ExportableModuleConfig,
      **kwargs,
  ):
    """Gets sample inputs."""
    del model_config
    sample_inputs = {}
    for prefill_length in export_config.prefill_lengths:
      inputs = {
          'input_pos': torch.arange(prefill_length, dtype=torch.int32),
      }
      sample_inputs[f'prefill_rope_{prefill_length}'] = (inputs, {})
    decode_inputs = {
        'input_pos': torch.arange(1, dtype=torch.int32),
    }
    sample_inputs['decode_rope'] = (decode_inputs, {})
    return sample_inputs
