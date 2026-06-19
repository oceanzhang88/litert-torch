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
"""Patch for LFM2."""

import contextlib
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.export_hf.model_ext.lfm2 import short_conv as short_conv_lib
import torch
from transformers.models.lfm2 import modeling_lfm2


class PatchedLfm2DecoderLayer(modeling_lfm2.Lfm2DecoderLayer):

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings=None,
      attention_mask=None,
      position_ids=None,
      past_key_values=None,
      **kwargs,
  ) -> torch.Tensor:
    residual = hidden_states
    if self.is_attention_layer:
      hidden_states, _ = self.self_attn(
          hidden_states=self.operator_norm(hidden_states),
          position_embeddings=position_embeddings,
          attention_mask=attention_mask,
          position_ids=position_ids,
          past_key_values=past_key_values,
          **kwargs,
      )
    else:
      cache_position = (
          position_ids.squeeze(0) if position_ids is not None else None
      )
      valid_mask = kwargs.get("valid_mask", None)
      hidden_states = self.conv(
          hidden_states=self.operator_norm(hidden_states),
          past_key_values=past_key_values,
          attention_mask=attention_mask,
          cache_position=cache_position,
          valid_mask=valid_mask,
      )
    hidden_states = hidden_states + residual
    hidden_states = hidden_states + self.feed_forward(
        self.ffn_norm(hidden_states)
    )

    return hidden_states


@patches_lib.register_patch(["lfm2"])
@contextlib.contextmanager
def lfm2_litert_patch():
  print("LFM2 patch applied.")
  original_short_conv = modeling_lfm2.Lfm2ShortConv
  modeling_lfm2.Lfm2ShortConv = short_conv_lib.Lfm2ShortConv

  original_decoder_layer = modeling_lfm2.Lfm2DecoderLayer
  modeling_lfm2.Lfm2DecoderLayer = PatchedLfm2DecoderLayer

  try:
    yield
  finally:
    modeling_lfm2.Lfm2ShortConv = original_short_conv
    modeling_lfm2.Lfm2DecoderLayer = original_decoder_layer
