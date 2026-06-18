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
"""Preprocess model for externalized rotary embedding."""

from typing import Any

from litert_torch.generative.export_hf.core import utils
import torch


class RotaryPositionEmbeddingInjector(torch.nn.Module):
  """Injects RotaryPositionEmbedding."""

  data: Any = None

  def __init__(self, config):
    super().__init__()
    self.config = config

  def forward(self, x, position_ids, layer_type=None):
    del x, position_ids

    assert self.data is not None
    if layer_type:
      return self.data[layer_type]
    return self.data


def inject_rotary_position_embedding(model):
  """Injects RotaryPositionEmbedding."""
  decoder_model = utils.get_decoder_module(
      model, "rotary_emb", allow_layers=True
  )
  if hasattr(decoder_model, "rotary_emb"):
    decoder_model.original_rotary_emb = decoder_model.rotary_emb
    decoder_model.rotary_emb = RotaryPositionEmbeddingInjector(
        decoder_model.config
    )
    return model

  first_rotary_emb = None
  shared_rotary_emb = RotaryPositionEmbeddingInjector(decoder_model.config)
  for decoder_layer in decoder_model.layers:
    attention = decoder_layer.self_attn
    if first_rotary_emb is None:
      first_rotary_emb = attention.rotary_emb
    attention.rotary_emb = shared_rotary_emb
  decoder_model.original_rotary_emb = first_rotary_emb
  decoder_model.rotary_emb = shared_rotary_emb
  return model
