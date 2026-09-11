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
"""Kokoro KModel definition."""

import copy
import dataclasses
import json
import logging as logger
from typing import Any

import huggingface_hub
from litert_torch.generative.export_hf.model_ext.kokoro_tts import istftnet
from litert_torch.generative.export_hf.model_ext.kokoro_tts import modules
import torch
import transformers


class KModel(torch.nn.Module):
  """Kokoro TTS PyTorch model.

  KModel has 2 main responsibilities:
  1. Init weights, downloading config.json + model.pth from HF if needed.
  2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

  You likely only need one KModel instance, and it can be reused across
  multiple KPipelines to avoid redundant memory allocation.

  Unlike KPipeline, KModel is language-blind.

  KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
  so there is no need to repeatedly download config.json outside of KModel.
  """

  MODEL_NAMES = {
      'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
      'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
  }

  DEFAULT_CONFIG: dict[str, Any] = {
      'vocab': {'a': 1, 'b': 2},
      'n_token': 178,
      'hidden_dim': 512,
      'style_dim': 128,
      'n_layer': 3,
      'max_dur': 50,
      'dropout': 0.2,
      'text_encoder_kernel_size': 5,
      'n_mels': 80,
      'plbert': {
          'hidden_size': 768,
          'num_attention_heads': 12,
          'intermediate_size': 2048,
          'max_position_embeddings': 512,
          'num_hidden_layers': 12,
          'dropout': 0.1,
      },
      'istftnet': {
          'upsample_rates': [10, 6],
          'upsample_kernel_sizes': [20, 12],
          'gen_istft_n_fft': 20,
          'gen_istft_hop_size': 5,
          'resblock_kernel_sizes': [3, 7, 11],
          'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
          'upsample_initial_channel': 512,
      },
  }

  def __init__(
      self,
      repo_id: str | None = None,
      config: dict[str, Any] | str | None = None,
      model: str | None = None,
      disable_complex: bool = False,
      use_random_weights: bool = False,
  ):
    super().__init__()
    if repo_id is None:
      repo_id = 'hexgrad/Kokoro-82M'
      print(
          f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}'"
          ' to suppress this warning.'
      )
    self.repo_id = repo_id
    if not isinstance(config, dict):
      if not config:
        if use_random_weights:
          config = copy.deepcopy(self.DEFAULT_CONFIG)
        else:
          logger.debug('No config provided, downloading from HF')
          config = huggingface_hub.hf_hub_download(
              repo_id=repo_id, filename='config.json'
          )
      if isinstance(config, str):
        with open(config, 'r', encoding='utf-8') as r:
          config = json.load(r)
          logger.debug('Loaded config: %s', config)
    self.vocab = config['vocab']
    self.bert = modules.CustomAlbert(
        transformers.AlbertConfig(
            vocab_size=config['n_token'], **config['plbert']
        )
    )
    self.bert_encoder = torch.nn.Linear(
        self.bert.config.hidden_size, config['hidden_dim']
    )
    self.context_length = self.bert.config.max_position_embeddings
    self.predictor = modules.ProsodyPredictor(
        style_dim=config['style_dim'],
        d_hid=config['hidden_dim'],
        nlayers=config['n_layer'],
        max_dur=config['max_dur'],
        dropout=config['dropout'],
    )
    self.text_encoder = modules.TextEncoder(
        channels=config['hidden_dim'],
        kernel_size=config['text_encoder_kernel_size'],
        depth=config['n_layer'],
        n_symbols=config['n_token'],
    )
    self.decoder = istftnet.Decoder(
        dim_in=config['hidden_dim'],
        style_dim=config['style_dim'],
        dim_out=config['n_mels'],
        disable_complex=disable_complex,
        **config['istftnet'],
    )
    if not use_random_weights:
      if not model:
        model = huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id]
        )
      for key, state_dict in torch.load(
          model, map_location='cpu', weights_only=True
      ).items():
        assert hasattr(self, key), key
        try:
          getattr(self, key).load_state_dict(state_dict)
        except RuntimeError:
          logger.debug('Did not load %s from state_dict', key)
          state_dict = {k[7:]: v for k, v in state_dict.items()}
          getattr(self, key).load_state_dict(state_dict, strict=False)

  @property
  def device(self):
    return self.bert.device

  @dataclasses.dataclass
  class Output:
    audio: torch.FloatTensor
    pred_dur: torch.LongTensor | None = None

  @torch.no_grad()
  def forward_with_tokens(
      self,
      input_ids: torch.LongTensor,
      ref_s: torch.FloatTensor,
      speed: float = 1,
  ) -> tuple[torch.FloatTensor, torch.LongTensor]:
    input_lengths = torch.full(
        (input_ids.shape[0],),
        input_ids.shape[-1],
        device=input_ids.device,
        dtype=torch.long,
    )

    text_mask = (
        torch.arange(input_lengths.max())
        .unsqueeze(0)
        .expand(input_lengths.shape[0], -1)
        .type_as(input_lengths)
    )
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(
        self.device
    )
    bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x = self.predictor.lstm(d, input_lengths)
    duration = self.predictor.duration_proj(x)

    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
    indices = torch.repeat_interleave(
        torch.arange(input_ids.shape[1], device=self.device), pred_dur
    )
    pred_aln_trg = torch.zeros(
        (input_ids.shape[1], indices.shape[0]), device=self.device
    )
    pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
    en = d.transpose(-1, -2) @ pred_aln_trg
    f0_pred, n_pred = self.predictor.F0Ntrain(en, s)
    t_en = self.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg
    audio = self.decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze()
    return audio, pred_dur

  def forward(
      self,
      phonemes: str,
      ref_s: torch.FloatTensor,
      speed: float = 1,
      return_output: bool = False,
  ) -> 'KModel.Output | torch.FloatTensor':
    input_ids = [
        self.vocab[p]
        for p in phonemes
        if p in self.vocab and self.vocab[p] is not None
    ]
    logger.debug('phonemes: %s -> input_ids: %s', phonemes, input_ids)
    assert len(input_ids) + 2 <= self.context_length, (
        len(input_ids) + 2,
        self.context_length,
    )
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
    ref_s = ref_s.to(self.device)
    audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
    audio = audio.squeeze().cpu()
    pred_dur = pred_dur.cpu() if pred_dur is not None else None
    logger.debug('pred_dur: %s', pred_dur)
    return (
        self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio
    )


class KModelForONNX(torch.nn.Module):
  """Wrapper around KModel for ONNX export."""

  def __init__(self, kmodel: KModel):
    super().__init__()
    self.kmodel = kmodel

  def forward(
      self,
      input_ids: torch.LongTensor,
      ref_s: torch.FloatTensor,
      speed: float = 1,
  ) -> tuple[torch.FloatTensor, torch.LongTensor]:
    waveform, duration = self.kmodel.forward_with_tokens(
        input_ids, ref_s, speed
    )
    return waveform, duration
