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
"""Kokoro acoustic, prosody, and duration predictor modules."""

# https://github.com/yl4579/StyleTTS2/blob/main/models.py
from litert_torch.generative.export_hf.model_ext.kokoro_tts import istftnet
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import transformers


class LinearNorm(nn.Module):
  """Linear layer with Xavier uniform initialization."""

  def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
    super().__init__()
    self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.xavier_uniform_(
        self.linear_layer.weight,
        gain=nn.init.calculate_gain(w_init_gain),  # pyrefly: ignore[bad-argument-type]
    )

  def forward(self, x):
    return self.linear_layer(x)


class LayerNorm(nn.Module):
  """Channel-first 1D LayerNorm."""

  def __init__(self, channels, eps=1e-5):
    super().__init__()
    self.channels = channels
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(channels))
    self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    x = x.transpose(1, -1)
    x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
    return x.transpose(1, -1)


class CustomFusedLSTM(nn.Module):
  """Single-layer unidirectional fused LSTM with precomputed input projection."""

  def __init__(self, input_size: int, hidden_size: int):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
    self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
    self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
    self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1.0 / (self.hidden_size**0.5) if self.hidden_size > 0 else 0.0
    for weight in self.parameters():
      nn.init.uniform_(weight, -stdv, stdv)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, D]
    b, t_steps, _ = x.shape
    h_dim = self.hidden_size
    # 1 single BMM for all T steps:
    x_proj = F.linear(
        x, self.weight_ih, self.bias_ih + self.bias_hh
    )  # [B, T, 4*H]

    h = torch.zeros((b, h_dim), dtype=x.dtype, device=x.device)
    c = torch.zeros((b, h_dim), dtype=x.dtype, device=x.device)

    x_proj_steps = torch.unbind(x_proj, dim=1)
    h_list = []
    for t in range(t_steps):
      # 1 single recurrent linear transformation per step:
      gates = x_proj_steps[t] + F.linear(h, self.weight_hh)  # [B, 4*H]
      i, f, g, o = gates.chunk(4, dim=-1)
      i = torch.sigmoid(i)
      f = torch.sigmoid(f)
      g = torch.tanh(g)
      o = torch.sigmoid(o)
      c = f * c + i * g
      h = o * torch.tanh(c)
      h_list.append(h)

    return torch.stack(h_list, dim=1)


class BidirectionalFusedLSTM(nn.Module):
  """Bidirectional wrapper around CustomFusedLSTM."""

  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.lstm_fwd = CustomFusedLSTM(input_size, hidden_size)
    self.lstm_bwd = CustomFusedLSTM(input_size, hidden_size)
    self.hidden_size = hidden_size

  def _load_from_state_dict(
      self,
      state_dict,
      prefix,
      local_metadata,
      strict,
      missing_keys,
      unexpected_keys,
      error_msgs,
  ):
    for name in ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"]:
      fwd_key = prefix + name
      rev_key = prefix + name + "_reverse"
      mapped_name = name.removesuffix("_l0")
      if fwd_key in state_dict:
        state_dict[prefix + "lstm_fwd." + mapped_name] = state_dict.pop(fwd_key)
      if rev_key in state_dict:
        state_dict[prefix + "lstm_bwd." + mapped_name] = state_dict.pop(rev_key)
    super()._load_from_state_dict(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    )

  def forward(self, x, seq_lengths, mask=None):
    t_steps = x.shape[1]
    time_arange = torch.arange(t_steps, device=x.device)
    idx_b = (seq_lengths.reshape(-1, 1) - 1 - time_arange.unsqueeze(0)).clamp(
        min=0, max=t_steps - 1
    )

    x_fwd = self.lstm_fwd(x)

    x_b = torch.gather(x, 1, idx_b.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    if mask is not None:
      x_b = x_b * (1.0 - mask.float().unsqueeze(-1))

    x_bwd_b = self.lstm_bwd(x_b)

    x_bwd = torch.gather(
        x_bwd_b, 1, idx_b.unsqueeze(-1).expand(-1, -1, self.hidden_size)
    )
    if mask is not None:
      x_bwd = x_bwd * (1.0 - mask.float().unsqueeze(-1))

    x_out = torch.cat([x_fwd, x_bwd], dim=-1)
    return x_out


class TextEncoder(nn.Module):
  """Kokoro CNN + BiLSTM text encoder."""

  def __init__(
      self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)
  ):
    super().__init__()
    self.embedding = nn.Embedding(n_symbols, channels)
    padding = (kernel_size - 1) // 2
    self.cnn = nn.ModuleList()
    for _ in range(depth):
      self.cnn.append(
          nn.Sequential(
              weight_norm(
                  nn.Conv1d(
                      channels,
                      channels,
                      kernel_size=kernel_size,
                      padding=padding,
                  )
              ),
              LayerNorm(channels),
              actv,
              nn.Dropout(0.2),
          )
      )
    self.lstm = BidirectionalFusedLSTM(channels, channels // 2)

  def forward(self, x, input_lengths, m):
    x = self.embedding(x)  # [B, T, emb]
    x = x.transpose(1, 2)  # [B, emb, T]
    m_inv = (1.0 - m.float()).unsqueeze(1)
    x = x * m_inv
    for c in self.cnn:
      x = c(x)
      x = x * m_inv
    x = x.transpose(1, 2)  # [B, T, chn]
    x = self.lstm(x, input_lengths, mask=m)
    x = x.transpose(-1, -2)
    x = x * m_inv
    return x


class AdaLayerNorm(nn.Module):
  """Adaptive Layer Normalization conditioned on speaker style."""

  def __init__(self, style_dim, channels, eps=1e-5):
    super().__init__()
    self.channels = channels
    self.eps = eps
    self.fc = nn.Linear(style_dim, channels * 2)

  def forward(self, x, s):
    x = x.transpose(-1, -2)
    x = x.transpose(1, -1)
    h = self.fc(s)
    h = h.view(h.size(0), h.size(1), 1)
    gamma, beta = torch.chunk(h, chunks=2, dim=1)
    gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
    x = F.layer_norm(x, (self.channels,), eps=self.eps)
    x = (1 + gamma) * x + beta
    return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):
  """Duration, F0, and energy prosody predictor."""

  # pylint: disable=invalid-name
  def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
    super().__init__()
    self.text_encoder = DurationEncoder(
        sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
    )
    self.lstm = BidirectionalFusedLSTM(d_hid + style_dim, d_hid // 2)
    self.duration_proj = LinearNorm(d_hid, max_dur)
    self.shared = BidirectionalFusedLSTM(d_hid + style_dim, d_hid // 2)
    self.F0 = nn.ModuleList()
    self.F0.append(
        istftnet.AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout)
    )
    self.F0.append(
        istftnet.AdainResBlk1d(
            d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
        )
    )
    self.F0.append(
        istftnet.AdainResBlk1d(
            d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout
        )
    )
    self.N = nn.ModuleList()
    self.N.append(
        istftnet.AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout)
    )
    self.N.append(
        istftnet.AdainResBlk1d(
            d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
        )
    )
    self.N.append(
        istftnet.AdainResBlk1d(
            d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout
        )
    )
    self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
    self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

  def forward(self, texts, style, text_lengths, alignment, m):
    d = self.text_encoder(texts, style, text_lengths, m)
    x = self.lstm(d, text_lengths, mask=m)
    duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=False))
    en = d.transpose(-1, -2) @ alignment
    return duration.squeeze(-1), en

  def F0Ntrain(self, x, s):
    x = self.shared(
        x.transpose(-1, -2),
        torch.full(
            (x.shape[0],), x.shape[-1], device=x.device, dtype=torch.long
        ),
    )

    f0 = x.transpose(-1, -2)
    for block in self.F0:
      f0 = block(f0, s)
    f0 = self.F0_proj(f0)
    n_pred = x.transpose(-1, -2)
    for block in self.N:
      n_pred = block(n_pred, s)
    n_pred = self.N_proj(n_pred)
    return f0.squeeze(1), n_pred.squeeze(1)

  # pylint: enable=invalid-name


class DurationEncoder(nn.Module):
  """Duration encoder stacking BidirectionalFusedLSTM and AdaLayerNorm."""

  def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
    super().__init__()
    self.lstms = nn.ModuleList()
    for _ in range(nlayers):
      self.lstms.append(BidirectionalFusedLSTM(d_model + sty_dim, d_model // 2))
      self.lstms.append(AdaLayerNorm(sty_dim, d_model))
    self.dropout = dropout
    self.d_model = d_model
    self.sty_dim = sty_dim

  def forward(self, x, style, text_lengths, m):
    m_inv_01 = (1.0 - m.float()).unsqueeze(-1).transpose(0, 1)
    m_inv_12 = (1.0 - m.float()).unsqueeze(-1).transpose(-1, -2)
    x = x.permute(2, 0, 1)
    s = style.expand(x.shape[0], x.shape[1], -1)
    x = torch.cat([x, s], dim=-1)
    x = x * m_inv_01
    x = x.transpose(0, 1)
    x = x.transpose(-1, -2)
    for block in self.lstms:
      if isinstance(block, AdaLayerNorm):
        x = block(x.transpose(-1, -2), style).transpose(-1, -2)
        x = torch.cat([x, s.permute(1, 2, 0)], dim=1)
        x = x * m_inv_12
      else:
        x = x.transpose(-1, -2)
        x = block(x, text_lengths, mask=m)
        x = F.dropout(x, p=self.dropout, training=False)
        x = x.transpose(-1, -2)

    return x.transpose(-1, -2)


# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
class CustomAlbert(transformers.AlbertModel):
  """Custom ALBERT wrapper returning last_hidden_state."""

  def forward(self, *args, **kwargs):
    outputs = super().forward(*args, **kwargs)
    return outputs.last_hidden_state
