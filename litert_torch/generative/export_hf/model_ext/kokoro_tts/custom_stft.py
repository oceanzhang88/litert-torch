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
"""Custom STFT/iSTFT implementation using 1D convolutions for LiteRT export."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CustomSTFT(nn.Module):
  """STFT/iSTFT without unfold/complex ops, using conv1d and conv_transpose1d.

  - forward STFT => Real-part conv1d + Imag-part conv1d
  - inverse STFT => Real-part conv_transpose1d + Imag-part conv_transpose1d +
  sum
  - avoids F.unfold, so easier to export to ONNX / TFLite
  - uses replicate or constant padding for 'center=True' to approximate
  'reflect'
    (reflect is not supported for dynamic shapes in ONNX)
  """

  def __init__(
      self,
      filter_length=800,
      hop_length=200,
      win_length=800,
      window="hann",
      center=True,
      pad_mode="constant",
  ):
    super().__init__()
    self.filter_length = filter_length
    self.hop_length = hop_length
    self.win_length = win_length
    self.n_fft = filter_length
    self.center = center
    self.pad_mode = pad_mode

    # Number of frequency bins for real-valued STFT with onesided=True
    self.freq_bins = self.n_fft // 2 + 1

    # Build window
    assert window == "hann", window
    window_tensor = torch.hann_window(
        win_length, periodic=True, dtype=torch.float32
    )
    if self.win_length < self.n_fft:
      # Zero-pad up to n_fft
      extra = self.n_fft - self.win_length
      window_tensor = F.pad(window_tensor, (0, extra))
    elif self.win_length > self.n_fft:
      window_tensor = window_tensor[: self.n_fft]
    self.register_buffer("window", window_tensor)

    # Precompute forward DFT (real, imag)
    # PyTorch stft uses e^{-j 2 pi k n / N} => real=cos(...), imag=-sin(...)
    n = np.arange(self.n_fft)
    k = np.arange(self.freq_bins)
    angle = 2 * np.pi * np.outer(k, n) / self.n_fft  # shape (freq_bins, n_fft)
    dft_real = np.cos(angle)
    dft_imag = -np.sin(angle)  # note negative sign

    # Combine window and dft => shape (freq_bins, filter_length)
    # We'll make 2 conv weight tensors of shape (freq_bins, 1, filter_length).
    forward_window = window_tensor.numpy()  # shape (n_fft,)
    forward_real = dft_real * forward_window  # (freq_bins, n_fft)
    forward_imag = dft_imag * forward_window

    # Convert to PyTorch
    forward_real_torch = torch.from_numpy(forward_real).float()
    forward_imag_torch = torch.from_numpy(forward_imag).float()

    # Register as Conv1d weight => (out_channels, in_channels, kernel_size)
    # out_channels = freq_bins, in_channels=1, kernel_size=n_fft
    self.register_buffer("weight_forward_real", forward_real_torch.unsqueeze(1))
    self.register_buffer("weight_forward_imag", forward_imag_torch.unsqueeze(1))

    # Precompute inverse DFT
    # Real iFFT formula => scale = 1/n_fft, doubling for bins 1..freq_bins-2 if
    # n_fft even.
    bin_scale = np.ones(self.freq_bins, dtype=np.float32)
    bin_scale[1:-1] = 2.0

    ola_sum = 1.5 * (self.n_fft / 4.0 / self.hop_length)
    inv_scale = (1.0 / self.n_fft) / ola_sum
    n = np.arange(self.n_fft)
    angle_t = (
        2 * np.pi * np.outer(n, k) / self.n_fft
    )  # shape (n_fft, freq_bins)
    idft_cos = np.cos(angle_t).T  # => (freq_bins, n_fft)
    idft_sin = np.sin(angle_t).T  # => (freq_bins, n_fft)

    # Multiply by window again for typical overlap-add
    # We also incorporate the scale factor 1/n_fft and OLA envelope division
    inv_window = window_tensor.numpy() * inv_scale
    backward_real = (
        idft_cos * bin_scale[:, None]
    ) * inv_window  # (freq_bins, n_fft)
    backward_imag = (idft_sin * bin_scale[:, None]) * inv_window

    # We'll implement iSTFT as real+imag conv_transpose with stride=hop.
    self.register_buffer(
        "weight_backward_real",
        torch.from_numpy(backward_real).float().unsqueeze(1),
    )
    self.register_buffer(
        "weight_backward_imag",
        torch.from_numpy(backward_imag).float().unsqueeze(1),
    )

  def transform(self, waveform: torch.Tensor):
    """Computes forward STFT and returns (magnitude, phase) of shape (B, freq_bins, frames)."""
    # waveform shape => (B, T) or (B, 1, T). conv1d expects (B, 1, T).
    x = waveform.unsqueeze(1) if waveform.dim() == 2 else waveform
    # Optional center pad on 3D tensor to produce 3x2 padding for GPU delegate
    if self.center:
      pad_len = self.n_fft // 2
      x = F.pad(x, (pad_len, pad_len), mode=self.pad_mode)
    # Convolution to get real part => shape (B, freq_bins, frames)
    real_out = F.conv1d(
        x,
        self.weight_forward_real,
        bias=None,
        stride=self.hop_length,
        padding=0,
    )
    # Imag part
    imag_out = F.conv1d(
        x,
        self.weight_forward_imag,
        bias=None,
        stride=self.hop_length,
        padding=0,
    )

    # magnitude, phase
    magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)
    # High-precision polynomial atan2 approximation for full GPU delegation.
    ay = torch.abs(imag_out)
    ax = torch.abs(real_out)
    eps = 1e-12
    invert = ay > ax
    # Use min/max rather than `where(invert, ax/ay, ay/ax)`. The polynomial
    # below is only valid for r in [0, 1]; the `where` form evaluates *both*
    # ratios, and the unselected one can be enormous (measured up to 3e11 when
    # the other component is ~0). If `invert` and the selected ratio are
    # computed with even sub-ULP differences -- as happens on GPU whenever
    # ay ~= ax -- the branch can pick a ratio > 1, and the odd polynomial then
    # diverges wildly. min/max keeps r in [0, 1] by construction, so a flipped
    # comparison is harmless (at ay == ax both quadrant branches agree on pi/4).
    amin = torch.minimum(ay, ax)
    amax = torch.maximum(ay, ax)
    r = amin / (amax + eps)
    c1 = 0.99997726
    c3 = -0.33262347
    c5 = 0.19354346
    c7 = -0.11643287
    c9 = 0.05265332
    c11 = -0.01172120
    r2 = r * r
    angle = r * (c1 + r2 * (c3 + r2 * (c5 + r2 * (c7 + r2 * (c9 + r2 * c11)))))
    pi = 3.141592653589793
    angle = torch.where(invert, 0.5 * pi - angle, angle)
    angle = torch.where(real_out < 0.0, pi - angle, angle)
    phase = torch.where(imag_out < 0.0, -angle, angle)
    return magnitude, phase

  def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, length=None):
    """Inverse STFT => returns waveform shape (B, T)."""
    # magnitude, phase => (B, freq_bins, frames)
    # Re-create real/imag => shape (B, freq_bins, frames)
    real_part = magnitude * torch.cos(phase)
    imag_part = magnitude * torch.sin(phase)

    # real iSTFT => convolve with "backward_real", "backward_imag", and sum
    # We'll do 2 conv_transpose calls, each giving (B, 1, time),
    # then add them => (B, 1, time).
    real_rec = F.conv_transpose1d(
        real_part,
        self.weight_backward_real,  # shape (freq_bins, 1, filter_length)
        bias=None,
        stride=self.hop_length,
        padding=0,
    )
    imag_rec = F.conv_transpose1d(
        imag_part,
        self.weight_backward_imag,
        bias=None,
        stride=self.hop_length,
        padding=0,
    )
    # sum => (B, 1, time)
    waveform = (
        real_rec - imag_rec
    )  # typical real iFFT has minus for imaginary part

    # If we used "center=True" in forward, we should remove pad
    if self.center:
      pad_len = self.n_fft // 2
      # Because of transposed convolution, total length might have extra samples
      # We remove `pad_len` from start & end if possible
      waveform = waveform[..., pad_len:-pad_len]

    # If a specific length is desired, clamp
    if length is not None:
      waveform = waveform[..., :length]

    # shape => (B, T)
    return waveform

  def forward(self, x: torch.Tensor):
    """Full STFT -> iSTFT pass: returns time-domain reconstruction."""
    mag, phase = self.transform(x)
    return self.inverse(mag, phase, length=x.shape[-1])
