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
"""iSTFTNet vocoder modules adapted from StyleTTS2 for LiteRT export."""

# ADAPTED from https://github.com/yl4579/StyleTTS2/blob/main/Modules/istftnet.py
import math
from litert_torch.generative.export_hf.model_ext.kokoro_tts import custom_stft
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


# https://github.com/yl4579/StyleTTS2/blob/main/Modules/utils.py
def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
  return int((kernel_size * dilation - dilation) / 2)


class AdaIN1d(nn.Module):
  """Adaptive Instance Normalization 1D layer."""

  def __init__(self, style_dim, num_features):
    super().__init__()
    # affine should be False, however there's a bug in the old
    # torch.onnx.export (not newer dynamo) that causes the channel dimension to
    # be lost if affine=False. When affine is true, there's additional
    # learnably parameters. This shouldn't really matter setting it to True,
    # since we're in inference mode
    self.norm = nn.InstanceNorm1d(num_features, affine=True)
    self.fc = nn.Linear(style_dim, num_features * 2)

  def forward(self, x, s, mask=None):
    if mask is not None:
      if mask.shape[2] != x.shape[2]:
        ratio = (x.shape[2] + mask.shape[2] - 1) // mask.shape[2]
        if ratio > 1:
          mask = torch.stack([mask] * ratio, dim=-1).view(
              mask.shape[0], mask.shape[1], -1
          )
        if mask.shape[2] < x.shape[2]:
          mask = F.pad(mask, (x.shape[2] - mask.shape[2], 0))
        elif mask.shape[2] > x.shape[2]:
          mask = mask[:, :, : x.shape[2]]
      valid_count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
      x_valid = x * mask
      mean = x_valid.sum(dim=-1, keepdim=True) / valid_count
      var = ((x_valid - mean) * mask).pow(2).sum(
          dim=-1, keepdim=True
      ) / valid_count
      x_norm = (x - mean) / (var + 1e-5).sqrt() * mask
      h = self.fc(s)
      h = h.view(h.size(0), h.size(1), 1)
      gamma, beta = torch.chunk(h, chunks=2, dim=1)
      return ((1 + gamma) * x_norm + beta) * mask
    else:
      x_norm = self.norm(x)
      h = self.fc(s)
      h = h.view(h.size(0), h.size(1), 1)
      gamma, beta = torch.chunk(h, chunks=2, dim=1)
      return (1 + gamma) * x_norm + beta


class AdaINResBlock1(nn.Module):
  """Residual block with AdaIN1d and Snake1D activations."""

  def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
    super().__init__()
    self.convs1 = nn.ModuleList([
        weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                1,
                dilation=dilation[0],
                padding=get_padding(kernel_size, dilation[0]),
            )
        ),
        weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                1,
                dilation=dilation[1],
                padding=get_padding(kernel_size, dilation[1]),
            )
        ),
        weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                1,
                dilation=dilation[2],
                padding=get_padding(kernel_size, dilation[2]),
            )
        ),
    ])
    self.convs1.apply(init_weights)
    self.convs2 = nn.ModuleList([
        weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
        ),
        weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
        ),
        weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
        ),
    ])
    self.convs2.apply(init_weights)
    self.adain1 = nn.ModuleList([
        AdaIN1d(style_dim, channels),
        AdaIN1d(style_dim, channels),
        AdaIN1d(style_dim, channels),
    ])
    self.adain2 = nn.ModuleList([
        AdaIN1d(style_dim, channels),
        AdaIN1d(style_dim, channels),
        AdaIN1d(style_dim, channels),
    ])
    self.alpha1 = nn.ParameterList([
        nn.Parameter(torch.ones(1, channels, 1))
        for _ in range(len(self.convs1))
    ])
    self.alpha2 = nn.ParameterList([
        nn.Parameter(torch.ones(1, channels, 1))
        for _ in range(len(self.convs2))
    ])

  def forward(self, x, s, mask=None):
    for c1, c2, n1, n2, a1, a2 in zip(
        self.convs1,
        self.convs2,
        self.adain1,
        self.adain2,
        self.alpha1,
        self.alpha2,
    ):
      xt = n1(x, s, mask)
      xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
      xt = c1(xt)
      xt = n2(xt, s, mask)
      xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
      xt = c2(xt)
      x = xt + x
    return x


class TorchSTFT(nn.Module):
  """PyTorch native STFT/iSTFT wrapper."""

  def __init__(
      self, filter_length=800, hop_length=200, win_length=800, window='hann'
  ):
    super().__init__()
    self.filter_length = filter_length
    self.hop_length = hop_length
    self.win_length = win_length
    assert window == 'hann', window
    self.window = torch.hann_window(
        win_length, periodic=True, dtype=torch.float32
    )

  def transform(self, input_data):
    forward_transform = torch.stft(
        input_data,
        self.filter_length,
        self.hop_length,
        self.win_length,
        window=self.window.to(input_data.device),
        return_complex=True,
    )
    return torch.abs(forward_transform), torch.angle(forward_transform)

  def inverse(self, magnitude, phase):
    inverse_transform = torch.istft(
        magnitude * torch.exp(phase * 1j),
        self.filter_length,
        self.hop_length,
        self.win_length,
        window=self.window.to(magnitude.device),
    )
    return inverse_transform.unsqueeze(
        -2
    )  # unsqueeze to stay consistent with conv_transpose1d implementation

  def forward(self, input_data):
    self.magnitude, self.phase = self.transform(input_data)
    reconstruction = self.inverse(self.magnitude, self.phase)
    return reconstruction


class SineGen(nn.Module):
  """Definition of sine generator.

  SineGen(samp_rate, harmonic_num = 0,
          sine_amp = 0.1, noise_std = 0.003,
          voiced_threshold = 0,
          flag_for_pulse=False)
  samp_rate: sampling rate in Hz
  harmonic_num: number of harmonic overtones (default 0)
  sine_amp: amplitude of sine-wavefrom (default 0.1)
  noise_std: std of Gaussian noise (default 0.003)
  voiced_thoreshold: F0 threshold for U/V classification (default 0)
  flag_for_pulse: this SinGen is used inside PulseGen (default False)
  Note: when flag_for_pulse is True, the first time step of a voiced
      segment is always sin(torch.pi) or cos(0)
  """

  def __init__(
      self,
      samp_rate,
      upsample_scale,
      harmonic_num=0,
      sine_amp=0.1,
      noise_std=0.003,
      voiced_threshold=0,
      flag_for_pulse=False,
  ):
    super().__init__()
    self.sine_amp = sine_amp
    self.noise_std = noise_std
    self.harmonic_num = harmonic_num
    self.dim = self.harmonic_num + 1
    self.sampling_rate = samp_rate
    self.voiced_threshold = voiced_threshold
    self.flag_for_pulse = flag_for_pulse
    self.upsample_scale = upsample_scale

  def _f02uv(self, f0):
    # generate uv signal
    uv = (f0 > self.voiced_threshold).type(torch.float32)
    return uv

  def _f02sine(self, f0_values):
    """Converts F0 values to sine waveforms.

    Args:
      f0_values: (batchsize, length, dim) where dim indicates fundamental tone
        and overtones.

    Returns:
      Sine waveforms tensor.
    """
    # convert to F0 in rad. The interger part n can be ignored
    # because 2 * torch.pi * n doesn't affect phase
    rad = f0_values / self.sampling_rate
    rad_values = rad - torch.floor(rad)
    # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
    if not self.flag_for_pulse:
      s_scale = int(self.upsample_scale)
      batch_size, l_full, dim = rad_values.shape
      l_down = l_full // s_scale
      rad_down = rad_values[:, : l_down * s_scale, :].view(
          batch_size, l_down, s_scale, dim
      )[:, :, 0, :]
      # Accumulate phase in *cycles* (not radians) and wrap into [0, 1).
      # sin() is 2*pi periodic so wrapping is exact, but it keeps the argument
      # bounded. Without it the accumulator reaches ~2.4e5 rad, far outside the
      # |x| <= 2^16 range where OpenCL guarantees sin() accuracy, which makes
      # GPU output diverge from CPU and corrupts the harmonic excitation.
      cyc_down = torch.cumsum(rad_down, dim=1) * float(s_scale)
      cyc_down = cyc_down - torch.floor(cyc_down)

      # Per-substep increment in cycles. This equals (x1 - x0) / S, but is taken
      # straight from rad_down so it is not affected by the wrap above.
      k = torch.arange(
          s_scale, device=rad_values.device, dtype=rad_values.dtype
      ).view(1, 1, s_scale, 1)
      # Broadcast instead of building S tensors and stacking them: the previous
      # `torch.stack([...for k in range(S)])` lowered to a CONCATENATION with S
      # (=300) inputs, which overflows the OpenCL kernel argument list on
      # Adreno.
      stacked = (
          cyc_down[:, :-1, :].unsqueeze(2) + k * rad_down[:, 1:, :].unsqueeze(2)
      ).reshape(batch_size, (l_down - 1) * s_scale, dim)
      last_block = (
          cyc_down[:, -1:, :].unsqueeze(2)
          + k * rad_down[:, -1:, :].unsqueeze(2)
      ).reshape(batch_size, s_scale, dim)
      phase = torch.cat([stacked, last_block], dim=1)
      phase = phase - torch.floor(phase)
      sines = torch.sin(phase * (2.0 * torch.pi))
    else:
      # If necessary, make sure that the first time step of every
      # voiced segments is sin(pi) or cos(0)
      # This is used for pulse-train generation
      # identify the last time step in unvoiced segments
      uv = self._f02uv(f0_values)
      uv_1 = torch.roll(uv, shifts=-1, dims=1)
      uv_1[:, -1, :] = 1
      u_loc = (uv < 1) * (uv_1 > 0)
      # get the instantanouse phase
      tmp_cumsum = torch.cumsum(rad_values, dim=1)
      # different batch needs to be processed differently
      for idx in range(f0_values.shape[0]):
        temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
        temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
        # stores the accumulation of i.phase within
        # each voiced segments
        tmp_cumsum[idx, :, :] = 0
        tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
      # rad_values - tmp_cumsum: remove the accumulation of i.phase
      # within the previous voiced segment.
      i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
      # get the sines
      sines = torch.cos(i_phase * 2 * torch.pi)
    return sines

  def forward(self, f0):
    """Generates sine_tensor, uv, and noise from F0.

    Args:
      f0: tensor(batchsize=1, length, dim=1), f0 for unvoiced steps should be 0.

    Returns:
      Tuple of (sine_waves, uv, noise).
    """
    # fundamental component
    fn = torch.multiply(
        f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device)
    )
    # generate sine waveforms
    sine_waves = self._f02sine(fn) * self.sine_amp
    # generate uv signal
    uv = self._f02uv(f0)
    # noise: for unvoiced should be similar to sine_amp
    #        std = self.sine_amp/3 -> max value ~ self.sine_amp
    #        for voiced regions is self.noise_std
    noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
    t_arr = torch.arange(
        sine_waves.shape[-1], device=sine_waves.device, dtype=sine_waves.dtype
    ).view(1, 1, -1)
    pseudo_randn = torch.sin(t_arr * 123.456) * torch.cos(t_arr * 789.123)
    noise = noise_amp * pseudo_randn
    # first: set the unvoiced part to 0 by uv
    # then: additive noise
    sine_waves = sine_waves * uv + noise
    return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
  """SourceModule for hn-nsf.

  SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
               add_noise_std=0.003, voiced_threshod=0)
  sampling_rate: sampling_rate in Hz
  harmonic_num: number of harmonic above F0 (default: 0)
  sine_amp: amplitude of sine source signal (default: 0.1)
  add_noise_std: std of additive Gaussian noise (default: 0.003)
      note that amplitude of noise in unvoiced is decided
      by sine_amp
  voiced_threshold: threhold to set U/V given F0 (default: 0)
  Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
  F0_sampled (batchsize, length, 1)
  Sine_source (batchsize, length, 1)
  noise_source (batchsize, length 1)
  uv (batchsize, length, 1)
  """

  def __init__(
      self,
      sampling_rate,
      upsample_scale,
      harmonic_num=0,
      sine_amp=0.1,
      add_noise_std=0.003,
      voiced_threshod=0,
  ):
    super().__init__()
    self.sine_amp = sine_amp
    self.noise_std = add_noise_std
    # to produce sine waveforms
    self.l_sin_gen = SineGen(
        sampling_rate,
        upsample_scale,
        harmonic_num,
        sine_amp,
        add_noise_std,
        voiced_threshod,
    )
    # to merge source harmonics into a single excitation
    self.l_linear = nn.Linear(harmonic_num + 1, 1)
    self.l_tanh = nn.Tanh()

  def forward(self, x):
    """Generates harmonic source, noise source, and uv mask from F0_sampled."""
    # source for harmonic branch
    with torch.no_grad():
      sine_wavs, uv, _ = self.l_sin_gen(x)
    sine_merge = self.l_tanh(self.l_linear(sine_wavs))
    # source for noise branch, in the same shape as uv
    t_arr = torch.arange(uv.shape[-1], device=uv.device, dtype=uv.dtype).view(
        1, 1, -1
    )
    pseudo_randn = torch.sin(t_arr * 123.456) * torch.cos(t_arr * 789.123)
    noise = pseudo_randn * self.sine_amp / 3
    return sine_merge, noise, uv


class F0Upsample1d(nn.Module):
  """Nearest neighbor 1D upsampling via torch.stack to eliminate GATHER_ND for GPU delegation."""

  def __init__(self, scale_factor: int):
    super().__init__()
    self.scale_factor = int(scale_factor)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    b, c, l = x.shape
    return torch.stack([x] * self.scale_factor, dim=-1).view(
        b, c, l * self.scale_factor
    )


class Generator(nn.Module):
  """iSTFTNet generator module."""

  def __init__(
      self,
      style_dim,
      resblock_kernel_sizes,
      upsample_rates,
      upsample_initial_channel,
      resblock_dilation_sizes,
      upsample_kernel_sizes,
      gen_istft_n_fft,
      gen_istft_hop_size,
      disable_complex=False,
  ):
    super().__init__()
    self.num_kernels = len(resblock_kernel_sizes)
    self.num_upsamples = len(upsample_rates)
    self.m_source = SourceModuleHnNSF(
        sampling_rate=24000,
        upsample_scale=math.prod(upsample_rates) * gen_istft_hop_size,
        harmonic_num=8,
        voiced_threshod=10,
    )
    self.f0_upsamp = F0Upsample1d(
        scale_factor=math.prod(upsample_rates) * gen_istft_hop_size
    )
    self.noise_convs = nn.ModuleList()
    self.noise_res = nn.ModuleList()
    self.ups = nn.ModuleList()
    for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
      self.ups.append(
          weight_norm(
              nn.ConvTranspose1d(
                  upsample_initial_channel // (2**i),
                  upsample_initial_channel // (2 ** (i + 1)),
                  k,
                  u,
                  padding=(k - u) // 2,
              )
          )
      )
    self.resblocks = nn.ModuleList()
    for i in range(len(self.ups)):
      ch = upsample_initial_channel // (2 ** (i + 1))
      for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
        self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
      c_cur = upsample_initial_channel // (2 ** (i + 1))
      if i + 1 < len(upsample_rates):
        stride_f0 = math.prod(upsample_rates[i + 1 :])
        self.noise_convs.append(
            nn.Conv1d(
                gen_istft_n_fft + 2,
                c_cur,
                kernel_size=stride_f0 * 2,
                stride=stride_f0,
                padding=(stride_f0 + 1) // 2,
            )
        )
        self.noise_res.append(AdaINResBlock1(c_cur, 7, [1, 3, 5], style_dim))
      else:
        self.noise_convs.append(
            nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1)
        )
        self.noise_res.append(AdaINResBlock1(c_cur, 11, [1, 3, 5], style_dim))
    self.post_n_fft = gen_istft_n_fft
    ch_post = upsample_initial_channel // (2 ** len(upsample_rates))
    self.conv_post = weight_norm(
        nn.Conv1d(ch_post, self.post_n_fft + 2, 7, 1, padding=3)
    )
    self.ups.apply(init_weights)
    self.conv_post.apply(init_weights)
    self.reflection_pad = nn.ConstantPad1d((1, 0), 0.0)
    if disable_complex:
      self.stft = custom_stft.CustomSTFT(
          filter_length=gen_istft_n_fft,
          hop_length=gen_istft_hop_size,
          win_length=gen_istft_n_fft,
      )
    else:
      self.stft = TorchSTFT(
          filter_length=gen_istft_n_fft,
          hop_length=gen_istft_hop_size,
          win_length=gen_istft_n_fft,
      )

  def forward(self, x, s, har, mask=None):
    for i in range(self.num_upsamples):
      x = F.leaky_relu(x, negative_slope=0.1)
      x_source = self.noise_convs[i](har)
      x_source = self.noise_res[i](x_source, s, mask)
      x = self.ups[i](x)
      if i == self.num_upsamples - 1:
        x = F.pad(x, (1, 0))
      x = x + x_source
      xs = sum(
          self.resblocks[i * self.num_kernels + j](x, s, mask)
          for j in range(self.num_kernels)
      )
      x = xs / self.num_kernels

    x = F.leaky_relu(x)
    x = self.conv_post(x)
    spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
    phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
    if getattr(self, 'return_spec_phase', False):
      return spec, phase
    return self.stft.inverse(spec, phase)


class UpSample1d(nn.Module):
  """1D nearest-neighbor upsampling module."""

  def __init__(self, layer_type):
    super().__init__()
    self.layer_type = layer_type

  def forward(self, x):
    if self.layer_type == 'none':
      return x
    else:
      b, c, l = x.shape
      return torch.stack([x, x], dim=-1).view(b, c, l * 2)


class InterleavedConvTranspose1d(nn.Module):
  """Lowers 1D transposed convolution to zero-interleaving + Conv1d.

  Ensures 100% full OpenCL GPU delegation without TransposeConv texture shape
  mismatches, runtime weight flipping (GatherNd), or slice_scatter (SelectV2).
  """

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride,
      padding=0,
      groups=1,
      bias=True,
  ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.groups = groups
    self.pad_left = kernel_size - 1 - padding
    self.pad_right = padding
    self.conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=groups,
        bias=bias,
    )

  def forward(self, x):
    b, c, l = x.shape
    if self.stride == 1:
      x_inter = x
    elif self.stride == 2:
      x_zeros = torch.zeros_like(x)
      x_inter = torch.stack([x, x_zeros], dim=-1).view(b, c, l * 2)
    else:
      zeros = torch.zeros(
          (b, c, l, self.stride - 1), device=x.device, dtype=x.dtype
      )
      x_inter = torch.cat([x.unsqueeze(-1), zeros], dim=-1).view(
          b, c, l * self.stride
      )
    if self.pad_left > 0 or self.pad_right > 0:
      x_inter = F.pad(x_inter, (self.pad_left, self.pad_right))
    return self.conv(x_inter)


class AdainResBlk1d(nn.Module):
  """AdaIN 1D residual block with optional upsampling."""

  def __init__(
      self,
      dim_in,
      dim_out,
      style_dim=64,
      actv=nn.LeakyReLU(0.2),
      upsample='none',
      dropout_p=0.0,
  ):
    super().__init__()
    self.actv = actv
    self.upsample_type = upsample
    self.upsample = UpSample1d(upsample)
    self.learned_sc = dim_in != dim_out
    self._build_weights(dim_in, dim_out, style_dim)
    self.dropout = nn.Dropout(dropout_p)
    if upsample == 'none':
      self.pool = nn.Identity()
    else:
      self.pool = weight_norm(
          nn.ConvTranspose1d(
              dim_in,
              dim_in,
              kernel_size=3,
              stride=2,
              groups=dim_in,
              padding=1,
              output_padding=1,
          )
      )

  def _build_weights(self, dim_in, dim_out, style_dim):
    self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
    self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
    self.norm1 = AdaIN1d(style_dim, dim_in)
    self.norm2 = AdaIN1d(style_dim, dim_out)
    if self.learned_sc:
      self.conv1x1 = weight_norm(
          nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)
      )

  def _shortcut(self, x):
    x = self.upsample(x)
    if self.learned_sc:
      x = self.conv1x1(x)
    return x

  def _residual(self, x, s, mask=None):
    x = self.norm1(x, s, mask)
    x = self.actv(x)
    x = self.pool(x)
    x = self.conv1(self.dropout(x))
    x = self.norm2(x, s, mask)
    x = self.actv(x)
    x = self.conv2(self.dropout(x))
    return x

  def forward(self, x, s, mask=None):
    out = self._residual(x, s, mask)
    out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2.0))
    return out


class Decoder(nn.Module):
  """Kokoro iSTFTNet Decoder module."""

  def __init__(
      self,
      dim_in,
      style_dim,
      dim_out,
      resblock_kernel_sizes,
      upsample_rates,
      upsample_initial_channel,
      resblock_dilation_sizes,
      upsample_kernel_sizes,
      gen_istft_n_fft,
      gen_istft_hop_size,
      disable_complex=False,
  ):
    super().__init__()
    del dim_out
    self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
    self.decode = nn.ModuleList()
    self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
    self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
    self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
    self.decode.append(
        AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True)
    )
    self.F0_conv = weight_norm(  # pylint: disable=invalid-name
        nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
    )
    self.N_conv = weight_norm(  # pylint: disable=invalid-name
        nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
    )
    self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(512, 64, kernel_size=1)))
    self.generator = Generator(
        style_dim,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
        disable_complex=disable_complex,
    )

  def forward(  # pylint: disable=invalid-name
      self,
      asr,
      F0_curve,
      N,
      s,
      harmonic_features=None,
      l_speech=None,
  ):
    if l_speech is not None:
      time_f = torch.arange(
          asr.shape[-1], device=asr.device, dtype=torch.float32
      ).unsqueeze(0)
      mask = (time_f < l_speech.float().reshape(-1, 1)).float().unsqueeze(1)
    else:
      mask = None

    if harmonic_features is None:
      f0 = self.generator.f0_upsamp(F0_curve[:, None]).transpose(1, 2)
      har_source, _, _ = self.generator.m_source(f0)
      har_source = har_source.transpose(1, 2).squeeze(1)
      har_spec, har_phase = self.generator.stft.transform(har_source)
      harmonic_features = torch.cat([har_spec, har_phase], dim=1)

    f0_down = self.F0_conv(F0_curve.unsqueeze(1))
    n_down = self.N_conv(N.unsqueeze(1))
    x = torch.cat([asr, f0_down, n_down], dim=1)
    x = self.encode(x, s, mask)
    asr_res = self.asr_res(asr)
    res = True
    for block in self.decode:
      if res:
        x = torch.cat([x, asr_res, f0_down, n_down], dim=1)
      x = block(x, s, mask)
      if block.upsample_type != 'none':
        res = False
    x = self.generator(x, s, harmonic_features, mask)
    return x
