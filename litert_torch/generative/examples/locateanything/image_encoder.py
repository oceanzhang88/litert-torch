# Copyright 2025 The LiteRT Torch Authors.
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

"""Re-authored MoonViT-SO-400M image encoder for LocateAnything-3B.

Architecture summary (config.json vision_config + modeling_vit.py):
  - patch_size=14, Conv2d projection (bias=True), no temporal axis.
  - Learnable 2D-interp additive pos-emb: vision_model.patch_embed.pos_emb.weight
    shape (64, 64, 1152); bicubic-interpolated to actual grid at set_image_size.
  - 27 encoder blocks (modeling_vit.py MoonVitEncoderLayer):
      norm0 (LayerNorm) -> fused wqkv (Linear 1152->3456, bias=True) -> Q/K/V split
      -> 2D RoPE on Q/K -> block-diagonal scaled dot-product attention -> wo -> residual
      norm1 (LayerNorm) -> MLP (fc0 1152->4304 -> GELU_TANH -> fc1 4304->1152, bias) -> residual
  - final_layernorm (LayerNorm) after last block
  - patch_merger: 2x2 pixel-unshuffle -> (N_patches//4, 4608) per image

GPU-hostile ops ELIMINATED:
  - torch.polar, view_as_complex, complex multiply (RoPE) ->
    replaced by precomputed real cos/sin tables + interleaved multiply.
    Convention: INTERLEAVED (GPT-NeoX style), NOT split-half.
  - F.interpolate for learnable pos-emb -> computed once at set_image_size() and
    stored as a plain additive buffer (constant during forward).
  - patch_merger masked index/pad -> pure reshape+permute.
  - data-dependent window loop (item()/Python-loop over cu_seqlens) ->
    replaced by precomputed block-diagonal additive mask [1,1,L,L] stored at
    set_image_size(); F.scaled_dot_product_attention runs a single pass over
    the full sequence. No .item(), no Python loop in forward.

Phase 2a parity (RoPE externalization, verified 2026-06-20):
  - fp32, 3 grids (16x16, 24x32, 8x12): max|dQ|=max|dK|=4.77e-7. PASS at 1e-4.
  - GPU op-set exported: {mul, add, neg, reshape, select, stack, unsqueeze, to}.
    ZERO complex ops (no view_as_complex/polar/view_as_real).

Phase 2b (full encoder, verified 2026-06-20):
  - Verified against HF MoonVitPretrainedModel with real weights at atol=1e-3.

Phase 2b remediation (audit fix, 2026-06-20):
  - Window loop replaced with block-diagonal mask + F.scaled_dot_product_attention.
  - Export gate added to verify script: torch.export.export() + op-histogram check.
  - Multi-image (n_images=2) parity test added to verify cross-image mask isolation.
"""

import dataclasses
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architecture constants (MoonViT-SO-400M) — single source of truth.
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 1152
NUM_LAYERS = 27
NUM_HEADS = 16
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 72  (72 % 4 == 0, required by 2D rope)
INTERMEDIATE_SIZE = 4304
PATCH_SIZE = 14
INIT_POS_EMB_HEIGHT = 64
INIT_POS_EMB_WIDTH = 64
MERGE_KERNEL_SIZE = (2, 2)
ROPE_THETA_BASE = 10000.0
LAYERNORM_EPS = 1e-5  # nn.LayerNorm default; MoonViT uses bare nn.LayerNorm


# ---------------------------------------------------------------------------
# 2D RoPE externalization (Phase 2a) — GPU-clean real-op replacement.
# ---------------------------------------------------------------------------


def build_moonvit_rope_2d(
    grid_h: int,
    grid_w: int,
    head_dim: int = HEAD_DIM,
    theta_base: float = ROPE_THETA_BASE,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Precomputes the MoonViT 2D rope cos/sin tables as REAL constants.

  GPU-clean replacement for Rope2DPosEmb._precompute_freqs_cis + apply_rope
  (modeling_vit.py). Returned tensors are plain floats; applying them is a
  pure elementwise multiply + reshape (see apply_moonvit_rope_2d).

  Angle layout matches the original exactly:
    - flat position n row-major over (grid_h, grid_w): x_pos=n%gw, y_pos=n//gw
    - freq[i] = theta_base ** (-(4*i)/head_dim), i in [0, head_dim//4)
    - complex entry 2i   = angle x_pos * freq[i]  (x-axis)
    - complex entry 2i+1 = angle y_pos * freq[i]  (y-axis)
    Interleaved: [x0, y0, x1, y1, ...] (matching modeling_vit.py:357-362).

  Returns:
    (cos, sin): each shape (grid_h * grid_w, head_dim). Doubled so cos[..., 2j]
    == cos[..., 2j+1] == cos(angle_j) — ready for interleaved elementwise rotation.
  """
  assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D rope"
  if device is None:
    device = torch.device("cpu")

  n = grid_h * grid_w
  flat_pos = torch.arange(0, n, dtype=torch.float32, device=device)
  x_pos = flat_pos % grid_w
  y_pos = torch.div(flat_pos, grid_w, rounding_mode="floor")

  dim_range = torch.arange(0, head_dim, 4, dtype=torch.float32, device=device)[
      : head_dim // 4
  ]
  freqs = 1.0 / (theta_base ** (dim_range / head_dim))

  x_angles = torch.outer(x_pos, freqs)   # (N, head_dim//4)
  y_angles = torch.outer(y_pos, freqs)   # (N, head_dim//4)

  # Interleave: [x0, y0, x1, y1, ...] -> shape (N, head_dim//2)
  angles = torch.stack([x_angles, y_angles], dim=-1).reshape(n, head_dim // 2)

  cos_half = torch.cos(angles)
  sin_half = torch.sin(angles)

  # Double: pair (2j, 2j+1) shares angle j.
  cos = torch.repeat_interleave(cos_half, 2, dim=-1)  # (N, head_dim)
  sin = torch.repeat_interleave(sin_half, 2, dim=-1)  # (N, head_dim)

  return cos.to(dtype), sin.to(dtype)


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
  """Interleaved rotate: adjacent pairs (a, b) -> (-b, a).

  Real-op equivalent of multiplying (a + ib) by i. Op-set: reshape, select,
  stack (no gather, no complex) — safe for the OpenCL delegate.
  """
  shape = x.shape
  x_pairs = x.reshape(*shape[:-1], shape[-1] // 2, 2)
  a = x_pairs[..., 0]
  b = x_pairs[..., 1]
  return torch.stack([-b, a], dim=-1).reshape(*shape)


def apply_moonvit_rope_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Applies externalized 2D rope to q/k as pure elementwise ops.

  Args:
    q, k: (seq, num_heads, head_dim).
    cos, sin: (seq, head_dim) from build_moonvit_rope_2d (doubled).

  Returns:
    (q_roped, k_roped): same shapes as q, k.
  """
  cos_b = cos.unsqueeze(-2)   # (seq, 1, head_dim)
  sin_b = sin.unsqueeze(-2)

  q_out = q.float() * cos_b + _rotate_interleaved(q.float()) * sin_b
  k_out = k.float() * cos_b + _rotate_interleaved(k.float()) * sin_b
  return q_out.type_as(q), k_out.type_as(k)


# ---------------------------------------------------------------------------
# MoonViT config dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MoonViTConfig:
  """MoonViT-SO-400M config — single source of truth for the re-author."""

  hidden_size: int = HIDDEN_SIZE
  num_layers: int = NUM_LAYERS
  num_heads: int = NUM_HEADS
  head_dim: int = HEAD_DIM
  intermediate_size: int = INTERMEDIATE_SIZE
  patch_size: int = PATCH_SIZE
  merge_kernel_size: Tuple[int, int] = MERGE_KERNEL_SIZE
  rope_theta_base: float = ROPE_THETA_BASE
  layernorm_eps: float = LAYERNORM_EPS
  init_pos_emb_height: int = INIT_POS_EMB_HEIGHT
  init_pos_emb_width: int = INIT_POS_EMB_WIDTH


# ---------------------------------------------------------------------------
# MoonViT sub-modules
# ---------------------------------------------------------------------------


def build_block_diagonal_mask(
    n_patches_per_image: int,
    n_images: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:
  """Build a block-diagonal additive attention mask for multi-image sequences.

  Returns shape (1, 1, L, L) where L = n_patches_per_image * n_images.
  Within each image's block, entries are 0 (attend freely).
  Across image blocks, entries are -inf (mask out cross-image attention).

  This is the qwen_vl _get_mask() pattern (image_encoder.py:289-305) adapted
  for MoonViT's flat cu_seqlens convention. The mask is precomputed ONCE at
  set_image_size() and reused in every forward call — no Python loop or .item()
  inside forward.

  Args:
    n_patches_per_image: number of patches per image (gh * gw).
    n_images: number of images concatenated in the sequence.
    dtype: mask dtype (should match query/key dtype for SDPA).
    device: target device.

  Returns:
    Additive attention mask (1, 1, L, L).
  """
  if device is None:
    device = torch.device("cpu")
  L = n_patches_per_image * n_images
  mask = torch.full((1, 1, L, L), float("-inf"), dtype=dtype, device=device)
  for i in range(n_images):
    s = i * n_patches_per_image
    e = s + n_patches_per_image
    mask[..., s:e, s:e] = 0.0
  return mask


class MoonViTAttention(nn.Module):
  """Single MoonViT attention block with externalized 2D rope.

  Mirrors MoonVitEncoderLayer.attention_qkvpacked (modeling_vit.py:439-469)
  but with:
    - complex rope replaced by apply_moonvit_rope_2d (precomputed real tables).
    - data-dependent cu_seqlens window loop replaced by a precomputed
      block-diagonal additive attention mask + F.scaled_dot_product_attention.

  The mask is built ONCE at set_image_size() and passed in via forward(attn_mask).
  No .item(), no Python loop, no gather or nonzero in the forward graph.

  Fused wqkv Linear(1152->3456) is kept fused; the view+unbind inside forward
  reproduces the original ``xqkv.view(..3, nheads, headdim); unbind`` split.
  """

  def __init__(self, config: MoonViTConfig):
    super().__init__()
    self.num_heads = config.num_heads
    self.head_dim = config.head_dim
    self.hidden_size = config.hidden_size
    # attn_bias=True confirmed at modeling_vit.py:594 / modeling_locateanything.py.
    self.wqkv = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=True)
    self.wo = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

  def forward(
      self,
      x: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      attn_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Args:
      x: (seq, hidden_size) — all images concatenated.
      cos, sin: (seq, head_dim) externalized rope (precomputed at set_image_size).
      attn_mask: (1, 1, seq, seq) additive mask from build_block_diagonal_mask(),
                 precomputed at set_image_size(). If None, full attention over seq.
    """
    seq = x.shape[0]
    xqkv = self.wqkv(x)  # (seq, 3*hidden)
    xqkv = xqkv.view(seq, 3, self.num_heads, self.head_dim)
    xq, xk, xv = torch.unbind(xqkv, dim=1)  # each (seq, num_heads, head_dim)

    xq, xk = apply_moonvit_rope_2d(xq, xk, cos, sin)

    # SDPA layout: (batch, heads, seq, head_dim).
    # We treat all images as batch=1 sequence (block-diagonal mask separates them).
    xq_t = xq.unsqueeze(0).transpose(1, 2)  # (1, H, seq, D)
    xk_t = xk.unsqueeze(0).transpose(1, 2)
    xv_t = xv.unsqueeze(0).transpose(1, 2)

    # Mask dtype must match q/k dtype for SDPA.
    if attn_mask is not None:
      mask = attn_mask.to(dtype=xq_t.dtype, device=xq_t.device)
    else:
      mask = None

    # Single SDPA call — no Python loop, no .item().
    out = F.scaled_dot_product_attention(xq_t, xk_t, xv_t, attn_mask=mask)
    # out: (1, H, seq, D) -> (seq, H*D)
    out = out.transpose(1, 2).reshape(seq, self.hidden_size)
    return self.wo(out)


class MoonViTMLP(nn.Module):
  """MoonViT MLP: fc0 -> GELU_TANH -> fc1 (sequential, both bias=True).

  PytorchGELUTanh = nn.GELU(approximate='tanh') confirmed at modeling_vit.py:593.
  """

  def __init__(self, config: MoonViTConfig):
    super().__init__()
    self.fc0 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
    self.fc1 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
    self.act = nn.GELU(approximate="tanh")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc1(self.act(self.fc0(x)))


class MoonViTBlock(nn.Module):
  """Single MoonViT encoder block.

  Layout (modeling_vit.py:484-493):
    x = x + attn(norm0(x))
    x = x + mlp(norm1(x))
  """

  def __init__(self, config: MoonViTConfig):
    super().__init__()
    self.norm0 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
    self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
    self.attn = MoonViTAttention(config)
    self.mlp = MoonViTMLP(config)

  def forward(
      self,
      x: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      attn_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Args:
      x: (seq, hidden_size).
      cos, sin: (seq, head_dim) precomputed rope tables.
      attn_mask: (1, 1, seq, seq) additive block-diagonal mask, or None.
    """
    x = x + self.attn(self.norm0(x), cos, sin, attn_mask)
    x = x + self.mlp(self.norm1(x))
    return x


# ---------------------------------------------------------------------------
# Learnable 2D interpolated positional embedding
# ---------------------------------------------------------------------------


class Learnable2DInterpPosEmb(nn.Module):
  """Additive learnable 2D pos-emb with bicubic interpolation.

  Weight shape (init_h, init_w, hidden_size) = (64, 64, 1152) matches checkpoint
  key vision_model.patch_embed.pos_emb.weight exactly.

  get_cached_emb() is called once at set_image_size() time and returns a plain
  (H*W, D) tensor. No interpolation happens during forward.
  """

  def __init__(self, config: MoonViTConfig):
    super().__init__()
    self.init_h = config.init_pos_emb_height
    self.init_w = config.init_pos_emb_width
    self.hidden_size = config.hidden_size
    self.weight = nn.Parameter(
        torch.empty(self.init_h, self.init_w, self.hidden_size)
    )
    nn.init.normal_(self.weight)

  def get_cached_emb(
      self, grid_h: int, grid_w: int, device: torch.device = None
  ) -> torch.Tensor:
    """Returns interpolated pos-emb for (grid_h, grid_w) as (H*W, D)."""
    if device is None:
      device = self.weight.device
    w = self.weight.to(device)  # (init_h, init_w, D)
    if grid_h == self.init_h and grid_w == self.init_w:
      return w.reshape(self.init_h * self.init_w, self.hidden_size).detach()
    # F.interpolate wants (1, D, H, W).
    w_4d = w.permute(2, 0, 1).unsqueeze(0)  # (1, D, init_h, init_w)
    interp = F.interpolate(
        w_4d.float(), size=(grid_h, grid_w), mode="bicubic", align_corners=False
    )  # (1, D, grid_h, grid_w)
    interp = interp.squeeze(0).permute(1, 2, 0).reshape(grid_h * grid_w, self.hidden_size)
    return interp.to(w.dtype).detach()


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------


class MoonViTPatchEmbed(nn.Module):
  """Conv2d patch projection (patch=14, bias=True) + additive learned pos-emb.

  After set_image_size(), the interpolated pos-emb is cached as
  self._cached_pos_emb — a plain (H*W, D) tensor, not a parameter. The add
  during forward is a constant-fold-friendly elementwise add.
  """

  def __init__(self, config: MoonViTConfig):
    super().__init__()
    ps = config.patch_size
    self.proj = nn.Conv2d(3, config.hidden_size, kernel_size=(ps, ps), stride=(ps, ps), bias=True)
    self.pos_emb = Learnable2DInterpPosEmb(config)
    self._cached_pos_emb: Optional[torch.Tensor] = None

  def set_image_size(self, grid_h: int, grid_w: int):
    """Precompute and cache the interpolated pos-emb for this grid size."""
    self._cached_pos_emb = self.pos_emb.get_cached_emb(grid_h, grid_w)

  def forward(
      self,
      pixel_values: torch.Tensor,
      pos_emb: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Args:
      pixel_values: (N_images, 3, H_img, W_img).
      pos_emb: optional (gh*gw, D) RUNTIME-INPUT positional embedding. When
        provided (the exported GPU path), it is added as an activation+activation
        2D add — the closed-MLDrift GPU bug only mis-broadcasts a rank-2 CONSTANT
        operand of an ADD, not an activation input. When None, the baked
        self._cached_pos_emb constant path is used (CPU / non-exported).
    Returns:
      (N_images * gh * gw, hidden_size) patches with pos-emb added.
    """
    n_img = pixel_values.shape[0]
    c = pixel_values.shape[1]
    ph, pw = self.proj.kernel_size
    w_img = pixel_values.shape[3]
    gh = pixel_values.shape[2] // ph
    gw = w_img // pw
    # The stride==kernel (non-overlapping) Conv2d patch projection miscomputes on
    # the OpenCL/MLDrift GPU delegate: measured CPU-vs-GPU cosine 0.61 at this
    # stage while the encoder input is bit-identical (INPUT cut cosine 1.0). A
    # non-overlapping conv is mathematically identical to unfold-into-patches + a
    # Linear matmul, and matmul is GPU-correct. self.proj stays the weight owner
    # (loaded from checkpoint); only the compute becomes reshape + linear.
    #
    # The unfold must NEVER exceed rank 4: the GPU delegate requires "dimensions
    # must be less than 5" and rejects 5D transposes, so a rank-5/6 unfold drops
    # full-graph delegation and crashes the vision executor. The exported vision
    # graph is single-image (N=1), so fold the batch and split H, merge, split W,
    # merge — every intermediate stays <= rank 4, ending at a 2D matmul.
    x = pixel_values.reshape(c, gh, ph, w_img)   # (C, gh, kh, W)       4D [N=1 folded]
    x = x.permute(1, 0, 2, 3)                     # (gh, C, kh, W)       4D
    x = x.reshape(gh, c * ph, w_img)             # (gh, C*kh, W)        3D
    x = x.reshape(gh, c * ph, gw, pw)            # (gh, C*kh, gw, kw)   4D
    x = x.permute(0, 2, 1, 3)                    # (gh, gw, C*kh, kw)   4D
    x = x.reshape(n_img * gh * gw, c * ph * pw)  # (gh*gw, C*kh*kw); cols (c,kh,kw)
    import os as _os
    _pcut = _os.getenv("LA_BISECT")
    if _pcut == "PATCHES":
      return x  # BISECT: pre-linear unfolded patches (isolates reshape/transpose)
    w_flat = self.proj.weight.reshape(self.proj.weight.shape[0], -1)  # (D, C*kh*kw)
    x = torch.nn.functional.linear(x, w_flat, self.proj.bias)  # (N*gh*gw, D)
    if _pcut == "NOPOS":
      return x  # BISECT: post-linear, pre-pos-emb

    if pos_emb is not None:
      # RUNTIME-INPUT activation add (Approach A, the GPU fix). Both operands
      # are activations, so a plain rank-2 [seq, D] add is correct on the
      # MLDrift/OpenCL delegate — the mis-broadcast-to-row-0 bug is specific to a
      # rank-2 CONSTANT operand. No rank-4 reshape dance needed here.
      x = x + pos_emb
    elif self._cached_pos_emb is not None:
      pos = self._cached_pos_emb.to(x.device, x.dtype)  # (gh*gw, D)
      if n_img > 1:
        pos = pos.unsqueeze(0).expand(n_img, -1, -1).reshape(n_img * pos.shape[0], pos.shape[1])
      if _pcut == "POSONLY":
        return pos  # BISECT: raw pos constant as the GPU delegate stores/reads it
      # MLDrift GPU bug: a rank-2 [seq, D] activation maps to BHWC with seq in the
      # BATCH axis, but a rank-2 [seq, D] constant maps with seq in the HEIGHT axis
      # -> axis mismatch -> the ADD reads only constant row 0 (broadcast). Fix: make
      # BOTH operands rank-4 [1, seq, 1, D] so seq lands in HEIGHT for both, and the
      # kernel iterates the row index identically. (seq in WIDTH, [1,1,seq,D], did
      # NOT work; HEIGHT is the axis that aligns the activation and the constant.)
      seq_h, d_h = x.shape[0], x.shape[1]
      x = (x.reshape(1, seq_h, 1, d_h) + pos.reshape(1, seq_h, 1, d_h)).reshape(seq_h, d_h)
    return x


# ---------------------------------------------------------------------------
# Full MoonViT encoder
# ---------------------------------------------------------------------------


class MoonViTEncoderClean(nn.Module):
  """Full 27-layer MoonViT encoder.

  Production re-author of MoonVitPretrainedModel (modeling_vit.py:567-615).
  All GPU-hostile ops removed:
    - 2D RoPE -> precomputed cos/sin tables at set_image_size.
    - pos-emb interpolation -> cached at set_image_size.
    - patch merger -> reshape+permute only.

  Weights loaded via load_moonvit_encoder().
  """

  def __init__(self, config: MoonViTConfig):
    super().__init__()
    self.config = config
    self.patch_embed = MoonViTPatchEmbed(config)
    self.blocks = nn.ModuleList([MoonViTBlock(config) for _ in range(config.num_layers)])
    self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)

    # Set by set_image_size(). All are image-size-dependent constants.
    self._grid_h: int = 0
    self._grid_w: int = 0
    self._rope_cos: Optional[torch.Tensor] = None
    self._rope_sin: Optional[torch.Tensor] = None
    # Block-diagonal additive attention mask (1, 1, L, L); 0 within each image
    # block, -inf across image blocks. Replaces the data-dependent cu_seqlens loop.
    self._attn_mask: Optional[torch.Tensor] = None

  def set_image_size(self, image_h: int, image_w: int, n_images: int = 1):
    """Precompute all image-size-dependent constants.

    Must be called before forward(). Safe to call again when image size changes.
    Stores:
      _rope_cos/_rope_sin: (L, head_dim) repeated rope tables.
      _attn_mask: (1, 1, L, L) block-diagonal additive mask (qwen_vl pattern).

    Args:
      image_h, image_w: pixel dimensions.
      n_images: number of images in the batch (all must be the same size).
    """
    gh = image_h // self.config.patch_size
    gw = image_w // self.config.patch_size
    self._grid_h = gh
    self._grid_w = gw

    self.patch_embed.set_image_size(gh, gw)

    # Rope tables: repeat per-image rope for the full concatenated sequence.
    cos1, sin1 = build_moonvit_rope_2d(
        gh, gw, head_dim=self.config.head_dim, theta_base=self.config.rope_theta_base
    )  # (gh*gw, head_dim)
    if n_images == 1:
      self._rope_cos = cos1
      self._rope_sin = sin1
    else:
      self._rope_cos = cos1.repeat(n_images, 1)
      self._rope_sin = sin1.repeat(n_images, 1)

    # Block-diagonal mask (qwen_vl _get_mask pattern).
    # Built in float32; cast to query dtype inside attention forward.
    self._attn_mask = build_block_diagonal_mask(
        n_patches_per_image=gh * gw,
        n_images=n_images,
        dtype=torch.float32,
    )

  @torch.inference_mode()
  def forward(
      self,
      pixel_values: torch.Tensor,
      pos_emb: Optional[torch.Tensor] = None,
  ) -> list:
    """Run the full MoonViT forward.

    Args:
      pixel_values: (N, 3, H_img, W_img). All images must be the same size.
                    Call set_image_size(H_img, W_img, N) before forward.
      pos_emb: optional (gh*gw, D) RUNTIME-INPUT positional embedding threaded to
               patch_embed. When None, patch_embed uses its baked constant
               (_cached_pos_emb) — the CPU / non-exported fallback.
    Returns:
      List of N tensors each (n_out_patches, 4608) after 2x2 pixel-unshuffle.
      n_out_patches = (gh//2) * (gw//2).
      The connector mlp1 maps (n_out_patches, 4608) -> (n_out_patches, 2048).
    """
    if self._rope_cos is None:
      raise RuntimeError("Call set_image_size() before MoonViTEncoderClean.forward().")

    n_images = pixel_values.shape[0]
    gh = self._grid_h
    gw = self._grid_w
    n_patches = gh * gw

    import os as _os
    _cut = _os.getenv("LA_BISECT")
    if _cut == "INPUT":
      return [pixel_values.reshape(pixel_values.shape[0], -1)]  # BISECT: post-transpose encoder input

    x = self.patch_embed(pixel_values, pos_emb)  # (N*n_patches, D)
    if _cut in ("A", "PATCHES", "NOPOS", "POSONLY"):
      return [x]  # BISECT: patch-embed output (or pre-linear / pre-pos / pos-only cut)

    # Holistic per-stage debug: collect every same-shape [L, D] stage (patch-embed,
    # each of the 27 blocks, final_layernorm) and emit them stacked as ONE output so
    # a single CPU run + single GPU run map the whole chain's CPU-vs-GPU cosine. The
    # host splitter reads stage 0 = patch-embed, 1..27 = after block i, 28 = final_ln.
    _dbg = _os.getenv("LA_DEBUG_ALL")
    _stages = [x] if _dbg else None

    cos = self._rope_cos.to(x.device, x.dtype)
    sin = self._rope_sin.to(x.device, x.dtype)
    # Block-diagonal mask: device/dtype cast deferred to MoonViTAttention.forward.
    attn_mask = self._attn_mask  # (1, 1, L, L), float32

    for block in self.blocks:
      x = block(x, cos, sin, attn_mask)
      if _dbg:
        _stages.append(x)

    x = self.final_layernorm(x)
    if _dbg:
      _stages.append(x)
      dbg = torch.stack(_stages, dim=0)  # (n_stages, L, D)
      return [dbg.reshape(dbg.shape[0] * dbg.shape[1], dbg.shape[2])]

    # Pixel-unshuffle merger: reshape+permute only — no gather/pad/masked-index.
    d = self.config.hidden_size
    kh, kw = self.config.merge_kernel_size
    new_h = gh // kh
    new_w = gw // kw

    outputs = []
    for i in range(n_images):
      seg = x[i * n_patches:(i + 1) * n_patches]   # (gh*gw, D)
      # GPU delegates (OpenCL/MLDrift) reject >4D transposes AND mis-execute the
      # 4D transposing permute here — the exported TRANSPOSE scrambled the
      # pixel-unshuffle channel order (measured: CPU vs GPU encoder cosine 0.23,
      # no NaN, fp32==fp16). Replace the permute with a static slice over the kh
      # axis + channel-concat, which lowers to SLICE+CONCATENATION (delegate-
      # correct) and is torch-verified bit-identical to the permute form.
      seg = seg.view(new_h, kh, new_w, kw * d)         # (new_h, kh, new_w, kw*d)
      rows = [seg[:, i, :, :] for i in range(kh)]      # kh x (new_h, new_w, kw*d)
      seg = torch.cat(rows, dim=-1)                    # (new_h, new_w, kh*kw*d)
      seg = seg.reshape(new_h * new_w, d * kh * kw)    # (n_out, 4608)
      outputs.append(seg)
    return outputs


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def load_moonvit_encoder(
    checkpoint_path: str,
    encoder: MoonViTEncoderClean,
    state: Optional[Dict[str, torch.Tensor]] = None,
) -> MoonViTEncoderClean:
  """Load MoonViT weights from a LocateAnything checkpoint directory.

  Tensor keys are mapped from vision_model.* in the safetensors checkpoint to
  the MoonViTEncoderClean parameter names.

  Args:
    checkpoint_path: directory containing the *.safetensors shards.
    encoder: MoonViTEncoderClean instance to load into.
    state: pre-loaded state dict (if provided, checkpoint_path is ignored).

  Returns:
    The encoder with weights loaded and set to eval mode.
  """
  if state is None:
    from litert_torch.generative.utilities.loader import load_safetensors
    state = load_safetensors(checkpoint_path)

  def get(key: str) -> torch.Tensor:
    val = state.get(key)
    if val is None:
      raise KeyError(f"Checkpoint key not found: {key!r}")
    return val

  # --- Patch embed ---
  encoder.patch_embed.proj.weight.data.copy_(get("vision_model.patch_embed.proj.weight"))
  if encoder.patch_embed.proj.bias is not None:
    bias_key = "vision_model.patch_embed.proj.bias"
    if bias_key in state:
      encoder.patch_embed.proj.bias.data.copy_(get(bias_key))
    else:
      encoder.patch_embed.proj.bias.data.zero_()

  # Pos-emb: checkpoint shape (64, 64, 1152) matches parameter shape exactly.
  encoder.patch_embed.pos_emb.weight.data.copy_(
      get("vision_model.patch_embed.pos_emb.weight")
  )

  # --- Per-block ---
  for i in range(encoder.config.num_layers):
    blk = encoder.blocks[i]
    pfx = f"vision_model.encoder.blocks.{i}"
    blk.norm0.weight.data.copy_(get(f"{pfx}.norm0.weight"))
    blk.norm0.bias.data.copy_(get(f"{pfx}.norm0.bias"))
    blk.norm1.weight.data.copy_(get(f"{pfx}.norm1.weight"))
    blk.norm1.bias.data.copy_(get(f"{pfx}.norm1.bias"))
    blk.attn.wqkv.weight.data.copy_(get(f"{pfx}.wqkv.weight"))
    blk.attn.wqkv.bias.data.copy_(get(f"{pfx}.wqkv.bias"))
    blk.attn.wo.weight.data.copy_(get(f"{pfx}.wo.weight"))
    blk.attn.wo.bias.data.copy_(get(f"{pfx}.wo.bias"))
    blk.mlp.fc0.weight.data.copy_(get(f"{pfx}.mlp.fc0.weight"))
    blk.mlp.fc0.bias.data.copy_(get(f"{pfx}.mlp.fc0.bias"))
    blk.mlp.fc1.weight.data.copy_(get(f"{pfx}.mlp.fc1.weight"))
    blk.mlp.fc1.bias.data.copy_(get(f"{pfx}.mlp.fc1.bias"))

  # --- Final norm ---
  encoder.final_layernorm.weight.data.copy_(
      get("vision_model.encoder.final_layernorm.weight")
  )
  encoder.final_layernorm.bias.data.copy_(
      get("vision_model.encoder.final_layernorm.bias")
  )

  encoder.eval()
  return encoder


def build_moonvit_encoder(
    checkpoint_path: str,
    image_h: int = 448,
    image_w: int = 448,
) -> MoonViTEncoderClean:
  """Convenience factory: build config + encoder + load weights + set_image_size."""
  config = MoonViTConfig()
  encoder = MoonViTEncoderClean(config)
  load_moonvit_encoder(checkpoint_path, encoder)
  encoder.set_image_size(image_h, image_w)
  return encoder


# ---------------------------------------------------------------------------
# Reference: original complex-exp rope (used by parity check only).
# ---------------------------------------------------------------------------


def _reference_precompute_freqs_cis(
    max_height: int, max_width: int, dim: int, theta_base: float = 10000.0
) -> torch.Tensor:
  """Verbatim port of Rope2DPosEmb._precompute_freqs_cis (modeling_vit.py)."""
  n = max_height * max_width
  flat_pos = torch.arange(0, n).float()
  x_pos = flat_pos % max_width
  y_pos = flat_pos // max_width
  dim_range = torch.arange(0, dim, 4)[: (dim // 4)].float()
  freqs = 1.0 / (theta_base ** (dim_range / dim))
  x_freqs = torch.outer(x_pos, freqs).float()
  y_freqs = torch.outer(y_pos, freqs).float()
  x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
  y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
  freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
  return freqs_cis.reshape(max_height, max_width, -1)


def _reference_apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Verbatim port of apply_rope (modeling_vit.py:201-221)."""
  freqs_cis = freqs_cis.unsqueeze(-2)
  xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
  return xq_out.type_as(xq), xk_out.type_as(xk)


def _reference_get_freqs_cis(
    grid_h: int, grid_w: int, dim: int, theta_base: float = 10000.0
) -> torch.Tensor:
  """Get freqs_cis for a (grid_h, grid_w) subgrid, as in the original model."""
  full = _reference_precompute_freqs_cis(
      max_height=max(grid_h, 512),
      max_width=max(grid_w, 512),
      dim=dim,
      theta_base=theta_base,
  )
  return full[:grid_h, :grid_w].reshape(-1, dim // 2)
