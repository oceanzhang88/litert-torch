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
"""Exportable modules for FastVLM (llava_qwen2) vision encoder and adapter.

FastVLM pairs a FastViT-HD image tower with a LLaVA-style projector on a Qwen2
text model, and ships the whole architecture as `trust_remote_code` rather than
as a transformers module. The checkpoint's own `encode_images` is the entire
vision contract:

    image_features = self.get_model().get_vision_tower()(images)
    image_features = self.get_model().mm_projector(image_features)

so the two exportables below are that pair, split at the tensor that travels
between them. The tower already flattens its `[B, C, H, W]` output to
`[B, H*W, C]` in `feature_select`, which is why the encoder needs no reshape of
its own and the adapter receives soft tokens directly.
"""

from litert_torch.generative.export_hf.core import exportable_module as exportable_module_base
import torch
import torch.nn.functional as torch_functional


def get_image_processor(model: torch.nn.Module):
  """Returns the processor FastVLM built for itself at load time.

  `AutoImageProcessor.from_pretrained` reads `preprocessor_config.json`, and
  FastVLM ships without one on purpose -- Apple's own tooling states that "LLaVA
  trainer does not save all the states needed for auto inference" and writes the
  file back as a separate step. With the file absent, transformers infers a
  processor from the config and reaches for `config.vision_config`, which a
  LLaVA-Qwen2 config does not carry.

  Asking the tower is better than reading the file even when the file exists:
  the processor comes from the same module as the weights and cannot disagree
  with them. FastVLM's uses identity normalization (mean 0, std 1), which a
  CLIP-default guess would silently break.
  """
  return _vision_tower(model).image_processor


def _vision_tower(model: torch.nn.Module) -> torch.nn.Module:
  """Returns the FastViT-HD tower, via the checkpoint's own accessor."""
  return model.get_model().get_vision_tower()


def _dummy_pixel_values(
    model: torch.nn.Module, image_processor
) -> torch.Tensor:
  """Builds one image the tower will accept, at the tower's own resolution.

  The resolution is read off the tower rather than the config: FastVLM encodes
  it in the tower name (`mobileclip_l_1024`), and `MobileCLIPVisionTower`
  resolves it with `int(vision_tower.split("_")[-1])` at load time. Asking the
  loaded module is therefore the one source that cannot disagree with the
  weights actually in memory.
  """
  if image_processor is None:
    raise ValueError(
        'Image processor is required for exporting the FastVLM vision tower.'
    )
  resolution = _vision_tower(model).input_image_size
  return image_processor(
      images=[torch.zeros((1, 3, resolution, resolution))],
      return_tensors='pt',
  ).pixel_values


def _scale_output_channels(layer: torch.nn.Module, scale: torch.Tensor) -> None:
  """Multiplies a layer's output channels by `scale`, in place.

  Works for both `nn.Linear` (weight `[out, in]`) and `nn.Conv2d` (weight
  `[out, in, kh, kw]`) because the output channel is the leading dimension of
  each, which is also the axis FastViT's `[dim, 1, 1]` layer scale broadcasts
  over.
  """
  flat = scale.detach().reshape(-1)
  weight = layer.weight
  with torch.no_grad():
    weight.mul_(flat.reshape([-1] + [1] * (weight.dim() - 1)))
    if layer.bias is not None:
      layer.bias.mul_(flat)


class _BatchOneFastVlmMhsa(torch.nn.Module):
  """FastViT MHSA with no rank-five intermediates.

  FastVLM's runtime contract fixes the vision batch to one, but the checkpoint's
  remote ``MHSA.forward`` retains that singleton while it splits packed QKV::

      [1, N, 3 * C] -> [1, N, 3, heads, head_dim]

  LiteRT's GPU delegate rejects every tensor above rank four. The official
  FastVLM portable graph splits packed QKV before introducing the head axis,
  keeps every intermediate at rank four or lower, and materializes K-transpose
  before its batch matmul. This module performs that same algebra::

      [1, N, 3 * C] -> 3 * [1, N, heads, head_dim]
      Q @ K: [heads, N, head_dim] @ [heads, head_dim, N]

  It reuses the checkpoint's exact Linear and Dropout modules, so parameter
  identity, state-dict paths, dtype, device, values, and training state are
  preserved.  Only the batch-one graph expression changes.
  """

  def __init__(self, original: torch.nn.Module):
    super().__init__()
    self.head_dim = int(original.head_dim)
    self.num_heads = int(original.num_heads)
    self.scale = float(original.scale)
    self.qkv = original.qkv
    self.attn_drop = original.attn_drop
    self.proj = original.proj
    self.proj_drop = original.proj_drop
    self.train(original.training)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
      raise ValueError(
          'FastVLM vision export requires rank-four [B, C, H, W] input to MHSA;'
          f' got {list(x.shape)}.'
      )
    batch, channels, height, width = x.shape
    if batch != 1:
      raise ValueError(
          'FastVLM vision export supports batch size one;'
          f' got batch={batch}.'
      )
    tokens = height * width
    x = torch.flatten(x, start_dim=2).transpose(-2, -1)
    query, key, value = self.qkv(x).chunk(3, dim=-1)
    query = (
        query.reshape(batch, tokens, self.num_heads, self.head_dim)
        .transpose(1, 2)
        .reshape(batch * self.num_heads, tokens, self.head_dim)
    )
    key = (
        key.reshape(batch, tokens, self.num_heads, self.head_dim)
        .permute(0, 2, 3, 1)
        .reshape(batch * self.num_heads, self.head_dim, tokens)
    )
    value = (
        value.reshape(batch, tokens, self.num_heads, self.head_dim)
        .transpose(1, 2)
        .reshape(batch * self.num_heads, tokens, self.head_dim)
    )

    attention = (query * self.scale) @ key
    attention = attention.reshape(
        batch, self.num_heads, tokens, tokens
    )
    attention = self.attn_drop(attention.softmax(dim=-1)).reshape(
        batch * self.num_heads, tokens, tokens
    )
    x = (attention @ value).reshape(
        batch, self.num_heads, tokens, self.head_dim
    )
    x = x.transpose(1, 2).reshape(batch, tokens, channels)
    x = self.proj_drop(self.proj(x))
    return x.transpose(-2, -1).reshape(batch, channels, height, width)


class _BatchOneSplitDepthwiseMultiplierConv(torch.nn.Module):
  """Expresses a depthwise multiplier as multiplier-one branches.

  FastViT's four patch transitions and final expansion use grouped Conv2d with
  ``groups == in_channels`` and ``out_channels == 2 * in_channels``. PyTorch
  lowers that directly to a TFLite depthwise convolution with multiplier two.
  Google's working FastVLM graph instead has two multiplier-one depthwise
  convolutions, reshapes their batch-one outputs to ``[C, 1, H, W]``,
  concatenates the multiplier axis, then reshapes back to ``[1, 2*C, H, W]``.

  The branch weights use ``weight[j::multiplier]`` because grouped-convolution
  output channels are ordered input-major: ``input * multiplier + j``. The
  concatenate/reshape below restores exactly that order without a rank-five
  intermediate. Weight and bias remain the checkpoint's original Parameters,
  preserving state-dict paths and object identity.
  """

  def __init__(self, original: torch.nn.Conv2d):
    super().__init__()
    if original.groups != original.in_channels:
      raise ValueError('split depthwise convolution requires groups=in_channels')
    if original.out_channels % original.in_channels:
      raise ValueError('depthwise output channels must divide input channels')
    self.in_channels = int(original.in_channels)
    self.out_channels = int(original.out_channels)
    self.depth_multiplier = self.out_channels // self.in_channels
    if self.depth_multiplier <= 1:
      raise ValueError('split depthwise convolution requires multiplier > 1')
    if original.padding_mode != 'zeros':
      raise ValueError('split depthwise convolution supports zero padding only')
    self.kernel_size = original.kernel_size
    self.stride = original.stride
    self.padding = original.padding
    self.dilation = original.dilation
    self.weight = original.weight
    self.bias = original.bias
    self.train(original.training)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
      raise ValueError(
          'FastVLM split depthwise convolution requires rank-four input;'
          f' got {list(x.shape)}.'
      )
    if x.shape[0] != 1:
      raise ValueError(
          'FastVLM vision export supports batch size one;'
          f' got batch={x.shape[0]}.'
      )
    branches = []
    for branch_index in range(self.depth_multiplier):
      branch_bias = (
          None
          if self.bias is None
          else self.bias[branch_index :: self.depth_multiplier]
      )
      branch = torch_functional.conv2d(
          x,
          self.weight[branch_index :: self.depth_multiplier],
          branch_bias,
          self.stride,
          self.padding,
          self.dilation,
          self.in_channels,
      )
      branches.append(branch.squeeze(0).unsqueeze(1))
    interleaved = torch.cat(branches, dim=1)
    return interleaved.reshape(
        1, self.out_channels, interleaved.shape[-2], interleaved.shape[-1]
    )


def adapt_fastvit_mhsa_for_litert_gpu(module: torch.nn.Module) -> int:
  """Replaces concrete FastVLM MHSA children with the batch-one expression.

  Matching the checkpoint's concrete class name confines the adaptation to the
  FastViT tower.  Modules that merely expose QKV-shaped attributes elsewhere in
  a multimodal checkpoint are left untouched.
  """
  adapted = 0

  def replace_children(parent: torch.nn.Module) -> None:
    nonlocal adapted
    for child_name, child in list(parent.named_children()):
      if type(child).__name__ == 'MHSA':
        setattr(parent, child_name, _BatchOneFastVlmMhsa(child))
        adapted += 1
      else:
        replace_children(child)

  replace_children(module)
  return adapted


def adapt_fastvit_depthwise_multipliers_for_litert(
    module: torch.nn.Module,
) -> int:
  """Splits concrete multiplier depthwise Conv2d children recursively."""
  adapted = 0

  def replace_children(parent: torch.nn.Module) -> None:
    nonlocal adapted
    for child_name, child in list(parent.named_children()):
      if (
          isinstance(child, torch.nn.Conv2d)
          and child.groups == child.in_channels
          and child.out_channels > child.in_channels
      ):
        setattr(
            parent,
            child_name,
            _BatchOneSplitDepthwiseMultiplierConv(child),
        )
        adapted += 1
      else:
        replace_children(child)

  replace_children(module)
  return adapted


def fold_layer_scale(module: torch.nn.Module) -> int:
  """Folds FastViT's layer scale into the weights that feed it.

  FastViT scales each residual branch by a learned per-channel parameter
  initialised at 1e-5:

      x = x + drop_path(layer_scale_1 * token_mixer(norm(x)))   # AttentionBlock
      x = x + drop_path(layer_scale   * convffn(x))             # RepMixerBlock

  Quantized, that multiply is unrepresentable on Hexagon. The constant is ~1e-5,
  so the op's output scale lands about five orders of magnitude below its input
  scale, and QNN refuses to construct it -- `ElementWiseBinary_... could not
  create op` inside `AttentionBlock_0`, then `RouterX86 graph prepare failed 12`,
  which aborts the whole bundle. No activation width fixes it; the scale *ratio*
  is the problem, not the precision.

  Since each scale is per-output-channel and immediately follows a linear op,
  `s * f(x)` folds exactly into `f` -- scale the output rows of the weight and
  the bias by `s`. The multiply then disappears from the graph rather than
  being made representable, and `use_layer_scale=False` routes the forward down
  the branch that no longer emits it.

  Exact up to float rounding, so calibration measures the folded graph and the
  portable bundle is unaffected in behaviour.
  """
  folded = 0
  for block in module.modules():
    if not getattr(block, 'use_layer_scale', False):
      continue
    convffn = getattr(block, 'convffn', None)
    if convffn is None or not hasattr(convffn, 'fc2'):
      continue

    if hasattr(block, 'layer_scale_1') and hasattr(block, 'layer_scale_2'):
      # AttentionBlock: MHSA's output projection, then the ConvFFN.
      token_mixer = getattr(block, 'token_mixer', None)
      if token_mixer is None or not hasattr(token_mixer, 'proj'):
        continue
      _scale_output_channels(token_mixer.proj, block.layer_scale_1)
      _scale_output_channels(convffn.fc2, block.layer_scale_2)
    elif hasattr(block, 'layer_scale'):
      # RepMixerBlock: one scale, on the ConvFFN branch only.
      _scale_output_channels(convffn.fc2, block.layer_scale)
    else:
      continue

    block.use_layer_scale = False
    folded += 1
  return folded


class LiteRTExportableModuleForFastVlmVisionEncoder(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for the FastVLM (FastViT-HD) vision encoder."""

  def __init__(self, model: torch.nn.Module, export_config):
    super().__init__(export_config)
    self.model = model
    tower = _vision_tower(model)
    adapted = adapt_fastvit_mhsa_for_litert_gpu(tower)
    if adapted:
      print(
          'Adapted FastViT MHSA to batch-one rank-four form in '
          f'{adapted} block(s).'
      )
    split_depthwise = adapt_fastvit_depthwise_multipliers_for_litert(tower)
    if split_depthwise:
      print(
          'Split FastViT depthwise multipliers into multiplier-one branches in '
          f'{split_depthwise} convolution(s).'
      )
    folded = fold_layer_scale(tower)
    if folded:
      print(f'Folded FastViT layer scale into {folded} block(s) before export.')

  def forward(
      self,
      images,
  ):
    # The graph takes NHWC and the tower takes NCHW, so the transpose lives here
    # rather than in the caller. LiteRT-LM checks the encoder's input layout at
    # conversation creation -- `vision_litert_compiled_model_executor.cc` reads
    # `dimensions[3]` of a rank-4 input and refuses anything outside 3..4
    # channels:
    #
    #     FAILED_PRECONDITION: Expected encoder input tensor to have 3 or 4
    #     channels but got 1024
    #
    # which is what a PyTorch-native [1, 3, 1024, 1024] export produces, since
    # the runtime reads the trailing width as the channel count. The engine
    # builds fine and the failure lands one step later, on the first
    # conversation.
    images = images.permute(0, 3, 1, 2)
    return {'features': _vision_tower(self.model)(images)}

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    # Currently we only support batch size = 1.
    dummy_image = _dummy_pixel_values(
        self.model, kwargs.get('image_processor', None)
    )
    resolution = int(dummy_image.shape[-1])
    # NHWC, matching what the runtime feeds. The processor returns NCHW because
    # that is what the tower consumes; `forward` transposes back.
    inputs = {'images': dummy_image.permute(0, 2, 3, 1).contiguous()}
    # Keyed on the spatial size, not on `shape[-1]` -- under NHWC that trailing
    # dimension is the channel count, and the signature would be named
    # `vision_3`. The runtime resolves this graph by the `vision_` prefix
    # (`kVisionLengthPrefix`), and Google's FastVLM bundle names it `vision_1024`.
    return {f'vision_{resolution}': (inputs, {})}


class LiteRTExportableModuleForFastVlmVisionAdapter(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for the FastVLM multimodal projector."""

  def __init__(self, model: torch.nn.Module, export_config, tokenizer):
    super().__init__(export_config)
    self.model = model
    self.tokenizer = tokenizer

  def forward(
      self,
      soft_tokens,
  ):
    return {'mm_embedding': self.model.get_model().mm_projector(soft_tokens)}

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    # Currently we only support batch size = 1.
    dummy_image = _dummy_pixel_values(
        self.model, kwargs.get('image_processor', None)
    )
    # One real tower forward rather than a shape computed from config. The token
    # count is a property of the FastViT downsampling stack, not a documented
    # constant, so deriving it from `image_size // patch_size` would be a second
    # source of truth that can silently drift from the weights.
    with torch.no_grad():
      features = _vision_tower(self.model)(dummy_image)
    inputs = {'soft_tokens': torch.zeros_like(features, dtype=torch.float32)}
    return {'vision_adapter': (inputs, {})}
