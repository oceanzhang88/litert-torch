"""LocateAnything MoonViT vision encoder and mlp1 adapter exportables."""

from __future__ import annotations

from types import MethodType
import sys

from litert_torch.generative.export_hf.core import exportable_module as exportable_module_base
import torch


DEFAULT_EXPORT_IMAGE_SIZE = 448
DEFAULT_EXPORT_GRID_HW = (32, 32)
DEFAULT_IMAGE_MEAN = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_STD = (0.5, 0.5, 0.5)


def _feature_tensor(features):
    """Normalize MoonViT output to a single tensor for the adapter boundary."""

    if hasattr(features, "last_hidden_state"):
        features = features.last_hidden_state
    if isinstance(features, (list, tuple)):
        features = [item.last_hidden_state if hasattr(item, "last_hidden_state") else item for item in features]
        return torch.cat(features, dim=0)
    return features


def _fixed_export_grid_hws(pixel_values: torch.Tensor) -> torch.Tensor:
    """Return the first fixed-shape MoonViT grid used for LiteRT export."""

    return torch.tensor([DEFAULT_EXPORT_GRID_HW], dtype=torch.int32, device=pixel_values.device)


def _fixed_patch_kernel(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a fixed Conv2D identity kernel for 14x14 RGB patch extraction."""

    patch_size = 14
    channels = 3
    patch_values = channels * patch_size * patch_size
    return torch.eye(
        patch_values,
        dtype=dtype,
        device=device,
    ).view(
        patch_values,
        channels,
        patch_size,
        patch_size,
    )


def _nhwc_image_to_locateanything_patches(
    images: torch.Tensor,
    patch_kernel: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert LiteRT-LM's image-like NHWC input to MoonViT patches.

    LiteRT-LM's high-level vision executor expects the first vision encoder input
    to look like an image tensor: HWC or NHWC with 3/4 channels in the last
    dimension. LocateAnything's Python processor normally emits normalized
    MoonViT patches shaped [num_patches, 3, 14, 14]. Moving that patchification
    behind this exported graph boundary lets the runtime feed ordinary image data
    while the model still receives the tensor shape its weights expect.
    """

    if images.dim() == 3:
        images = images.unsqueeze(0)
    image = images[0:1, :, :, :3].permute(0, 3, 1, 2)
    mean = torch.tensor(DEFAULT_IMAGE_MEAN, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(DEFAULT_IMAGE_STD, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    image = (image - mean) / std
    patch_size = 14
    channels = 3
    if patch_kernel is None:
        patch_kernel = _fixed_patch_kernel(dtype=image.dtype, device=image.device)
    patches = torch.nn.functional.conv2d(image, patch_kernel, stride=patch_size)
    patches = patches.permute(0, 2, 3, 1).reshape(-1, channels, patch_size, patch_size)
    return patches.contiguous()


def _fixed_pos_emb_forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
    """Export-safe fixed-grid positional embedding for the first 448px proof."""

    del grid_hws
    if not hasattr(self, "export_pos_emb_32x32"):
        raise RuntimeError("LocateAnything export requires precomputed export_pos_emb_32x32.")
    pos_emb = self.export_pos_emb_32x32
    if pos_emb.dtype != x.dtype or pos_emb.device != x.device:
        pos_emb = pos_emb.to(dtype=x.dtype, device=x.device)
    return x + pos_emb


def _precompute_fixed_pos_emb_32x32(pos_emb_module: torch.nn.Module) -> torch.Tensor:
    """Precompute MoonViT's first fixed-grid position embedding outside export."""

    shape = DEFAULT_EXPORT_GRID_HW
    if shape == tuple(pos_emb_module.weight.shape[:-1]):
        pos_emb = pos_emb_module.weight.flatten(end_dim=1)
    else:
        pos_emb = (
            torch.nn.functional.interpolate(
                pos_emb_module.weight.permute((2, 0, 1)).unsqueeze(0),
                size=shape,
                mode=pos_emb_module.interpolation_mode,
            )
            .squeeze(0)
            .permute((1, 2, 0))
            .flatten(end_dim=1)
        )
    return pos_emb.detach().contiguous()


def _fixed_get_freqs_cis(self, grid_hws: torch.Tensor) -> torch.Tensor:
    """Export-safe fixed-grid 2D rotary frequencies for the first 448px proof."""

    del grid_hws
    if self.freqs_cis is None:
        self.freqs_cis = self._precompute_freqs_cis(torch.device("cpu"))
    height, width = DEFAULT_EXPORT_GRID_HW
    return self.freqs_cis[:height, :width].reshape(-1, self.dim // 2)


def _fixed_patch_merger(x: torch.Tensor, merge_kernel_size: list[int] | tuple[int, int]) -> list[torch.Tensor]:
    """Export-safe fixed-grid patch merger for one 448px image."""

    d_model = x.size(-1)
    height, width = DEFAULT_EXPORT_GRID_HW
    kernel_height, kernel_width = merge_kernel_size
    if (int(kernel_height), int(kernel_width)) != (2, 2):
        raise ValueError("LocateAnything LiteRT vision export currently supports only 2x2 patch merging.")
    new_height = height // kernel_height
    new_width = width // kernel_width
    seq = x[: height * width]
    grid = seq.reshape(height, width, d_model)
    merged = torch.cat(
        (
            grid[0::2, 0::2, :],
            grid[0::2, 1::2, :],
            grid[1::2, 0::2, :],
            grid[1::2, 1::2, :],
        ),
        dim=-1,
    )
    return [merged.reshape(new_height * new_width, -1).contiguous()]


def _fixed_single_sequence_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None = None,
    k_cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Export-safe full attention for one fixed 448px image sequence."""

    del q_cu_seqlens, k_cu_seqlens
    seq_length = q.shape[0]
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    attn_output = attn_output.transpose(0, 1)
    return attn_output.reshape(seq_length, -1)


def _fixed_apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cossin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply precomputed 2D RoPE with real-valued ops.

    The LiteRT lowering path previously lifted separate same-shaped `cos` and
    `sin` buffers as constants and generated a TFLite graph that reused the
    cosine tensor for both paths. Packing them into one last-dimension-paired
    buffer makes the two lanes share one owner tensor while remaining distinct
    slices inside the graph.
    """

    cos = cossin[..., 0].unsqueeze(-2)
    sin = cossin[..., 1].unsqueeze(-2)

    def apply_one(x: torch.Tensor) -> torch.Tensor:
        pair = x.float().view(*x.shape[:-1], -1, 2)
        even = pair[..., 0]
        odd = pair[..., 1]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2).type_as(x)

    return apply_one(xq), apply_one(xk)


def _fixed_attention_qkvpacked(self, x: torch.Tensor, cu_seqlens=None, rope_freqs_cis=None):
    """Export-safe MoonViT block attention for one fixed 448px image."""

    del cu_seqlens, rope_freqs_cis
    xqkv = self.wqkv(x)
    qkv_shape = xqkv.size()[:-1] + (
        3,
        self.num_heads,
        self.hidden_size_per_attention_head,
    )
    xqkv = xqkv.view(*qkv_shape)
    xq, xk, xv = torch.unbind(xqkv, dim=-3)
    xq, xk = _fixed_apply_rope(xq, xk, self.export_rope_cossin)
    return self.wo(_fixed_single_sequence_attention(xq, xk, xv))


def _fixed_encoder_forward(self, hidden_states: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
    """Export-safe MoonViT encoder forward for one fixed 448px image."""

    del grid_hws
    for block in self.blocks:
        hidden_states = block(hidden_states, None, rope_freqs_cis=None)
    return self.final_layernorm(hidden_states)


def _fixed_vision_model_forward(self, pixel_values: torch.Tensor, grid_hws: torch.Tensor):
    """Export-safe fixed-grid MoonViT forward for the first 448px proof."""

    export_grid_hws = _fixed_export_grid_hws(pixel_values)
    hidden_states = self.patch_embed(pixel_values, export_grid_hws)
    hidden_states = self.encoder(hidden_states, export_grid_hws)
    return _fixed_patch_merger(hidden_states, self.merge_kernel_size)


def _patch_fixed_vision_model(vision_model: torch.nn.Module) -> None:
    """Install fixed-grid export helpers on the loaded MoonViT instance."""

    module = sys.modules.get(type(vision_model).__module__)
    if module is None:
        raise ValueError("LocateAnything MoonViT module is not loaded.")
    module.VL_VISION_ATTENTION_FUNCTIONS["sdpa"] = _fixed_single_sequence_attention
    module.VL_VISION_ATTENTION_FUNCTIONS["eager"] = _fixed_single_sequence_attention
    freqs_cis = vision_model.encoder.rope_2d._precompute_freqs_cis(torch.device("cpu"))  # pylint: disable=protected-access
    height, width = DEFAULT_EXPORT_GRID_HW
    freqs_cis = freqs_cis[:height, :width].reshape(-1, vision_model.encoder.rope_2d.dim // 2)
    rope_cos = freqs_cis.real.to(dtype=torch.float32)
    rope_sin = freqs_cis.imag.to(dtype=torch.float32)
    rope_cossin = torch.stack((rope_cos, rope_sin), dim=-1).contiguous()
    fixed_pos_emb = _precompute_fixed_pos_emb_32x32(vision_model.patch_embed.pos_emb)
    if "export_pos_emb_32x32" in vision_model.patch_embed.pos_emb._buffers:
        vision_model.patch_embed.pos_emb._buffers["export_pos_emb_32x32"] = fixed_pos_emb
    else:
        vision_model.patch_embed.pos_emb.register_buffer(
            "export_pos_emb_32x32", fixed_pos_emb, persistent=False
        )
    vision_model.patch_embed.pos_emb.forward = MethodType(
        _fixed_pos_emb_forward, vision_model.patch_embed.pos_emb
    )
    vision_model.encoder.rope_2d.get_freqs_cis = MethodType(
        _fixed_get_freqs_cis, vision_model.encoder.rope_2d
    )
    vision_model.encoder.forward = MethodType(_fixed_encoder_forward, vision_model.encoder)
    for block in vision_model.encoder.blocks:
        if "export_rope_cossin" in block._buffers:
            block._buffers["export_rope_cossin"] = rope_cossin
        else:
            block.register_buffer("export_rope_cossin", rope_cossin, persistent=False)
        block.attention_qkvpacked = MethodType(_fixed_attention_qkvpacked, block)
    vision_model.forward = MethodType(_fixed_vision_model_forward, vision_model)


class LiteRTExportableModuleForLocateAnythingVisionEncoder(
    exportable_module_base.ExportableModuleBase
):
    """Exportable module for LocateAnything's MoonViT encoder."""

    def __init__(self, model: torch.nn.Module, export_config):
        super().__init__(export_config)
        self.model = model
        self.register_buffer("export_patch_kernel", _fixed_patch_kernel(), persistent=False)
        _patch_fixed_vision_model(self.model.vision_model)

    def forward(self, images):
        pixel_values = _nhwc_image_to_locateanything_patches(images, self.export_patch_kernel)
        export_grid_hws = _fixed_export_grid_hws(pixel_values)
        features = self.model.vision_model(pixel_values=pixel_values, grid_hws=export_grid_hws)
        return {"features": _feature_tensor(features)}

    def get_sample_inputs(self, model_config, **kwargs):
        image_processor = kwargs.get("image_processor")
        if image_processor is None:
            raise ValueError("Image processor is required for exporting LocateAnything vision encoder.")
        image_size = int(kwargs.get("locateanything_sample_image_size", DEFAULT_EXPORT_IMAGE_SIZE))
        if image_size != DEFAULT_EXPORT_IMAGE_SIZE:
            raise ValueError(
                "LocateAnything first LiteRT vision export is fixed at "
                f"{DEFAULT_EXPORT_IMAGE_SIZE}px; dynamic image sizes are a later W-unit."
            )
        del image_processor
        inputs = {"images": torch.zeros((1, image_size, image_size, 3), dtype=torch.float32)}
        return {f"vision_locateanything_{image_size}": (inputs, {})}


class LiteRTExportableModuleForLocateAnythingVisionAdapter(
    exportable_module_base.ExportableModuleBase
):
    """Exportable module for LocateAnything's `mlp1` multimodal projector."""

    def __init__(self, model: torch.nn.Module, export_config, tokenizer):
        super().__init__(export_config)
        del tokenizer
        self.model = model

    def forward(self, features):
        return {"mm_embedding": self.model.mlp1(features)}

    def get_sample_inputs(self, model_config, **kwargs):
        vision_config = getattr(model_config, "vision_config", None)
        if vision_config is None:
            raise ValueError("LocateAnything vision adapter requires model_config.vision_config.")
        merge_kernel_size = getattr(vision_config, "merge_kernel_size", [2, 2])
        hidden_size = int(getattr(vision_config, "hidden_size"))
        input_hidden_size = hidden_size * int(merge_kernel_size[0]) * int(merge_kernel_size[1])
        sample_tokens = int(kwargs.get("locateanything_sample_soft_tokens", 256))
        inputs = {"features": torch.zeros((sample_tokens, input_hidden_size), dtype=torch.float32)}
        return {"vision_adapter": (inputs, {})}
