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

"""Re-authored LocateAnything-3B full VLM: encoder + connector + decoder.

Architecture:
  MoonViT-SO-400M encoder (image_encoder.py)
  -> pixel-unshuffle merger (built into encoder.forward())
  -> MoonViTConnector (mlp1): LayerNorm(4608) -> Linear(4608->2048) -> GELU -> Linear(2048->2048)
  -> image token splice at image_token_index=151665 via torch.cat (no masked_scatter)
  -> Qwen2.5-3B decoder (decoder.py)

Image token injection convention (export-safe static layout):
  torch.export cannot handle data-dependent slicing (finding where image tokens
  are at runtime).  For the pixel-prefill export signature we use a static layout:
  the connector output goes at the START of the sequence (position 0..n_img-1)
  and the text tokens follow, matching qwen_vl's convention and the LiteRT-LM
  runtime's expected prompt layout.  n_img_tokens = image_embeds.shape[0] is a
  compile-time constant fixed by set_image_size() before tracing.

Connector weight keys in the LocateAnything-3B safetensors:
  mlp1.0.weight / mlp1.0.bias  -- LayerNorm(4608)
  mlp1.1.weight / mlp1.1.bias  -- Linear(4608->2048)
  mlp1.3.weight / mlp1.3.bias  -- Linear(2048->2048)  (index 3: GELU has no weights)

GELU variant: nn.GELU() (approximate='none'), matching modeling_locateanything.py:139
which uses nn.GELU() default (not 'tanh').

Phase 3 verify tolerance: atol~1e-1 on full VLM pixel path (logits at last text
position), consistent with qwen_vl verify.py:129.
"""

import dataclasses
import glob
import os
from typing import Optional, Tuple

import torch
from torch import nn

from litert_torch.generative.examples.locateanything import decoder as dec_module
from litert_torch.generative.examples.locateanything import image_encoder as enc_module
import litert_torch.generative.layers.kv_cache as kv_utils
from litert_torch.generative.utilities import export_config as export_cfg


# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------

IMAGE_TOKEN_INDEX = 151665      # config.json image_token_index
CONNECTOR_IN_DIM = 4608         # vit_hidden_size(1152) * merge_kernel_size(2*2) = 4608
CONNECTOR_HIDDEN_DIM = 2048     # llm_hidden_size = 2048
CONNECTOR_OUT_DIM = 2048        # same as llm_hidden_size


# ---------------------------------------------------------------------------
# MoonViT Connector (mlp1)
# ---------------------------------------------------------------------------


class MoonViTConnector(nn.Module):
    """MLP connector bridging MoonViT encoder to the Qwen2.5-3B decoder.

    Mirrors modeling_locateanything.py mlp1 (lines 136-141):
        nn.Sequential(
            nn.LayerNorm(vit_hidden_size * 4),   # 4608
            nn.Linear(vit_hidden_size*4, llm_hidden_size),  # 4608->2048
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),    # 2048->2048
        )

    GELU is nn.GELU(approximate='none') — the HF source uses the default
    constructor, NOT the tanh approximation used in the MoonViT MLP blocks.
    """

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(CONNECTOR_IN_DIM)
        self.fc1 = nn.Linear(CONNECTOR_IN_DIM, CONNECTOR_HIDDEN_DIM)
        self.act = nn.GELU()     # exact (not tanh) — matches mlp1 in HF source
        self.fc2 = nn.Linear(CONNECTOR_HIDDEN_DIM, CONNECTOR_OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
          x: (n_out_patches, 4608) — pixel-unshuffled encoder output.
        Returns:
          (n_out_patches, 2048) — ready for token injection.
        """
        return self.fc2(self.act(self.fc1(self.norm(x))))

    def load_weights(self, state: dict) -> None:
        """Load mlp1.* keys from a flat safetensors state dict."""
        def _copy(param: nn.Parameter, key: str):
            t = state.get(key)
            if t is None:
                raise KeyError(f"Connector weight not found in checkpoint: {key!r}")
            param.data.copy_(t)

        _copy(self.norm.weight, "mlp1.0.weight")
        _copy(self.norm.bias,   "mlp1.0.bias")
        _copy(self.fc1.weight,  "mlp1.1.weight")
        _copy(self.fc1.bias,    "mlp1.1.bias")
        _copy(self.fc2.weight,  "mlp1.3.weight")
        _copy(self.fc2.bias,    "mlp1.3.bias")


# ---------------------------------------------------------------------------
# Full LocateAnything VLM
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LocateAnythingConfig:
    """Config for the full re-authored LocateAnything-3B VLM."""

    image_token_index: int = IMAGE_TOKEN_INDEX
    encoder_image_h: int = 448
    encoder_image_w: int = 448
    kv_cache_max_len: int = 2048


class LocateAnythingModel(nn.Module):
    """Full re-authored LocateAnything-3B VLM.

    Components:
      encoder: MoonViTEncoderClean (image_encoder.py)
      connector: MoonViTConnector (mlp1)
      decoder: Decoder (decoder.py, Qwen2.5-3B)

    Image token injection:
      The HF reference injects image embeddings at positions where
      input_ids == image_token_index (see modeling_locateanything.py:232-234).
      We replicate this with a torch.cat slice-inject pattern that avoids
      masked_scatter (not GPU-safe):
        1. Find the image token run start/end in input_ids.
        2. Embed the full token sequence.
        3. Replace the image-token slice with connector output.
        4. Forward the merged embeddings through the decoder.

    Caller contract:
      - Call set_image_size(H, W, n_images) on the encoder before forward().
      - The image token block in input_ids must be exactly n_out_patches long
        and contiguous.
      - pixel_values: (n_images, 3, H, W) — all same size.
    """

    def __init__(self, config: LocateAnythingConfig, mask_cache_size: int = 0):
        super().__init__()
        moonvit_cfg = enc_module.MoonViTConfig()
        self.encoder = enc_module.MoonViTEncoderClean(moonvit_cfg)
        self.connector = MoonViTConnector()
        self.decoder = dec_module.Decoder(
            dec_module.get_decoder_config(), mask_cache_size
        )
        self.config = config

    @torch.inference_mode()
    def forward(
        self,
        tokens: torch.Tensor,
        input_pos: torch.Tensor,
        kv_cache: kv_utils.KVCache,
        pixel_values: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        export_config: Optional[export_cfg.ExportConfig] = None,
    ) -> dict:
        """VLM forward.

        Text-only (decode step or text prefill without image):
          Pass pixel_values=None. Delegates to decoder.forward() directly.

        Image prefill:
          1. Run encoder -> connector to produce image_embeds.
          2. Embed all tokens (including image placeholder tokens).
          3. Replace the image placeholder run with image_embeds via torch.cat.
          4. Forward merged embeds through decoder.

        Args:
          tokens: (1, seq_len) int64 token ids.
          input_pos: (seq_len,) int64 positions.
          kv_cache: KVCache from kv_utils.
          pixel_values: (n_images, 3, H, W) or None.
          mask: optional causal mask (if None, decoder builds it from kv_cache).
          export_config: ExportConfig for litert export (None during verify).

        Returns:
          dict with "logits" key — shape (1, seq_len, vocab_size).
        """
        if pixel_values is None:
            return self.decoder.forward(
                tokens=tokens,
                input_pos=input_pos,
                kv_cache=kv_cache,
                input_embeds=None,
                mask=mask,
                export_config=export_config,
            )

        # --- Image path ---

        # 1. Run encoder (set_image_size must already be called externally).
        encoder_outputs = self.encoder.forward(pixel_values)
        # encoder_outputs: list of (n_out_patches, 4608) tensors, one per image.

        # 2. Concat multi-image encoder outputs and run connector.
        vit_feats = torch.cat(encoder_outputs, dim=0)   # (total_out_patches, 4608)
        image_embeds = self.connector(vit_feats)          # (total_out_patches, 2048)

        # 3. Embed all tokens.
        input_embeds = self.decoder.tok_embedding(tokens)  # (1, seq_len, 2048)

        # 4. Inject image embeds at image token positions via STATIC slice-cat.
        #
        # Export constraint: torch.export cannot handle data-dependent branches
        # (e.g. finding where image tokens are at runtime).  We use the same
        # static-layout assumption as qwen_vl: the image token block starts at
        # the FIRST position of the sequence (index 0).  For the pixel prefill
        # signature this matches the "image-first" prompt layout used by the
        # LiteRT-LM inference runtime.
        #
        # n_img_tokens is a compile-time constant derived from image_embeds.shape[0]
        # which is fixed by set_image_size() before tracing.
        n_img_tokens = image_embeds.shape[0]   # static at trace time
        img_block = image_embeds.unsqueeze(0).to(dtype=input_embeds.dtype)  # (1, n_img, D)
        suffix = input_embeds[:, n_img_tokens:, :]   # (1, text_len, D)
        input_embeds = torch.cat([img_block, suffix], dim=1)  # (1, n_img+text_len, D)

        # 5. Forward merged embeds through decoder.
        return self.decoder.forward(
            tokens=None,
            input_pos=input_pos,
            kv_cache=kv_cache,
            input_embeds=input_embeds,
            mask=mask,
            export_config=export_config,
        )



# ---------------------------------------------------------------------------
# Factory / builder
# ---------------------------------------------------------------------------


def load_full_model_state(checkpoint_path: str) -> dict:
    """Load all safetensors shards from checkpoint_path into one flat state dict.

    Skips the mmap/index dance — just loads all *.safetensors into CPU memory.
    For the LocateAnything-3B (~3.83B params, ~7.2 GB bf16) this requires ~8 GB RAM.
    """
    from safetensors import safe_open
    state = {}
    pattern = os.path.join(checkpoint_path, "*.safetensors")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files in {checkpoint_path}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
    return state


def build_full_model(
    checkpoint_path: str,
    mask_cache_size: int = 0,
    image_h: int = 448,
    image_w: int = 448,
) -> LocateAnythingModel:
    """Build and return the full re-authored LocateAnything-3B VLM.

    Loads all three components (encoder, connector, decoder) from the shared
    safetensors checkpoint in checkpoint_path.

    Args:
      checkpoint_path: directory with model-00001-of-00002.safetensors etc.
      mask_cache_size: KV cache / mask cache size (default 0 = no mask cache).
      image_h, image_w: image dimensions for set_image_size.

    Returns:
      LocateAnythingModel in eval() mode with all weights loaded.
    """
    config = LocateAnythingConfig(encoder_image_h=image_h, encoder_image_w=image_w)
    model = LocateAnythingModel(config, mask_cache_size)

    # Load all shards once and share the state dict across sub-loaders.
    state = load_full_model_state(checkpoint_path)

    # --- Encoder ---
    enc_module.load_moonvit_encoder(checkpoint_path, model.encoder, state=state)
    model.encoder.set_image_size(image_h, image_w)

    # --- Connector ---
    model.connector.load_weights(state)

    # --- Decoder (via litert-torch ModelLoader) ---
    from litert_torch.generative.utilities import loader as loading_utils
    # The ModelLoader uses load_safetensors internally, but since we already
    # have the state we bypass the file I/O by using a custom loader.
    def _preloaded_loader(_path):
        return state
    loader = loading_utils.ModelLoader(
        checkpoint_path, dec_module.TENSOR_NAMES, custom_loader=_preloaded_loader
    )
    missing, unexpected = loader.load(model.decoder, strict=False)
    if missing:
        real_missing = [k for k in missing if "inv_freq" not in k]
        if real_missing:
            raise RuntimeError(
                f"Decoder weights missing from checkpoint: {real_missing[:10]}"
            )

    model.eval()
    return model
