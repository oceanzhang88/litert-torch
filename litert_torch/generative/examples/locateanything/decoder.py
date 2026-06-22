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

"""Re-authored Qwen2.5-3B decoder for LocateAnything-3B.

Architecture: 36 layers, h=2048, 16Q/2KV GQA, RoPE θ=1e6, RMSNorm eps=1e-6,
SwiGLU (FeedForwardType.GATED + SILU), QKV bias present, tie_word_embeddings=True,
vocab=152681 (includes 1001 coordinate tokens 151677-152677).

Tensor name prefix in the safetensors checkpoint: language_model.*
The ModelLoader maps these from the LocateAnything checkpoint layout.
"""

from typing import Optional, Tuple

import litert_torch.generative.custom_ops.dynamic_slice  # noqa: F401 (registers litert_torch::dynamic_slice)
from litert_torch.generative.layers import kv_cache as kv_utils
import litert_torch.generative.layers.model_config as cfg
from litert_torch.generative.utilities import model_builder
import litert_torch.generative.utilities.loader as loading_utils
from litert_torch.generative.utilities import export_config as export_cfg
import torch

# ---------------------------------------------------------------------------
# Architecture constants — single source of truth
# ---------------------------------------------------------------------------

# Decoder dims (Qwen2.5-3B backbone inside LocateAnything-3B)
VOCAB_SIZE = 152681          # extended vocab: +1001 coordinate tokens
NUM_LAYERS = 36
EMBEDDING_DIM = 2048
NUM_HEADS = 16
NUM_KV_HEADS = 2             # GQA: 16 Q / 2 KV
HEAD_DIM = EMBEDDING_DIM // NUM_HEADS  # 128
INTERMEDIATE_SIZE = 11008
ROPE_THETA = 1_000_000.0
RMS_NORM_EPS = 1e-6
MAX_SEQ_LEN = 32768

# LocateAnything image/grounding token ids (for reference; decoder itself is
# token-agnostic — these appear in the input_ids stream only)
IMAGE_TOKEN_INDEX = 151665
BOX_START_TOKEN  = 151668
BOX_END_TOKEN    = 151669
COORD_TOKEN_START = 151677
COORD_TOKEN_END   = 152677  # inclusive; 1001 tokens total

# ---------------------------------------------------------------------------
# TensorNames — maps LocateAnything checkpoint keys to loader convention
#
# The checkpoint stores decoder tensors under language_model.model.* (body)
# and language_model.lm_head.weight (head). QKV are separate (not fused) with
# bias tensors present on q/k/v_proj.  The loader's _fuse_qkv() concatenates
# them into a single qkv_projection in [Q|K|V] order.
# ---------------------------------------------------------------------------

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    # Feedforward (SwiGLU gated: gate=w1, up=w3, down=w2 in litert convention)
    ff_up_proj   = "language_model.model.layers.{}.mlp.up_proj",
    ff_down_proj = "language_model.model.layers.{}.mlp.down_proj",
    ff_gate_proj = "language_model.model.layers.{}.mlp.gate_proj",
    # Attention projections (separate Q/K/V — loader fuses them)
    attn_query_proj  = "language_model.model.layers.{}.self_attn.q_proj",
    attn_key_proj    = "language_model.model.layers.{}.self_attn.k_proj",
    attn_value_proj  = "language_model.model.layers.{}.self_attn.v_proj",
    attn_output_proj = "language_model.model.layers.{}.self_attn.o_proj",
    # Norms
    pre_attn_norm  = "language_model.model.layers.{}.input_layernorm",
    post_attn_norm = "language_model.model.layers.{}.post_attention_layernorm",
    # Embedding, final norm, LM head
    embedding  = "language_model.model.embed_tokens",
    final_norm = "language_model.model.norm",
    lm_head    = "language_model.lm_head",
)


# ---------------------------------------------------------------------------
# ModelConfig — drives both the re-authored model and the loader
# ---------------------------------------------------------------------------

def get_decoder_config() -> cfg.ModelConfig:
    """Returns ModelConfig for the Qwen2.5-3B decoder inside LocateAnything-3B."""
    attn_config = cfg.AttentionConfig(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_query_groups=NUM_KV_HEADS,
        rotary_base=ROPE_THETA,
        rotary_percentage=1.0,
        qkv_use_bias=True,   # LocateAnything has q/k/v_proj.bias — atypical for Qwen2.5
    )
    ff_config = cfg.FeedForwardConfig(
        type=cfg.FeedForwardType.GATED,
        activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
        intermediate_size=INTERMEDIATE_SIZE,
    )
    norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.RMS_NORM,
        epsilon=RMS_NORM_EPS,
    )
    block_config = cfg.TransformerBlockConfig(
        attn_config=attn_config,
        ff_config=ff_config,
        pre_attention_norm_config=norm_config,
        post_attention_norm_config=norm_config,
    )
    config = cfg.ModelConfig(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        embedding_dim=EMBEDDING_DIM,
        block_configs=block_config,
        final_norm_config=norm_config,
        # tie_word_embeddings=True: the loader loads lm_head.weight from the
        # checkpoint (language_model.lm_head.weight — which is NOT the same
        # tensor object as embed_tokens.weight in the file, but IS identical in
        # value due to tied-weight convention). We load both keys explicitly via
        # lm_head in TensorNames and set lm_head_share_weight_with_embedding=False
        # so that strict=True in the loader doesn't complain about embed_tokens
        # appearing twice. After loading, lm_head.weight == embed_tokens.weight
        # in value (bit-identical since the checkpoint writes both).
        lm_head_share_weight_with_embedding=False,
    )
    return config


def get_fake_decoder_config() -> cfg.ModelConfig:
    """Returns a tiny ModelConfig for unit tests (not for real weights)."""
    config = get_decoder_config()
    config.vocab_size = 128
    config.num_layers = 2
    config.block_config(0).ff_config.intermediate_size = 64
    config.max_seq_len = 128
    return config


# ---------------------------------------------------------------------------
# Decoder class — thin subclass of DecoderOnlyModel
# ---------------------------------------------------------------------------

class Decoder(model_builder.DecoderOnlyModel):
    """Re-authored Qwen2.5-3B decoder for LocateAnything-3B.

    Identical structure to QwenVL Decoder: supports both pure-text forward
    (tokens) and mixed-modal forward (input_embeds pre-merged by the full VLM).
    For Phase 1 we only use the text path; Phase 3 wires input_embeds from the
    vision encoder + connector.
    """

    @torch.inference_mode
    def forward(
        self,
        tokens: torch.Tensor,
        input_pos: torch.Tensor,
        kv_cache: kv_utils.KVCache,
        input_embeds: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        export_config: Optional[export_cfg.ExportConfig] = None,
    ) -> dict:
        if input_embeds is None:
            _, seq_len = tokens.size()
            assert self.config.max_seq_len >= seq_len, (
                f"Cannot forward sequence of length {seq_len}, max seq length is"
                f" only {self.config.max_seq_len}"
            )
            input_embeds = self.tok_embedding(tokens)

        if rope is None:
            attn_config = self.config.block_config(0).attn_config
            n_elem = int(attn_config.rotary_percentage * attn_config.head_dim)
            rope = self.config.build_rope(
                input_pos, n_elem, attn_config.rotary_base
            )

        if mask is None:
            assert kv_cache is not None, "KV cache must be provided."
            # GPU-friendly mask lookup. input_pos is a contiguous range (see the
            # KV-cache update contract), so the causal-mask rows are a dynamic
            # slice, not a gather. index_select lowers to GATHER_ND, unsupported by
            # the OpenCL/MLDrift GPU delegate (engine creation fails at
            # llm_litert_compiled_model_executor.cc:1951). The dynamic_slice custom
            # op lowers to stablehlo.dynamic_slice (delegate-supported) and its
            # static-shape register_fake avoids torch.export unbacked-symbol errors
            # that a bare narrow(dim, dynamic_start, len) would raise.
            # Fold the key-length trim (formerly mask[:, :, :, :kv_len], which
            # lowered to a GPU-unsupported "SLICE: Constant input") directly into
            # the dynamic_slice key dimension.
            seq_len = input_pos.shape[0]
            d0 = self.mask_cache.shape[0]
            d1 = self.mask_cache.shape[1]
            kv_len = kv_cache.get_max_seq_len()
            zero = torch.zeros([], dtype=torch.int32)
            pos = input_pos.to(torch.int32)[0].reshape([])
            mask = torch.ops.litert_torch.dynamic_slice(
                self.mask_cache,
                [zero, zero, pos, zero],
                [d0, d1, seq_len, kv_len],
            )

        return self._forward_with_embeds(
            input_embeds,
            rope,
            mask,
            input_pos,
            kv_cache,
            export_config=export_config,
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_decoder(
    checkpoint_path: str,
    mask_cache_size: int = 0,
) -> Decoder:
    """Loads and returns the re-authored LocateAnything decoder.

    The LocateAnything checkpoint is a multi-modal archive containing
    language_model.*, vision_model.*, and mlp1.* tensors. We load only the
    language_model.* subset defined in TENSOR_NAMES; the rest are intentionally
    ignored (strict=False). This is correct because Phase 1 only re-authors the
    decoder; vision tensors will be handled by image_encoder.py in Phase 2.

    Args:
        checkpoint_path: Path to the directory containing the LocateAnything-3B
            safetensors shards (model-00001-of-00002.safetensors, etc.).
        mask_cache_size: Size of the static causal-mask cache. Must be > 0 for
            the decoder to be usable without an external mask. Typically set to
            the KV cache max length used during verification.

    Returns:
        An initialized Decoder in eval mode.
    """
    config = get_decoder_config()
    decoder = Decoder(config, mask_cache_size)
    loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
    # strict=False: vision_model.* and mlp1.* tensors in the checkpoint are
    # intentionally unmapped — only language_model.* is loaded here.
    missing, unexpected = loader.load(decoder, strict=False)
    if missing:
        raise RuntimeError(
            f"Decoder weights missing from checkpoint: {missing[:10]}..."
        )
    decoder.eval()
    return decoder
