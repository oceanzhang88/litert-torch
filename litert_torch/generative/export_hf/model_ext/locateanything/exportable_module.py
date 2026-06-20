"""LocateAnything text exportables for LiteRT Torch.

LocateAnything is a multimodal wrapper. Its decoder-only text core lives at
`model.language_model`, while image soft tokens are spliced as external embeddings.
The generic LiteRT Torch external-embedder exportables are the right primitive once
they receive the inner Qwen2 language model instead of the full LocateAnything wrapper.
"""

from __future__ import annotations

from types import MethodType
import sys

from litert_torch.backend import optimization_barrier as litert_optimization_barrier
from litert_torch.generative.export_hf.core import attention as litert_attention
from litert_torch.generative.export_hf.core.external_emb import exportable_module as external_emb_module
from litert_torch.generative.export_hf.core.split_cache import attention as litert_split_cache_attention
from litert_torch.generative.export_hf.core.split_cache import exportable_module as split_cache_module
import torch


def _language_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return LocateAnything's inner decoder-only language model."""

    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise ValueError("LocateAnything export requires `model.language_model`.")
    return language_model


def _export_safe_no_prefix_drop(position_ids: torch.Tensor) -> torch.Tensor:
    """Torch-export-safe replacement for LocateAnything's prefix drop detector."""

    batch_size = position_ids.shape[0]
    return torch.full((batch_size,), -1, dtype=torch.long, device=position_ids.device)


def _patch_export_safe_prefix_helper(language_model: torch.nn.Module) -> None:
    """Patch vendored Qwen2's data-dependent prefix helper for eval export."""

    qwen_model = getattr(language_model, "model", None)
    module = sys.modules.get(type(qwen_model).__module__) if qwen_model is not None else None
    if module is not None and hasattr(module, "find_prefix_seq_length_by_pe"):
        module.find_prefix_seq_length_by_pe = _export_safe_no_prefix_drop


def _drop_locateanything_unsupported_inputs(inputs: dict) -> dict:
    """Remove LiteRT/HF cache kwargs unsupported by LocateAnything's vendored Qwen2."""

    inputs.pop("cache_position", None)
    inputs.pop("k_ts_idx", None)
    inputs.pop("v_ts_idx", None)
    inputs.pop("valid_mask", None)
    return inputs


def _vendored_qwen_module(qwen_model: torch.nn.Module):
    """Return the loaded LocateAnything Qwen module object."""

    module = sys.modules.get(type(qwen_model).__module__)
    if module is None:
        raise ValueError("LocateAnything vendored Qwen module is not loaded.")
    return module


def _export_safe_prepare_attention_mask(
    qwen_model: torch.nn.Module,
    attention_mask: torch.Tensor | None,
    input_shape,
    inputs_embeds: torch.Tensor,
    past_key_values_length,
):
    """Prepare an additive attention mask without data-dependent shape guards."""

    if attention_mask is not None and len(attention_mask.shape) == 4:
        # Keep LocateAnything's proven W1 CPU/GPU semantics: LiteRT-LM's
        # external-embedder runner feeds a visible-position mask, while the
        # vendored Qwen attention expects an additive blocked-position mask.
        # Passing the 4D mask through made the migrated text graph produce
        # no-box/corrupt text even though the package sections were present.
        inverted_mask = 1.0 - attention_mask
        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
        )

    module = _vendored_qwen_module(qwen_model)
    return module._prepare_4d_causal_attention_mask(  # pylint: disable=protected-access
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
        sliding_window=qwen_model.config.sliding_window,
    )


def _export_safe_rope_from_position_ids(
    rotary_emb: torch.nn.Module,
    position_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Qwen RoPE values for absolute positions without table gather."""

    if not hasattr(rotary_emb, "inv_freq"):
        raise ValueError("LocateAnything export requires Qwen-style rotary inv_freq.")
    inv_freq = rotary_emb.inv_freq.to(device=device, dtype=torch.float32)
    # The sample export uses concrete position values; without a barrier the
    # converter can fold the formula branch away and leave runtime ``input_pos``
    # connected only to KV-cache offsets. Keep absolute positions as graph data
    # before deriving COS/SIN from them.
    position_ids = litert_optimization_barrier.optimization_barrier(position_ids)
    positions = position_ids.to(device=device, dtype=inv_freq.dtype)
    freqs = positions.unsqueeze(-1) * inv_freq.reshape(1, 1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _export_safe_apply_position_aligned_rotary(
    module,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None,
    unsqueeze_dim: int = 1,
):
    """Apply already position-aligned Qwen RoPE values.

    LocateAnything's regular CPU/GPU graph uses this helper after computing
    formula RoPE rows from ``inv_freq * position_ids`` so the known-good LiteRT
    conversion keeps the runtime-position path:
    input_pos -> RESHAPE -> CAST -> RESHAPE -> MUL -> CONCATENATION -> COS/SIN
    without cached-table ``GATHER_ND``. The split-cache lane also uses this
    helper after the caller supplies RoPE rows aligned to ``position_ids``.
    """

    if position_ids is None:
        raise ValueError("LocateAnything LiteRT export requires RoPE position ids.")
    if position_ids.ndim != 2:
        raise ValueError("LocateAnything LiteRT export expects 2D RoPE position ids.")

    batch_size = position_ids.shape[0]
    seq_length = position_ids.shape[1]
    head_dim = cos.shape[-1]
    cos = cos.reshape(batch_size, seq_length, head_dim)
    sin = sin.reshape(batch_size, seq_length, head_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (module.rotate_half(q) * sin)
    k_embed = (k * cos) + (module.rotate_half(k) * sin)
    return q_embed, k_embed


def _export_safe_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """LocateAnything Qwen attention forward using LiteRT transposed attention."""

    del kwargs  # Unused by the export-safe path.
    if output_attentions:
        raise NotImplementedError("LocateAnything LiteRT export does not expose attentions.")

    module = _vendored_qwen_module(self)
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Keep the converter-proven formula RoPE shape for regular CPU/GPU export.
    # The known-good TFLite graph lowers this to:
    # input_pos -> RESHAPE -> CAST -> RESHAPE -> MUL -> CONCATENATION -> COS/SIN,
    # instead of a cached-table GATHER_ND over [max_position_embeddings, dim].
    cos, sin = _export_safe_rope_from_position_ids(
        self.rotary_emb,
        position_ids,
        dtype=value_states.dtype,
        device=value_states.device,
    )
    query_states, key_states = _export_safe_apply_position_aligned_rotary(
        module,
        query_states,
        key_states,
        cos,
        sin,
        position_ids,
    )
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError("LocateAnything LiteRT export requires attention layer_idx.")
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attn_output, _ = litert_attention.transposed_attention(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        k_ts_idx=self.export_k_ts_idx,
        v_ts_idx=self.export_v_ts_idx,
    )
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def _export_safe_split_cache_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """LocateAnything Qwen attention forward using LiteRT split-cache attention."""

    del use_cache, kwargs  # Split-cache export always feeds an explicit LiteRT cache.
    if output_attentions:
        raise NotImplementedError("LocateAnything LiteRT split-cache export does not expose attentions.")
    if position_ids is None:
        raise ValueError("LocateAnything LiteRT split-cache export requires external RoPE position ids.")
    if past_key_value is None:
        raise ValueError("LocateAnything LiteRT split-cache export requires a LiteRT split cache.")

    module = _vendored_qwen_module(self)
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = _export_safe_apply_position_aligned_rotary(
        module,
        query_states, key_states, cos, sin, position_ids
    )

    if self.layer_idx is None:
        raise ValueError("LocateAnything LiteRT split-cache export requires attention layer_idx.")
    cache_kwargs = {"sin": sin, "cos": cos}
    key_states, value_states = past_key_value.update(
        key_states, value_states, self.layer_idx, cache_kwargs
    )

    attn_output, _ = litert_split_cache_attention.split_cache_attention(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        k_ts_idx=self.export_k_ts_idx,
        v_ts_idx=self.export_v_ts_idx,
    )
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def _export_safe_qwen2_model_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    visual_features: torch.FloatTensor | None = None,
    image_token_index: int | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
):
    """Export-safe LocateAnything Qwen2Model forward for external embeddings."""

    module = _vendored_qwen_module(self)
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Cannot specify both input_ids and inputs_embeds.")
    if input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("LocateAnything export requires input_ids or inputs_embeds.")

    use_legacy_cache = False
    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, module.Cache)
        if use_legacy_cache:
            if past_key_values is None:
                past_key_values = module.DynamicCache()
            else:
                past_key_values = module.DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_seq_length()

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.image_processing(input_ids, visual_features, image_token_index)

    attention_mask = _export_safe_prepare_attention_mask(
        self,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return module.BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def _patch_export_safe_language_model(
    language_model: torch.nn.Module, export_config, *, split_cache: bool = False
) -> None:
    """Install LocateAnything export-safe forward methods on the loaded instance."""

    qwen_model = getattr(language_model, "model", None)
    if qwen_model is None:
        raise ValueError("LocateAnything language model has no inner Qwen2 model.")
    qwen_model.forward = MethodType(_export_safe_qwen2_model_forward, qwen_model)
    attention_forward = (
        _export_safe_split_cache_attention_forward
        if split_cache
        else _export_safe_attention_forward
    )
    for decoder_layer in qwen_model.layers:
        attention = decoder_layer.self_attn
        attention.export_k_ts_idx = export_config.k_ts_idx
        attention.export_v_ts_idx = export_config.v_ts_idx
        attention.forward = MethodType(attention_forward, attention)


class LiteRTExportableModuleForLocateAnythingPrefillExternalEmbedder(
    external_emb_module.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder
):
    """Prefill exportable using externalized image/text embeddings."""

    def __init__(self, model: torch.nn.Module, export_config):
        language_model = _language_model(model)
        _patch_export_safe_prefix_helper(language_model)
        _patch_export_safe_language_model(language_model, export_config)
        super().__init__(language_model, export_config)

    def forward(self, embeddings, input_pos, kv_cache, mask, **kwargs):
        inputs = self.adapt_inputs(
            None,
            embeddings,
            input_pos,
            kv_cache,
            mask,
            use_bool_mask=self.export_config.extra_kwargs.get("use_bool_mask", False),
            **kwargs,
        )
        inputs |= self.attention_kwargs()
        inputs = _drop_locateanything_unsupported_inputs(inputs)
        output = self.model(**inputs)
        return {"kv_cache": output.past_key_values}


class LiteRTExportableModuleForLocateAnythingGenerateExternalEmbedder(
    external_emb_module.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder
):
    """Decode exportable using externalized image/text embeddings."""

    def __init__(self, model: torch.nn.Module, export_config):
        language_model = _language_model(model)
        _patch_export_safe_prefix_helper(language_model)
        _patch_export_safe_language_model(language_model, export_config)
        super().__init__(language_model, export_config)

    def forward(self, embeddings, input_pos, kv_cache, mask, **kwargs):
        inputs = self.adapt_inputs(
            None,
            embeddings,
            input_pos,
            kv_cache,
            mask,
            use_bool_mask=self.export_config.extra_kwargs.get("use_bool_mask", False),
            **kwargs,
        )
        inputs |= self.attention_kwargs()
        inputs = _drop_locateanything_unsupported_inputs(inputs)
        output = self.model(**inputs)
        return {"kv_cache": output.past_key_values, "logits": output.logits}


def _locateanything_rope_table(rope_tensor: torch.Tensor) -> torch.Tensor:
    """Return RoPE table shape expected by LocateAnything's vendored Qwen2."""

    if rope_tensor.ndim == 4 and rope_tensor.shape[0] == 1 and rope_tensor.shape[2] == 1:
        return rope_tensor.squeeze(0).squeeze(1)
    if rope_tensor.ndim == 3 and rope_tensor.shape[0] == 1:
        return rope_tensor.squeeze(0)
    return rope_tensor


def _adapt_locateanything_split_cache_inputs(exportable, embeddings, pos_emb, mask, kv_cache):
    """Adapt split-cache inputs and fix external RoPE table shape for LocateAnything."""

    inputs = split_cache_module.LiteRTSplitCacheExportableModuleForDecoderOnlyLM.adapt_inputs(
        exportable, embeddings, pos_emb, mask, kv_cache
    )
    language_model = exportable.model.model
    language_model.rotary_emb.data = (
        _locateanything_rope_table(pos_emb["cos"]),
        _locateanything_rope_table(pos_emb["sin"]),
    )
    return inputs


class LiteRTExportableModuleForLocateAnythingPrefillSplitCache(
    split_cache_module.LiteRTSplitCacheExportableModuleForDecoderOnlyLMPrefill
):
    """Prefill exportable using external embeddings, external RoPE, and LiteRT split cache."""

    def __init__(self, model: torch.nn.Module, export_config):
        language_model = _language_model(model)
        _patch_export_safe_prefix_helper(language_model)
        _patch_export_safe_language_model(language_model, export_config, split_cache=True)
        super().__init__(language_model, export_config)

    def adapt_inputs(self, embeddings, pos_emb, mask, kv_cache):
        return _adapt_locateanything_split_cache_inputs(self, embeddings, pos_emb, mask, kv_cache)

    def forward(self, embeddings, pos_emb, mask, kv_cache):
        inputs = self.adapt_inputs(embeddings, pos_emb, mask, kv_cache)
        inputs |= self.attention_kwargs()
        inputs = _drop_locateanything_unsupported_inputs(inputs)
        output = self.model(**inputs)
        return self.post_process_kv_cache(output.past_key_values)


class LiteRTExportableModuleForLocateAnythingGenerateSplitCache(
    split_cache_module.LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate
):
    """Decode exportable using external embeddings, external RoPE, and LiteRT split cache."""

    def __init__(self, model: torch.nn.Module, export_config):
        language_model = _language_model(model)
        _patch_export_safe_prefix_helper(language_model)
        _patch_export_safe_language_model(language_model, export_config, split_cache=True)
        super().__init__(language_model, export_config)

    def adapt_inputs(self, embeddings, pos_emb, mask, kv_cache):
        return _adapt_locateanything_split_cache_inputs(self, embeddings, pos_emb, mask, kv_cache)

    def forward(self, embeddings, pos_emb, mask, kv_cache):
        inputs = self.adapt_inputs(embeddings, pos_emb, mask, kv_cache)
        inputs |= self.attention_kwargs()
        inputs = _drop_locateanything_unsupported_inputs(inputs)
        output = self.model(**inputs)
        return self.post_process_kv_cache(output.past_key_values) | {"logits": output.logits}
