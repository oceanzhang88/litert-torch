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

"""Phase 1 verification: re-authored decoder vs HF reference (text-only).

Uses transformers.Qwen2ForCausalLM (upstream, no relative-import issues) as the
reference, loaded from the LocateAnything-3B safetensors by stripping the
"language_model." prefix and renaming "model.norm" -> "model.norm.weight".

Usage:
    /path/to/nightsoul-python/bin/python \
        -m litert_torch.generative.examples.locateanything.verify \
        --checkpoint /path/to/LocateAnything-3B/

    Or pass no flags — it reads LOCATEANYTHING_CHECKPOINT from env, falling
    back to the default path in Assets/ModelStore.
"""

import argparse
import logging
import os
import sys
import types

# ---------------------------------------------------------------------------
# LLVM conflict bypass (macOS only).
#
# litert_torch/__init__.py imports ai_edge_litert which loads a LLVM C
# extension.  litert_torch.generative.layers.lora imports tensorflow.lite
# schema which loads a different LLVM C extension.  When both are loaded in
# the same process on macOS the two LLVM instances both try to register the
# same CommandLine option ("info-output-file") and crash.
#
# For verification we do not need LoRA or the top-level litert_torch convert
# interface.  Bypassing them entirely (via sys.modules stubs) lets all the
# generative sub-modules load cleanly.
# ---------------------------------------------------------------------------
def _bypass_llvm_conflicts():
    """Inject stubs before any litert_torch sub-module import."""
    # 1. Stub the top-level package so its __init__.py (which loads
    #    ai_edge_litert._pywrap_litert_interpreter_wrapper) never runs.
    if "litert_torch" not in sys.modules:
        stub = types.ModuleType("litert_torch")
        stub.__path__ = [
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                *[".."] * 4,  # up to litert_torch/ root
                "litert_torch",
            )
        ]
        stub.__package__ = "litert_torch"
        stub.__spec__ = None
        sys.modules["litert_torch"] = stub

    # 2. Stub lora so tensorflow.lite C extension is never loaded.  LoRA is
    #    not used during pure-text verification.
    lora_key = "litert_torch.generative.layers.lora"
    if lora_key not in sys.modules:
        fake_lora = types.ModuleType(lora_key)
        # Provide the attrs that attention.py and model_builder.py reference.
        class _FakeLoRA:  # noqa: N801
            pass
        fake_lora.LoRA = _FakeLoRA
        fake_lora.LoRAEntry = _FakeLoRA
        fake_lora.AttentionLoRA = _FakeLoRA
        sys.modules[lora_key] = fake_lora


_bypass_llvm_conflicts()

import torch  # noqa: E402 (must come after bypass)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("locateanything.verify")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    return torch.nn.functional.cosine_similarity(
        af.unsqueeze(0), bf.unsqueeze(0)
    ).item()


# ---------------------------------------------------------------------------
# HF reference: Qwen2ForCausalLM loaded from LocateAnything safetensors
#
# Strategy: load the raw safetensors, filter keys with prefix
# "language_model.", strip that prefix, then load into a standard
# transformers.Qwen2ForCausalLM initialized from the text_config in
# config.json.  This avoids the relative-import problems in the bundled
# modeling_qwen2.py and the peft dependency in modeling_locateanything.py.
# ---------------------------------------------------------------------------

def _load_hf_decoder(checkpoint_path: str) -> torch.nn.Module:
    """Loads transformers.Qwen2ForCausalLM from LocateAnything-3B weights."""
    import json
    from transformers import Qwen2ForCausalLM, Qwen2Config
    from safetensors import safe_open
    import glob

    log.info("Loading config.json from: %s", checkpoint_path)
    config_file = os.path.join(checkpoint_path, "config.json")
    with open(config_file) as f:
        full_config = json.load(f)
    text_cfg = full_config["text_config"]

    # Build Qwen2Config matching the LocateAnything decoder exactly.
    qwen_config = Qwen2Config(
        vocab_size=text_cfg["vocab_size"],          # 152681
        hidden_size=text_cfg["hidden_size"],         # 2048
        intermediate_size=text_cfg["intermediate_size"],  # 11008
        num_hidden_layers=text_cfg["num_hidden_layers"],  # 36
        num_attention_heads=text_cfg["num_attention_heads"],  # 16
        num_key_value_heads=text_cfg["num_key_value_heads"],  # 2
        max_position_embeddings=text_cfg["max_position_embeddings"],  # 32768
        rms_norm_eps=text_cfg["rms_norm_eps"],       # 1e-6
        rope_theta=text_cfg["rope_theta"],           # 1e6
        tie_word_embeddings=text_cfg["tie_word_embeddings"],  # True
        hidden_act=text_cfg["hidden_act"],           # "silu"
        attention_dropout=text_cfg.get("attention_dropout", 0.0),
        torch_dtype="float32",   # fp32 for numeric parity comparison
    )

    log.info(
        "Qwen2Config: vocab=%d layers=%d h=%d heads=%d/%d",
        qwen_config.vocab_size,
        qwen_config.num_hidden_layers,
        qwen_config.hidden_size,
        qwen_config.num_attention_heads,
        qwen_config.num_key_value_heads,
    )

    # Instantiate model directly on CPU (NOT via meta+to_empty).
    #
    # Pitfall: using `torch.device("meta") + to_empty(device="cpu")` zeroes
    # ALL buffers including register_buffer(..., persistent=False) ones like
    # rotary_emb.inv_freq and rotary_emb.original_inv_freq.  Since inv_freq is
    # NOT stored in the safetensors checkpoint (it is computed, not trained),
    # load_state_dict(strict=False) leaves it at zero → cos=1.0, sin=0.0 for
    # every position → RoPE silently does nothing → logits are totally wrong.
    #
    # Instantiating directly on CPU avoids this: __init__ calls
    # compute_default_rope_parameters which populates inv_freq correctly.
    log.info("Instantiating Qwen2ForCausalLM (direct CPU, no meta)...")
    hf_model = Qwen2ForCausalLM(qwen_config)

    # Load the safetensors, extract language_model.* keys, strip prefix.
    log.info("Loading safetensors from: %s", checkpoint_path)
    pattern = os.path.join(checkpoint_path, "*.safetensors")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files in {checkpoint_path}")
    log.info("Found %d shard(s): %s", len(shard_files), [os.path.basename(f) for f in shard_files])

    lm_state = {}
    for shard in shard_files:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("language_model."):
                    # Strip the "language_model." prefix to get standard keys
                    stripped = key[len("language_model."):]
                    lm_state[stripped] = f.get_tensor(key).float()  # fp32

    log.info("Extracted %d language_model tensors", len(lm_state))

    # Load into the HF model. strict=False because computed buffers (inv_freq,
    # original_inv_freq) are not in the checkpoint and must NOT be overwritten.
    missing, unexpected = hf_model.load_state_dict(lm_state, strict=False)
    if missing:
        # Warn but don't fail — rotary buffers like inv_freq are computed, not stored.
        inv_freq_missing = [k for k in missing if "inv_freq" in k]
        real_missing = [k for k in missing if "inv_freq" not in k]
        if real_missing:
            log.warning("Missing (non-inv_freq) keys in HF model: %s", real_missing[:10])
        if inv_freq_missing:
            log.info("inv_freq keys (computed, OK to be missing): %d", len(inv_freq_missing))
    if unexpected:
        log.warning("Unexpected keys in state dict: %s", unexpected[:10])

    # Verify inv_freq is non-zero (sanity check after load).
    inv_freq = hf_model.model.rotary_emb.inv_freq
    if inv_freq.abs().max().item() < 1e-9:
        raise RuntimeError(
            "rotary_emb.inv_freq is zero after load — RoPE will be silent. "
            "Do NOT use meta+to_empty loading for this model."
        )
    log.info("rotary_emb.inv_freq OK: max=%.4f", inv_freq.max().item())

    hf_model.eval()
    log.info("HF reference decoder loaded. Type: %s", type(hf_model).__name__)
    return hf_model


# ---------------------------------------------------------------------------
# HF-side wrapper
# ---------------------------------------------------------------------------

class HFDecoderWrapper:
    """Wraps Qwen2ForCausalLM for text-only logit extraction."""

    def __init__(self, hf_model: torch.nn.Module):
        self.model = hf_model

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns logits shape (1, seq_len, vocab_size)."""
        out = self.model(
            input_ids=input_ids,
            attention_mask=None,
            use_cache=False,
        )
        return out.logits


# ---------------------------------------------------------------------------
# Re-authored side wrapper
# ---------------------------------------------------------------------------

class ReauthoredDecoderWrapper:
    """Wraps the litert-torch Decoder for text-only logit extraction."""

    def __init__(self, decoder, kv_cache_max_len: int = 256):
        from litert_torch.generative.layers import kv_cache as kv_utils
        from litert_torch.generative.utilities import export_config as export_cfg

        self.decoder = decoder
        self.kv_cache_max_len = kv_cache_max_len
        self.export_config = export_cfg.ExportConfig(output_logits_on_prefill=True)
        self._kv_utils = kv_utils

    def _fresh_kv_cache(self):
        return self._kv_utils.KVCache.from_model_config(
            self.kv_cache_max_len, self.decoder.config
        )

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns logits shape (1, seq_len, vocab_size)."""
        seq_len = input_ids.shape[1]
        input_pos = torch.arange(0, seq_len, dtype=torch.int)
        kv_cache = self._fresh_kv_cache()
        out = self.decoder.forward(
            tokens=input_ids,
            input_pos=input_pos,
            kv_cache=kv_cache,
            export_config=self.export_config,
        )
        return out["logits"]


# ---------------------------------------------------------------------------
# Main verify routine
# ---------------------------------------------------------------------------

def verify_decoder(checkpoint_path: str, kv_cache_max_len: int = 256):
    log.info("=== Phase 1: Decoder parity (text-only) ===")
    log.info("Checkpoint: %s", checkpoint_path)

    # ------------------------------------------------------------------
    # Load HF reference decoder
    # ------------------------------------------------------------------
    hf_model = _load_hf_decoder(checkpoint_path)
    hf_wrapper = HFDecoderWrapper(hf_model)

    # ------------------------------------------------------------------
    # Load re-authored decoder
    # ------------------------------------------------------------------
    log.info("Building re-authored decoder from: %s", checkpoint_path)
    try:
        from litert_torch.generative.examples.locateanything.decoder import (
            build_decoder,
        )
        reauthored = build_decoder(checkpoint_path, mask_cache_size=kv_cache_max_len)
        log.info("Re-authored decoder built. num_layers=%d", reauthored.config.num_layers)
    except Exception as e:
        log.error("Failed to build re-authored decoder: %s", e)
        raise

    re_wrapper = ReauthoredDecoderWrapper(reauthored, kv_cache_max_len=kv_cache_max_len)

    # ------------------------------------------------------------------
    # Test cases: sequences with token IDs in known ranges
    # ------------------------------------------------------------------
    test_cases = [
        ("text_ids_short",   [1, 2, 3, 4]),
        ("text_ids_medium",  list(range(1, 17))),
        # coord/grounding token IDs that the decoder must support
        ("coord_token_ids",  [151665, 151668, 151677, 152677, 151669]),
    ]

    all_passed = True
    results = {}
    for name, input_ids_list in test_cases:
        seq_len = len(input_ids_list)
        # Pad to kv_cache_max_len so both sides get the same input shape
        padded = input_ids_list + [0] * (kv_cache_max_len - seq_len)
        input_ids = torch.tensor([padded], dtype=torch.long)

        log.info("--- Test case: %s (seq_len=%d) ---", name, seq_len)

        logits_hf = hf_wrapper.forward(input_ids)
        logits_re = re_wrapper.forward(input_ids)

        # Compare at the last real (non-padded) token position
        pos = seq_len - 1
        lhf = logits_hf[0, pos, :].float()
        lre = logits_re[0, pos, :].float()

        max_abs = _max_abs(lhf, lre)
        cos    = _cosine(lhf, lre)
        top1_hf = lhf.argmax().item()
        top1_re = lre.argmax().item()
        passed = max_abs < 1e-3 and cos > 0.9999

        results[name] = {
            "max_abs": max_abs,
            "cosine": cos,
            "top1_hf": top1_hf,
            "top1_re": top1_re,
            "passed": passed,
        }

        log.info(
            "  max_abs=%.6e  cosine=%.8f  top1_hf=%d  top1_re=%d  %s",
            max_abs, cos, top1_hf, top1_re,
            "PASS" if passed else "FAIL",
        )
        if not passed:
            all_passed = False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== DECODER PARITY SUMMARY ===")
    for name, r in results.items():
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {name:30s}: max_abs={r['max_abs']:.3e}  cos={r['cosine']:.8f}  "
            f"top1: {r['top1_hf']}->{r['top1_re']}  [{status}]"
        )
    print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    default_ckpt = os.environ.get(
        "LOCATEANYTHING_CHECKPOINT",
        "/Users/ocean/TheTraveler/TheBridge/Assets/ModelStore/LocateAnything-3B",
    )
    parser = argparse.ArgumentParser(
        description="Phase 1 — verify re-authored LocateAnything decoder vs HF"
    )
    parser.add_argument(
        "--checkpoint",
        default=default_ckpt,
        help="Path to LocateAnything-3B weights directory",
    )
    parser.add_argument(
        "--kv-cache-max-len",
        type=int,
        default=256,
        help="KV cache / sequence pad length (default: 256)",
    )
    args = parser.parse_args()

    passed = verify_decoder(
        checkpoint_path=args.checkpoint,
        kv_cache_max_len=args.kv_cache_max_len,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
