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

"""Convert the re-authored LocateAnything-3B VLM to TFLite / LiteRT-LM.

Mirrors qwen_vl/convert_to_tflite.py but adapted for LocateAnything's
standard (N, 3, H, W) pixel_values convention (vs qwen_vl's flattened patches).

Key export parameters for 448x448 (the standard LocateAnything inference size):
  patch_size = 14  -> grid_h = grid_w = 448 // 14 = 32
  merge_kernel = 2x2 -> n_out_patches = (32//2) * (32//2) = 16*16 = 256
  pixel_values_size = torch.Size([1, 3, 448, 448])  (1 image, RGB)
  pixel_seq_len = 256  (connector output tokens injected into the sequence)

KV cache sizing:
  kv_cache_max_len must be > max(prefill_seq_lens) + pixel_seq_len.
  Default: 1280 satisfies 1024 + 256 = 1280.

Usage (standalone, no absl flags — use argparse):
  uv run python -m litert_torch.generative.examples.locateanything.convert_to_tflite \
    --checkpoint /path/to/LocateAnything-3B \
    --output_path /path/to/artifacts/ \
    [--output_format tflite|litertlm] \
    [--quantize none|dynamic_int8|fp16] \
    [--tokenizer_path /path/to/tokenizer.json] \
    [--prefill_seq_lens 128 256 512 1024] \
    [--kv_cache_max_len 1280] \
    [--image_height 448] \
    [--image_width 448]
"""

import argparse
import os
import sys
import types


# ---------------------------------------------------------------------------
# LLVM conflict bypass (same pattern as verify.py — needed on macOS because
# litert_torch/__init__.py and litert_torch.generative.layers.lora both pull
# in conflicting LLVM shared libraries).
# ---------------------------------------------------------------------------

def _bypass_llvm_conflicts():
    """Inject stubs before any litert_torch sub-module import.

    Problem: litert_torch/__init__.py imports ai_edge_litert which loads one
    LLVM C extension.  litert_torch.generative.layers.lora imports tensorflow.lite
    schema which loads a different LLVM C extension.  When both are loaded in the
    same process on macOS the two LLVM instances both try to register the same
    CommandLine option ("info-output-file") and crash.

    Solution:
    1. Stub the top-level litert_torch module so its __init__.py never runs.
       The sub-package __path__ is preserved so sub-modules can still be
       imported normally via Python's regular import machinery.
    2. Stub litert_torch.generative.layers.lora so tensorflow.lite C extension
       is never loaded.  LoRA is not used during conversion of LocateAnything.

    NOTE: the real litert_torch._convert.interface.Converter is still accessible
    after step 1 because the stub only blocks __init__.py execution, not the
    physical package directory.
    """
    if "litert_torch" not in sys.modules:
        stub = types.ModuleType("litert_torch")
        # Preserve __path__ so sub-packages resolve from the real file tree.
        stub.__path__ = [
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                *[".."] * 4,
                "litert_torch",
            )
        ]
        stub.__package__ = "litert_torch"
        stub.__spec__ = None
        sys.modules["litert_torch"] = stub
        # The bypassed __init__.py would bind `litert_torch.config`. The
        # optimize_layout_transposes pass (triggered by 4D layout permutes such
        # as the GPU-friendly pixel-unshuffle merger) reads litert_torch.config.
        # _config.py only imports logging/os (no LLVM C-extension), so importing
        # it here is safe and restores the attribute the stub would otherwise lack.
        from litert_torch._config import config as _real_config
        stub.config = _real_config

    lora_key = "litert_torch.generative.layers.lora"
    if lora_key not in sys.modules:
        fake_lora = types.ModuleType(lora_key)

        class _FakeLoRA:  # noqa: N801
            pass

        fake_lora.LoRA = _FakeLoRA
        fake_lora.LoRAEntry = _FakeLoRA
        fake_lora.AttentionLoRA = _FakeLoRA
        sys.modules[lora_key] = fake_lora


# Bypass BEFORE any litert_torch imports.
_bypass_llvm_conflicts()

# Now safe to import the generative sub-modules.
import torch

from litert_torch.generative.examples.locateanything import locateanything as la_module
from litert_torch.generative.examples.locateanything import decoder as dec_module
from litert_torch.generative.utilities import converter
from litert_torch.generative.utilities import export_config as export_cfg
from litert_torch.generative.utilities import litertlm_builder


# ---------------------------------------------------------------------------
# Constants (LocateAnything-3B default image size)
# ---------------------------------------------------------------------------

_DEFAULT_IMAGE_H = 448
_DEFAULT_IMAGE_W = 448
_PATCH_SIZE = 14
_MERGE_KERNEL = 2   # both h and w


def compute_pixel_seq_len(image_h: int, image_w: int) -> int:
    """Number of connector output tokens for one image of (image_h, image_w)."""
    gh = image_h // _PATCH_SIZE
    gw = image_w // _PATCH_SIZE
    return (gh // _MERGE_KERNEL) * (gw // _MERGE_KERNEL)


def compute_pixel_values_size(image_h: int, image_w: int, n_images: int = 1) -> torch.Size:
    """Shape of pixel_values tensor: (n_images, 3, H, W)."""
    return torch.Size([n_images, 3, image_h, image_w])


# ---------------------------------------------------------------------------
# Argument parsing (no absl dependency — simpler than flags for a script)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert re-authored LocateAnything-3B to TFLite / LiteRT-LM."
    )
    p.add_argument(
        "--checkpoint",
        default=os.path.join(
            os.path.dirname(__file__),
            "../../../../../../Assets/ModelStore/LocateAnything-3B",
        ),
        help="Path to LocateAnything-3B safetensors checkpoint directory.",
    )
    p.add_argument(
        "--output_path",
        default="/tmp/",
        help="Output directory for .tflite / .litertlm.",
    )
    p.add_argument(
        "--output_name_prefix",
        default="locateanything",
        help="Prefix for the output file name.",
    )
    p.add_argument(
        "--output_format",
        choices=["tflite", "litertlm"],
        default="tflite",
        help="Output format: 'tflite' (default) or 'litertlm' (requires tokenizer).",
    )
    p.add_argument(
        "--quantize",
        default="dynamic_int8",
        choices=["none", "dynamic_int8", "weight_only_int8", "fp16",
                 "dynamic_int4_block32", "dynamic_int4_block128"],
        help="Quantization strategy.",
    )
    p.add_argument(
        "--tokenizer_path",
        default=None,
        help="Path to tokenizer.json (HF format). Required for --output_format litertlm.",
    )
    p.add_argument(
        "--base_llm_metadata",
        default=None,
        help="Path to a base LlmMetadata .textproto/.pb. NOTE: supplying a base drops the "
             "auto-detected vision encoder section (TF_LITE_VISION_ENCODER); prefer --stop_token_ids.",
    )
    p.add_argument(
        "--stop_token_ids",
        type=int,
        nargs="+",
        default=[151645],
        help="Stop token ids added to the auto-generated metadata (default: [151645] = <|im_end|>). "
             "The runtime requires stop tokens; this keeps the auto vision-encoder section intact.",
    )
    p.add_argument(
        "--prefill_seq_lens",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="List of prefill sequence lengths (multiple = multiple signatures).",
    )
    p.add_argument(
        "--kv_cache_max_len",
        type=int,
        default=1280,
        help="KV cache max length (must be > max(prefill_seq_lens) + pixel_seq_len).",
    )
    p.add_argument(
        "--image_height",
        type=int,
        default=_DEFAULT_IMAGE_H,
        help="Input image height in pixels.",
    )
    p.add_argument(
        "--image_width",
        type=int,
        default=_DEFAULT_IMAGE_W,
        help="Input image width in pixels.",
    )
    p.add_argument(
        "--n_images",
        type=int,
        default=1,
        help="Number of images per export (default 1).",
    )
    p.add_argument(
        "--mask_as_input",
        action="store_true",
        help="Pass causal mask as an explicit input tensor (for GPU-dynamic-shape mode).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Validate.
    pixel_seq_len = compute_pixel_seq_len(args.image_height, args.image_width)
    pixel_values_size = compute_pixel_values_size(
        args.image_height, args.image_width, args.n_images
    )
    max_prefill = max(args.prefill_seq_lens)
    required_kv = max_prefill + pixel_seq_len
    if args.kv_cache_max_len <= required_kv:
        print(
            f"ERROR: kv_cache_max_len={args.kv_cache_max_len} must be > "
            f"max_prefill({max_prefill}) + pixel_seq_len({pixel_seq_len}) = {required_kv}. "
            f"Increase --kv_cache_max_len."
        )
        sys.exit(1)

    if args.output_format == "litertlm" and not args.tokenizer_path:
        print("ERROR: --tokenizer_path is required for --output_format litertlm.")
        sys.exit(1)

    print(f"[convert] checkpoint:       {args.checkpoint}")
    print(f"[convert] output_path:      {args.output_path}")
    print(f"[convert] output_format:    {args.output_format}")
    print(f"[convert] quantize:         {args.quantize}")
    print(f"[convert] image:            {args.image_height}x{args.image_width}, n={args.n_images}")
    print(f"[convert] pixel_seq_len:    {pixel_seq_len}")
    print(f"[convert] pixel_values_sz:  {tuple(pixel_values_size)}")
    print(f"[convert] prefill_seq_lens: {args.prefill_seq_lens}")
    print(f"[convert] kv_cache_max_len: {args.kv_cache_max_len}")
    print(f"[convert] mask_as_input:    {args.mask_as_input}")

    # Build model.
    print("\n[convert] Loading model weights ...")
    pytorch_model = la_module.build_full_model(
        checkpoint_path=args.checkpoint,
        mask_cache_size=args.kv_cache_max_len,
        image_h=args.image_height,
        image_w=args.image_width,
    )
    pytorch_model.eval()
    print("[convert] Model loaded OK.")

    # ExportConfig.
    ec = export_cfg.ExportConfig(mask_as_input=args.mask_as_input)

    # Decoder config (for KV cache construction inside converter).
    decoder_config = dec_module.get_decoder_config()

    os.makedirs(args.output_path, exist_ok=True)

    if args.output_format == "tflite":
        print("\n[convert] Converting to TFLite ...")
        out_file = converter.convert_to_tflite(
            pytorch_model,
            output_path=args.output_path,
            output_name_prefix=args.output_name_prefix,
            prefill_seq_len=args.prefill_seq_lens,
            kv_cache_max_len=args.kv_cache_max_len,
            pixel_values_size=pixel_values_size,
            pixel_seq_len=pixel_seq_len,
            quantize=args.quantize,
            config=decoder_config,
            export_config=ec,
        )
        print(f"[convert] TFLite written: {out_file}")
        sz = os.path.getsize(out_file)
        print(f"[convert] File size: {sz / 1e6:.1f} MB")
    else:
        print("\n[convert] Converting to LiteRT-LM (.litertlm) ...")
        if not litertlm_builder.is_litertlm_builder_available():
            print(
                "ERROR: litert_lm_builder is not available in this Python environment. "
                "Install it via: pip install litert-lm-builder"
            )
            sys.exit(1)
        converter.convert_to_litert(
            pytorch_model,
            output_path=args.output_path,
            output_name_prefix=args.output_name_prefix,
            prefill_seq_len=args.prefill_seq_lens,
            kv_cache_max_len=args.kv_cache_max_len,
            pixel_values_size=pixel_values_size,
            pixel_seq_len=pixel_seq_len,
            quantize=args.quantize,
            config=decoder_config,
            export_config=ec,
            output_format="litertlm",
            hf_tokenizer_model_path=args.tokenizer_path,
            base_llm_metadata_path=args.base_llm_metadata,
            stop_token_ids=args.stop_token_ids,
        )
        # litertlm name follows the same stem as tflite.
        quant_sfx = converter.create_quantize_suffix(args.quantize)
        kv = args.kv_cache_max_len
        expected_stem = f"{args.output_name_prefix}_{quant_sfx}_ekv{kv}"
        litertlm_path = os.path.join(args.output_path, f"{expected_stem}.litertlm")
        if os.path.exists(litertlm_path):
            sz = os.path.getsize(litertlm_path)
            print(f"[convert] LiteRT-LM written: {litertlm_path}")
            print(f"[convert] File size: {sz / 1e6:.1f} MB")
        else:
            # Search for it.
            candidates = [f for f in os.listdir(args.output_path) if f.endswith(".litertlm")]
            print(f"[convert] LiteRT-LM candidates in {args.output_path}: {candidates}")

    print("[convert] Done.")


if __name__ == "__main__":
    main()
