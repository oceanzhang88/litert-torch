# LocateAnything LiteRT Torch model extension

This directory is the in-tree LiteRT Torch model extension for LocateAnything:

```text
litert_torch.generative.export_hf.model_ext.locateanything
```

It is not a deployable phone runtime by itself and it is not the `.litertlm` artifact.
It is the exporter implementation that `litert-torch export_hf` loads from the
owned `4d-pocket/litert-torch` checkout.

## Owned section map

| LiteRT-LM section | LocateAnything source | Extension file |
|---|---|---|
| `PREFILL_DECODE` | `model.language_model` | `exportable_module.py` |
| `EMBEDDER` | `model.language_model.get_input_embeddings()` | upstream generic exporter |
| `AUX` / `tf_lite_aux` | external RoPE, split attention masks, cache update helpers for LiteRT-LM NPU executor | `exportable_module.py` split-cache exportables plus upstream auxiliary exporter |
| `VISION_ENCODER` | `images [1,448,448,3] -> internal MoonViT patches -> model.vision_model(pixel_values, grid_hws)` | `vision_exportable.py` |
| `VISION_ADAPTER` | `model.mlp1(features)` | `vision_exportable.py` |
| metadata | Gemma3 image processor envelope plus Qwen chat template and LocateAnything tokens | `metadata_builder.py` |

## W1 fixed-image export contract

The first package proof deliberately targets a fixed, image-only LocateAnything lane:

```text
image size: 448x448
MoonViT grid: 32x32
merged soft tokens: 256
decode: slow greedy autoregressive
```

The extension patches the loaded model instance during export:

- text export installs a LocateAnything-safe Qwen2 forward and LiteRT transposed
  self-attention path for external image/text embeddings;
- NPU text export uses LocateAnything-specific split-cache prefill/decode exportables
  so upstream LiteRT Torch can emit the `auxiliary.tflite` RoPE/mask/cache section
  that LiteRT-LM packages as `tf_lite_aux`;
- vision export accepts the LiteRT-LM image processor envelope as an NHWC
  `[1, 448, 448, 3]` image tensor, derives the fixed MoonViT patch tensor and grid
  inside the exportable, precomputes one packed real RoPE cos/sin buffer, precomputes the
  fixed 32x32 position embedding as `export_pos_emb_32x32`, and uses one-image full
  attention to avoid Python `.tolist()`, runtime position-embedding interpolation, and
  unsupported complex lowering;
- `decord` is shimmed only for macOS image export because video is not part of W1.

W1's current phone-owner package is the position-buffer export at:

```text
Assets/ModelStore/LocateAnything-3B-litertlm-experiment/litert-torch-probe-posbuf-rawvision/model.litertlm
```

The package keeps text/embedder sections quantized but leaves vision encoder/adapter raw:
the earlier quantized vision sections failed compiler fidelity after raw TFLite passed.
Local LiteRT-LM Mac CPU smoke returns bbox-parity LocateAnything box tokens for the
synthetic red-square fixture. HF CPU slow decode returns:

```text
<ref>Locate the red square</ref><box><279><342><721><785></box><|im_end|>
```

LiteRT-LM returns:

```text
<ref>Locate the red square</ref><box><278><342><721><785></box>
```

The comparison passes at IoU `0.9977426636568849`. The current saved
compiler-fidelity gate is:

```text
Assets/ModelStore/LocateAnything-3B-litertlm-experiment/parity-fixtures/red-square-vision-tflite.json
```

This package proves the `export_pos_emb_32x32` lowering without losing local behavior:
`litert-package-inspect` reports all six LiteRT-LM sections, `litert-vision-parity` passes
with vision mean abs `0.000004192`, and `litert-parity` passes HF-vs-LiteRT red-square
bbox parity with IoU `1.0`. Its side-lane inventory removes the old MoonViT
`BROADCAST_TO` and `GATHER_ND` blockers (`8 -> 0` and `16 -> 0`) while reducing
`RESHAPE` (`641 -> 631`) and `TRANSPOSE` (`129 -> 112`). It is now also the physical
phone-view CPU ROI owner: `R3CY80SA4PN` loaded the posbuf package in the Android ROI app
and returned one `golf ball` bbox for `01-golf-ball.png`. It also has a hybrid
CPU-main / vision-NPU diagnostic receipt on the same S26 Ultra, but that receipt does not
claim full NPU serving because the text backend remains CPU.

The next MoonViT GPU source repair has landed and has local export proof, but it is not a
serving claim yet. `_nhwc_image_to_locateanything_patches()` now uses a fixed-grid Conv2D
identity extractor, and `_fixed_patch_merger()` uses rank-3 slicing plus concat instead of
the older five-dimensional reshape/permute paths. The archive export harness runs
`tests.litert.test_vision_exportable` and confirms both the Conv2D patchifier and the 2x2
patch merger match the legacy fixed-grid layout. The fresh
`litert-torch-probe-posbuf-rawvision-convpatch-kernelbuf-merge4d` export completed and
packages as a standard six-section CPU/GPU `.litertlm`; `npu_aux` is intentionally absent
from that regular package.

Current merge4d result: packaged `vision_encoder.tflite` vs patched PyTorch mean abs
`0.000004192`, packaged `vision_adapter.tflite` vs patched PyTorch mean abs
`0.000010326`, and FlatBuffers rank audit reports `RESHAPE` / `TRANSPOSE`
`rank_gt4_count=0` with `max_rank=4`. The Mac currently has no ADB-visible phone, and full
GPU still needs a fresh physical receipt, so this remains source/export/parity evidence,
not `real-litertlm-present-runtime-bbox-physical-android-gpu-ok`.

The current Qwen text GPU lowering lives in the same extension module. The
`merge4d-qwen-ropegather` experiment proved that `torch.index_select` removed the old
INT64 cast but still lowered cached RoPE rows to `GATHER_ND`. The follow-up
`merge4d-qwen-ropeformula` export computes regular RoPE rows from `inv_freq *
position_ids` and treats split-cache external RoPE rows as already position-aligned. Its
artifact-local `prefill_decode.json` inventory reports `GATHER_ND=0`, `SELECT=0`,
`COS=2`, `SIN=2`, `CAST=2`, and `operator_count=3847`; the current `CAST` samples are
`INT32 -> FLOAT32` RoPE-position conversions, not the historical `INT64` table-gather
reason. `LESS` is not part of the current artifact-local blocker list. This is the current
Qwen text graph proof, but it is not a GPU-serving claim until the physical S26 Ultra
accepts `backend=gpu`, `vision_backend=gpu` and returns parseable boxes.
Dynamic image sizes, video, optimized decoding, and Android GPU graph lowering remain later
work. The NPU aux export lane now has a ropefix package at
`litert-torch-probe-posbuf-rawvision-npu-aux-ropefix/model.litertlm`; package inspection
reports `npu_aux` / `tf_lite_aux` ready, the embedder export contains the
`decode_embedder` signature, and the 4d-pocket owner test proves legacy Qwen
`decode_rope` now gathers by absolute `input_pos` instead of reusing row zero. Full NPU
remains unclaimed: older aux receipts reached
exit `0` with no accelerator errors but no parseable box, while the first ropefix physical
probe exited `255` with empty stdout/stderr and the S26 Ultra disappeared from ADB; a
later macOS USB recheck saw `MSM_UPLOAD` instead of an Android/Samsung ADB target. At
least ten S26 Ultra images still must pass `backend=npu`,
`vision_backend=npu` after `R3CY80SA4PN` is visible again.

## Required in-tree dispatch

LiteRT Torch upstream special-cases Gemma-family vision exportables. This in-tree
extension keeps these dispatch points wired to `locateanything.registry` when
`model_config.model_type == "locateanything"`:

```text
model_ext/exportables.py:get_prefill_decode_exportables
model_ext/exportables.py:get_vision_exportables
model_ext/metadata_builder.py:get_metadata_builder
```

The loader dispatch must also stay in-tree before exportables run:

```text
core/export_lib.py:load_model
```

For `image_text_to_text`, upstream LiteRT Torch selects
`transformers.AutoModelForImageTextToText`. LocateAnything only registers its custom class
under `AutoModel`, so the loader must choose `AutoModel` for `model_type=locateanything`
or expose a working override before source weights can load. The current package proof uses
the owned checkout path, not an in-place site-packages monkey patch.

For NPU packaging, LocateAnything must not reject LiteRT Torch `split_cache=True`.
`locateanything.registry.get_prefill_decode_exportables` dispatches split-cache exports to
`LiteRTExportableModuleForLocateAnythingPrefillSplitCache` and
`LiteRTExportableModuleForLocateAnythingGenerateSplitCache`; those classes preserve the
LocateAnything-safe Qwen forward patches while using LiteRT split-cache attention. The
standard upstream `export_auxiliary_model()` then owns the `auxiliary.tflite` artifact that
the LiteRT-LM builder packages as `TfLiteModelType.AUX`.

## Artifact gate

Liyue may only claim a finished LocateAnything LiteRT-LM package after the export folder
contains real exporter outputs:

```text
model.tflite
embedder.tflite
auxiliary.tflite
vision_encoder.tflite
vision_adapter.tflite
tokenizer.json or tokenizer.spiece
llm_metadata.pb or llm_metadata.pbtext
model.litertlm
```

After those artifacts exist, inspect the package through Liyue's owner rail:

```bash
python -m cli.main locate litert-package-inspect Assets/ModelStore/LocateAnything-3B
```

The command checks `LITERTLM` magic and parses official `litert-lm-peek` output for LLM
metadata, tokenizer, prefill/decode, embedder, vision encoder, and vision adapter sections.
For `backend=npu`, package inspection must additionally report `npu_aux` / `tf_lite_aux`;
the standard six-section CPU/GPU package is not enough for LiteRT-LM's NPU executor. The
current NPU aux package satisfies that package gate, and the embedder export now carries
the `decode_embedder` alias required by LiteRT-LM's NPU multimodal lookup path.
After `claim=package-inspection-ok`, `locate litert-plan` switches from export/build
blockers to the remaining unproven runtime gates. For the current package those gates are:

- all-NPU parseable box output before any full-NPU claim;
- Android GPU graph lowering before GPU serving claim; current physical-phone posbuf GPU
  receipts are red because GPU lanes hit unsupported delegate ops.
