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
"""calibration_data_lib.py -- v2.1.0

Builds static-range (SRQ) calibration samples for
`export_hf.core.export_lib.quantize_model()`'s `calibration_data` parameter
(NPU campaign phase 7: HTP legalizes static INT8xINT8 FULLY_CONNECTED, not
FP32xINT8 hybrid -- see export_lib.quantize_model()'s docstring), from a
tokenizer + prompt strings + a real, already-exported (pre-quantization)
tflite model.

Design: introspect the float tflite's own signatures via
`ai_edge_litert.interpreter` (the same mechanism
`ai_edge_quantizer.utils.tfl_interpreter_utils.create_random_dataset` uses
for its random calibration data) instead of re-deriving shapes from the
torch-level `ExportableModuleConfig`. This keeps the builder correct across
signature-input flattening (e.g. `kv_cache` pytree flattening into
per-layer tensors with generated names) without ever needing to know the
exact flattened tensor names ahead of time.

Two input-role contracts, auto-detected per model (see
`_get_input_role_mode`) from the SAME signature introspection -- no config
flag needed:

TOKEN-ID CONTRACT (the ORIGINAL/v1 path, unchanged below -- models whose
prefill/decode signatures take a "token"-named input directly, e.g. a
model exported WITHOUT `externalize_embedder=True`). Role heuristic per
signature input, matched case-insensitively against the tflite SignatureDef
arg name (the dict key `get_input_details()` returns, NOT the internal
flatbuffer tensor `detail['name']`, which may be mangled by tracing):
  - name contains "token"                    -> real tokenized prompt ids
  - name contains "input_pos" or "position"  -> real position ids (arange)
  - name contains "mask"                     -> causal mask, self-consistent
                                                 with an empty/zero cache
  - anything else (kv_cache_*, ...)          -> zeros, matching a fresh/empty
                                                 cache
This path treats every sample as an independent, fresh, empty-cache forward
pass (no cross-chunk KV chaining). Fine for models that take token ids
directly, since "token"/"mask" are still real, varied signal there.

EXTERNALIZED-EMBEDDER CONTRACT (v2, added phase 7d): models exported with
`split_cache=True, externalize_embedder=True` (see
`core/split_cache/*`, `core/external_emb/exportable_module.py`,
`core/external_rope/exportable_module.py`) do NOT have a "token"-named
input on the main graph AT ALL. Their real inputs are pre-computed
`embeddings`/`per_layer_embeddings` (from a sibling embedder.tflite /
per_layer_embedder.tflite), pre-computed RoPE tables `pos_emb_*` and
attention masks `mask_global`/`mask_local` (from a sibling
auxiliary.tflite), plus a persistent `kv_cache_*` the caller chains across
calls. Under the token-id role heuristic above, NONE of these names match
"token"/"input_pos"/"position" (only "mask_global"/"mask_local" match
"mask", and even that hit degenerate under `_infer_seq_len`'s token-based
seq_len inference falling back to 1) -- so the v1 path zero-filled
`embeddings`, `per_layer_embeddings`, every `pos_emb_*`, and every
`kv_cache_*` for this contract. Root-caused during the phase 7c/7d NPU
campaign coherence gate (Golf-Assistant-VLM, 2026-07-02): min=max=0 per
sample collapsed SRQ's per-tensor scale to a numerically-degenerate epsilon
(observed: scale=3.9215683651783184e-12, representable real range
[0, 1e-9]) -- mask's own 0.0/-100.0 values both quantized to the same int8
code, embeddings/pos_emb of ordinary magnitude saturated to +-127, and the
resulting quantized model's logits saturated uniformly at -128 across the
full vocab on every decode step (see the phase 7c/7d gate writeup).

v2 detects this contract (`_get_input_role_mode`) and, when detected,
drives the REAL host-side pipeline instead: tokenizer -> embedder.tflite ->
per_layer_embedder.tflite -> auxiliary.tflite (rope + mask) -> main model,
chaining a REAL, host-owned KV cache across REAL prefill chunks AND a
handful of real greedy decode steps per prompt (`_ExternalEmbedderPipeline`
/ `_collect_external_embedder_calibration_data`), recording the exact
input dict fed to each `prefill_128`/`decode` call as one calibration
sample. This is the same driving pattern proven correct by the phase-7c
decode coherence gate
(Golf-Assistant-VLM/.../decode_probe.py -- float model produced fully
coherent output through it), factored in here so calibration-time and
inference-time driving cannot drift apart. Calibration always runs against
the pre-quantization FLOAT model (see `build_calibration_data`'s
`tflite_model_path` doc), so this new path's own signature calls are
always float32-to-float32 in practice; the tiny `_is_quantized` /
`_quantize_if_needed` / `_dequantize_if_needed` helpers below exist for
robustness/generality anyway (e.g. re-running this against an
already-partially-quantized graph) rather than being load-bearing today.

Note on `_ExternalEmbedderPipeline`/its quant-boundary helpers: this
mirrors (does NOT import) the equivalent
`try_run_signature_with_quant_dequant` / `is_quantized` /
`try_get_quantized_input` / `try_get_dequantized_output` pattern in
`export_hf.experimental.calib.sampling_executor` (that module's own
`Executor` is the fuller reference for driving an ALREADY-quantized model
at inference time, e.g. for a decode gate). Not imported directly because
`sampling_executor` transitively pulls in its sibling `tokenizer.py`'s
PIL/sentencepiece/transformers dependency via `experimental.calib`'s own
tokenizer_lib -- too heavy to add to this core-path module (used by
`export_lib.quantize_model()`) for ~50 lines of pure quant/dequant helper
logic. `sampling_executor.py:56` (old stale docstring reference target)
does exist in this repo (`litert_torch/generative/export_hf/experimental/
calib/sampling_executor.py`, upstream commit `daad5d3`) -- it was simply
never deployed into the phase-7 Omen venv's site-packages twin, which is
why phase 7c's search for it there came up empty. Both are patched/synced
together as of this v2.0.0 bump; see phase 7d dispatch notes for the sha
cross-check between the Mac fork commit and the Omen hot-patch.

v1/v2 scope: text-only, matching the phase-7 dispatch ("text-only
calibration is acceptable for v1 (vision tower is a separate graph)").
Neither path touches vision-encoder signatures.

v2.1.0 (phase 7e, 2026-07-02): the v2 external-embedder pipeline fixed the
zero-fill degeneracy above, but the decode gate STILL came back GARBAGE via
a second, narrower mechanism: golf-only calibration under-covers the
*logits* tensor's true activation range specifically (golf held-out
prompt's true top logit sat within one quantization step of the calibrated
ceiling; a generic prompt's true top logit exceeded that ceiling by 9.75
units, since no non-golf text was ever in the calibration set). Phase 7e's
combined fix is two-part: (1) `default_generic_prompts()` below adds 8
out-of-domain prompts (factual completion / dialogue / list / short
instruction) rendered through the same chat-template scaffold as the golf
prompts, so calibration covers a text distribution wider than one domain;
(2) the logits/LM-head FULLY_CONNECTED op is separately scoped OUT of int8
activation quantization entirely in the phase 7e recipe (weight-only int8,
float32 activations -- see `recipes/wa_srq_logits_weightonly.json` and the
phase 7e dispatch), which structurally removes the saturation failure mode
for that tensor regardless of calibration coverage. This module only owns
fix (1); fix (2) lives in the recipe JSON consumed by
`export_lib.quantize_model()`, not in this file.
"""

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from ai_edge_litert import interpreter as tfl_interpreter
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor

CalibrationSample = dict[str, np.ndarray]
CalibrationData = dict[str, list[CalibrationSample]]


# -----------------------------------------------------------------------------
# Golf-Assistant-VLM prompt curation.
#
# Mirrors (does NOT import -- litert-torch is a vendored fork and stays
# dependency-free of sibling CompanyWork products) the labeling prompt in
# CompanyWork/Golf-Assistant-VLM/src/teacher_prompt.py SYSTEM_PROMPT +
# build_user_prompt(), v1.0.0, as read 2026-07-02. If that file's schema
# changes, this mirror will drift; it only needs to stay "real enough" for
# SRQ activation-range coverage, not byte-identical to the training prompt.
# -----------------------------------------------------------------------------

_GOLF_SYSTEM_PROMPT = """\
You label egocentric (smart-glasses, first-person) golf clips for training an
on-device assistant. Each clip already passed a quality gate: it is first-person
and the ball is visible at address. You receive frames spanning the clip, the
impact time, and grounding priors (audio-contact signature, ball-track span).

Produce ONE JSON object with exactly two top-level keys: "summary", "feedback".

== SUMMARY -- honest perception. ==
Fill every field from the enums given. This surface is INTERNAL: when the
frames do not show something, write "unknown" -- never guess. Hard rules:
- If the ball-track prior says the ball left the frame (tracked none/partial),
  est_result MUST be "unknown". A full swing from glasses POV almost never
  shows flight or landing; that is expected, not a defect.
- Putts and chips usually keep the ball in frame to the end: then est_result
  MUST be a real read (holed/close/short/long/left/right), and start_dir too.
- contact must agree with the audio prior; if they conflict, use "unknown"
  and lower confidence.
- retake.recommended=true only for genuinely unusable captures.

== FEEDBACK -- what the golfer HEARS. Persona is given per request. ==
The golfer must end happier or better. Content law, in order:
1. NEVER mention what you could not see. Banned phrases and their shapes:
   "I couldn't see", "I don't have", "unknown", "out of frame", "no info",
   "I lost the ball". Honesty lives in the summary by omission here.
2. Ground every claim in a summary fact. Never invent a result: if
   est_result is unknown, do not praise or blame the OUTCOME -- speak to
   contact, tempo, setup, or rhythm, which you did observe.
3. coach = ONE specific, actionable, forward cue from what IS known.
   On a mishit: give ONLY the corrective cue -- never name the miss
   (no "thin/fat/toe/heel" in the golfer's ear).
4. companion = emotional value, adaptive to the shot: after a good one,
   warmth plus at most one light non-technical nudge; after a bad one,
   pure comfort and presence -- zero instruction.
5. Never name a club to the golfer (club_guess stays internal).
6. Tone matches result: good -> celebrate plainly; poor -> forward and warm,
   never harsh. Maximum 2 sentences; it is spoken audio in their ear.

Output ONLY the JSON object. No markdown fences, no commentary.
"""

_GOLF_PERSONAS = ("coach", "companion")

# 8 representative shot scenarios (drawn from teacher_prompt.py's own
# FEWSHOTS priors plus a couple of additional edge cases: a no-contact
# practice swing and a genuinely ambiguous retake-flavored clip).
_GOLF_PRIORS = [
    {"audio_contact": "soft_tock", "ball_track": "full", "shot_hint": "putt"},
    {
        "audio_contact": "crisp_strike",
        "ball_track": "partial",
        "shot_hint": "full_swing",
    },
    {
        "audio_contact": "clicky_thin",
        "ball_track": "none",
        "shot_hint": "full_swing",
    },
    {
        "audio_contact": "heavy_thud",
        "ball_track": "none",
        "shot_hint": "full_swing",
    },
    {
        "audio_contact": "crisp_strike",
        "ball_track": "full",
        "shot_hint": "chip_or_pitch",
    },
    {
        "audio_contact": "soft_tock",
        "ball_track": "partial",
        "shot_hint": "putt",
    },
    {
        "audio_contact": "none",
        "ball_track": "none",
        "shot_hint": "practice_swing",
    },
    {"audio_contact": "unclear", "ball_track": "none", "shot_hint": "unknown"},
]


def _golf_user_prompt(persona, priors, impact_t_s, n_frames, headpose=None):
  """Mirrors teacher_prompt.build_user_prompt() (see module docstring)."""
  assert persona in _GOLF_PERSONAS, persona
  lines = [
      f"persona: {persona}",
      f"impact_t_s: {impact_t_s}",
      f"frames: {n_frames} (uniform over the clip window)",
      f"priors: {priors}",
  ]
  if headpose is not None:
    lines.append(f"headpose: {headpose}")
  lines.append("Label this clip. JSON only.")
  return "\n".join(lines)


def _render_golf_prompt(tokenizer: Any | None, user_prompt: str) -> str:
  """Renders one (system, user) turn pair to a final prompt string.

  If `tokenizer` is given and exposes a chat template, renders through it
  (matches real inference-time token distribution -- Gemma tokenizers
  accept a "system" role directly, confirmed against the Gemma-4-E2B
  checkpoint's tokenizer_config.json + chat_template.jinja). Falls back to
  plain concatenation if no tokenizer/template is available, or if the
  template rejects a system turn.
  """
  if tokenizer is not None and getattr(tokenizer, "chat_template", None):
    try:
      return tokenizer.apply_chat_template(
          [
              {"role": "system", "content": _GOLF_SYSTEM_PROMPT},
              {"role": "user", "content": user_prompt},
          ],
          tokenize=False,
          add_generation_prompt=True,
      )
    except Exception:  # pylint: disable=broad-except
      try:
        return tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": _GOLF_SYSTEM_PROMPT + "\n\n" + user_prompt,
            }],
            tokenize=False,
            add_generation_prompt=True,
        )
      except Exception:  # pylint: disable=broad-except
        pass
  return _GOLF_SYSTEM_PROMPT + "\n\n" + user_prompt


def default_golf_assistant_prompts(tokenizer: Any | None = None) -> list[str]:
  """Returns 16 real-domain calibration prompt strings.

  8 shot-scenario priors x {coach, companion} personas, each rendered as a
  full SYSTEM_PROMPT + one build_user_prompt() turn, curated from
  Golf-Assistant-VLM's teacher-labeling prompt (see module docstring) so
  SRQ calibration statistics reflect this project's real deployment text
  distribution instead of synthetic filler text.

  Args:
    tokenizer: Optional loaded HF tokenizer. When given and it exposes a
      chat template, prompts are chat-template-rendered (system + user
      turns); otherwise prompts are a plain SYSTEM_PROMPT + "\\n\\n" +
      user_turn string.
  """
  prompts = []
  for priors in _GOLF_PRIORS:
    for persona in _GOLF_PERSONAS:
      user_prompt = _golf_user_prompt(
          persona, priors, impact_t_s=5.0, n_frames=8
      )
      prompts.append(_render_golf_prompt(tokenizer, user_prompt))
  return prompts


# -----------------------------------------------------------------------------
# Generic / out-of-domain prompt curation (phase 7e, added v2.1.0).
#
# Deliberately OUTSIDE the golf domain: golf-only calibration left the
# logits/LM-head tensor's activation range under-covered (see module
# docstring, "v2.1.0 (phase 7e...)"). These 8 prompts span factual
# completion, dialogue, list, and short-instruction shapes -- the mix the
# phase 7e dispatch asked for -- rendered through the same chat-template
# scaffold as `default_golf_assistant_prompts` so prompt CONTENT is the only
# varying dimension, not the rendering path.
# -----------------------------------------------------------------------------

_GENERIC_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. Respond directly and"
    " concisely."
)

# 2 prompts x 4 shapes = 8, matching the phase 7e dispatch's "~6-8 prompts,
# mix: factual completion, dialogue, list, short instruction."
_GENERIC_USER_PROMPTS = (
    # factual completion
    "Complete this fact: The largest planet in our solar system is",
    "Complete this fact: Water boils at a temperature of",
    # dialogue
    (
        "Continue this dialogue naturally:\n"
        "User: What time does the store close?\n"
        "Assistant: The store closes at 9 PM. Anything else I can help"
        " with?\n"
        "User: Do you have parking nearby?\n"
        "Assistant:"
    ),
    (
        "Continue this dialogue naturally:\n"
        "User: I can't find my keys anywhere.\n"
        "Assistant: Let's think it through -- where did you last have"
        " them?\n"
        "User: I think in my jacket pocket.\n"
        "Assistant:"
    ),
    # list
    "List three benefits of regular exercise, one per line.",
    "List the ingredients needed to make a basic omelette, one per line.",
    # short instruction
    "Write a one-sentence summary of why recycling matters.",
    "Translate the phrase 'good morning' into French.",
)


def default_generic_prompts(tokenizer: Any | None = None) -> list[str]:
  """Returns 8 generic/out-of-domain calibration prompt strings.

  Covers factual completion, dialogue, list, and short-instruction shapes,
  deliberately outside the golf domain, so SRQ calibration statistics for
  shared/general-purpose ops (esp. the LM head / logits projection -- see
  module docstring, "v2.1.0 (phase 7e...)") cover a wider activation range
  than golf-only text produces. Meant to be combined with
  `default_golf_assistant_prompts()`'s 16 prompts, not used alone.

  Args:
    tokenizer: Optional loaded HF tokenizer. Same chat-template-with-fallback
      rendering as `default_golf_assistant_prompts`.
  """
  return [
      _render_generic_prompt(tokenizer, user_prompt)
      for user_prompt in _GENERIC_USER_PROMPTS
  ]


def _render_generic_prompt(tokenizer: Any | None, user_prompt: str) -> str:
  """Renders one generic (system, user) turn pair to a final prompt string.

  Mirrors `_render_golf_prompt`'s chat-template-with-fallback logic (see
  that function's docstring), using `_GENERIC_SYSTEM_PROMPT` instead of the
  golf one.
  """
  if tokenizer is not None and getattr(tokenizer, "chat_template", None):
    try:
      return tokenizer.apply_chat_template(
          [
              {"role": "system", "content": _GENERIC_SYSTEM_PROMPT},
              {"role": "user", "content": user_prompt},
          ],
          tokenize=False,
          add_generation_prompt=True,
      )
    except Exception:  # pylint: disable=broad-except
      try:
        return tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": _GENERIC_SYSTEM_PROMPT + "\n\n" + user_prompt,
            }],
            tokenize=False,
            add_generation_prompt=True,
        )
      except Exception:  # pylint: disable=broad-except
        pass
  return _GENERIC_SYSTEM_PROMPT + "\n\n" + user_prompt


# -----------------------------------------------------------------------------
# Signature introspection + calibration sample construction.
# -----------------------------------------------------------------------------


def _infer_seq_len(input_details: dict[str, Any]) -> int:
  """Token-input sequence length for this signature (e.g. 128, or 1 for decode)."""
  for name, detail in input_details.items():
    if "token" in name.lower():
      return int(detail["shape"][-1])
  return 1


def _cache_length(input_details: dict[str, Any]) -> int:
  """Cache length (mask's last dim), 0 if this signature has no mask input."""
  for name, detail in input_details.items():
    if "mask" in name.lower():
      return int(detail["shape"][-1])
  return 0


def _causal_mask(
    shape: tuple[int, ...], dtype: np.dtype, start_pos: int, seq_len: int
) -> np.ndarray:
  """Causal mask bounded to this chunk's own span (see v1 simplification)."""
  cache_length = shape[-1]
  query_len = shape[-2]
  key_pos = np.arange(cache_length)
  query_pos = start_pos + np.arange(query_len)
  attend = (
      (key_pos[None, :] <= query_pos[:, None])
      & (key_pos[None, :] >= start_pos)
      & (key_pos[None, :] < start_pos + seq_len)
  )
  attend = attend.reshape((1,) * (len(shape) - 2) + attend.shape)
  if np.issubdtype(dtype, np.bool_):
    filled = attend
  else:
    neg = (
        float(np.finfo(np.float32).min)
        if np.issubdtype(dtype, np.floating)
        else -1e9
    )
    filled = np.where(attend, 0.0, neg)
  return np.broadcast_to(filled, shape).astype(dtype)


def _build_sample(
    input_details: dict[str, Any],
    start_pos: int,
    token_chunk: np.ndarray,
    seq_len: int,
) -> CalibrationSample:
  """Builds one {input_arg_name: np.ndarray} sample for a real token chunk."""
  position_ids = np.arange(start_pos, start_pos + seq_len, dtype=np.int64)
  sample = {}
  for name, detail in input_details.items():
    lname = name.lower()
    shape = tuple(int(d) for d in detail["shape"])
    dtype = np.dtype(detail["dtype"])
    if "token" in lname:
      sample[name] = token_chunk.reshape(shape).astype(dtype)
    elif "input_pos" in lname or "position" in lname:
      sample[name] = position_ids.reshape(shape).astype(dtype)
    elif "mask" in lname:
      sample[name] = _causal_mask(shape, dtype, start_pos, seq_len)
    else:
      # kv_cache (any flattened per-layer name) or anything unrecognized:
      # zero-fill, matching a fresh/empty cache -- see module docstring.
      sample[name] = np.zeros(shape, dtype=dtype)
  return sample


def _chunk_prompt(
    tokenizer: Any, prompt: str, seq_len: int
) -> list[tuple[int, np.ndarray]]:
  """Splits one tokenized prompt into (start_pos, token_chunk) windows.

  Non-overlapping windows of `seq_len` real tokens (last window zero-
  padded if short), so calibration exercises the prompt's *entire* content
  instead of a head-truncated slice -- real Golf-Assistant-VLM prompts
  (system + user turn) are longer than a single prefill_128 window, let
  alone a tiny validation window.
  """
  ids = np.asarray(
      tokenizer(prompt, add_special_tokens=False)["input_ids"], dtype=np.int64
  )
  if ids.size == 0:
    return []
  chunks = []
  for start in range(0, max(int(ids.size), seq_len), seq_len):
    chunk = ids[start : start + seq_len]
    if chunk.size == 0:
      break
    if chunk.size < seq_len:
      chunk = np.pad(chunk, (0, seq_len - chunk.size))
    chunks.append((start, chunk))
    if start + seq_len >= ids.size:
      break
  return chunks


# -----------------------------------------------------------------------------
# v2: externalized-embedder contract -- real host-side pipeline driving.
# See module docstring's "EXTERNALIZED-EMBEDDER CONTRACT" section for the
# root-cause story and the sampling_executor.py provenance note.
# -----------------------------------------------------------------------------


def _get_input_role_mode(input_details: dict[str, Any]) -> str:
  """Returns 'external_embedder' or 'token_id' for one signature's inputs.

  external_embedder: has an 'embeddings'-named input and NO 'token'-named
  input (the split_cache + externalize_embedder=True contract). token_id:
  everything else (the original/v1 contract this module was built for).
  """
  names_lower = {name.lower() for name in input_details}
  has_embeddings = any("embeddings" in n for n in names_lower)
  has_token = any("token" in n for n in names_lower)
  if has_embeddings and not has_token:
    return "external_embedder"
  return "token_id"


def _is_quantized(tensor_detail: dict[str, Any]) -> bool:
  """Mirrors sampling_executor.is_quantized (see module docstring note)."""
  return bool(len(tensor_detail["quantization_parameters"]["scales"]))


def _quantize_if_needed(
    data: np.ndarray, detail: dict[str, Any]
) -> np.ndarray:
  """Quantizes `data` to `detail`'s dtype iff it's float32 and `detail` is
  quantized; otherwise returns `data` unchanged. Mirrors
  sampling_executor.try_get_quantized_input for a single tensor."""
  if data.dtype == np.float32 and _is_quantized(detail):
    params = qtyping.UniformQuantParams.from_tfl_tensor_details(detail)
    return uniform_quantize_tensor.uniform_quantize(data, params)
  return data


def _dequantize_if_needed(
    data: np.ndarray, detail: dict[str, Any]
) -> np.ndarray:
  """Dequantizes `data` to float32 iff `detail` is quantized; otherwise
  returns `data` unchanged. Mirrors
  sampling_executor.try_get_dequantized_output for a single tensor."""
  if _is_quantized(detail):
    params = qtyping.UniformQuantParams.from_tfl_tensor_details(detail)
    return uniform_quantize_tensor.uniform_dequantize(data, params).astype(
        np.float32
    )
  return data


class _ExternalEmbedderPipeline:
  """Drives the real embedder/per_layer_embedder/auxiliary/main pipeline.

  Same driving contract as Golf-Assistant-VLM's phase-7c decode_probe.py
  (proven correct there: the float model produced fully coherent output
  through this exact chaining), reused here so calibration-time driving
  cannot drift from inference-time driving. The KV cache buffer is kept in
  float32 throughout: calibration always drives the PRE-quantization float
  model (see `build_calibration_data`'s `tflite_model_path` doc), whose
  kv_cache_* tensors are natively float32, so no quant-boundary special-
  casing is needed here (unlike a decode GATE against an already-quantized
  model, which decode_probe.py handles separately).
  """

  def __init__(self, main_interp, embedder_interp, ple_interp, aux_interp):
    self.main = main_interp
    self.embedder = embedder_interp
    self.ple = ple_interp
    self.aux = aux_interp

    dec_in = self.main.get_signature_runner("decode").get_input_details()
    self.kv_names = sorted(n for n in dec_in if n.startswith("kv_cache_"))
    self.cache_shapes = {
        n: tuple(int(x) for x in dec_in[n]["shape"]) for n in self.kv_names
    }
    self.reset_cache()

  def reset_cache(self):
    self.cache = {
        n: np.zeros(self.cache_shapes[n], dtype=np.float32)
        for n in self.kv_names
    }

  def _write_slice(self, name, slice_val, start_pos, chunk_len):
    buf = self.cache[name]
    if name.startswith("kv_cache_k_"):
      buf[..., start_pos : start_pos + chunk_len, :] = slice_val
    else:
      buf[..., start_pos : start_pos + chunk_len] = slice_val

  def _embed(self, token_ids_2d, is_prefill):
    sig = "prefill_embedder_128" if is_prefill else "decode_embedder"
    emb = self.embedder.get_signature_runner(sig)(token_ids=token_ids_2d)[
        "embeddings"
    ]
    sig2 = (
        "prefill_per_layer_embedder_128"
        if is_prefill
        else "decode_per_layer_embedder"
    )
    ple = self.ple.get_signature_runner(sig2)(token_ids=token_ids_2d)[
        "embeddings"
    ]
    return emb.astype(np.float32), ple.astype(np.float32)

  def _rope(self, input_pos_1d, is_prefill):
    sig = "prefill_rope_128" if is_prefill else "decode_rope"
    return self.aux.get_signature_runner(sig)(input_pos=input_pos_1d)

  def _mask(self, input_tokens_2d, time_step, valid_mask_2d, is_prefill):
    sig = "prefill_mask_128" if is_prefill else "decode_mask"
    return self.aux.get_signature_runner(sig)(
        input_tokens=input_tokens_2d,
        time_step=np.array(time_step, dtype=np.int32),
        valid_mask=valid_mask_2d,
    )

  def _build_inputs(self, token_ids_2d, start_pos, is_prefill, valid_mask):
    """Builds the exact {input_arg_name: array} dict for one signature call.

    Every value is an INDEPENDENT COPY, never a live reference into
    `self.cache` or into an interpreter's internal output buffer. This
    matters here in a way it didn't for decode_probe.py's gate: a gate
    consumes each `inputs` dict once and discards it, but a calibration
    collector ACCUMULATES these dicts across many prefill/decode calls
    (`_collect_external_embedder_calibration_data`'s `per_sig_samples`)
    -- `self.cache[n]` is mutated in place every subsequent call
    (`_write_slice`), and TFLite signature runners commonly reuse their
    own internal output buffers across `invoke()` calls, so an uncopied
    reference recorded now would silently read back as a LATER call's
    values once accumulation finishes. `.astype(np.float32)` already
    copies (numpy's default `copy=True`) for `emb`/`ple`; everything else
    needs an explicit `.copy()`.
    """
    chunk_len = token_ids_2d.shape[1]
    emb, ple = self._embed(token_ids_2d, is_prefill)
    positions = (
        np.arange(start_pos, start_pos + chunk_len, dtype=np.int32)
        if is_prefill
        else np.array([start_pos], dtype=np.int32)
    )
    pos_emb = self._rope(positions, is_prefill)
    mask_out = self._mask(token_ids_2d, start_pos, valid_mask, is_prefill)
    inputs = {
        "embeddings": emb,
        "per_layer_embeddings": ple,
        "pos_emb_cos": pos_emb["pos_emb_cos"].copy(),
        "pos_emb_sin": pos_emb["pos_emb_sin"].copy(),
        "pos_emb_local_cos": pos_emb["pos_emb_local_cos"].copy(),
        "pos_emb_local_sin": pos_emb["pos_emb_local_sin"].copy(),
        "mask_global": mask_out["mask_global"].copy(),
        "mask_local": mask_out["mask_local"].copy(),
    }
    for n in self.kv_names:
      inputs[n] = self.cache[n].copy()
    return inputs

  def prefill_chunk(self, ids_padded, start_pos, real_len):
    """Runs one prefill_128 call; returns (recorded_inputs, output_dict)."""
    chunk_len = len(ids_padded)
    token_ids_2d = np.asarray(ids_padded, dtype=np.int32).reshape(
        1, chunk_len
    )
    valid_mask = np.zeros((1, chunk_len), dtype=bool)
    valid_mask[0, :real_len] = True
    inputs = self._build_inputs(token_ids_2d, start_pos, True, valid_mask)
    runner = self.main.get_signature_runner("prefill_128")
    in_details = runner.get_input_details()
    sig_inputs = {
        k: _quantize_if_needed(v, in_details[k]) for k, v in inputs.items()
    }
    raw_out = runner(**sig_inputs)
    out_details = runner.get_output_details()
    out = {
        k: _dequantize_if_needed(v, out_details[k]) for k, v in raw_out.items()
    }
    for n in self.kv_names:
      slice_name = "kv_slice_" + n[len("kv_cache_") :]
      self._write_slice(n, out[slice_name], start_pos, chunk_len)
    return inputs, out

  def decode_one(self, token_id, start_pos):
    """Runs one decode call; returns (recorded_inputs, logits)."""
    token_ids_2d = np.array([[token_id]], dtype=np.int32)
    valid_mask = np.ones((1, 1), dtype=bool)
    inputs = self._build_inputs(token_ids_2d, start_pos, False, valid_mask)
    runner = self.main.get_signature_runner("decode")
    in_details = runner.get_input_details()
    sig_inputs = {
        k: _quantize_if_needed(v, in_details[k]) for k, v in inputs.items()
    }
    raw_out = runner(**sig_inputs)
    out_details = runner.get_output_details()
    out = {
        k: _dequantize_if_needed(v, out_details[k]) for k, v in raw_out.items()
    }
    for n in self.kv_names:
      slice_name = "kv_slice_" + n[len("kv_cache_") :]
      self._write_slice(n, out[slice_name], start_pos, 1)
    return inputs, out["logits"]


def _collect_external_embedder_calibration_data(
    tflite_model_path: str,
    embedder_model_path: str,
    per_layer_embedder_model_path: str,
    auxiliary_model_path: str,
    tokenizer: Any,
    prompts: Sequence[str],
    signature_keys: Sequence[str] | None,
    max_samples_per_signature: int,
    n_decode_steps_per_prompt: int,
) -> CalibrationData:
  """Real-pipeline calibration collector for the externalized-embedder
  contract -- see module docstring. Drives each prompt through a REAL
  chained prefill (however many 128-token chunks it needs) plus one
  redundant decode() on the prompt's own last token (seeds real logits;
  mirrors both decode_probe.py and sampling_executor.Executor.prefill_chunk's
  own time_step-=1-on-last-chunk convention) plus `n_decode_steps_per_prompt`
  further real greedy decode steps, recording the exact input dict fed to
  every prefill_128/decode call as one calibration sample.
  """
  import gc

  PAD_ID = 0
  CHUNK = 128

  main_interp = tfl_interpreter.Interpreter(model_path=tflite_model_path)
  main_interp.allocate_tensors()
  available = main_interp.get_signature_list()
  keys = list(signature_keys) if signature_keys else list(available)
  for key in keys:
    if key not in available:
      raise ValueError(
          f"Signature {key!r} not found in {tflite_model_path}. Available:"
          f" {sorted(available)}"
      )

  embedder_interp = tfl_interpreter.Interpreter(model_path=embedder_model_path)
  embedder_interp.allocate_tensors()
  ple_interp = tfl_interpreter.Interpreter(
      model_path=per_layer_embedder_model_path
  )
  ple_interp.allocate_tensors()
  aux_interp = tfl_interpreter.Interpreter(model_path=auxiliary_model_path)
  aux_interp.allocate_tensors()

  pipeline = _ExternalEmbedderPipeline(
      main_interp, embedder_interp, ple_interp, aux_interp
  )

  per_sig_samples: dict[str, list[CalibrationSample]] = {k: [] for k in keys}

  for prompt in prompts:
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not ids:
      continue
    pipeline.reset_cache()
    pos = 0
    n = len(ids)
    while pos < n:
      real_len = min(CHUNK, n - pos)
      chunk = list(ids[pos : pos + real_len])
      if len(chunk) < CHUNK:
        chunk = chunk + [PAD_ID] * (CHUNK - len(chunk))
      inputs, _ = pipeline.prefill_chunk(chunk, pos, real_len)
      if "prefill_128" in per_sig_samples:
        per_sig_samples["prefill_128"].append(inputs)
      pos += real_len

    inputs, logits = pipeline.decode_one(ids[-1], n - 1)
    if "decode" in per_sig_samples:
      per_sig_samples["decode"].append(inputs)

    cur_pos = n
    cur_logits = logits
    for _ in range(n_decode_steps_per_prompt):
      next_id = int(np.argmax(cur_logits))
      inputs, cur_logits = pipeline.decode_one(next_id, cur_pos)
      if "decode" in per_sig_samples:
        per_sig_samples["decode"].append(inputs)
      cur_pos += 1

  result: CalibrationData = {}
  for key in keys:
    samples = per_sig_samples.get(key, [])
    if len(samples) > max_samples_per_signature:
      stride_idx = sorted(
          set(
              np.linspace(0, len(samples) - 1, max_samples_per_signature)
              .round()
              .astype(int)
              .tolist()
          )
      )
      samples = [samples[i] for i in stride_idx]
    result[key] = samples

  # Explicit teardown: these 4 interpreters (float main model + embedder +
  # per_layer_embedder + auxiliary) are only needed for collection, and the
  # caller's subsequent ai_edge_quantizer.calibrate() sweep needs headroom
  # (phase 7d memory note: prior run peaked 56.3GB against a 54GiB+16G-swap
  # ceiling) -- don't let them linger past this function's return.
  del main_interp, embedder_interp, ple_interp, aux_interp, pipeline
  del per_sig_samples
  gc.collect()

  return result


def build_calibration_data(
    tflite_model_path: str,
    tokenizer: Any,
    prompts: Sequence[str],
    signature_keys: Sequence[str] | None = None,
    max_samples_per_signature: int = 32,
    embedder_model_path: str | None = None,
    per_layer_embedder_model_path: str | None = None,
    auxiliary_model_path: str | None = None,
    n_decode_steps_per_prompt: int = 3,
) -> CalibrationData:
  """Builds ai_edge_quantizer-shaped calibration data from real prompts.

  Auto-detects, per `_get_input_role_mode`, whether `tflite_model_path`
  takes token ids directly (original/v1 path, unchanged) or is the
  externalized-embedder contract (v2 path -- see module docstring). Callers
  do not need to know which; both paths return the same
  `{signature_key: [{input_arg_name: np.ndarray}, ...]}` shape.

  Args:
    tflite_model_path: Path to the exported (pre-quantization) float tflite
      model. Its signatures are introspected directly via
      `ai_edge_litert.interpreter` -- this is what makes the result robust
      to kv_cache pytree-flattened tensor names without this function ever
      needing to know them ahead of time.
    tokenizer: A loaded HF tokenizer for the model being calibrated (e.g.
      `transformers.AutoTokenizer.from_pretrained(checkpoint_dir)`).
    prompts: Prompt strings to tokenize; see `default_golf_assistant_prompts`.
      Each prompt is chunked to every signature's own token-input length, so
      one prompt can contribute multiple calibration samples per signature.
    signature_keys: Which tflite signatures to build samples for (e.g.
      `["prefill_128", "decode"]`). Defaults to every signature in the
      model.
    max_samples_per_signature: Caps the (prompt x chunk-or-step) cross
      product per signature via even striding, so a handful of long
      prompts can't balloon a "decode" (single-token-window) signature
      into thousands of near-duplicate samples. Tune upward (e.g. 128-256)
      for a production calibration pass; 32 is sized for a fast
      validation pass and is memory-conscious for the v2 path (each
      sample's kv_cache_* alone is ~240MB at this model's 4096-token
      cache x 15 layers x float32 -- see phase 7d memory note in
      `_collect_external_embedder_calibration_data`).
    embedder_model_path: v2 path only. Defaults to `embedder.tflite` next
      to `tflite_model_path`.
    per_layer_embedder_model_path: v2 path only. Defaults to
      `per_layer_embedder.tflite` next to `tflite_model_path`.
    auxiliary_model_path: v2 path only. Defaults to `auxiliary.tflite`
      next to `tflite_model_path`.
    n_decode_steps_per_prompt: v2 path only. Real greedy decode steps
      chained per prompt (beyond the one redundant last-prompt-token
      decode call), so decode-signature calibration samples see a
      genuinely populated kv_cache, not just prefill-fresh state.

  Returns:
    `{signature_key: [{input_arg_name: np.ndarray}, ...]}`, directly usable
    as `ai_edge_quantizer.quantizer.Quantizer.calibrate()`'s
    `calibration_data` argument, and as
    `export_lib.quantize_model()`'s / `ExportableModuleConfig
    .calibration_data`'s value.

  Raises:
    ValueError: If a requested signature_key is not in the model.
    FileNotFoundError: v2 path, if a sibling embedder/per_layer_embedder/
      auxiliary model isn't found next to `tflite_model_path` and no
      explicit override path was given.
  """
  interp = tfl_interpreter.Interpreter(model_path=tflite_model_path)
  interp.allocate_tensors()
  available = interp.get_signature_list()
  keys = list(signature_keys) if signature_keys else list(available)

  modes = set()
  for key in keys:
    if key not in available:
      raise ValueError(
          f"Signature {key!r} not found in {tflite_model_path}. Available:"
          f" {sorted(available)}"
      )
    modes.add(
        _get_input_role_mode(interp.get_signature_runner(key).get_input_details())
    )
  if len(modes) > 1:
    raise ValueError(
        f"Mixed input-role modes across signatures {keys}: {modes} --"
        " unsupported (a model should be all-token-id or"
        " all-externalized-embedder)."
    )
  mode = modes.pop() if modes else "token_id"

  if mode == "external_embedder":
    # Free this mode-detection interpreter's ~model-sized allocation before
    # the collector below loads its OWN copy of tflite_model_path -- avoids
    # briefly holding two full copies of the (large, float, pre-quantization)
    # model at once. See phase 7d memory note: prior run peaked 56.3GB
    # against a 54GiB+16G-swap ceiling.
    import gc

    del interp
    gc.collect()

    model_dir = os.path.dirname(tflite_model_path)
    embedder_model_path = embedder_model_path or os.path.join(
        model_dir, "embedder.tflite"
    )
    per_layer_embedder_model_path = per_layer_embedder_model_path or os.path.join(
        model_dir, "per_layer_embedder.tflite"
    )
    auxiliary_model_path = auxiliary_model_path or os.path.join(
        model_dir, "auxiliary.tflite"
    )
    for path, label in (
        (embedder_model_path, "embedder"),
        (per_layer_embedder_model_path, "per_layer_embedder"),
        (auxiliary_model_path, "auxiliary"),
    ):
      if not os.path.exists(path):
        raise FileNotFoundError(
            f"{tflite_model_path} uses the externalized-embedder contract"
            f" (detected via signature introspection -- see"
            f" _get_input_role_mode) but its sibling {label} model was not"
            f" found at {path!r}. Pass {label}_model_path=... explicitly"
            " if it lives elsewhere."
        )
    return _collect_external_embedder_calibration_data(
        tflite_model_path,
        embedder_model_path,
        per_layer_embedder_model_path,
        auxiliary_model_path,
        tokenizer,
        prompts,
        keys,
        max_samples_per_signature,
        n_decode_steps_per_prompt,
    )

  # token_id mode: original v1 path, unchanged.
  calibration_data: CalibrationData = {}
  for key in keys:
    runner = interp.get_signature_runner(key)
    input_details = runner.get_input_details()
    seq_len = _infer_seq_len(input_details)

    all_chunks: list[tuple[int, np.ndarray]] = []
    for prompt in prompts:
      all_chunks.extend(_chunk_prompt(tokenizer, prompt, seq_len))

    if len(all_chunks) > max_samples_per_signature:
      stride_idx = sorted(
          set(
              np.linspace(0, len(all_chunks) - 1, max_samples_per_signature)
              .round()
              .astype(int)
              .tolist()
          )
      )
      all_chunks = [all_chunks[i] for i in stride_idx]

    calibration_data[key] = [
        _build_sample(input_details, start_pos, chunk, seq_len)
        for start_pos, chunk in all_chunks
    ]
  return calibration_data
