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
"""calibration_data_lib.py -- v1.0.0

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

Role heuristic per signature input, matched case-insensitively against the
tflite SignatureDef arg name (the dict key `get_input_details()` returns,
NOT the internal flatbuffer tensor `detail['name']`, which may be mangled by
tracing):
  - name contains "token"                    -> real tokenized prompt ids
  - name contains "input_pos" or "position"  -> real position ids (arange)
  - name contains "mask"                     -> causal mask, self-consistent
                                                 with an empty/zero cache (see
                                                 "v1 simplification" below)
  - anything else (kv_cache_*, ...)          -> zeros, matching a fresh/empty
                                                 cache

v1 simplification (documented, not a silent gap): every calibration sample
-- whether a prefill chunk or a single decode step -- is treated as an
independent, fresh, empty-cache forward pass. The mask is bounded to
`[start_pos, start_pos + seq_len)` (this chunk's own span only, never
earlier "history"), which is the self-consistent choice given the cache
input is always zero-filled: a mask that claimed earlier positions were
attendable while the cache tensor holding them is zero would manufacture a
fake attend-to-zeros signal. Net effect: real, varied token content and
real absolute (ROPE-relevant) positions drive FC activation ranges from
this project's actual domain text; genuine cross-chunk / cross-decode-step
KV chaining (replaying real forward output back into the next sample's
cache input, mirroring `export_hf.experimental.calib.sampling_executor`'s
runtime chaining) is NOT exercised. See NPU campaign phase-7 dispatch notes
for the follow-up if the full run's decode-side activations look under-
covered.

v1 scope: text-only, matching the phase-7 dispatch ("text-only calibration
is acceptable for v1 (vision tower is a separate graph)"). This builder
does not touch vision-encoder or embedder signatures.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from ai_edge_litert import interpreter as tfl_interpreter

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


def build_calibration_data(
    tflite_model_path: str,
    tokenizer: Any,
    prompts: Sequence[str],
    signature_keys: Sequence[str] | None = None,
    max_samples_per_signature: int = 32,
) -> CalibrationData:
  """Builds ai_edge_quantizer-shaped calibration data from real prompts.

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
    max_samples_per_signature: Caps the (prompt x chunk) cross product per
      signature via even striding, so a handful of long prompts can't
      balloon a "decode" (single-token-window) signature into thousands of
      near-duplicate samples. Tune upward (e.g. 128-256) for a production
      calibration pass; 32 is sized for a fast v1/validation pass.

  Returns:
    `{signature_key: [{input_arg_name: np.ndarray}, ...]}`, directly usable
    as `ai_edge_quantizer.quantizer.Quantizer.calibrate()`'s
    `calibration_data` argument, and as
    `export_lib.quantize_model()`'s / `ExportableModuleConfig
    .calibration_data`'s value.

  Raises:
    ValueError: If a requested signature_key is not in the model.
  """
  interp = tfl_interpreter.Interpreter(model_path=tflite_model_path)
  interp.allocate_tensors()
  available = interp.get_signature_list()
  keys = list(signature_keys) if signature_keys else list(available)

  calibration_data: CalibrationData = {}
  for key in keys:
    if key not in available:
      raise ValueError(
          f"Signature {key!r} not found in {tflite_model_path}. Available:"
          f" {sorted(available)}"
      )
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
