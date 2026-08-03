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
"""Patch for FastVLM (llava_qwen2).

FastVLM ships as `trust_remote_code`, so its `forward` is authored against the
transformers release that existed when the checkpoint was published rather than
against this exporter. Three things follow from that, and only one of them
announces itself:

  1. The signature has no `**kwargs`, so every runtime argument the exporter
     adds is a hard `TypeError`. `valid_mask` is the one that fires first --
     `adapt_inputs` attaches it unconditionally
     (`core/exportable_module.py:155`).

  2. `cache_position` is accepted as a parameter and then never forwarded to
     `super().forward()`. That one is silent: the export succeeds and produces
     a graph whose cache positions were never wired, which is far more
     expensive to discover than a crash.

  3. The exporter's generic RMSNorm replacement is installed through a
     transformers class decorator. If a caller materializes the FastVLM
     tokenizer first, transformers imports Qwen2 before that decorator is
     replaced, and every already-constructed norm stays as inline SUM/RSQRT
     arithmetic. The graph is numerically valid but loses the `odml.rms_norm`
     boundaries expected by the portable GPU delegate.

The patch replaces `forward` with one that keeps FastVLM's own multimodal branch
intact and forwards everything else, including `cache_position`, to the Qwen2
implementation underneath. It also adapts already-loaded Qwen2 RMSNorm module
instances to the exporter's composite implementation, preserving the original
parameters and restoring every module when the export context closes.
"""

import contextlib
from litert_torch.generative.export_hf.core import patches as core_patches
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
import torch


class FastVlmRMSNorm(core_patches.RMSNorm):
  """Composite RMSNorm adapter for an already-constructed Qwen2 norm."""

  def __init__(self, original: torch.nn.Module):
    epsilon = getattr(
        original, "variance_epsilon", getattr(original, "eps", None)
    )
    if epsilon is None:
      raise ValueError(
          f"{type(original).__name__} has no RMSNorm epsilon attribute."
      )
    if not hasattr(original, "weight"):
      raise ValueError(
          f"{type(original).__name__} has no RMSNorm weight parameter."
      )

    hidden_size = int(original.weight.shape[0])
    super().__init__(hidden_size, eps=float(epsilon))
    # Reuse the exact Parameter rather than copying it. This keeps parameter
    # identity, state-dict names, device, and checkpoint values unchanged.
    self.weight = original.weight
    self.train(original.training)


def _replace_inline_qwen2_rms_norms(model: torch.nn.Module):
  """Replaces loaded Qwen2 norms and returns enough state to restore them."""
  replaced_modules = []

  def replace_children(parent: torch.nn.Module):
    for child_name, child in list(parent.named_children()):
      # Match the checkpoint's concrete text-model class rather than every
      # RMSNorm-shaped module in the multimodal tree. A FastViT module that
      # happens to expose `weight` and `eps` is not part of this contract.
      if type(child).__name__ == "Qwen2RMSNorm":
        replacement = FastVlmRMSNorm(child)
        setattr(parent, child_name, replacement)
        replaced_modules.append((parent, child_name, child))
      else:
        replace_children(child)

  try:
    replace_children(model)
  except Exception:
    for parent, child_name, original in reversed(replaced_modules):
      setattr(parent, child_name, original)
    raise
  return replaced_modules


@patches_lib.register_model_patch(["llava_qwen2"])
@contextlib.contextmanager
def llava_qwen2_litert_patch(model, export_config):
  """Makes FastVLM's remote-code forward accept the exporter's call shape."""
  del export_config  # No export-config-dependent behaviour here.
  print("FastVLM (llava_qwen2) patch applied.")

  model_cls = type(model)
  original_forward = model_cls.forward

  def patched_forward(
      self,
      input_ids=None,
      attention_mask=None,
      position_ids=None,
      past_key_values=None,
      inputs_embeds=None,
      labels=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      images=None,
      image_sizes=None,
      return_dict=None,
      cache_position=None,
      **kwargs,
  ):
    # Untouched from upstream: with no embeddings supplied, images still have
    # to be folded into the token stream before the text model sees them.
    if inputs_embeds is None:
      (
          input_ids,
          position_ids,
          attention_mask,
          past_key_values,
          inputs_embeds,
          labels,
      ) = self.prepare_inputs_labels_for_multimodal(
          input_ids,
          position_ids,
          attention_mask,
          past_key_values,
          labels,
          images,
          image_sizes,
      )

    # `super(model_cls, self)` rather than a bare `super()`: this function is
    # defined outside the class body, so there is no implicit __class__ cell to
    # resolve against.
    return super(model_cls, self).forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

  model_cls.forward = patched_forward
  replaced_norms = []
  try:
    # Instance adaptation closes the import-order hole left by the generic
    # class-decorator hook: it works whether Qwen2 was imported before or after
    # LiteRT-Torch and affects this model only.
    replaced_norms = _replace_inline_qwen2_rms_norms(model)
    if replaced_norms:
      print(
          "FastVLM RMSNorm composite adaptation applied to "
          f"{len(replaced_norms)} module(s)."
      )
    yield
  finally:
    for parent, child_name, original in reversed(replaced_norms):
      setattr(parent, child_name, original)
    model_cls.forward = original_forward
