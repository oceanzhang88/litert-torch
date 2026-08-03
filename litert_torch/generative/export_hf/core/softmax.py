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
"""Attention softmax graph policy shared by LiteRT export paths."""

from litert_torch.backend import composite
import torch
import torch.nn.functional as F


def apply_softmax(
    logits: torch.Tensor,
    *,
    dim: int = -1,
    use_composite: bool = True,
) -> torch.Tensor:
  """Applies softmax with the selected exported operator boundary.

  `use_composite=True` preserves LiteRT-Torch's current `odml.softmax`
  StableHLO composite. `False` deliberately leaves the ATen softmax unmarked so
  the ordinary Torch-TFL decomposition lowers it to a builtin TFLite SOFTMAX in
  the signature graph. The latter is a compatibility contract for delegates
  that cannot prepare `STABLEHLO_COMPOSITE` softmax nodes.

  Args:
    logits: Attention logits.
    dim: Dimension over which softmax is applied.
    use_composite: Whether to outline the softmax as an ODML composite.

  Returns:
    Softmax probabilities with the same numerical definition in either form.
  """
  if not use_composite:
    return F.softmax(logits, dim=dim)

  builder = composite.StableHLOCompositeBuilder(
      name="odml.softmax", attr={"axis": dim}
  )
  logits = builder.mark_inputs(logits)
  probs = F.softmax(logits, dim=dim)
  return builder.mark_outputs(probs)
