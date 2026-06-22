# Copyright 2024 The LiteRT Torch Authors.
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
# Dynamic slice op for GPU-friendly dynamic-offset reads (e.g. causal-mask rows).
#
# A plain `tensor.narrow(dim, dynamic_start, static_len)` fails torch.export with
# PendingUnbackedSymbolNotFound, because the dynamic start becomes a data-dependent
# (unbacked) symbol. Wrapping the read in a custom op whose register_fake returns a
# fixed-shape FakeTensor sidesteps that — same pattern as dynamic_update_slice — and
# lowers to stablehlo.dynamic_slice, which the OpenCL/MLDrift GPU delegate supports
# (unlike GATHER_ND produced by index_select).
from typing import Sequence
from litert_torch.backend import lowerings
from litert_converter.mlir import ir
from litert_converter.mlir.dialects import stablehlo
import torch


@torch.library.custom_op("litert_torch::dynamic_slice", mutates_args=())
def dynamic_slice(
    in_tensor: torch.Tensor,
    start_indices: Sequence[torch.Tensor],
    slice_sizes: Sequence[int],
) -> torch.Tensor:
  out = in_tensor
  for dim, size in enumerate(slice_sizes):
    start = int(start_indices[dim].item())
    out = out.narrow(dim, start, size)
  return out.clone()


@dynamic_slice.register_fake
def _(in_tensor, start_indices, slice_sizes):
  # Static output shape avoids creating unbacked symbols during export.
  return in_tensor.new_empty(tuple(int(s) for s in slice_sizes))


@lowerings.lower(torch.ops.litert_torch.dynamic_slice)
def _dynamic_slice_lower(
    lctx,
    in_tensor: ir.Value,
    start_indices: Sequence[ir.Value],
    slice_sizes: Sequence[int],
):
  return stablehlo.dynamic_slice(in_tensor, start_indices, list(slice_sizes))
