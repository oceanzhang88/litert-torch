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

import importlib
from typing import Any

from litert_torch._config import config
from litert_torch._convert.to_channel_last_io import to_channel_last_io
from litert_torch.model import Model
from litert_torch.version import __version__


# Keep compiler-only tools independent from the conversion dependency graph.
# Importing any ``litert_torch.*`` submodule executes this package initializer.
# Eagerly importing ``_convert.interface`` therefore pulled AI Edge Quantizer
# (and, depending on its generation, TensorFlow) into the standalone Qualcomm
# AOT packer even though that process only rewrites an existing LiteRT model.
#
# These remain the same public module attributes; only their first access is
# deferred.  Normal ``litert_torch.convert(...)`` and
# ``from litert_torch import convert`` callers observe the same objects.
_LAZY_CONVERSION_EXPORTS = {
    "convert": ("litert_torch._convert.interface", "convert"),
    "experimental_add_compilation_backend": (
        "litert_torch._convert.interface",
        "experimental_add_compilation_backend",
    ),
    "signature": ("litert_torch._convert.interface", "signature"),
}


def __getattr__(name: str) -> Any:
  target = _LAZY_CONVERSION_EXPORTS.get(name)
  if target is None:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  module_name, attribute_name = target
  value = getattr(importlib.import_module(module_name), attribute_name)
  globals()[name] = value
  return value


def __dir__() -> list[str]:
  return sorted(set(globals()) | set(_LAZY_CONVERSION_EXPORTS))


def load(path: str) -> Model:
  """Imports an litert_torch model from disk.

  Args:
    path: The path to the serialized litert_torch model.

  Returns:
    An litert_torch.model.Model object.
  """
  return Model.load(path)
