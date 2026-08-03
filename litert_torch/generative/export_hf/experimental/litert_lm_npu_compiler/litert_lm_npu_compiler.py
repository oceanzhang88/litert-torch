# Copyright 2026 The LiteRT Torch Authors. All Rights Reserved.
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
"""Tool for compiling TFLite models inside .litertlm file for NPU."""

import contextlib
import importlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

try:
  import tomllib
except ImportError:
  import tomli as tomllib
from typing import Any, Sequence

from absl import app
from absl import flags
from absl import logging
from litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler import litert_lm_npu_compiler_configs

gfile = None

try:
  from ai_edge_litert.aot.core import aot_types
except ImportError:
  # LiteRT 2.0.x used ``types`` for this public AOT model wrapper.  The
  # Qualcomm 2.39 compiler generation predates the 2.1.x rename, so a compiler
  # lane intentionally pinned to that generation must accept both spellings.
  from ai_edge_litert.aot.core import types as aot_types
from ai_edge_litert.aot.core.apply_plugin import ApplyPlugin
from ai_edge_litert import schema_py_generated as schema_fb
import litert_lm_builder as litertlm_builder
from litert_lm_builder import litertlm_peek


def _open(path: str | pathlib.Path, mode: str = 'r'):
  if gfile:
    return gfile.Open(str(path), mode)
  return open(path, mode)


FLAGS = flags.FLAGS


def _define_flag(define_fn, name, *args, **kwargs):
  try:
    return define_fn(name, *args, **kwargs)
  except flags.DuplicateFlagError:
    return flags.FLAGS[name]


_define_flag(
    flags.DEFINE_string, 'input_litertlm', None, 'Path to input .litertlm file.'
)
_define_flag(
    flags.DEFINE_string,
    'output_litertlm',
    None,
    'Path to output .litertlm file.',
)
_define_flag(
    flags.DEFINE_enum,
    'backend',
    None,
    ['qualcomm', 'mediatek'],
    'Target NPU backend.',
)
_define_flag(
    flags.DEFINE_string,
    'soc_model',
    None,
    'Target SoC model (e.g., SM8850, MT6993).',
)
_define_flag(
    flags.DEFINE_string,
    'compile_configs',
    None,
    'JSON string or path to JSON file containing compilation flags for each'
    ' model type.',
)
_define_flag(
    flags.DEFINE_string,
    'model_name',
    None,
    'Model name to use default configurations (e.g., gemma4_2b).',
)
_define_flag(
    flags.DEFINE_bool,
    'disable_weight_sharing',
    False,
    'Disable weight sharing for Qualcomm backend compiler.',
)
_define_flag(
    flags.DEFINE_bool,
    'disable_aux_compilation',
    False,
    'Disable compiling aux model.',
)


#: Backend -> the distribution that ships that vendor's host compiler libs.
#: Separate distributions from `ai_edge_litert`, which is why ApplyPlugin's
#: default library search (a walk of the ai_edge_litert package tree) never
#: finds them.
_SDK_PACKAGES = {
    'qualcomm': 'ai_edge_litert_sdk_qualcomm',
    'mediatek': 'ai_edge_litert_sdk_mediatek',
}

#: Vendor SDK roots accepted without a Python wheel.  Qualcomm's public SDK is
#: published as a versioned QAIRT archive; keeping that archive as the compiler
#: body avoids installing one SDK wheel and then replacing its 3.6 GB ``data``
#: directory with another version.  The package lookup below remains the
#: compatibility path for ordinary upstream installations.
_SDK_ROOTS = {
    'qualcomm': ('QNN_SDK_ROOT', pathlib.Path('lib/x86_64-linux-clang')),
}

# Optional package root for a compiler wheel that is deliberately isolated from
# the runtime/conversion wheel imported by this process.  Qualcomm's public
# QAIRT 2.39.3 body exposes QNN System API 1.5: LiteRT 2.1.0rc1 can compile
# SM8850 against it, while the current LiteRT/quantizer stack requires 2.1.6 and
# that release's plugin requires System API 1.11.  A target directory installed
# with ``pip --target`` lets conversion stay current without loading two
# incompatible ``ai_edge_litert`` packages into one interpreter.
_AOT_COMPILER_ROOT_ENV = 'LITERT_AOT_COMPILER_ROOT'


def _resolve_sdk_libs_path(backend: str) -> str | None:
  """Locates the vendor SDK libs the compiler plugin links against.

  `aot_compile` reaches the plugin through the vendor backend, which resolves
  this and hands it to ApplyPlugin as `sdk_libs_path`. Calling ApplyPlugin
  directly, as this module does, skips that step -- so without this the plugin
  cannot dlopen libQnnHtp.so and the only report is "failed to apply plugin".

  A declared vendor root is authoritative and fail-closed: silently falling
  back to a differently versioned Python SDK after a typo in ``QNN_SDK_ROOT``
  would compile with the wrong QNN generation.  Returns None only when neither
  a root nor an SDK package is present, which remains a real configuration for
  callers that populate ``LD_LIBRARY_PATH`` themselves.
  """
  root_spec = _SDK_ROOTS.get(backend)
  if root_spec:
    environment_name, relative_libs = root_spec
    root = os.environ.get(environment_name)
    if root:
      libs = pathlib.Path(os.path.abspath(os.path.expanduser(root))) / relative_libs
      if not libs.is_dir():
        raise FileNotFoundError(
            f'{environment_name}={root!r} has no SDK library directory at {libs}'
        )
      return str(libs)

  package = _SDK_PACKAGES.get(backend)
  if not package:
    return None
  try:
    sdk = importlib.import_module(package)
  except ImportError:
    return None
  path_to_sdk_libs = getattr(sdk, 'path_to_sdk_libs', None)
  if path_to_sdk_libs is None:
    return None
  return str(path_to_sdk_libs())


def _resolve_isolated_compiler_assets(
    backend: str,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path] | None:
  """Returns an explicitly isolated apply binary and plugin directory.

  The environment value names the ``ai_edge_litert`` package root itself, not
  its site-packages parent.  A declared root is authoritative and fail-closed;
  falling back to the imported wheel after one missing file would silently
  switch QNN ABI generations.
  """
  value = os.environ.get(_AOT_COMPILER_ROOT_ENV)
  if not value:
    return None

  root = pathlib.Path(os.path.abspath(os.path.expanduser(value)))
  binary = root / 'tools' / 'apply_plugin_main'
  plugin_dir = root / 'vendors' / backend / 'compiler'
  # LiteRT renamed its packaged runtime C library after 2.1.0rc1.  Accept the
  # two real wheel layouts, but still require one to live inside the declared
  # root so the subprocess cannot bind to the ambient system LiteRT by accident.
  runtime_candidates = (
      root / 'libLiteRt.so',
      root / 'libLiteRtRuntimeCApi.so',
  )
  runtime = next((path for path in runtime_candidates if path.is_file()), None)
  missing = [
      str(path)
      for path in (binary, plugin_dir)
      if not path.exists()
  ]
  if runtime is None:
    missing.append(' or '.join(str(path) for path in runtime_candidates))
  if missing:
    raise FileNotFoundError(
        f'{_AOT_COMPILER_ROOT_ENV}={value!r} lacks isolated compiler assets: '
        + ', '.join(missing)
    )
  if not binary.is_file() or not plugin_dir.is_dir():
    raise FileNotFoundError(
        f'{_AOT_COMPILER_ROOT_ENV}={value!r} has invalid compiler asset types'
    )
  return root, binary, plugin_dir


def _run_isolated_apply_plugin(
    *,
    compiler_assets: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    soc_manufacturer: str,
    soc_model: str,
    sdk_libs_path: str | None,
    subgraphs_to_compile: list[int] | None,
    flags_dict: dict[str, Any],
) -> None:
  """Runs a version-isolated LiteRT compiler without changing Python imports."""
  compiler_root, binary, plugin_dir = compiler_assets
  args = [
      str(binary),
      '--cmd=apply',
      f'--model={input_path}',
      f'--o={output_path}',
      f'--soc_manufacturer={soc_manufacturer}',
      f'--soc_model={soc_model}',
      '--err=--',
      f'--libs={plugin_dir}',
  ]
  args.extend(f'--{key}={value}' for key, value in flags_dict.items())
  if subgraphs_to_compile:
    args.append(
        '--subgraphs=' + ','.join(str(index) for index in subgraphs_to_compile)
    )

  env = os.environ.copy()
  search_paths = [str(compiler_root)]
  if sdk_libs_path:
    search_paths.append(sdk_libs_path)
  if env.get('LD_LIBRARY_PATH'):
    search_paths.append(env['LD_LIBRARY_PATH'])
  env['LD_LIBRARY_PATH'] = os.pathsep.join(search_paths)

  result = subprocess.run(
      args,
      check=False,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      env=env,
  )
  if result.returncode:
    log_file = tempfile.NamedTemporaryFile(
        suffix='.error', mode='w', delete=False
    )
    log_file.write(result.stdout)
    log_file.close()
    raise ValueError(
        'isolated apply_plugin failed to compile '
        f'{input_path} with exit {result.returncode}. See {log_file.name}'
    )
  if not output_path.is_file() or output_path.stat().st_size < 8:
    raise ValueError(f'{output_path} is not a TFLite model.')
  with output_path.open('rb') as output_file:
    header = output_file.read(8)
  if header[4:8] != b'TFL3':
    raise ValueError(f'{output_path} is not a TFLite model.')


def _get_soc_manufacturer(backend: str) -> str:
  if backend == 'qualcomm':
    return 'Qualcomm'
  elif backend == 'mediatek':
    return 'MediaTek'
  else:
    raise ValueError(f'Unsupported backend: {backend}')


def _resolve_subgraphs_to_compile(
    model_path: pathlib.Path, model_type: str
) -> list[int] | None:
  """Resolve which subgraphs to compile based on model type and signatures."""
  if model_type != 'aux':
    return None

  try:
    model_bytes = model_path.read_bytes()
    model = schema_fb.Model.GetRootAsModel(model_bytes, 0)
    if not model.SignatureDefsLength():
      logging.warning('No signatures found in aux model.')
      return None

    subgraphs_to_compile = []
    for i in range(model.SignatureDefsLength()):
      sig = model.SignatureDefs(i)
      key = sig.SignatureKey()
      if isinstance(key, bytes):
        key = key.decode('utf-8')
      if re.search(r'(prefill_mask|decode_mask)', key):
        subgraphs_to_compile.append(sig.SubgraphIndex())

    if subgraphs_to_compile:
      logging.info(
          'Resolved subgraphs to compile for aux model: %s',
          subgraphs_to_compile,
      )
      return sorted(list(set(subgraphs_to_compile)))

  except Exception as e:  # pylint: disable=broad-except
    logging.exception('Failed to parse aux model for signatures')

  return None


_DEFAULT_INTERMEDIATE_ARTIFACTS_DIR = 'intermediate_compiler_artifacts'


def compile_litertlm(
    input_litertlm: str | pathlib.Path,
    output_litertlm: str | pathlib.Path,
    backend: str,
    soc_model: str,
    compile_configs: str | dict[str, Any] | None = None,
    model_name: str | None = None,
    disable_weight_sharing: bool = False,
    disable_aux_compilation: bool = False,
    intermediate_dir: str | pathlib.Path | None = None,
    keep_temporary_files: bool = False,
) -> None:
  """Compiles LiteRT-LM models inside a package for the target NPU.

  Args:
    input_litertlm: Path to the input .litertlm file.
    output_litertlm: Path to the output .litertlm file.
    backend: Target backend ('qualcomm' or 'mediatek').
    soc_model: SoC model ID (e.g. 'SM8850', 'MT6993').
    compile_configs: Optional JSON string or dictionary representing compilation
      configs.
    model_name: Optional model name to resolve model-specific default configs.
    disable_weight_sharing: If True, disables weight sharing for Qualcomm.
    disable_aux_compilation: If True, disables compiling auxiliary model.

  Raises:
    ValueError: If compile options are invalid.
    RuntimeError: If compilation fails.
  """
  configs = {}
  if compile_configs:
    if isinstance(compile_configs, dict):
      configs = compile_configs
    elif compile_configs.startswith('{'):
      configs = json.loads(compile_configs)
    else:
      with _open(compile_configs, 'r') as f:
        configs = json.load(f)
  else:
    # Resolve default configs
    if model_name:
      key = (backend, model_name)
      if key in litert_lm_npu_compiler_configs.MODEL_SPECIFIC_DEFAULTS:
        configs = litert_lm_npu_compiler_configs.MODEL_SPECIFIC_DEFAULTS[key]
        logging.info(
            'Using model-specific defaults for %s on %s',
            model_name,
            backend,
        )
      else:
        configs = litert_lm_npu_compiler_configs.GENERIC_DEFAULT_CONFIGS.get(
            backend, {}
        )
        logging.info(
            'No model-specific defaults for %s, using generic defaults for %s',
            model_name,
            backend,
        )
    else:
      configs = litert_lm_npu_compiler_configs.GENERIC_DEFAULT_CONFIGS.get(
          backend, {}
      )
      logging.info('Using generic defaults for %s', backend)

  soc_manufacturer = _get_soc_manufacturer(backend)
  isolated_compiler_assets = _resolve_isolated_compiler_assets(backend)
  if isolated_compiler_assets:
    logging.info(
        'Using isolated LiteRT AOT compiler root: %s',
        isolated_compiler_assets[0],
    )

  if intermediate_dir:
    temp_dir_path = pathlib.Path(intermediate_dir)
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    ctx = contextlib.nullcontext(str(temp_dir_path))
  elif keep_temporary_files:
    temp_dir_path = (
        pathlib.Path(str(output_litertlm)).parent
        / _DEFAULT_INTERMEDIATE_ARTIFACTS_DIR
    )
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    ctx = contextlib.nullcontext(str(temp_dir_path))
  else:
    ctx = tempfile.TemporaryDirectory()

  with ctx as temp_dir:
    temp_dir_path = pathlib.Path(temp_dir)
    logging.info('Unpacking %s to %s', input_litertlm, temp_dir)

    litertlm_peek.peek_litertlm_file(str(input_litertlm), temp_dir, sys.stdout)

    toml_path = temp_dir_path / 'model.toml'
    if not toml_path.exists():
      raise RuntimeError(
          f'Failed to unpack, model.toml not found in {temp_dir}'
      )

    with _open(toml_path, 'r') as f:
      toml_data = tomllib.loads(f.read())

    compiled_types: list[str] = []
    seen_types: list[str] = []

    if 'section' in toml_data:
      for section in toml_data['section']:
        if section.get('section_type') == 'TFLiteModel':
          model_type = section.get('model_type')
          relative_data_path = section.get('data_path')
          model_path = temp_dir_path / relative_data_path

          if model_type:
            seen_types.append(model_type)

          should_compile = False
          extra_flags: list[str] = []

          if model_type in configs:
            if isinstance(configs[model_type], list):
              should_compile = True
              extra_flags = list(configs[model_type])
            elif isinstance(configs[model_type], dict):
              should_compile = configs[model_type].get('compile', True)
              raw_flags = configs[model_type].get('flags', [])
              if isinstance(raw_flags, list):
                extra_flags = list(raw_flags)

          if (
              should_compile
              and backend == 'qualcomm'
              and model_type == 'prefill_decode'
          ):
            if not disable_weight_sharing:
              if not any(
                  'qualcomm_enable_weight_sharing' in f for f in extra_flags
              ):
                extra_flags = list(extra_flags)
                extra_flags.append('--qualcomm_enable_weight_sharing=true')
            else:
              extra_flags = [
                  f
                  for f in extra_flags
                  if 'qualcomm_enable_weight_sharing' not in f
              ]
              extra_flags = list(extra_flags)
              extra_flags.append('--qualcomm_enable_weight_sharing=false')

          if model_type == 'aux' and disable_aux_compilation:
            should_compile = False

          if should_compile and model_type == 'aux' and backend != 'qualcomm':
            raise ValueError(
                'Compiling aux model is only supported for Qualcomm backend.'
            )

          if should_compile:
            logging.info('Compiling model %s (%s)', model_type, model_path)

            subgraphs = _resolve_subgraphs_to_compile(model_path, model_type)

            kwargs = {}
            for flag in extra_flags:
              flag = flag.lstrip('-')
              if '=' in flag:
                k, v = flag.split('=', 1)
                if v.lower() == 'true':
                  v = True
                elif v.lower() == 'false':
                  v = False
                else:
                  try:
                    v = int(v)
                  except ValueError:
                    pass
                kwargs[k] = v
              else:
                kwargs[flag] = True

            compiled_model_path = model_path.with_suffix('.compiled.tflite')

            # The vendor SDK ships as its own distribution, so it is not under
            # the ai_edge_litert tree that ApplyPlugin's default
            # `construct_ld_library_path()` walks -- the compiler plugin then
            # fails to dlopen libQnnHtp.so and reports only "failed to apply
            # plugin". `aot_compile`'s backend resolves this by passing
            # sdk_libs_path; calling ApplyPlugin directly skips that, which is
            # why the pip-installed SDK alone was never enough here.
            sdk_libs_path = _resolve_sdk_libs_path(backend)
            if sdk_libs_path:
              logging.info('Using %s SDK libs: %s', backend, sdk_libs_path)
            else:
              logging.warning(
                  'No SDK package found for backend %s; the plugin will only '
                  'load if LD_LIBRARY_PATH already points at the vendor libs',
                  backend,
              )

            try:
              if isolated_compiler_assets:
                _run_isolated_apply_plugin(
                    compiler_assets=isolated_compiler_assets,
                    input_path=model_path,
                    output_path=compiled_model_path,
                    soc_manufacturer=soc_manufacturer,
                    soc_model=soc_model,
                    sdk_libs_path=sdk_libs_path,
                    subgraphs_to_compile=subgraphs,
                    flags_dict=kwargs,
                )
              else:
                input_model = aot_types.Model(path=str(model_path))
                output_model = aot_types.Model(path=str(compiled_model_path))
                compiler = ApplyPlugin(
                    subgraphs_to_compile=subgraphs,
                    experimental_capture_stderr=True,
                )

                # Resolve plugin directory from the imported ai_edge_litert
                # package only in the ordinary same-version path.  The isolated
                # path resolves its plugin beside its own executable above.
                try:
                  apply_plugin_module = sys.modules[ApplyPlugin.__module__]
                  file_path = (
                      getattr(apply_plugin_module, '__file__', None)
                      if apply_plugin_module
                      else None
                  )
                  if file_path is not None:
                    apply_plugin_file = pathlib.Path(file_path)
                    package_root = apply_plugin_file.parent.parent.parent
                    plugin_dir = package_root / 'vendors' / backend / 'compiler'
                    if not plugin_dir.exists():
                      # Fallback for monorepo source tree layout
                      plugin_dir = (
                          package_root.parent / 'vendors' / backend / 'compiler'
                      )

                    if plugin_dir.exists():
                      logging.info('Found plugin directory: %s', plugin_dir)
                      kwargs['libs'] = str(plugin_dir)
                    else:
                      logging.warning(
                          'Could not find plugin directory for backend %s',
                          backend,
                      )
                except Exception as e:  # pylint: disable=broad-except
                  logging.warning('Failed to resolve plugin directory: %s', e)

                compiler(
                    input_model=input_model,
                    output_model=output_model,
                    soc_manufacturer=soc_manufacturer,
                    soc_model=soc_model,
                    sdk_libs_path=sdk_libs_path,
                    **kwargs,
                )
              os.replace(compiled_model_path, model_path)
              logging.info('Successfully compiled %s', model_type)
              compiled_types.append(str(model_type))
            except Exception as e:
              if model_path.exists():
                logging.error(
                    'Model path %s size: %d',
                    model_path,
                    model_path.stat().st_size,
                )
              else:
                logging.error('Model path %s does not exist!', model_path)
              logging.error('Failed to compile %s: %s', model_type, e)
              raise

    # Repacking nothing is the failure this whole tool exists to avoid: the
    # output is a valid bundle, named for a SoC, that runs entirely on CPU.
    # Nothing downstream can tell it apart from a compiled one without counting
    # dispatch ops, so refuse here instead of shipping it.
    if not compiled_types:
      raise ValueError(
          'NPU compilation matched no sections, so the output would be an '
          'uncompiled copy of the input. Bundle model types: '
          f'{sorted(set(seen_types))}; configured model types: '
          f'{sorted(configs)}.'
      )
    logging.info('Compiled sections: %s', compiled_types)

    logging.info('Repacking to %s', output_litertlm)
    output_dir = os.path.dirname(output_litertlm)
    if output_dir and not os.path.exists(output_dir):
      os.makedirs(output_dir, exist_ok=True)
    builder = litertlm_builder.LitertLmFileBuilder.from_toml_file(
        str(toml_path)
    )
    with _open(output_litertlm, 'wb') as f:
      builder.build(f)
    logging.info('Done')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not (
      FLAGS.input_litertlm
      and FLAGS.output_litertlm
      and FLAGS.backend
      and FLAGS.soc_model
  ):
    raise app.UsageError(
        'Missing required flags: --input_litertlm, --output_litertlm,'
        ' --backend, --soc_model'
    )

  try:
    compile_litertlm(
        input_litertlm=FLAGS.input_litertlm,
        output_litertlm=FLAGS.output_litertlm,
        backend=FLAGS.backend,
        soc_model=FLAGS.soc_model,
        compile_configs=FLAGS.compile_configs,
        model_name=FLAGS.model_name,
        disable_weight_sharing=FLAGS.disable_weight_sharing,
        disable_aux_compilation=FLAGS.disable_aux_compilation,
    )
  except ValueError as e:
    raise app.UsageError(str(e)) from e


if __name__ == '__main__':
  app.run(main)
