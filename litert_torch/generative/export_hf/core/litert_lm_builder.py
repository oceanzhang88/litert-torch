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
"""Litert LM builder for packing exported models."""

import dataclasses
import logging
import os
import re

from litert_torch import progress
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.experimental.minijinja_transpile import transpile as transpile_lib
from litert_torch.generative.export_hf.model_ext import metadata_builder as metadata_builder_lib

import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2
from litert_lm_builder.runtime.proto import sampler_params_pb2

_PH = 'KIMAIRA'

_STOP_TOKEN_PREFIXES = [
    ' ',
    '.',
    ',',
    '?',
    '!',
    ':',
    ';',
    '"',
    "'",
    ')',
    ']',
    '}',
    '*',
]


def parse_chat_template(tokenizer):
  """Parses chat template."""
  if tokenizer.chat_template is None:
    return None
  try:
    messages = [
        {'role': 'system', 'content': _PH},
    ]
    sys_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=False,
    )
    sys_prompt_parts = sys_prompt.split(_PH)
    no_sys_prompt = False
    if len(sys_prompt_parts) == 1:
      sys_prompt_parts = [sys_prompt_parts[0], '']
      no_sys_prompt = True
    if len(sys_prompt_parts) != 2:
      raise ValueError(
          f'System prompt {_PH} not found in chat template: {sys_prompt}'
      )
    if sys_prompt_parts[0].startswith(str(tokenizer.bos_token)):
      sys_prompt_parts[0] = sys_prompt_parts[0][len(tokenizer.bos_token) :]

    if no_sys_prompt:
      messages = [{'role': 'user', 'content': _PH}]
    else:
      messages.append({'role': 'user', 'content': _PH})
    user_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=False,
    )
    if not user_prompt.startswith(sys_prompt):
      raise ValueError('Cannot guess user prompt from prompt template.')
    user_prompt_substr = user_prompt[len(sys_prompt) :]
    user_prompt_parts = user_prompt_substr.split(_PH)
    if len(user_prompt_parts) != 2:
      raise ValueError(
          f'User prompt {_PH} not found in chat template: {user_prompt_substr}'
      )
    messages.append({'role': 'assistant', 'content': _PH})
    model_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=False,
    )
    if not model_prompt.startswith(user_prompt):
      raise ValueError('Cannot guess model prompt from prompt template.')
    model_prompt_substr = model_prompt[len(user_prompt) :]
    model_prompt_parts = model_prompt_substr.split(_PH)
    if len(model_prompt_parts) != 2:
      raise ValueError(
          f'Model prompt {_PH} not found in chat template:'
          f' {model_prompt_substr}'
      )
    return sys_prompt_parts, user_prompt_parts, model_prompt_parts
  except ValueError as e:
    print(f'Failed to parse chat template: {e}')
    return (None, None), (None, None), (None, None)
  except Exception as e:  # pylint: disable=broad-except
    print(f'Failed to parse chat template: {e}')
    return (None, None), (None, None), (None, None)


#: FastVLM's vision tower names its own input resolution, e.g. `mobileclip_l_1024`.
_FAST_VLM_TOWER_RESOLUTION = re.compile(r'_(\d{3,4})$')

#: The resolution every published FastVLM checkpoint has shipped with so far.
_FAST_VLM_DEFAULT_RESOLUTION = 1024


def _fast_vlm_image_tensor_size(config) -> tuple[int, int]:
  """Returns the (height, width) the image preprocessor must resize to.

  Read from the checkpoint rather than assumed: the preprocessor resizes every
  image to this, so a wrong value does not fail loudly -- it feeds the vision
  tower off-size tensors and quietly degrades every caption the model produces.
  """
  for attr in ('image_size', 'vision_tower_input_size'):
    value = getattr(config, attr, None)
    if isinstance(value, int) and value > 0:
      return value, value

  tower = getattr(config, 'mm_vision_tower', None) or ''
  match = _FAST_VLM_TOWER_RESOLUTION.search(str(tower))
  if match:
    size = int(match.group(1))
    return size, size

  logging.warning(
      'FastVLM image resolution not declared in config (mm_vision_tower=%r);'
      ' falling back to %d. Verify against the checkpoint.',
      tower,
      _FAST_VLM_DEFAULT_RESOLUTION,
  )
  return _FAST_VLM_DEFAULT_RESOLUTION, _FAST_VLM_DEFAULT_RESOLUTION


def build_llm_metadata(
    source_model_artifacts: export_lib.SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    chat_templates: tuple[tuple, tuple, tuple] | str,  # pylint: disable=g-bare-generic,
    exported_model_artifacts: export_lib.ExportedModelArtifacts,
    litert_lm_model_type_override: str | None = None,
):
  """Builds LLM metadata."""
  model = source_model_artifacts.model
  tokenizer = source_model_artifacts.tokenizer
  context_length = export_config.cache_length

  llm_metadata = llm_metadata_pb2.LlmMetadata()

  if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token:
    if isinstance(tokenizer.bos_token, int):
      llm_metadata.start_token.token_ids.ids.append(tokenizer.bos_token)
    elif isinstance(tokenizer.bos_token, str):
      llm_metadata.start_token.token_str = tokenizer.bos_token
    else:
      llm_metadata.start_token.token_str = str(tokenizer.bos_token)

  stop_tokens = set()
  gen_config = getattr(model, 'generation_config', None)
  # Tracks whether a usable id was actually taken from the generation config,
  # not merely whether the attribute exists. `GenerationConfig` always defines
  # `eos_token_id`, so testing with `hasattr` reports success even when the
  # value is None -- which is what FastVLM's generation_config.json leaves it
  # as. That suppressed the tokenizer fallback below and shipped a bundle whose
  # only stop token was the chat template's "<|im_end|>\n" suffix and its
  # punctuation-prefixed variants, with no bare "<|im_end|>" among them, so
  # generation ran to the token limit on every turn.
  has_eos_token_id = False
  if gen_config:
    eos_token_id = getattr(gen_config, 'eos_token_id', None)
    if isinstance(eos_token_id, int):
      stop_tokens.add(eos_token_id)
      has_eos_token_id = True
    elif isinstance(eos_token_id, list):
      for token_id in eos_token_id:
        stop_tokens.add(token_id)
      has_eos_token_id = bool(eos_token_id)
  if (
      not has_eos_token_id
      and hasattr(tokenizer, 'eos_token')
      and tokenizer.eos_token
  ):
    stop_tokens.add(tokenizer.eos_token)

  if isinstance(chat_templates, tuple):
    _, _, model_prompt_parts = chat_templates
    if model_prompt_parts[1]:
      stop_tokens.add(model_prompt_parts[1])
  else:
    parsed_templates = parse_chat_template(tokenizer)
    if parsed_templates is not None:
      _, _, model_prompt_parts = parsed_templates
      if model_prompt_parts[1]:
        stop_tokens.add(model_prompt_parts[1])

  expanded_stop_tokens = set()
  for stop_token in stop_tokens:
    if isinstance(stop_token, str):
      expanded_stop_tokens.add(stop_token)
      # Generates punctuation-prefixed variations to handle SentencePiece
      # greedy token merging.
      for prefix in _STOP_TOKEN_PREFIXES:
        expanded_stop_tokens.add(prefix + stop_token)
    else:
      expanded_stop_tokens.add(stop_token)

  for stop_token in expanded_stop_tokens:
    if isinstance(stop_token, int):
      tu = llm_metadata.stop_tokens.add()
      tu.token_ids.ids.append(stop_token)
    elif isinstance(stop_token, str):
      tu = llm_metadata.stop_tokens.add()
      tu.token_str = stop_token

  sampler_top_k = export_config.sampler_top_k
  sampler_top_p = export_config.sampler_top_p
  sampler_temperature = export_config.sampler_temperature

  final_top_k = (
      sampler_top_k
      if sampler_top_k is not None
      else (getattr(gen_config, 'top_k', None) if gen_config else None)
  )
  final_top_p = (
      sampler_top_p
      if sampler_top_p is not None
      else (getattr(gen_config, 'top_p', None) if gen_config else None)
  )
  final_temperature = (
      sampler_temperature
      if sampler_temperature is not None
      else (getattr(gen_config, 'temperature', None) if gen_config else None)
  )

  # Only describe a sampler when there is something to describe. `do_sample`
  # alone is not enough: a config can set it while leaving top_k, top_p and
  # temperature at None, and writing `sampler_params` from that produces a
  # message whose type is TOP_P and whose k, p and temperature are proto3
  # defaults -- k=0, p=0.0, temperature=0.0. That is worse than writing nothing,
  # because `engine_settings.cc` only fills in defaults when the field is
  # *absent*; present-but-empty is taken at face value, and TopPSampler::Create
  # rejects k=0 outright. Leaving it unset lets the runtime choose per backend:
  # k=1/p=0.95/temperature=1.0 on CPU and GPU, and TYPE_UNSPECIFIED on NPU,
  # where the compiled decode graph does its own greedy sampling.
  if (
      final_top_k is not None
      or final_top_p is not None
      or final_temperature is not None
  ):
    sampler_params = llm_metadata.sampler_params
    if final_top_k is not None:
      sampler_params.k = final_top_k
    if final_top_p is not None:
      sampler_params.p = final_top_p
    if final_temperature is not None:
      sampler_params.temperature = final_temperature

    if final_top_k == 1:
      sampler_params.type = sampler_params_pb2.SamplerParameters.GREEDY
    elif final_top_p is not None:
      sampler_params.type = sampler_params_pb2.SamplerParameters.TOP_P
    else:
      sampler_params.type = sampler_params_pb2.SamplerParameters.TOP_K

  if gen_config and getattr(gen_config, 'suppress_tokens', None):
    suppress_tokens = gen_config.suppress_tokens
    for s_token in suppress_tokens:
      llm_metadata.suppress_tokens.ids.append(s_token)

  if chat_templates is not None:
    if isinstance(chat_templates, str):
      if export_config.experimental_transpile_chat_template_for_minijinja:
        chat_templates = transpile_lib.transpile_jinja2(chat_templates)
      llm_metadata.jinja_prompt_template = chat_templates
    else:
      sys_prompt_parts, user_prompt_parts, model_prompt_parts = chat_templates
      pairs = []
      if sys_prompt_parts[0] is not None:
        pairs.append((sys_prompt_parts, llm_metadata.prompt_templates.system))
      if user_prompt_parts[0] is not None:
        pairs.append((user_prompt_parts, llm_metadata.prompt_templates.user))
      if model_prompt_parts[0] is not None:
        pairs.append((model_prompt_parts, llm_metadata.prompt_templates.model))
      for pts, fld in pairs:
        fld.prefix = pts[0]
        fld.suffix = pts[1]

  llm_metadata.max_num_tokens = context_length
  if export_config.llm_metadata_max_num_tokens_override:
    llm_metadata.max_num_tokens = (
        export_config.llm_metadata_max_num_tokens_override
    )

  model_type = litert_lm_model_type_override or model.config.model_type

  match (model_type):
    case 'fast_vlm' | 'llava_qwen2':
      # The runtime selects its data processor from this tag, so a FastVLM
      # bundle that falls through to generic_model below loads without error
      # and then never routes an image through the vision path -- a failure
      # only visible as a model that silently ignores every picture.
      height, width = _fast_vlm_image_tensor_size(model.config)
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(
              fast_vlm=llm_model_type_pb2.FastVlm(
                  image_tensor_height=height,
                  image_tensor_width=width,
              )
          )
      )
    case 'qwen3':
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(qwen3=llm_model_type_pb2.Qwen3())
      )
    case 'qwen2' | 'qwen2p5':
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(qwen2p5=llm_model_type_pb2.Qwen2p5())
      )
    case 'gemma3':
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(gemma3=llm_model_type_pb2.Gemma3())
      )
    case 'function_gemma':
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(
              function_gemma=llm_model_type_pb2.FunctionGemma()
          )
      )
    case 'gemma3n':
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(gemma3n=llm_model_type_pb2.Gemma3N())
      )
    case _:
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(
              generic_model=llm_model_type_pb2.GenericModel()
          )
      )

  # Model specific metadata builders.
  if not litert_lm_model_type_override:
    metadata_builder = metadata_builder_lib.get_metadata_builder(model.config)  # pyrefly: ignore[bad-argument-type]
    llm_metadata = metadata_builder(
        source_model_artifacts,
        export_config,
        exported_model_artifacts,
        llm_metadata,
    )
  # Check thinking channel is properly set up.
  if isinstance(chat_templates, str):
    if '<think>' in chat_templates and not llm_metadata.channels:
      channel = llm_metadata.channels.add()
      channel.channel_name = 'thought'
      channel.start = '<think>'
      channel.end = '</think>'

  return llm_metadata


@progress.task('Package model')
def package_model(
    source_model_artifacts: export_lib.SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: export_lib.ExportedModelArtifacts,
):
  """Packs models to LiteRT LM."""
  work_dir = export_config.work_dir
  output_dir = export_config.output_dir
  use_jinja_template = export_config.use_jinja_template
  litert_lm_model_type_override = export_config.litert_lm_model_type_override
  tokenizer = source_model_artifacts.tokenizer
  tokenizer_model_path = exported_model_artifacts.tokenizer_model_path
  if export_config.tokenizer_path_override:
    tokenizer_model_path = export_config.tokenizer_path_override
  if export_config.jinja_chat_template_override:
    if os.path.exists(export_config.jinja_chat_template_override):
      chat_templates_path = export_config.jinja_chat_template_override
    else:
      try:
        # pylint: disable=g-import-not-at-top
        import huggingface_hub
      except ImportError as e:
        raise ImportError(
            'Please install huggingface_hub to use remote chat template.'
        ) from e
      repo_id = export_config.jinja_chat_template_override
      filename = 'chat_template.jinja'
      chat_templates_path = huggingface_hub.hf_hub_download(
          repo_id=repo_id, filename=filename
      )
    with open(chat_templates_path, 'rt') as f:
      chat_templates = f.read()
  elif use_jinja_template:
    chat_templates = getattr(tokenizer, 'chat_template', '')
  else:
    chat_templates = parse_chat_template(tokenizer)
  if not chat_templates:
    print('WARNING: Chat template is not found. Using empty template.')
  if export_config.litert_lm_llm_metadata_override:
    if os.path.exists(export_config.litert_lm_llm_metadata_override):
      llm_metadata_path = export_config.litert_lm_llm_metadata_override
    else:
      llm_metadata_path = os.path.join(work_dir, 'llm_metadata.pbtext')  # pyrefly: ignore[no-matching-overload]
      with open(llm_metadata_path, 'w') as f:
        f.write(export_config.litert_lm_llm_metadata_override)
  else:
    llm_metadata = build_llm_metadata(
        source_model_artifacts,
        export_config,
        chat_templates,
        exported_model_artifacts,
        litert_lm_model_type_override,
    )
    llm_metadata_path = os.path.join(work_dir, 'llm_metadata.pb')  # pyrefly: ignore[no-matching-overload]
    with open(llm_metadata_path, 'wb') as f:
      f.write(llm_metadata.SerializeToString())

  executor_metadata_builder = (
      metadata_builder_lib.get_executor_metadata_builder(
          source_model_artifacts.model.config  # pyrefly: ignore[bad-argument-type]
      )
  )
  if executor_metadata_builder:
    executor_metadata = executor_metadata_builder(
        source_model_artifacts,
        export_config,
        exported_model_artifacts,
    )
    executor_metadata_path = os.path.join(work_dir, 'executor_metadata.pb')  # pyrefly: ignore[no-matching-overload]
    with open(executor_metadata_path, 'wb') as f:
      f.write(executor_metadata.SerializeToString())
  else:
    executor_metadata_path = None

  builder = litertlm_builder.LitertLmFileBuilder()
  builder.add_system_metadata(
      litertlm_builder.Metadata(
          key='Authors',
          value='ODML',
          dtype=litertlm_builder.DType.STRING,
      )
  )
  builder.add_llm_metadata(llm_metadata_path)
  if executor_metadata_path:
    builder.add_executor_metadata(executor_metadata_path)
  assert (
      tokenizer_model_path is not None
  ), 'Exported tokenizer model path is not found.'
  if (
      export_config.tokenizer_format
      == exportable_module_config.TokenizerFormat.HUGGING_FACE
  ):
    builder.add_hf_tokenizer(tokenizer_model_path)
  elif (
      export_config.tokenizer_format
      == exportable_module_config.TokenizerFormat.SENTENCEPIECE
  ):
    builder.add_sentencepiece_tokenizer(tokenizer_model_path)
  elif tokenizer_model_path.endswith('.json'):
    builder.add_hf_tokenizer(tokenizer_model_path)
  else:
    builder.add_sentencepiece_tokenizer(tokenizer_model_path)
  if export_config.experimental_use_fp16:
    builder.add_tflite_model(
        exported_model_artifacts.prefill_decode_model_path,  # pyrefly: ignore[bad-argument-type]
        litertlm_builder.TfLiteModelType.PREFILL_DECODE,
        prefer_activation_type='fp32_fp16',
    )
  else:
    builder.add_tflite_model(
        exported_model_artifacts.prefill_decode_model_path,  # pyrefly: ignore[bad-argument-type]
        litertlm_builder.TfLiteModelType.PREFILL_DECODE,
    )
  if exported_model_artifacts.embedder_model_path:
    builder.add_tflite_model(
        exported_model_artifacts.embedder_model_path,
        litertlm_builder.TfLiteModelType.EMBEDDER,
    )
  if exported_model_artifacts.vision_encoder_model_path:
    builder.add_tflite_model(
        exported_model_artifacts.vision_encoder_model_path,
        litertlm_builder.TfLiteModelType.VISION_ENCODER,
    )
  if exported_model_artifacts.vision_adapter_model_path:
    builder.add_tflite_model(
        exported_model_artifacts.vision_adapter_model_path,
        litertlm_builder.TfLiteModelType.VISION_ADAPTER,
    )
  if exported_model_artifacts.eoi_model_path:
    builder.add_tflite_model(
        exported_model_artifacts.eoi_model_path,
        litertlm_builder.TfLiteModelType.END_OF_VISION,
    )
  if exported_model_artifacts.auxiliary_model_path:
    builder.add_tflite_model(
        exported_model_artifacts.auxiliary_model_path,
        litertlm_builder.TfLiteModelType.AUX,
    )
  if exported_model_artifacts.additional_model_paths:
    for (
        name,
        model_path,
    ) in exported_model_artifacts.additional_model_paths.items():
      if name == 'per_layer_embedder':
        model_type = litertlm_builder.TfLiteModelType.PER_LAYER_EMBEDDER
      else:
        raise ValueError(f'Unsupported additional model type: {name}')
      builder.add_tflite_model(
          model_path,
          model_type,
      )
  model_path = os.path.join(output_dir, 'model.litertlm')  # pyrefly: ignore[no-matching-overload]
  with open(model_path, 'wb') as f:
    builder.build(f)
  return dataclasses.replace(
      exported_model_artifacts,
      litert_lm_model_path=model_path,
  )
