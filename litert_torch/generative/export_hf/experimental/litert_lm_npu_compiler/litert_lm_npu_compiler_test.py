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
"""Tests for litert_lm_npu_compiler."""

import json
import os
import pathlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import flatbuffers
from litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler import litert_lm_npu_compiler

try:
  from ai_edge_litert.aot.core import aot_types
except ImportError:
  from ai_edge_litert.aot.core import types as aot_types
from ai_edge_litert import schema_py_generated as schema_fb
import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import llm_metadata_pb2


def build_dummy_tflite_model(signatures_info: list[tuple[str, int]]) -> bytes:
  """Creates a dummy flatbuffer model with specified signatures."""
  builder = flatbuffers.Builder(1024)

  # Create buffers
  schema_fb.BufferStart(builder)
  buffer0_offset = schema_fb.BufferEnd(builder)
  schema_fb.ModelStartBuffersVector(builder, 1)
  builder.PrependUOffsetTRelative(buffer0_offset)
  buffers_offset = builder.EndVector()

  # Create Operator Codes (required for valid operators)
  schema_fb.OperatorCodeStart(builder)
  schema_fb.OperatorCodeAddBuiltinCode(builder, schema_fb.BuiltinOperator.ADD)
  op_code_offset = schema_fb.OperatorCodeEnd(builder)
  schema_fb.ModelStartOperatorCodesVector(builder, 1)
  builder.PrependUOffsetTRelative(op_code_offset)
  op_codes_offset = builder.EndVector()

  # Create a dummy subgraph for each unique subgraph index in signatures
  subgraph_indices = sorted(list(set([idx for _, idx in signatures_info])))
  if not subgraph_indices:
    subgraph_indices = [0]

  subgraph_offsets = []
  for sg_idx in subgraph_indices:
    # We need some dummy tensors to make it valid
    schema_fb.TensorStart(builder)
    schema_fb.TensorAddType(builder, 0)
    schema_fb.TensorAddBuffer(builder, 0)
    tensor_offset = schema_fb.TensorEnd(builder)

    schema_fb.SubGraphStartTensorsVector(builder, 1)
    builder.PrependUOffsetTRelative(tensor_offset)
    tensors_offset = builder.EndVector()

    # Dummy Operator
    schema_fb.OperatorStart(builder)
    schema_fb.OperatorAddOpcodeIndex(builder, 0)
    op_offset = schema_fb.OperatorEnd(builder)

    schema_fb.SubGraphStartOperatorsVector(builder, 1)
    builder.PrependUOffsetTRelative(op_offset)
    ops_offset = builder.EndVector()

    schema_fb.SubGraphStart(builder)
    schema_fb.SubGraphAddTensors(builder, tensors_offset)
    schema_fb.SubGraphAddOperators(builder, ops_offset)
    subgraph_offset = schema_fb.SubGraphEnd(builder)
    subgraph_offsets.append(subgraph_offset)

  schema_fb.ModelStartSubgraphsVector(builder, len(subgraph_offsets))
  for sg_offset in reversed(subgraph_offsets):
    builder.PrependUOffsetTRelative(sg_offset)
  subgraphs_offset = builder.EndVector()

  # Create signature defs
  sig_offsets = []
  for key, sg_idx in signatures_info:
    sig_key_offset = builder.CreateString(key)
    schema_fb.SignatureDefStart(builder)
    schema_fb.SignatureDefAddSignatureKey(builder, sig_key_offset)
    schema_fb.SignatureDefAddSubgraphIndex(builder, sg_idx)
    sig_offset = schema_fb.SignatureDefEnd(builder)
    sig_offsets.append(sig_offset)

  schema_fb.ModelStartSignatureDefsVector(builder, len(sig_offsets))
  for sig_offset in reversed(sig_offsets):
    builder.PrependUOffsetTRelative(sig_offset)
  signature_defs_offset = builder.EndVector()

  # Create model
  schema_fb.ModelStart(builder)
  schema_fb.ModelAddVersion(builder, 3)
  schema_fb.ModelAddSubgraphs(builder, subgraphs_offset)
  schema_fb.ModelAddBuffers(builder, buffers_offset)
  schema_fb.ModelAddSignatureDefs(builder, signature_defs_offset)
  schema_fb.ModelAddOperatorCodes(builder, op_codes_offset)
  model_offset = schema_fb.ModelEnd(builder)
  builder.Finish(model_offset)

  return builder.Output()


class LitertLmNpuCompilerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir()
    # Production images intentionally select an isolated compiler through ENV.
    # Unit tests mock ApplyPlugin unless they explicitly exercise that route, so
    # keep the ambient image configuration from changing their execution path.
    compiler_env = mock.patch.dict(
        os.environ,
        {litert_lm_npu_compiler._AOT_COMPILER_ROOT_ENV: ''},
        clear=False,
    )
    compiler_env.start()
    self.addCleanup(compiler_env.stop)

  def test_sdk_root_precedes_the_python_sdk_package(self):
    sdk_root = pathlib.Path(self.test_dir) / 'qairt'
    sdk_libs = sdk_root / 'lib' / 'x86_64-linux-clang'
    sdk_libs.mkdir(parents=True)

    with mock.patch.dict(
        os.environ, {'QNN_SDK_ROOT': str(sdk_root)}, clear=False
    ), mock.patch.object(
        litert_lm_npu_compiler.importlib,
        'import_module',
        side_effect=AssertionError('Python SDK package must not be consulted'),
    ):
      resolved = litert_lm_npu_compiler._resolve_sdk_libs_path('qualcomm')

    self.assertEqual(resolved, str(sdk_libs))

  def test_declared_sdk_root_with_no_host_libs_fails_closed(self):
    sdk_root = pathlib.Path(self.test_dir) / 'wrong-qairt'
    sdk_root.mkdir()

    with mock.patch.dict(
        os.environ, {'QNN_SDK_ROOT': str(sdk_root)}, clear=False
    ):
      with self.assertRaisesRegex(
          FileNotFoundError, 'QNN_SDK_ROOT.*lib/x86_64-linux-clang'
      ):
        litert_lm_npu_compiler._resolve_sdk_libs_path('qualcomm')

  @parameterized.named_parameters(
      ('rc1_runtime_name', 'libLiteRt.so'),
      ('new_runtime_name', 'libLiteRtRuntimeCApi.so'),
  )
  def test_isolated_compiler_root_resolves_its_own_binary_and_plugin(
      self, runtime_name
  ):
    root = pathlib.Path(self.test_dir) / 'litert-rc1' / 'ai_edge_litert'
    binary = root / 'tools' / 'apply_plugin_main'
    plugin_dir = root / 'vendors' / 'qualcomm' / 'compiler'
    runtime = root / runtime_name
    binary.parent.mkdir(parents=True)
    plugin_dir.mkdir(parents=True)
    binary.write_bytes(b'compiler')
    runtime.write_bytes(b'runtime')

    with mock.patch.dict(
        os.environ,
        {litert_lm_npu_compiler._AOT_COMPILER_ROOT_ENV: str(root)},
        clear=False,
    ):
      resolved = litert_lm_npu_compiler._resolve_isolated_compiler_assets(
          'qualcomm'
      )

    self.assertEqual(resolved, (root, binary, plugin_dir))

  def test_declared_isolated_compiler_root_fails_closed(self):
    root = pathlib.Path(self.test_dir) / 'incomplete-litert'
    root.mkdir()

    with mock.patch.dict(
        os.environ,
        {litert_lm_npu_compiler._AOT_COMPILER_ROOT_ENV: str(root)},
        clear=False,
    ):
      with self.assertRaisesRegex(
          FileNotFoundError, 'LITERT_AOT_COMPILER_ROOT.*compiler assets'
      ):
        litert_lm_npu_compiler._resolve_isolated_compiler_assets('qualcomm')

  def test_isolated_apply_uses_versioned_assets_and_qnn_libs(self):
    root = pathlib.Path(self.test_dir) / 'litert-rc1' / 'ai_edge_litert'
    binary = root / 'tools' / 'apply_plugin_main'
    plugin_dir = root / 'vendors' / 'qualcomm' / 'compiler'
    input_path = pathlib.Path(self.test_dir) / 'input.tflite'
    output_path = pathlib.Path(self.test_dir) / 'output.tflite'
    sdk_libs = pathlib.Path(self.test_dir) / 'qairt' / 'lib'
    input_path.write_bytes(b'\x08\x00\x00\x00TFL3')
    captured = {}

    def fake_run(args, **kwargs):
      captured['args'] = args
      captured['env'] = kwargs['env']
      output_path.write_bytes(b'\x08\x00\x00\x00TFL3compiled')
      return litert_lm_npu_compiler.subprocess.CompletedProcess(
          args, 0, stdout='compiled'
      )

    with mock.patch.object(
        litert_lm_npu_compiler.subprocess, 'run', side_effect=fake_run
    ):
      litert_lm_npu_compiler._run_isolated_apply_plugin(
          compiler_assets=(root, binary, plugin_dir),
          input_path=input_path,
          output_path=output_path,
          soc_manufacturer='Qualcomm',
          soc_model='SM8850',
          sdk_libs_path=str(sdk_libs),
          subgraphs_to_compile=[1, 4],
          flags_dict={
              'qualcomm_optimization_level': 'O3',
              'qualcomm_enable_weight_sharing': True,
          },
      )

    self.assertIn(f'--libs={plugin_dir}', captured['args'])
    self.assertIn('--soc_model=SM8850', captured['args'])
    self.assertIn('--subgraphs=1,4', captured['args'])
    self.assertIn('--qualcomm_optimization_level=O3', captured['args'])
    self.assertIn('--qualcomm_enable_weight_sharing=True', captured['args'])
    self.assertTrue(
        captured['env']['LD_LIBRARY_PATH'].startswith(
            f'{root}{os.pathsep}{sdk_libs}'
        )
    )

  @parameterized.named_parameters(
      ('no_signatures', [], None),
      ('other_signatures', [('serving_default', 0)], None),
      (
          'mask_signatures',
          [('prefill_mask_128', 1), ('decode_mask', 4)],
          [1, 4],
      ),
      (
          'mixed_signatures',
          [('serving_default', 0), ('prefill_mask_128', 1), ('decode_mask', 4)],
          [1, 4],
      ),
  )
  def test_resolve_subgraphs_to_compile(self, signatures, expected):
    model_bytes = build_dummy_tflite_model(signatures)
    model_path = pathlib.Path(self.test_dir) / 'test_model.tflite'
    with open(model_path, 'wb') as f:
      f.write(model_bytes)

    resolved = litert_lm_npu_compiler._resolve_subgraphs_to_compile(
        model_path, 'aux'
    )
    self.assertEqual(resolved, expected)

  def test_resolve_subgraphs_to_compile_non_aux(self):
    model_bytes = build_dummy_tflite_model([('prefill_mask_128', 1)])
    model_path = pathlib.Path(self.test_dir) / 'test_model.tflite'
    with open(model_path, 'wb') as f:
      f.write(model_bytes)

    resolved = litert_lm_npu_compiler._resolve_subgraphs_to_compile(
        model_path, 'prefill_decode'
    )
    self.assertIsNone(resolved)

  @mock.patch(
      'litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler.litert_lm_npu_compiler.ApplyPlugin'
  )
  def test_main_flow(self, mock_apply_plugin_class):
    compiled_instances = []

    # Setup mock compiler side effect to create dummy compiled file
    def apply_plugin_init_side_effect(*args, **kwargs):
      instance_mock = mock.MagicMock()

      def compile_call_side_effect(*args, **kwargs):
        input_model = kwargs.get('input_model')
        output_model = kwargs.get('output_model')
        model_path = input_model.path
        compiled_path = output_model.path
        with open(model_path, 'rb') as orig_f:
          model_bytes = orig_f.read()
        with open(compiled_path, 'wb') as f:
          f.write(model_bytes)

      instance_mock.side_effect = compile_call_side_effect
      compiled_instances.append(instance_mock)
      return instance_mock

    mock_apply_plugin_class.side_effect = apply_plugin_init_side_effect

    # Create dummy models
    main_model_bytes = build_dummy_tflite_model([('serving_default', 0)])
    aux_model_bytes = build_dummy_tflite_model(
        [('prefill_mask_128', 1), ('decode_mask', 2)]
    )

    main_model_path = os.path.join(self.test_dir, 'model.tflite')
    aux_model_path = os.path.join(self.test_dir, 'aux.tflite')

    with open(main_model_path, 'wb') as f:
      f.write(main_model_bytes)
    with open(aux_model_path, 'wb') as f:
      f.write(aux_model_bytes)

    # Create dummy metadata and tokenizer
    metadata_path = os.path.join(self.test_dir, 'metadata.pb')
    metadata = llm_metadata_pb2.LlmMetadata()
    with open(metadata_path, 'wb') as f:
      f.write(metadata.SerializeToString())

    tokenizer_path = os.path.join(self.test_dir, 'tokenizer.model')
    with open(tokenizer_path, 'w') as f:
      f.write('dummy tokenizer content')

    # Build input .litertlm
    input_litertlm_path = os.path.join(self.test_dir, 'input.litertlm')
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        main_model_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        aux_model_path, litertlm_builder.TfLiteModelType.AUX
    )
    builder.add_llm_metadata(metadata_path)
    builder.add_sentencepiece_tokenizer(tokenizer_path)

    with open(input_litertlm_path, 'wb') as f:
      builder.build(f)

    # Output path
    output_litertlm_path = os.path.join(self.test_dir, 'output.litertlm')

    # Run litert_lm_npu_compiler
    litert_lm_npu_compiler.compile_litertlm(
        input_litertlm=input_litertlm_path,
        output_litertlm=output_litertlm_path,
        backend='qualcomm',
        soc_model='SM8550',
        compile_configs=json.dumps({
            'prefill_decode': ['--qualcomm_enable_weight_sharing=true'],
            'aux': ['--qualcomm_optimization_level=O3'],
        }),
    )

    # Verify ApplyPlugin was instantiated correctly for each model
    # We expect 2 compilations: prefill_decode and aux
    self.assertEqual(mock_apply_plugin_class.call_count, 2)

    # First call should be for prefill_decode (or aux, order depends on TOML, but let's check both)
    # Actually, we can check calls
    calls = mock_apply_plugin_class.call_args_list

    # One call should have subgraphs_to_compile=None (prefill_decode)
    # Another should have subgraphs_to_compile=[1, 2] (aux, resolved from signatures)
    subgraphs_compiled = [call[1].get('subgraphs_to_compile') for call in calls]
    self.assertIn(None, subgraphs_compiled)
    self.assertIn([1, 2], subgraphs_compiled)

    # Verify compiler calls
    self.assertEqual(len(compiled_instances), 2)

    compiler_calls = []
    for inst in compiled_instances:
      self.assertEqual(inst.call_count, 1)
      compiler_calls.append(inst.call_args)

    # Verify kwargs passed to compiler
    # For prefill_decode we passed --qualcomm_enable_weight_sharing=true
    # For aux we passed --qualcomm_optimization_level=O3
    passed_kwargs = [call[1] for call in compiler_calls]

    # We need to find which call is which
    prefill_decode_call = None
    aux_call = None
    for kw in passed_kwargs:
      if 'qualcomm_enable_weight_sharing' in kw:
        prefill_decode_call = kw
      if 'qualcomm_optimization_level' in kw:
        aux_call = kw

    self.assertIsNotNone(prefill_decode_call)
    self.assertIsNotNone(aux_call)

    self.assertTrue(prefill_decode_call['qualcomm_enable_weight_sharing'])
    self.assertEqual(aux_call['qualcomm_optimization_level'], 'O3')

    self.assertEqual(prefill_decode_call['soc_manufacturer'], 'Qualcomm')
    self.assertEqual(prefill_decode_call['soc_model'], 'SM8550')

    # Verify output file exists
    self.assertTrue(os.path.exists(output_litertlm_path))

  @mock.patch(
      'litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler.litert_lm_npu_compiler.ApplyPlugin'
  )
  def test_generic_defaults_qualcomm(self, mock_apply_plugin_class):
    compiled_instances = []

    def apply_plugin_init_side_effect(*args, **kwargs):
      instance_mock = mock.MagicMock()

      def compile_call_side_effect(*args, **kwargs):
        input_model = kwargs.get('input_model')
        output_model = kwargs.get('output_model')
        model_path = input_model.path
        compiled_path = output_model.path
        with open(model_path, 'rb') as orig_f:
          model_bytes = orig_f.read()
        with open(compiled_path, 'wb') as f:
          f.write(model_bytes)

      instance_mock.side_effect = compile_call_side_effect
      compiled_instances.append(instance_mock)
      return instance_mock

    mock_apply_plugin_class.side_effect = apply_plugin_init_side_effect

    main_model_bytes = build_dummy_tflite_model([('serving_default', 0)])
    aux_model_bytes = build_dummy_tflite_model([('prefill_mask_128', 1)])
    main_model_path = os.path.join(self.test_dir, 'model.tflite')
    aux_model_path = os.path.join(self.test_dir, 'aux.tflite')
    with open(main_model_path, 'wb') as f:
      f.write(main_model_bytes)
    with open(aux_model_path, 'wb') as f:
      f.write(aux_model_bytes)

    metadata_path = os.path.join(self.test_dir, 'metadata.pb')
    with open(metadata_path, 'wb') as f:
      f.write(llm_metadata_pb2.LlmMetadata().SerializeToString())
    tokenizer_path = os.path.join(self.test_dir, 'tokenizer.model')
    with open(tokenizer_path, 'w') as f:
      f.write('dummy')

    input_litertlm_path = os.path.join(self.test_dir, 'input.litertlm')
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        main_model_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        aux_model_path, litertlm_builder.TfLiteModelType.AUX
    )
    builder.add_llm_metadata(metadata_path)
    builder.add_sentencepiece_tokenizer(tokenizer_path)
    with open(input_litertlm_path, 'wb') as f:
      builder.build(f)

    output_litertlm_path = os.path.join(self.test_dir, 'output.litertlm')

    litert_lm_npu_compiler.compile_litertlm(
        input_litertlm=input_litertlm_path,
        output_litertlm=output_litertlm_path,
        backend='qualcomm',
        soc_model='SM8550',
    )

    self.assertLen(compiled_instances, 2)
    prefill_decode_call = None
    aux_call = None
    for inst in compiled_instances:
      inst.assert_called_once()
      args, kwargs = inst.call_args
      input_model = kwargs.get('input_model') or args[0]
      if 'aux' in str(input_model.path):
        aux_call = kwargs
      elif 'prefill_decode' in str(input_model.path):
        prefill_decode_call = kwargs

    self.assertIsNotNone(prefill_decode_call)
    self.assertIsNotNone(aux_call)

    self.assertEqual(prefill_decode_call['qualcomm_optimization_level'], 'O3')
    self.assertEqual(prefill_decode_call['qualcomm_log_level'], 'off')
    self.assertTrue(prefill_decode_call['qualcomm_enable_weight_sharing'])

    self.assertEqual(aux_call['qualcomm_optimization_level'], 'O3')
    self.assertEqual(aux_call['qualcomm_log_level'], 'off')
    self.assertNotIn('qualcomm_enable_weight_sharing', aux_call)

  @mock.patch(
      'litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler.litert_lm_npu_compiler.ApplyPlugin'
  )
  def test_generic_defaults_mediatek(self, mock_apply_plugin_class):
    compiled_instances = []

    def apply_plugin_init_side_effect(*args, **kwargs):
      instance_mock = mock.MagicMock()

      def compile_call_side_effect(**kwargs):
        input_model = kwargs.get('input_model')
        output_model = kwargs.get('output_model')
        model_path = input_model.path
        compiled_path = output_model.path
        with open(model_path, 'rb') as orig_f:
          model_bytes = orig_f.read()
        with open(compiled_path, 'wb') as f:
          f.write(model_bytes)

      instance_mock.side_effect = compile_call_side_effect
      compiled_instances.append(instance_mock)
      return instance_mock

    mock_apply_plugin_class.side_effect = apply_plugin_init_side_effect

    main_model_bytes = build_dummy_tflite_model([('serving_default', 0)])
    aux_model_bytes = build_dummy_tflite_model([('prefill_mask_128', 1)])
    main_model_path = os.path.join(self.test_dir, 'model.tflite')
    aux_model_path = os.path.join(self.test_dir, 'aux.tflite')
    with open(main_model_path, 'wb') as f:
      f.write(main_model_bytes)
    with open(aux_model_path, 'wb') as f:
      f.write(aux_model_bytes)

    metadata_path = os.path.join(self.test_dir, 'metadata.pb')
    with open(metadata_path, 'wb') as f:
      f.write(llm_metadata_pb2.LlmMetadata().SerializeToString())
    tokenizer_path = os.path.join(self.test_dir, 'tokenizer.model')
    with open(tokenizer_path, 'w') as f:
      f.write('dummy')

    input_litertlm_path = os.path.join(self.test_dir, 'input.litertlm')
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        main_model_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        aux_model_path, litertlm_builder.TfLiteModelType.AUX
    )
    builder.add_llm_metadata(metadata_path)
    builder.add_sentencepiece_tokenizer(tokenizer_path)
    with open(input_litertlm_path, 'wb') as f:
      builder.build(f)

    output_litertlm_path = os.path.join(self.test_dir, 'output.litertlm')

    litert_lm_npu_compiler.compile_litertlm(
        input_litertlm=input_litertlm_path,
        output_litertlm=output_litertlm_path,
        backend='mediatek',
        soc_model='mt6991',
    )

    # Only prefill_decode should be compiled, aux should be skipped (not in configs for mediatek)
    self.assertLen(compiled_instances, 1)
    inst = compiled_instances[0]
    inst.assert_called_once()
    kwargs = inst.call_args[1]
    self.assertTrue(kwargs['mediatek_enable_gemma_compiler_optimizations'])
    self.assertEqual(kwargs['mediatek_performance_mode_type'], 'turbo_boost')

  @mock.patch(
      'litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler.litert_lm_npu_compiler.ApplyPlugin'
  )
  def test_disable_weight_sharing(self, mock_apply_plugin_class):
    compiled_instances = []

    def apply_plugin_init_side_effect(*args, **kwargs):
      instance_mock = mock.MagicMock()

      def compile_call_side_effect(*args, **kwargs):
        input_model = kwargs.get('input_model')
        output_model = kwargs.get('output_model')
        model_path = input_model.path
        compiled_path = output_model.path
        with open(model_path, 'rb') as orig_f:
          model_bytes = orig_f.read()
        with open(compiled_path, 'wb') as f:
          f.write(model_bytes)

      instance_mock.side_effect = compile_call_side_effect
      compiled_instances.append(instance_mock)
      return instance_mock

    mock_apply_plugin_class.side_effect = apply_plugin_init_side_effect

    main_model_bytes = build_dummy_tflite_model([('serving_default', 0)])
    aux_model_bytes = build_dummy_tflite_model([('prefill_mask_128', 1)])
    main_model_path = os.path.join(self.test_dir, 'model.tflite')
    aux_model_path = os.path.join(self.test_dir, 'aux.tflite')
    with open(main_model_path, 'wb') as f:
      f.write(main_model_bytes)
    with open(aux_model_path, 'wb') as f:
      f.write(aux_model_bytes)

    metadata_path = os.path.join(self.test_dir, 'metadata.pb')
    with open(metadata_path, 'wb') as f:
      f.write(llm_metadata_pb2.LlmMetadata().SerializeToString())
    tokenizer_path = os.path.join(self.test_dir, 'tokenizer.model')
    with open(tokenizer_path, 'w') as f:
      f.write('dummy')

    input_litertlm_path = os.path.join(self.test_dir, 'input.litertlm')
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        main_model_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        aux_model_path, litertlm_builder.TfLiteModelType.AUX
    )
    builder.add_llm_metadata(metadata_path)
    builder.add_sentencepiece_tokenizer(tokenizer_path)
    with open(input_litertlm_path, 'wb') as f:
      builder.build(f)

    output_litertlm_path = os.path.join(self.test_dir, 'output.litertlm')

    litert_lm_npu_compiler.compile_litertlm(
        input_litertlm=input_litertlm_path,
        output_litertlm=output_litertlm_path,
        backend='qualcomm',
        soc_model='SM8550',
        disable_weight_sharing=True,
    )

    self.assertLen(compiled_instances, 2)

    prefill_decode_call = None
    for inst in compiled_instances:
      args, kwargs = inst.call_args
      input_model = kwargs.get('input_model') or args[0]
      if 'prefill_decode' in str(input_model.path):
        prefill_decode_call = kwargs

    self.assertIsNotNone(prefill_decode_call)
    self.assertFalse(prefill_decode_call['qualcomm_enable_weight_sharing'])

  def test_validation_aux_mediatek_raises(self):
    main_model_bytes = build_dummy_tflite_model([('serving_default', 0)])
    aux_model_bytes = build_dummy_tflite_model([('prefill_mask_128', 1)])
    main_model_path = os.path.join(self.test_dir, 'model.tflite')
    aux_model_path = os.path.join(self.test_dir, 'aux.tflite')
    with open(main_model_path, 'wb') as f:
      f.write(main_model_bytes)
    with open(aux_model_path, 'wb') as f:
      f.write(aux_model_bytes)

    metadata_path = os.path.join(self.test_dir, 'metadata.pb')
    with open(metadata_path, 'wb') as f:
      f.write(llm_metadata_pb2.LlmMetadata().SerializeToString())
    tokenizer_path = os.path.join(self.test_dir, 'tokenizer.model')
    with open(tokenizer_path, 'w') as f:
      f.write('dummy')

    input_litertlm_path = os.path.join(self.test_dir, 'input.litertlm')
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        main_model_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        aux_model_path, litertlm_builder.TfLiteModelType.AUX
    )
    builder.add_llm_metadata(metadata_path)
    builder.add_sentencepiece_tokenizer(tokenizer_path)
    with open(input_litertlm_path, 'wb') as f:
      builder.build(f)

    output_litertlm_path = os.path.join(self.test_dir, 'output.litertlm')

    with self.assertRaises(ValueError):
      litert_lm_npu_compiler.compile_litertlm(
          input_litertlm=input_litertlm_path,
          output_litertlm=output_litertlm_path,
          backend='mediatek',
          soc_model='mt6991',
          compile_configs=json.dumps({'aux': {'compile': True}}),
      )

  @mock.patch(
      'litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler.litert_lm_npu_compiler.ApplyPlugin'
  )
  def test_disable_aux_compilation(self, mock_apply_plugin_class):
    compiled_instances = []

    def apply_plugin_init_side_effect(*args, **kwargs):
      instance_mock = mock.MagicMock()

      def compile_call_side_effect(*args, **kwargs):
        input_model = kwargs.get('input_model')
        output_model = kwargs.get('output_model')
        model_path = input_model.path
        compiled_path = output_model.path
        with open(model_path, 'rb') as orig_f:
          model_bytes = orig_f.read()
        with open(compiled_path, 'wb') as f:
          f.write(model_bytes)

      instance_mock.side_effect = compile_call_side_effect
      compiled_instances.append(instance_mock)
      return instance_mock

    mock_apply_plugin_class.side_effect = apply_plugin_init_side_effect

    main_model_bytes = build_dummy_tflite_model([('serving_default', 0)])
    aux_model_bytes = build_dummy_tflite_model([('prefill_mask_128', 1)])
    main_model_path = os.path.join(self.test_dir, 'model.tflite')
    aux_model_path = os.path.join(self.test_dir, 'aux.tflite')
    with open(main_model_path, 'wb') as f:
      f.write(main_model_bytes)
    with open(aux_model_path, 'wb') as f:
      f.write(aux_model_bytes)

    metadata_path = os.path.join(self.test_dir, 'metadata.pb')
    with open(metadata_path, 'wb') as f:
      f.write(llm_metadata_pb2.LlmMetadata().SerializeToString())
    tokenizer_path = os.path.join(self.test_dir, 'tokenizer.model')
    with open(tokenizer_path, 'w') as f:
      f.write('dummy')

    input_litertlm_path = os.path.join(self.test_dir, 'input.litertlm')
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        main_model_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        aux_model_path, litertlm_builder.TfLiteModelType.AUX
    )
    builder.add_llm_metadata(metadata_path)
    builder.add_sentencepiece_tokenizer(tokenizer_path)
    with open(input_litertlm_path, 'wb') as f:
      builder.build(f)

    output_litertlm_path = os.path.join(self.test_dir, 'output.litertlm')

    litert_lm_npu_compiler.compile_litertlm(
        input_litertlm=input_litertlm_path,
        output_litertlm=output_litertlm_path,
        backend='qualcomm',
        soc_model='SM8550',
        disable_aux_compilation=True,
    )

    # Only prefill_decode should be compiled, aux should be skipped (disabled by flag)
    self.assertLen(compiled_instances, 1)
    inst = compiled_instances[0]
    inst.assert_called_once()
    args, kwargs = inst.call_args
    input_model = kwargs.get('input_model') or args[0]
    self.assertIn('prefill_decode', str(input_model.path))




if __name__ == '__main__':
  absltest.main()
