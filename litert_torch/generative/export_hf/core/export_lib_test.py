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
"""Tests for export_lib."""

import json
import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core.split_cache import (
    exportable_module as split_cache_module,
)


class ExportLibTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_maybe_patch_tokenizer_patches_incorrect_config(self):
    tokenizer_path = os.path.join(self.test_dir, "tokenizer.json")

    # BPE + Metaspace + BPE chars in vocab
    incorrect_config = {
        "model": {"type": "BPE", "vocab": {"ĠI": 1, "Ċ": 2, "hello": 3}},
        "pre_tokenizer": {"type": "Metaspace", "replacement": " "},
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {
                    "type": "Replace",
                    "pattern": {"String": " "},
                    "content": " ",
                }
            ],
        },
    }

    with open(tokenizer_path, "w") as f:
      json.dump(incorrect_config, f)

    export_lib._maybe_patch_tokenizer(tokenizer_path)

    with open(tokenizer_path, "r") as f:
      patched_config = json.load(f)

    self.assertEqual(patched_config["pre_tokenizer"]["type"], "ByteLevel")
    self.assertEqual(patched_config["decoder"]["type"], "ByteLevel")
    self.assertTrue(patched_config["pre_tokenizer"]["use_regex"])
    self.assertTrue(patched_config["decoder"]["use_regex"])

  def test_maybe_patch_tokenizer_skips_correct_config(self):
    tokenizer_path = os.path.join(self.test_dir, "tokenizer.json")

    # BPE + ByteLevel
    correct_config = {
        "model": {"type": "BPE", "vocab": {"ĠI": 1, "Ċ": 2}},
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
    }

    with open(tokenizer_path, "w") as f:
      json.dump(correct_config, f)

    export_lib._maybe_patch_tokenizer(tokenizer_path)

    with open(tokenizer_path, "r") as f:
      config = json.load(f)

    self.assertEqual(config["pre_tokenizer"]["type"], "ByteLevel")
    self.assertEqual(config["decoder"]["type"], "ByteLevel")
    self.assertNotIn("use_regex", config["pre_tokenizer"])

  def test_maybe_patch_tokenizer_skips_non_bpe(self):
    tokenizer_path = os.path.join(self.test_dir, "tokenizer.json")

    # WordPiece + Metaspace (dummy)
    config = {
        "model": {"type": "WordPiece", "vocab": {"hello": 1}},
        "pre_tokenizer": {"type": "Metaspace"},
    }

    with open(tokenizer_path, "w") as f:
      json.dump(config, f)

    export_lib._maybe_patch_tokenizer(tokenizer_path)

    with open(tokenizer_path, "r") as f:
      result_config = json.load(f)

    self.assertEqual(result_config["pre_tokenizer"]["type"], "Metaspace")

  def test_maybe_patch_tokenizer_skips_no_bpe_chars(self):
    tokenizer_path = os.path.join(self.test_dir, "tokenizer.json")

    # BPE + Metaspace but NO BPE chars in vocab
    config = {
        "model": {"type": "BPE", "vocab": {"hello": 1, "world": 2}},
        "pre_tokenizer": {"type": "Metaspace"},
    }

    with open(tokenizer_path, "w") as f:
      json.dump(config, f)

    export_lib._maybe_patch_tokenizer(tokenizer_path)

    with open(tokenizer_path, "r") as f:
      result_config = json.load(f)

    self.assertEqual(result_config["pre_tokenizer"]["type"], "Metaspace")

  def test_asr_export_config_rules(self):
    config = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_asr_model",
        task=export_lib.exportable_module_config.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
        split_cache=True,
        export_vision_encoder=True,
    )
    self.assertEqual(
        config.task,
        export_lib.exportable_module_config.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
    )
    self.assertTrue(config.export_audio_encoder)
    self.assertFalse(config.export_vision_encoder)
    self.assertFalse(config.split_cache)
    self.assertFalse(config.externalize_embedder)
    self.assertFalse(config.externalize_rope)
    self.assertEqual(config.input_sec, 1.0)
    self.assertEqual(config.stateful_after, -1)

  def test_gpu_dynamic_shapes_config_rules(self):
    # Only prefill
    config = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        prefill_lengths=[32, 128],
        cache_length=1024,
        enable_gpu_dynamic_prefill=True,
    )
    self.assertEqual(config.prefill_lengths, [37, 131])
    self.assertEqual(config.cache_length, 1024)

    # Only cache
    config = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        prefill_lengths=[32, 128],
        cache_length=1024,
        enable_gpu_dynamic_cache=True,
    )
    self.assertEqual(config.prefill_lengths, [32, 128])
    self.assertEqual(config.cache_length, 1031)

    # Both
    config = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        prefill_lengths=[32, 128],
        cache_length=1024,
        enable_gpu_dynamic_prefill=True,
        enable_gpu_dynamic_cache=True,
    )
    self.assertEqual(config.prefill_lengths, [37, 131])
    self.assertEqual(config.cache_length, 1031)

  def test_gpu_dynamic_shapes_conflict(self):
    with self.assertRaises(ValueError):
      export_lib.exportable_module_config.ExportableModuleConfig(
          model="dummy_model",
          enable_dynamic_shape=True,
          enable_gpu_dynamic_prefill=True,
      )
    with self.assertRaises(ValueError):
      export_lib.exportable_module_config.ExportableModuleConfig(
          model="dummy_model",
          enable_dynamic_shape=True,
          enable_gpu_dynamic_cache=True,
      )

  def test_sentencepiece_format_converts_directly(self):
    class Tokenizer:

      def save_pretrained(self, *args, **kwargs):
        raise AssertionError("JSON export must not run in SentencePiece mode")

    config = exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        work_dir=self.test_dir,
        tokenizer_format="sentencepiece",
    )
    source_artifacts = SimpleNamespace(tokenizer=Tokenizer())
    with mock.patch.object(
        export_lib.tokenizer_lib, "convert", return_value=b"sentencepiece"
    ) as convert:
      artifacts = export_lib.export_tokenizer(
          source_artifacts, config, export_lib.ExportedModelArtifacts()
      )

    convert.assert_called_once_with(source_artifacts.tokenizer)
    self.assertEqual(
        artifacts.tokenizer_model_path,
        os.path.join(self.test_dir, "tokenizer.spiece"),
    )
    with open(artifacts.tokenizer_model_path, "rb") as tokenizer_file:
      self.assertEqual(tokenizer_file.read(), b"sentencepiece")

  def test_block_embedder_recipe_preserves_float_calibration_path(self):
    float_path = os.path.join(self.test_dir, "embedder_calibration.tflite")
    quantized_path = os.path.join(
        self.test_dir, "embedder_calibration_quantized.tflite"
    )
    config = exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        split_cache=True,
        block_embedder_quantization_recipe="dynamic_wi8_afp32",
    )
    source = export_lib.ExportedModelArtifacts(
        calibration_embedder_model_path=float_path
    )
    with mock.patch.object(
        export_lib, "quantize_model", return_value=quantized_path
    ) as quantize:
      result = export_lib.prepare_shipped_block_embedder_model(
          None, config, source
      )

    quantize.assert_called_once_with(float_path, "dynamic_wi8_afp32")
    self.assertEqual(result.calibration_embedder_model_path, float_path)
    self.assertEqual(result.embedder_model_path, quantized_path)
    self.assertTrue(config.export_calibration_embedder)

  def test_calibration_embedder_export_disables_quantization(self):
    config = exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        task=exportable_module_config.ExportTask.IMAGE_TEXT_TO_TEXT,
        quantization_recipe="static_wi8_ai16",
        export_calibration_embedder=True,
    )
    with mock.patch.object(
        export_lib, "_export_embedder_impl", return_value="artifacts"
    ) as export_impl:
      result = export_lib.export_calibration_embedder_model(
          None, config, export_lib.ExportedModelArtifacts()
      )

    self.assertEqual(result, "artifacts")
    self.assertIsNone(export_impl.call_args.kwargs["quantization_recipe"])
    self.assertFalse(
        export_impl.call_args.args[1].single_token_embedder
    )

  def test_split_cache_mask_fill_is_checkpoint_owned(self):
    small = split_cache_module.SplitAttentionMaskBuilder(
        context_size=1280, mask_fill=-100.0
    )
    large = split_cache_module.SplitAttentionMaskBuilder(
        context_size=1280, mask_fill=-1e4
    )

    self.assertEqual(small.global_mask.mask_fill, -100.0)
    self.assertEqual(large.global_mask.mask_fill, -1e4)

  def test_split_cache_mask_sample_inputs_preserve_production_abi(self):
    config = exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        split_cache=True,
        prefill_lengths=[128],
    )
    production_builder = split_cache_module.SplitAttentionMaskBuilder(
        context_size=1280
    )
    current_runtime_builder = split_cache_module.SplitAttentionMaskBuilder(
        context_size=1280, pad_token=None
    )

    production_inputs = production_builder.get_sample_inputs(None, config)
    current_runtime_inputs = current_runtime_builder.get_sample_inputs(
        None, config
    )
    for signature_inputs, _ in production_inputs.values():
      self.assertEqual(
          set(signature_inputs), {"input_tokens", "time_step"}
      )
    for signature_inputs, _ in current_runtime_inputs.values():
      self.assertEqual(
          set(signature_inputs), {"input_tokens", "time_step", "valid_mask"}
      )

  def test_block_embedder_recipe_rejects_portable_executor(self):
    with self.assertRaisesRegex(ValueError, "requires split_cache=True"):
      exportable_module_config.ExportableModuleConfig(
          model="dummy_model",
          split_cache=False,
          block_embedder_quantization_recipe="dynamic_wi8_afp32",
      )

  def test_tokenizer_format_rejects_unknown_value(self):
    with self.assertRaisesRegex(ValueError, "Unsupported tokenizer_format"):
      exportable_module_config.ExportableModuleConfig(
          model="dummy_model", tokenizer_format="tokenizer-of-the-day"
      )


if __name__ == "__main__":
  absltest.main()
