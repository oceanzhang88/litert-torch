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
"""Tests for calibration prompt construction."""

from litert_torch.generative.export_hf.experimental.calib import quant_utils
from litert_torch.generative.export_hf.experimental.calib import tokenizer as tokenizer_lib

from absl.testing import absltest as googletest


class _FakeTransformersTokenizer:
  chat_template = 'checkpoint-owned-template'

  def apply_chat_template(
      self, messages, *, tokenize: bool, add_generation_prompt: bool
  ):
    assert tokenize is False
    assert add_generation_prompt is True
    return (
        '<|im_start|>system\nHelpful.<|im_end|>\n'
        '<|im_start|>user\n'
        f'{messages[0]["content"]}'
        '<|im_end|>\n<|im_start|>assistant\n'
    )


class _FakeTokenizer:

  def __init__(self, tx_tokenizer):
    self.tx_tokenizer = tx_tokenizer


class QuantUtilsTest(googletest.TestCase):

  def test_multimodal_request_uses_checkpoint_chat_template(self):
    request = quant_utils.get_example_prompt(
        {
            'data_items': [
                {'image_bytes': b'pixels'},
                {'text': 'Describe the golf swing.'},
            ]
        },
        tokenizer=_FakeTokenizer(_FakeTransformersTokenizer()),
    )

    self.assertIsInstance(request, tokenizer_lib.Request)
    self.assertEqual(
        request.contents,
        [
            tokenizer_lib.DataItem(
                text=(
                    '<|im_start|>system\nHelpful.<|im_end|>\n'
                    '<|im_start|>user\n'
                )
            ),
            tokenizer_lib.DataItem(image_bytes=b'pixels'),
            tokenizer_lib.DataItem(text='Describe the golf swing.'),
            tokenizer_lib.DataItem(
                text='<|im_end|>\n<|im_start|>assistant\n'
            ),
        ],
    )

  def test_sentencepiece_only_request_keeps_legacy_boundaries(self):
    request = quant_utils.get_example_prompt(
        {'data_items': [{'text': 'Hello'}]},
        tokenizer=_FakeTokenizer(None),
    )

    self.assertEqual(
        request.contents,
        [
            tokenizer_lib.DataItem(text=quant_utils.PROMPT_TEMPLATE_PREFIX),
            tokenizer_lib.DataItem(text='Hello'),
            tokenizer_lib.DataItem(text=quant_utils.PROMPT_TEMPLATE_SUFFIX),
        ],
    )

  def test_disabled_formatting_adds_no_boundaries(self):
    request = quant_utils.get_example_prompt(
        {'data_items': [{'image_bytes': b'pixels'}, {'text': 'Hello'}]},
        enable_formatting=False,
        tokenizer=_FakeTokenizer(_FakeTransformersTokenizer()),
    )

    self.assertEqual(
        request.contents,
        [
            tokenizer_lib.DataItem(image_bytes=b'pixels'),
            tokenizer_lib.DataItem(text='Hello'),
        ],
    )


if __name__ == '__main__':
  googletest.main()
