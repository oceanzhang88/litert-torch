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
"""Tests for Hugging Face tokenizer to SentencePiece conversion."""

import json
import os
import shutil
import tempfile

from absl.testing import absltest

from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib
from sentencepiece import sentencepiece_model_pb2 as spm_model


class _JsonBackedTokenizer:
  """Small Qwen-shaped tokenizer whose vocab_file is JSON, not protobuf."""

  def __init__(self, vocab_file: str):
    self.vocab_file = vocab_file
    self.vocab = {'<unk>': 0, 'hello': 1, 'world': 2}
    self.unk_token = '<unk>'
    self.bos_token = None
    self.eos_token = None
    self.pad_token = None
    self.special_tokens_map = {}

  def get_added_vocab(self):
    return {}

  def decode(self, ids):
    id_to_token = {id_: token for token, id_ in self.vocab.items()}
    return ''.join(id_to_token[id_] for id_ in ids)


class TokenizerToSentencePieceLibTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_json_vocab_file_reconstructs_instead_of_parsing_as_proto(self):
    vocab_file = os.path.join(self.test_dir, 'vocab.json')
    with open(vocab_file, 'w') as file:
      json.dump({'<unk>': 0, 'hello': 1, 'world': 2}, file)

    serialized = tokenizer_to_sentencepiece_lib.convert(
        _JsonBackedTokenizer(vocab_file), normalize_tokens='none'
    )
    model = spm_model.ModelProto.FromString(serialized)

    self.assertEqual(
        [piece.piece for piece in model.pieces],
        ['<unk>', 'hello', 'world'],
    )
    self.assertEqual(model.trainer_spec.vocab_size, 3)

  def test_sentencepiece_content_is_reused_regardless_of_filename(self):
    model = spm_model.ModelProto()
    model.trainer_spec.vocab_size = 1
    model.pieces.add(
        piece='<unk>',
        score=0.0,
        type=spm_model.ModelProto.SentencePiece.UNKNOWN,
    )
    serialized = model.SerializeToString()
    # Content, rather than an extension heuristic, owns this decision.
    vocab_file = os.path.join(self.test_dir, 'vocab.json')
    with open(vocab_file, 'wb') as file:
      file.write(serialized)

    tokenizer = type('Tokenizer', (), {'vocab_file': vocab_file})()

    self.assertEqual(
        tokenizer_to_sentencepiece_lib.convert(tokenizer), serialized
    )


if __name__ == '__main__':
  absltest.main()
