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
"""Kokoro TTS model export to TFLite flatbuffers via LiteRT-Torch."""

import os
import shutil
from typing import Any

import huggingface_hub
import litert_torch
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core.speech import tts_model
from litert_torch.generative.export_hf.model_ext.kokoro_tts import istftnet
from litert_torch.generative.export_hf.model_ext.kokoro_tts import model as kokoro_model
import numpy as np
import torch
from torch import nn

from ai_edge_quantizer import quantizer as quantizer_lib
from ai_edge_quantizer import recipe as recipe_lib


def _get_config_opt(
    export_config: exportable_module_config.ExportableModuleConfig,
    key: str,
    default: Any,
) -> Any:
  """Reads an option from `export_config.extra_kwargs` or `export_config`."""
  if key in export_config.extra_kwargs:
    val = export_config.extra_kwargs[key]
    if val is not None:
      return val
  val = getattr(export_config, key, None)
  return default if val is None else val


class KokoroAcousticFusedModule(nn.Module):
  """Wraps Kokoro duration, time-alignment, and prosody predictor for FX export."""

  def __init__(self, kmodel: kokoro_model.KModel, max_frames: int = 512):
    super().__init__()
    self.bert = kmodel.bert
    self.bert_encoder = kmodel.bert_encoder
    self.predictor = kmodel.predictor
    self.duration_encoder = kmodel.predictor.text_encoder
    self.duration_proj = kmodel.predictor.duration_proj
    self.shared = kmodel.predictor.shared
    self.text_encoder = kmodel.text_encoder
    self.max_frames = max_frames

  def forward(
      self,
      phoneme_ids: torch.LongTensor,
      speaker_style: torch.FloatTensor,
      phoneme_length: torch.LongTensor,
  ) -> dict[str, torch.Tensor]:
    _, seq_len = phoneme_ids.shape
    time_arange_t = torch.arange(seq_len, device=phoneme_ids.device)
    text_mask = time_arange_t.unsqueeze(0) >= phoneme_length.unsqueeze(1)
    attn_mask = 1 - text_mask.int()

    bert_dur = self.bert(phoneme_ids, attention_mask=attn_mask)
    d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
    d = self.duration_encoder(d_en, speaker_style, phoneme_length, text_mask)

    x = self.predictor.lstm(d, phoneme_length, mask=text_mask)

    duration = torch.sigmoid(self.duration_proj(x)).sum(axis=-1)
    pred_dur = torch.round(duration).clamp(min=1.0).to(torch.float32)
    pred_dur = pred_dur * (1.0 - text_mask.float())

    cumsum = torch.cumsum(pred_dur, dim=-1)
    start_frames = cumsum - pred_dur
    speech_frame_length = torch.clamp(
        cumsum[:, -1:], max=self.max_frames
    ).long()

    time_arange_f = torch.arange(
        self.max_frames, device=phoneme_ids.device
    ).unsqueeze(0)
    time_grid = time_arange_f.unsqueeze(1)
    start_expanded = start_frames.unsqueeze(-1)
    end_expanded = cumsum.unsqueeze(-1)

    aln = ((time_grid >= start_expanded) & (time_grid < end_expanded)).float()
    frame_mask = time_arange_f >= speech_frame_length
    valid_aln_mask = (1.0 - text_mask.float().unsqueeze(-1)) * (
        1.0 - frame_mask.float().unsqueeze(1)
    )
    aln_masked = aln * valid_aln_mask

    en = d.transpose(-1, -2) @ aln_masked
    en_masked = en * (1.0 - frame_mask.float().unsqueeze(1))

    x_in = en_masked.transpose(-1, -2)
    x_shared = self.shared(x_in, speech_frame_length, mask=frame_mask)

    f_mask = (1.0 - frame_mask.float()).unsqueeze(1)
    f0_pred = x_shared.transpose(-1, -2)
    for b in self.predictor.F0:
      f0_pred = b(f0_pred, speaker_style, f_mask)
    f0_pred = self.predictor.F0_proj(f0_pred).squeeze(1)

    n_pred = x_shared.transpose(-1, -2)
    for b in self.predictor.N:
      n_pred = b(n_pred, speaker_style, f_mask)
    n_pred = self.predictor.N_proj(n_pred).squeeze(1)

    t_en = self.text_encoder(phoneme_ids, phoneme_length, text_mask)
    asr = t_en @ aln_masked

    f0_mask = torch.arange(
        2 * self.max_frames, device=f0_pred.device
    ).unsqueeze(0) >= (2 * speech_frame_length)
    f0_pred = f0_pred * (1.0 - f0_mask.float())
    n_pred = n_pred * (1.0 - f0_mask.float())
    asr = asr * (1.0 - frame_mask.float().unsqueeze(1))

    return {
        "acoustic_features": asr,
        "pitch_contour": f0_pred,
        "energy_contour": n_pred,
        "speech_frame_length": speech_frame_length,
    }


class KokoroVocoderModule(nn.Module):
  """Wraps Kokoro iSTFTNet vocoder for FX export."""

  def __init__(self, kmodel: kokoro_model.KModel):
    super().__init__()
    self.decoder = kmodel.decoder
    self.decoder.generator.return_spec_phase = True

  def forward(
      self,
      acoustic_features: torch.FloatTensor,
      pitch_contour: torch.FloatTensor,
      energy_contour: torch.FloatTensor,
      speaker_style: torch.FloatTensor,
      speech_frame_length: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    sfl_f = speech_frame_length.float()
    t_frame = acoustic_features.shape[-1]
    time_arange_f = torch.arange(
        t_frame, device=acoustic_features.device, dtype=torch.float32
    ).unsqueeze(0)
    frame_mask = time_arange_f >= sfl_f.reshape(-1, 1)
    t_f0 = pitch_contour.shape[-1]
    f0_mask = torch.arange(
        t_f0, device=pitch_contour.device, dtype=torch.float32
    ).unsqueeze(0) >= (2.0 * sfl_f).reshape(-1, 1)

    asr_masked = acoustic_features * (1.0 - frame_mask.float().unsqueeze(1))
    f0_masked = pitch_contour * (1.0 - f0_mask.float())
    n_masked = energy_contour * (1.0 - f0_mask.float())

    f0 = self.decoder.generator.f0_upsamp(f0_masked[:, None]).transpose(1, 2)
    har_source, _, _ = self.decoder.generator.m_source(f0)
    har_source = har_source.transpose(1, 2).squeeze(1)
    har_spec, har_phase = self.decoder.generator.stft.transform(har_source)
    har = torch.cat([har_spec, har_phase], dim=1)

    spec, phase = self.decoder(
        asr_masked,
        f0_masked,
        n_masked,
        speaker_style,
        harmonic_features=har,
        l_speech=speech_frame_length,
    )
    return {
        "magnitude_spectrogram": spec,
        "phase_spectrogram": phase,
    }


def prepare_kmodel_for_export(
    kmodel: kokoro_model.KModel,
) -> kokoro_model.KModel:
  """Removes weight_norm and replaces ConvTranspose1d with InterleavedConvTranspose1d."""
  kmodel.eval()
  for m in kmodel.modules():
    try:
      torch.nn.utils.remove_weight_norm(m)
    except ValueError:
      pass
    try:
      torch.nn.utils.parametrize.remove_parametrizations(m, "weight")
    except ValueError:
      pass

  for module in list(kmodel.modules()):
    if isinstance(module, istftnet.AdainResBlk1d) and not isinstance(
        module.pool, (nn.Identity, istftnet.InterleavedConvTranspose1d)
    ):
      orig_pool = module.pool
      ict = istftnet.InterleavedConvTranspose1d(
          orig_pool.in_channels,
          orig_pool.out_channels,
          orig_pool.kernel_size[0],
          orig_pool.stride[0],
          padding=orig_pool.padding[0],
          groups=orig_pool.groups,
      )
      ict.conv.weight.data = orig_pool.weight.data.flip(-1)
      if ict.conv.bias is not None and orig_pool.bias is not None:
        ict.conv.bias.data = orig_pool.bias.data
      module.pool = ict

  for i in range(len(kmodel.decoder.generator.ups)):
    orig_up = kmodel.decoder.generator.ups[i]
    if not isinstance(orig_up, istftnet.InterleavedConvTranspose1d):
      ict = istftnet.InterleavedConvTranspose1d(
          orig_up.in_channels,
          orig_up.out_channels,
          orig_up.kernel_size[0],
          orig_up.stride[0],
          padding=orig_up.padding[0],
          groups=orig_up.groups,
      )
      ict.conv.weight.data = orig_up.weight.data.transpose(0, 1).flip(-1)
      if ict.conv.bias is not None and orig_up.bias is not None:
        ict.conv.bias.data = orig_up.bias.data
      kmodel.decoder.generator.ups[i] = ict

  return kmodel


class KokoroTts(tts_model.TtsModel):
  """Kokoro TTS exportable model class."""

  def __init__(
      self,
      model_path: str,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
      override_transformers: bool = False,
  ):
    super().__init__(model_path, export_config)
    del override_transformers
    use_random_weights = bool(
        export_config and export_config.use_random_weights
    )
    if os.path.exists(model_path):
      self.model_dir = model_path
      self.repo_id = "hexgrad/Kokoro-82M"
    elif use_random_weights:
      self.repo_id = model_path
      self.model_dir = ""
    else:
      self.repo_id = model_path
      self.model_dir = huggingface_hub.snapshot_download(
          repo_id=model_path,
          allow_patterns=["config.json", "*.pth", "voices/*.pt"],
      )

  def _apply_quantization(
      self, model_path: str, quant_recipe: str | None
  ) -> str:
    """Applies quantization recipe to exported TFLite flatbuffer."""
    if not quant_recipe or quant_recipe.lower() in ("none", "fp32", ""):
      return model_path
    print(f"Applying quantization recipe '{quant_recipe}' to {model_path}...")
    recipe_map = {
        "int4_weight_only": "weight_only_wi4_afp32",
        "int4": "dynamic_wi4_afp32",
        "q4": "dynamic_wi4_afp32",
        "int8_weight_only": "weight_only_wi8_afp32",
        "int8": "dynamic_wi8_afp32",
        "uint8": "dynamic_wi8_afp32",
        "q8": "dynamic_wi8_afp32",
        "fp16": "full_fp16_recipe",
    }
    mapped_recipe = recipe_map.get(quant_recipe.lower(), quant_recipe)
    q_path = model_path.removesuffix(".tflite") + "_quantized.tflite"
    qt = quantizer_lib.Quantizer(model_path)
    if mapped_recipe == "dynamic_wi4_afp32":
      recipe = recipe_lib.dynamic_wi4_afp32()
    elif mapped_recipe == "dynamic_wi8_afp32":
      recipe = recipe_lib.dynamic_wi8_afp32()
    elif mapped_recipe == "weight_only_wi4_afp32":
      recipe = recipe_lib.weight_only_wi4_afp32()
    elif mapped_recipe == "weight_only_wi8_afp32":
      recipe = recipe_lib.weight_only_wi8_afp32()
    elif hasattr(recipe_lib, mapped_recipe):
      recipe = getattr(recipe_lib, mapped_recipe)()
    else:
      return model_path
    qt.load_quantization_recipe(recipe)

    qt.quantize().export_model(q_path, overwrite=True)
    shutil.move(q_path, model_path)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(
        f"Quantized {model_path} with {mapped_recipe} ({file_size_mb:.2f} MB)"
    )
    return model_path

  def export(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> dict[str, str]:
    """Exports Kokoro acoustic, vocoder, and voice packs."""
    output_dir = export_config.output_dir or export_config.work_dir
    if not output_dir:
      raise ValueError("Either output_dir or work_dir must be specified.")
    os.makedirs(output_dir, exist_ok=True)
    artifacts = {}

    # Disable default torch.ops.aten.lstm.input decomposition to preserve Fused
    # LSTM.
    try:
      litert_torch.fx_infra.decomp.remove_pre_convert_decomp(
          torch.ops.aten.lstm.input
      )
    except Exception:  # pylint: disable=broad-exception-caught
      pass

    config_path = (
        os.path.join(self.model_dir, "config.json") if self.model_dir else None
    )
    pth_candidates = [
        "kokoro-v1_0.pth",
        "kokoro-v1_1-zh.pth",
        "kokoro.pth",
    ]
    pth_path = None
    if self.model_dir:
      for cand in pth_candidates:
        cand_path = os.path.join(self.model_dir, cand)
        if os.path.exists(cand_path):
          pth_path = cand_path
          break

    use_random_weights = bool(export_config.use_random_weights)
    print(
        f"Initializing Kokoro KModel from {self.model_dir or self.repo_id}"
        f" (config: {config_path}, pth: {pth_path},"
        f" use_random_weights: {use_random_weights})..."
    )
    kmodel = kokoro_model.KModel(
        repo_id=self.repo_id,
        config=(
            config_path if config_path and os.path.exists(config_path) else None
        ),
        model=pth_path,
        disable_complex=True,
        use_random_weights=use_random_weights,
    ).eval()
    prepare_kmodel_for_export(kmodel)

    raw_targets = _get_config_opt(export_config, "targets", ["all"])
    if isinstance(raw_targets, str):
      targets = {t.strip() for t in raw_targets.split(",") if t.strip()}
    else:
      targets = set(raw_targets)
    export_all = "all" in targets

    raw_buckets = _get_config_opt(export_config, "buckets", None)
    if not raw_buckets:
      buckets = [48, 64, 96]
    elif isinstance(raw_buckets, int):
      buckets = [raw_buckets]
    elif isinstance(raw_buckets, str):
      buckets = [int(b.strip()) for b in raw_buckets.split(",") if b.strip()]
    else:
      buckets = [int(b) for b in raw_buckets]

    default_bucket = int(_get_config_opt(export_config, "default_bucket", 64))
    if default_bucket not in buckets:
      default_bucket = buckets[0]

    export_vocoder_buckets = bool(
        _get_config_opt(export_config, "export_vocoder_buckets", True)
    )
    export_multisig = bool(
        _get_config_opt(export_config, "export_multisig", False)
    )
    quant_recipe = export_config.quantization_recipe
    export_acoustic = bool(
        _get_config_opt(
            export_config,
            "export_acoustic",
            export_all or "acoustic" in targets,
        )
    )
    export_vocoder = bool(
        _get_config_opt(
            export_config,
            "export_vocoder",
            export_all or "vocoder" in targets,
        )
    )
    export_voices = bool(export_all or "voices" in targets)
    export_espeak = bool(
        export_all or "espeak" in targets or "espeak_ng_data" in targets
    )

    # Acoustic output frames reserved per input phoneme token. This sets
    # `max_frames = frames_per_token * bucket`, which is the hard ceiling on how
    # much speech a single slice can emit. Measured English speech averages ~2.5
    # frames/token and reaches ~2.7 on long clauses, so the historical value of
    # 2 silently truncated any clause that filled its bucket. The runtime
    # derives usable token capacity as `max_frames / kFramesPerTokenBudget`, so
    # this must be >= that budget for the full bucket to be usable.
    frames_per_token = int(
        _get_config_opt(export_config, "frames_per_token", 2) or 2
    )

    # 1. Export Unified Acoustic Model (kokoro_acoustic.tflite)
    if export_acoustic:
      style_predictor = torch.zeros((1, 128), dtype=torch.float32)

      if export_multisig:
        print(
            f"Exporting multi-signature acoustic model for buckets {buckets}..."
        )
        ac_converter = None
        for b in buckets:
          max_frames = frames_per_token * b
          mod_ac = KokoroAcousticFusedModule(
              kmodel, max_frames=max_frames
          ).eval()
          sig_name = f"bucket_{b}"
          sample_kw = {
              "phoneme_ids": torch.zeros((1, b), dtype=torch.int64),
              "speaker_style": style_predictor,
              "phoneme_length": torch.tensor([b], dtype=torch.int64),
          }
          if ac_converter is None:
            ac_converter = litert_torch.signature(
                sig_name, mod_ac, sample_kwargs=sample_kw
            )
          else:
            ac_converter.signature(sig_name, mod_ac, sample_kwargs=sample_kw)

        ac_default_path = os.path.join(output_dir, "kokoro_acoustic.tflite")
        assert ac_converter is not None
        conv_ac = ac_converter.convert()
        conv_ac.export(ac_default_path)
        self._apply_quantization(ac_default_path, quant_recipe)
        artifacts["kokoro_acoustic"] = ac_default_path
      else:
        for b in buckets:
          max_frames = frames_per_token * b
          print(
              f"Exporting acoustic predictor for bucket T={b}"
              f" (max_frames={max_frames})..."
          )
          mod_ac = KokoroAcousticFusedModule(
              kmodel, max_frames=max_frames
          ).eval()
          ac_b_path = os.path.join(output_dir, f"kokoro_acoustic_{b}.tflite")
          input_ids = torch.zeros((1, b), dtype=torch.int64)
          seq_len = torch.tensor([b], dtype=torch.int64)
          conv_ac = litert_torch.convert(
              mod_ac,
              sample_kwargs={
                  "phoneme_ids": input_ids,
                  "speaker_style": style_predictor,
                  "phoneme_length": seq_len,
              },
          )
          conv_ac.export(ac_b_path)
          self._apply_quantization(ac_b_path, quant_recipe)
          artifacts[f"kokoro_acoustic_{b}"] = ac_b_path

        # Default fallback copy pointing to default_bucket
        ac_default_path = os.path.join(output_dir, "kokoro_acoustic.tflite")
        shutil.copyfile(
            os.path.join(
                output_dir, f"kokoro_acoustic_{default_bucket}.tflite"
            ),
            ac_default_path,
        )
        artifacts["kokoro_acoustic"] = ac_default_path

    # 2. Export Vocoder Models
    if export_vocoder:
      mod_voc = KokoroVocoderModule(kmodel).eval()
      if export_multisig:
        print(
            f"Exporting multi-signature vocoder model for buckets {buckets}..."
        )
        voc_converter = None
        for b in buckets:
          max_frames = frames_per_token * b
          sig_name = f"bucket_{b}"
          sample_kw = {
              "acoustic_features": torch.zeros(
                  (1, 512, max_frames), dtype=torch.float32
              ),
              "pitch_contour": torch.zeros(
                  (1, 2 * max_frames), dtype=torch.float32
              ),
              "energy_contour": torch.zeros(
                  (1, 2 * max_frames), dtype=torch.float32
              ),
              "speaker_style": torch.zeros((1, 128), dtype=torch.float32),
              "speech_frame_length": torch.tensor(
                  [max_frames], dtype=torch.int32
              ),
          }
          if voc_converter is None:
            voc_converter = litert_torch.signature(
                sig_name, mod_voc, sample_kwargs=sample_kw
            )
          else:
            voc_converter.signature(sig_name, mod_voc, sample_kwargs=sample_kw)

        voc_default_path = os.path.join(output_dir, "kokoro_vocoder.tflite")
        assert voc_converter is not None
        conv_voc = voc_converter.convert()
        conv_voc.export(voc_default_path)
        artifacts["kokoro_vocoder"] = voc_default_path
      elif export_vocoder_buckets:
        for b in buckets:
          max_frames = frames_per_token * b
          voc_b_path = os.path.join(output_dir, f"kokoro_vocoder_{b}.tflite")
          print(
              f"Exporting vocoder flatbuffer for bucket T={b}"
              f" (max_frames={max_frames}) to {voc_b_path}..."
          )
          conv_voc = litert_torch.convert(
              mod_voc,
              sample_kwargs={
                  "acoustic_features": torch.zeros(
                      (1, 512, max_frames), dtype=torch.float32
                  ),
                  "pitch_contour": torch.zeros(
                      (1, 2 * max_frames), dtype=torch.float32
                  ),
                  "energy_contour": torch.zeros(
                      (1, 2 * max_frames), dtype=torch.float32
                  ),
                  "speaker_style": torch.zeros((1, 128), dtype=torch.float32),
                  "speech_frame_length": torch.tensor(
                      [max_frames], dtype=torch.int32
                  ),
              },
          )
          conv_voc.export(voc_b_path)
          artifacts[f"kokoro_vocoder_{b}"] = voc_b_path

        voc_default_path = os.path.join(output_dir, "kokoro_vocoder.tflite")
        shutil.copyfile(
            os.path.join(output_dir, f"kokoro_vocoder_{default_bucket}.tflite"),
            voc_default_path,
        )
        artifacts["kokoro_vocoder"] = voc_default_path
      else:
        def_frames = frames_per_token * default_bucket
        voc_path = os.path.join(output_dir, "kokoro_vocoder.tflite")
        print(
            f"Exporting default vocoder flatbuffer (max_frames={def_frames}) to"
            f" {voc_path}..."
        )
        conv_voc = litert_torch.convert(
            mod_voc,
            sample_kwargs={
                "acoustic_features": torch.zeros(
                    (1, 512, def_frames), dtype=torch.float32
                ),
                "pitch_contour": torch.zeros(
                    (1, 2 * def_frames), dtype=torch.float32
                ),
                "energy_contour": torch.zeros(
                    (1, 2 * def_frames), dtype=torch.float32
                ),
                "speaker_style": torch.zeros((1, 128), dtype=torch.float32),
                "speech_frame_length": torch.tensor(
                    [def_frames], dtype=torch.int32
                ),
            },
        )
        conv_voc.export(voc_path)
        # Vocoder is preserved in FP32 for audio fidelity and full GPU OpenCL
        # delegation.
        artifacts["kokoro_vocoder"] = voc_path

    # 3. Export Voice Packs (.pt to binary .bin float32 format)
    if export_voices:
      voices_out_dir = os.path.join(output_dir, "voices")
      os.makedirs(voices_out_dir, exist_ok=True)
      voices_src_dir = (
          os.path.join(self.model_dir, "voices") if self.model_dir else None
      )
      voice_files = []
      if voices_src_dir and os.path.exists(voices_src_dir):
        voice_files = [
            os.path.join(voices_src_dir, f)
            for f in os.listdir(voices_src_dir)
            if f.endswith(".pt")
        ]
      if (
          not any(f.endswith("af_heart.pt") for f in voice_files)
          and not use_random_weights
      ):
        try:
          af_heart_pt = huggingface_hub.hf_hub_download(
              repo_id="hexgrad/Kokoro-82M", filename="voices/af_heart.pt"
          )
          voice_files.append(af_heart_pt)
        except Exception as e:  # pylint: disable=broad-exception-caught
          print(f"Warning: could not download default voice af_heart.pt: {e}")

      for vf in voice_files:
        v_name = os.path.splitext(os.path.basename(vf))[0]
        v_tensor = torch.load(vf, map_location="cpu", weights_only=True)
        if isinstance(v_tensor, torch.Tensor):
          bin_path = os.path.join(voices_out_dir, f"{v_name}.bin")
          v_tensor.detach().numpy().astype(np.float32).tofile(bin_path)
          artifacts[f"voice_{v_name}"] = bin_path

      if not voice_files and use_random_weights:
        bin_path = os.path.join(voices_out_dir, "af_heart.bin")
        np.zeros((510, 1, 256), dtype=np.float32).tofile(bin_path)
        artifacts["voice_af_heart"] = bin_path

    # 4. Export espeak-ng-data (from source model directory or Hugging Face Hub)
    if export_espeak:
      espeak_out_dir = os.path.join(output_dir, "espeak-ng-data")
      if not os.path.exists(espeak_out_dir):
        src_espeak = (
            os.path.join(self.model_dir, "espeak-ng-data")
            if self.model_dir
            else None
        )
        if src_espeak and os.path.isdir(src_espeak):
          print(
              f"Copying espeak-ng-data from source model directory {src_espeak}"
              f" to {espeak_out_dir}..."
          )
          shutil.copytree(src_espeak, espeak_out_dir, dirs_exist_ok=True)
          artifacts["espeak-ng-data"] = espeak_out_dir
        elif not use_random_weights:
          print(
              "espeak-ng-data not found in source model directory; fetching"
              " from Hugging Face Hub..."
          )
          try:
            dl_path = huggingface_hub.snapshot_download(
                repo_id="csukuangfj/kokoro-multi-lang-v1_0",
                allow_patterns=["espeak-ng-data/*"],
            )
            hf_src_espeak = os.path.join(dl_path, "espeak-ng-data")
            if os.path.isdir(hf_src_espeak):
              shutil.copytree(hf_src_espeak, espeak_out_dir, dirs_exist_ok=True)
              artifacts["espeak-ng-data"] = espeak_out_dir
              print(
                  "Successfully fetched and saved espeak-ng-data to"
                  f" {espeak_out_dir}"
              )
          except Exception as e:  # pylint: disable=broad-exception-caught
            print(
                "Warning: could not fetch espeak-ng-data from Hugging"
                f" Face: {e}"
            )
      else:
        artifacts["espeak-ng-data"] = espeak_out_dir

    return artifacts
