# Third-party software & pretrained-model notices

**Scope:** `Whisper ZH/RU Meeting Assistant` source preview v0.1.0-alpha (engine V3.3.5). Updated 2026-11-13. This file is an attribution and license inventory; **it is not a warranty that every upstream or downstream legal obligation has been satisfied**.

The project *orchestrates* external software and pretrained models. It does **not** claim to have invented or trained Whisper, Silero VAD, M2M100, pyannote, SepFormer, ECAPA-TDNN or DeepFilterNet. The released archive contains **no model weights, training datasets, user audio or user transcripts**. The top-level [MIT LICENSE](LICENSE) is intended only for code that the repository owner has the right to license; it **does not relicense third-party components**.

## Upstream models and their contributions

| Model (official source) | Upstream publisher / contributors | Role in this project | Upstream license | Usage status |
|---|---|---|---|---|
| [Whisper Large-v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | OpenAI | First-pass Chinese/Russian ASR; upstream model architecture and weights | MIT | Used externally; pretrained weights not included |
| [GGML Whisper Large-v3 Turbo Q5_0](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-large-v3-turbo-q5_0.bin) | ggerganov / whisper.cpp contributors; original Whisper by OpenAI | Quantized format of Whisper model used by local whisper.cpp runtime | MIT (as labeled by upstream model repository) | External local file; NOT bundled; verify provenance if replacing model |
| [Silero VAD / ggml-silero-v6.2.0](https://github.com/snakers4/silero-vad) | Silero Team; ggml conversion via whisper.cpp contributors | Voice activity detection (speech boundaries) | MIT | External VAD weights and whisper.cpp binary; NOT bundled |
| [M2M100 418M](https://huggingface.co/facebook/m2m100_418M) | Meta AI (facebook model publisher); M2M100 authors | Chinese ↔ Russian translation suggestions | MIT | Optional downloaded model; NOT bundled |
| [pyannote Segmentation 3.0](https://huggingface.co/pyannote/segmentation-3.0) | pyannote / Séverin Baroudi and contributors (see model card) | Overlapped speech detection, NOT verified speaker identity | MIT; gated access terms on Hugging Face | Optional gated download; requires accepted terms and read token; NOT bundled |
| [SepFormer WHAMR! 16k](https://huggingface.co/speechbrain/sepformer-whamr16k) | SpeechBrain contributors; pretrained speech separation model | Estimate two speech tracks for candidate overlap intervals | Apache-2.0 | Optional downloaded model; NOT bundled |
| [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) | Hendrik Schröter and DeepFilterNet contributors | Optional neural speech enhancement | MIT OR Apache-2.0 (upstream dual license; confirm specific artifacts before redistribution) | Optional separately installed model; NOT bundled |
| [ECAPA-TDNN VoxCeleb Speaker Embeddings](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | SpeechBrain contributors | Conditional/legacy reference-based speaker track matching; NOT the default V3.3.5 path | Apache-2.0 | Only loaded when optional speaker references are explicitly provided; NOT bundled |
| [pyannote Speaker Diarization 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | pyannote contributors | Legacy full diarization pathway retained in older first-pass implementation; disabled in V3.3.5 recommended path | MIT; gated access terms on Hugging Face | Legacy/optional; NOT default, NOT bundled |

**Verification notes:** License labels are taken from the linked upstream model cards or official license files. Model access rules (notably pyannote gates) are separate from the stated MIT license. The project may contain a legacy model-loading branch even when that model is not part of the documented default workflow. The GGML quantized file has a separate conversion/distribution provenance from the original OpenAI model. For a different replacement checkpoint, re-check that checkpoint's individual license.

## External runtime libraries / tools (not bundled)

| Dependency | Publisher / official source | License / important note |
|---|---|---|
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | ggml-org / contributors | [MIT](https://github.com/ggml-org/whisper.cpp/blob/master/LICENSE); executable built/installed separately |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | pyannote / contributors (CNRS) | [MIT](https://github.com/pyannote/pyannote-audio/blob/main/LICENSE) |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | SpeechBrain contributors | [Apache-2.0](https://github.com/speechbrain/speechbrain/blob/develop/LICENSE) |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Hugging Face contributors | [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| [PyTorch](https://github.com/pytorch/pytorch) | PyTorch contributors | [BSD-style](https://github.com/pytorch/pytorch/blob/main/LICENSE), with additional bundled notices |
| [FFmpeg / FFprobe](https://ffmpeg.org/) | FFmpeg project | [LGPL-2.1-or-later typically; GPL applies to some builds](https://ffmpeg.org/legal.html). **Do not describe every FFmpeg binary as MIT or assume an unknown build is LGPL-only.** |
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | Hugging Face contributors | [Upstream LICENSE](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Silero Team | [MIT](https://github.com/snakers4/silero-vad/blob/master/LICENSE) |
| [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | Hendrik Schröter / contributors | [MIT OR Apache-2.0](https://github.com/Rikorose/DeepFilterNet/blob/main/LICENSE) |

Other transitive Python dependencies (numpy, scipy, soundfile, sentencepiece, tokenizers, lightning, etc.) are installed separately and keep their respective licenses. Their exact license inventory depends on the selected versions and platform. If the repository ever begins shipping a binary package, wheel bundle, installer, vendored code or container image, generate a **versioned dependency SBOM and complete license/NOTICE bundle** for everything actually redistributed.

## Obligations for redistribution

1. **If distributing only orchestration code:** name upstream components, link to original model cards and licenses, and require users to obtain the models separately. Do not pretend their authors endorsed this project.
2. **If bundling MIT-licensed source or weights:** retain original copyright and license notices in copies or substantial portions as applicable.
3. **If bundling Apache-2.0 components:** carry the full Apache-2.0 license, maintain copyright, attribution and applicable NOTICE files, and label relevant modifications.
4. **If bundling FFmpeg or linking to its libraries:** review the actual FFmpeg build configuration and LGPL/GPL obligations; a simple source-only Python wrapper does not by itself distribute FFmpeg.
5. **For gated models:** a read token is not a redistribution license and should never be committed. Each user must accept upstream access terms where required.
6. **For recordings and transcripts:** permissions to use a speech model do not give rights to disclose meeting content, participants' voices, or commercially sensitive data.
7. **For repository source:** confirm authorship and rights for code written by or derived from others. Do not claim all files were written entirely by you unless substantiated.

See the [Chinese license-compliance checklist](docs/LICENSE_COMPLIANCE_CHECKLIST_ZH.md). This is not legal advice. Do not mark the repository public until the release checklist and source-ownership review have been completed.
