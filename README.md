# Whisper ZH/RU Meeting Assistant

**实验性本地会议转写流水线（V3.3.5）**：中文与俄文会议转写缓存整理、去磕绊、可追溯无关文本过滤、双向翻译辅助、重叠语音检测以及两路语音分离转写。

**English:** An experimental local Chinese/Russian meeting transcription pipeline with transcript cleanup, translation aids, overlap detection and two-stream speech separation. It is a **cache-oriented enhancement tool**; it is not a turnkey ASR installer or a verified production transcription system.

> Status: `v0.1.0-alpha` public source preview. Internal engine version: `V3.3.5`. Automated offline tests cover workflow and cache behavior; they do **not** establish real-world diarization, translation, or overlap-separation accuracy.

## What it does / 功能

| Function | Status and important limitations |
|---|---|
| Chinese/Russian first-pass ASR | Uses existing whisper.cpp + Silero VAD binaries and model files; **not bundled** |
| Ordered `S01`, `S02`... transcript labels | `S` is a sequential **segment ID**, not a unique person ID |
| Cleanup / hallucination filters | Conservative and auditable; may need manual review |
| Chinese ↔ Russian translation aids | Separate M2M100 model; derived translation is **not** an original utterance |
| Overlap detection | `pyannote/segmentation-3.0`; **no full speaker voiceprint clustering** |
| Separation + retranscription | SepFormer **two** candidate tracks; three or more overlapping people may not be recovered |
| Caching/resume | Reuses previously completed stages; individual failed chunks may rerun |

Models are downloaded separately; no user audio, transcript, Hugging Face token or model weights belong in this repository. Runtime telemetry is disabled and model hubs are offline by default. For an intentional first download only, set `WHISPER_MEETING_ALLOW_MODEL_DOWNLOADS=1`, download/cache the selected model, and then unset it. Model downloads never use meeting audio or transcripts as request inputs.


## Third-party model credits / 第三方模型来源

**This project integrates third-party pretrained models; none of their weights are included or claimed as original work.** The following are upstream works, not models trained by this repository:

| Component | Official publisher / source | License / access |
|---|---|---|
| ASR | [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3-turbo), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [GGML Q5 model](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-large-v3-turbo-q5_0.bin) | MIT (upstream) |
| Speech activity detection | [Silero Team — Silero VAD](https://github.com/snakers4/silero-vad) | MIT |
| Chinese–Russian translation | [Meta AI — M2M100](https://huggingface.co/facebook/m2m100_418M) | MIT |
| Overlap detection | [pyannote — segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) | MIT; gated access |
| Two-track speech separation | [SpeechBrain — SepFormer WHAMR 16k](https://huggingface.co/speechbrain/sepformer-whamr16k) | Apache-2.0 |
| Optional denoising | [Hendrik Schröter et al. — DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | MIT OR Apache-2.0 |

**Full model inventory (including retained legacy/conditional models and third-party libraries):** [THIRD_PARTY_NOTICES.md](whisper-zh-ru-meeting/THIRD_PARTY_NOTICES.md) · [Machine-readable inventory](whisper-zh-ru-meeting/third_party_models.json) · [中文许可核查清单](whisper-zh-ru-meeting/docs/LICENSE_COMPLIANCE_CHECKLIST_ZH.md). The project MIT `LICENSE` does **not** override any upstream license. Verify code ownership, attribution, and applicable redistribution terms before changing a private repository to public.

## Requirements

- macOS / Linux with Python 3.11 for heavy ML dependencies; Git, FFmpeg and FFprobe.
- Existing whisper.cpp `whisper-server` and Silero VAD binaries, and the GGML Whisper + VAD weights. Set paths via the first-pass CLI flags (see below).
- Optional dedicated virtual environments for pyannote segmentation and for the translation/separation worker. The latter can be installed with `bash setup_meeting_enhancement.sh`; installation can take time.
- Hugging Face read token (`HF_TOKEN`) with accepted terms for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) for overlap detection. Never hard-code or commit it.

Run the commands below from `whisper-zh-ru-meeting/` unless a command says otherwise.

### Quick local tests (no model download)

```bash
python3 -m compileall -q .
python3 -m unittest -v test_v335
```

### Route A: you already have V3.3 first-pass ASR caches (recommended for this alpha)

```bash
# Existing output directory must contain render_meta.json,
# chunk_plan.json and asr_cache/chunk_*.json with the original WAV present.
python3 check_v335.py --output-dir "$HOME/whisper-ru/output/meeting_v33_two_phase"

python3 meeting_v335.py render --output-dir "$HOME/whisper-ru/output/meeting_v33_two_phase"
python3 meeting_v335.py status --output-dir "$HOME/whisper-ru/output/meeting_v33_two_phase"

# Optional translate: requires enhancement environment
python3 meeting_v335.py translate --output-dir "$HOME/whisper-ru/output/meeting_v33_two_phase" --threads 2 --max-translations 20

# Optional overlap scan: requires pyannote segmentation environment and HF access
python3 meeting_v335.py detect --output-dir "$HOME/whisper-ru/output/meeting_v33_two_phase" --threads 2

# Optional candidate two-track separation; try a short budget first
python3 meeting_v335.py separate --output-dir "$HOME/whisper-ru/output/meeting_v33_two_phase" --threads 2 --whisper-threads 2 --max-separation-seconds 60
```

### Route B: begin with a new MP3 (advanced; prerequisites not bundled)

`whisper_zh_ru_v33.py` includes the V3.3 first-pass stage. Supply the paths for **your own** whisper.cpp server, Silero VAD binaries, and model weights, and disable full speaker diarization to create the cache expected by V3.3.5:

```bash
python3 whisper_zh_ru_v33.py   "/PATH/TO/your-meeting.mp3"   --phase fast --diarization off   --model "/PATH/TO/ggml-large-v3-turbo-q5_0.bin"   --server-binary "/PATH/TO/whisper-server"   --vad-binary "/PATH/TO/whisper-vad-speech-segments"   --vad-model "/PATH/TO/ggml-silero-v6.2.0.bin"   --output-dir "$HOME/whisper-ru/output/example-run"

python3 meeting_v335.py render --output-dir "$HOME/whisper-ru/output/example-run"
```

This new-audio route is supplied for advanced users; the released tests do not run the full live Whisper server and model pipeline. Validate it locally before calling it a one-command setup.

## Key files

- `meeting_v335.py`: cache-aware stage orchestrator (`render`, `translate`, `detect`, `separate`, `all`, `status`).
- `whisper_zh_ru_v33.py`, `whisper_zh_ru_v3.py`: first-pass Whisper and cache handling.
- `overlap_only_backend.py`: overlap detection only; does not identify people.
- `meeting_model_worker.py`: optional translation and separation worker.
- `privacy_defaults.py`: disables supported telemetry and makes model hubs offline unless a download is explicitly enabled.
- `disfluency_cleanup.py`, `transcript_guard.py`: text cleanup with an auditable original.
- `test_v335.py`: offline unit tests.
- `docs/GITHUB_OPEN_SOURCE_GUIDE_ZH.md`: step-by-step publishing tutorial (Chinese).

## Safety, privacy and reproducibility

- Only use meeting recordings that you are permitted to process. Never publish private audio or transcripts.
- Do not confuse an adjacent translation with a verified interpretation of the original speaker.
- Overlap detection finds candidates; separation quality depends on conditions and cannot recover every concurrent voice.
- The `S` labels are **not** speaker identities. Audio track 1/2 is local to a particular overlap segment.
- The repository **does not redistribute** Whisper, pyannote, M2M100 or SepFormer pretrained weights. Follow each upstream model license and access rules.

See [SECURITY.md](SECURITY.md), [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](whisper-zh-ru-meeting/CHANGELOG.md) and the [中文开源发布教程](whisper-zh-ru-meeting/docs/GITHUB_OPEN_SOURCE_GUIDE_ZH.md).

Original project code and upstream boundaries are recorded in [PROVENANCE.md](PROVENANCE.md). Automated checks reduce accidental disclosure and attribution mistakes, but cannot certify authorship or non-infringement.

## License

The original code in this repository is released under the [MIT License](LICENSE), subject to verification of ownership/third-party contributions by the repository publisher. Dependencies and model weights retain their own licenses. Review [THIRD_PARTY_NOTICES.md](whisper-zh-ru-meeting/THIRD_PARTY_NOTICES.md) before publishing or redistributing.
