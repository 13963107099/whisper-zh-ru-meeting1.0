# Changelog

## Unreleased — privacy and provenance hardening

- Added an optional local domain glossary (`--glossary-file`) and documented `--prompt`; defaults and decoding parameters are unchanged when neither is supplied.
- Disabled supported pyannote and Hugging Face telemetry at runtime and made model hubs offline by default; first-time model downloads now require `WHISPER_MEETING_ALLOW_MODEL_DOWNLOADS=1`.
- Added repository-wide ignore rules and expanded publication checks for meeting media, business documents, model weights, archives, credentials, personal absolute paths, and symlinks.
- Added a root license, security/contribution entry points, and `PROVENANCE.md` to distinguish original project work from upstream models, tools, algorithms, and third-party contributions.
- Pinned GitHub Actions to verified release commit SHAs and kept workflow credentials disabled.

## v0.1.0-alpha — public-source preview (engine V3.3.5)

- Published cache-aware sequential S-label rendering.
- Included bilingual assistance and independent overlap detection/separation entry points.
- Added privacy-oriented `.gitignore`, public README, license, CI and release guide.
- Automated offline tests are available; live quality and performance benchmarks are **not** included.

Known gaps: setup remains manual and depends on separately installed whisper.cpp, VAD/model weights and optional ML environments. S labels are segment IDs, not stable person identities.

- Added explicit upstream model names, publishers, official URLs, access rules and license references in `THIRD_PARTY_NOTICES.md`, `third_party_models.json` and `docs/LICENSE_COMPLIANCE_CHECKLIST_ZH.md`; updated README and offline checks. No model weights added.
