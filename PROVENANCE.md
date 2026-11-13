# Source provenance and original-work boundary

This record protects both upstream authors and the repository owner's original
work. It is an inventory, not proof of authorship and not legal advice.

## Project-specific work

The repository owner designates the project-specific Python and shell
orchestration, cache/resume flow, conservative transcript cleanup, audit output,
tests, and documentation in `whisper-zh-ru-meeting/` as original project work
licensed under the repository MIT License, but only to the extent that the owner
and contributors hold the rights to license those contributions. Copyright and
license notices must not be removed.

Commit history is the primary dated record of changes. Substantial contributions
must identify their author, source, and license. AI-assisted material still
requires human review for copied expressions, incompatible snippets, secrets,
and false authorship claims; AI assistance alone does not prove originality.

## Work not claimed as original

The project does not claim authorship or ownership of Whisper, whisper.cpp,
Silero VAD, M2M100, pyannote, SepFormer, SpeechBrain, ECAPA-TDNN,
DeepFilterNet, FFmpeg, their pretrained weights, training data, papers, or
upstream APIs and algorithms. They remain governed by their own licenses and
access terms. See `whisper-zh-ru-meeting/THIRD_PARTY_NOTICES.md` and
`whisper-zh-ru-meeting/third_party_models.json`.

No third-party source code or model weights are intentionally vendored in this
repository. If that changes, record the exact upstream URL and revision, author
or publisher, license, local modifications, and required NOTICE text before the
change is merged.

## Contribution declaration

By contributing, a contributor certifies that the contribution is their own
work or that they have the right to submit it under the project license; that
known third-party portions are clearly identified; and that the contribution
contains no private meeting content, credentials, personal data, or employer
confidential information. Use a Git sign-off (`git commit -s`) to record this
Developer Certificate of Origin-style declaration.

## Release gate

Keep the repository private until the publication check, tests, attribution
inventory, staged diff review, and source-ownership review all pass. These
automated checks reduce risk but cannot certify ownership or non-infringement.
