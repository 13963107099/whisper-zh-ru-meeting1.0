# Contributing

Thanks for considering improvements to the project. This is an alpha research prototype with deliberately limited scope. Before making a pull request:

1. Open an issue describing the problem with **synthetic data**, not a private recording.
2. Create a separate branch, e.g. `fix/overlap-cache`.
3. Add or update offline tests (`python3 -m unittest -v test_v335`).
4. Keep the original transcript and machine-derived translated/separated candidates distinct.
5. Never change S-segment IDs into unverified real speaker identities.
6. Explain performance changes with tested hardware, input duration and measurements; do not claim unverified accuracy.

Please do not commit model weights, audio, ASR caches, credentials, or personal information. See `SECURITY.md`.

7. If you add or change any upstream model, checkpoint, audio library, binary or vendored code, update `THIRD_PARTY_NOTICES.md`, `third_party_models.json` and `docs/LICENSE_COMPLIANCE_CHECKLIST_ZH.md` with the publisher, official URL, exact license, model ID and applicable access restrictions. Never bundle weights without a separate license audit.
8. Do not paste code, prompts, documents, or data from an employer, client, private repository, article, or other source unless you have documented permission and a compatible license.
9. Record substantial third-party material with its exact source revision and preserve required copyright, license, and NOTICE text. See `../PROVENANCE.md`.
10. Sign commits with `git commit -s`. The sign-off certifies that you wrote the contribution or have the right to submit it under this project's license, and that it contains no confidential or personal data.

AI assistance does not remove the contributor's responsibility to review for copied expression, incompatible licenses, secrets, and false originality claims. Automated checks are risk controls, not proof of ownership or legal clearance.
