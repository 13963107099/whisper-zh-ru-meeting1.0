# Security & privacy

**Never include user audio, meeting transcripts, real negotiation data or credentials in a public issue or commit.**

All meeting audio must be processed only where the recorder has permission. `HF_TOKEN` grants access to model repositories; pass it in the shell environment, never as a command-line argument or in code. Do not share terminal outputs containing actual tokens.

Runtime telemetry supported by Hugging Face Hub and pyannote is forcibly disabled by `privacy_defaults.py`. Model hubs are offline by default. Set `WHISPER_MEETING_ALLOW_MODEL_DOWNLOADS=1` only for an intentional first-time model download, then unset it. The download request may disclose the selected model identifier and account/token metadata to the model host, but meeting audio and transcripts are never download inputs. For the strongest boundary, download models separately, verify them, and run normal processing with outbound network access blocked at the operating-system or container level.

The `.gitignore` excludes common artifacts, but it is not a substitute for manual review. Before pushing run `git status --short`, `git diff --cached --name-only` and inspect any newly staged files. For accidental credential disclosure, immediately revoke or rotate the token; see https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository .

Security reports: please open a GitHub issue **without secrets or private recordings**, or contact the repository maintainer through a private channel when available. The initial alpha does not currently offer a published private-reporting mailbox.
