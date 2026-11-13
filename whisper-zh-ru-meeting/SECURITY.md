# Security & privacy

**Never include user audio, meeting transcripts, real negotiation data or credentials in a public issue or commit.**

All meeting audio must be processed only where the recorder has permission. `HF_TOKEN` grants access to model repositories; pass it in the shell environment, never as a command-line argument or in code. Do not share terminal outputs containing actual tokens.

The `.gitignore` excludes common artifacts, but it is not a substitute for manual review. Before pushing run `git status --short`, `git diff --cached --name-only` and inspect any newly staged files. For accidental credential disclosure, immediately revoke or rotate the token; see https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository .

Security reports: please open a GitHub issue **without secrets or private recordings**, or contact the repository maintainer through a private channel when available. The initial alpha does not currently offer a published private-reporting mailbox.
