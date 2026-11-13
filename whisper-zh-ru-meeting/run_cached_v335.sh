#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HOME/whisper-ru/output/meeting_v33_two_phase}"
exec python3 "$DIR/meeting_v335.py" render --output-dir "$OUT"
