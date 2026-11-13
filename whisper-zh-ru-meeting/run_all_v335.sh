#!/usr/bin/env bash
set -euo pipefail
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYANNOTE_METRICS_ENABLED=0
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HOME/whisper-ru/output/meeting_v33_two_phase}"
exec caffeinate -i python3 "$DIR/meeting_v335.py" all --output-dir "$OUT" --threads 2 --whisper-threads 2 --max-translations 20 --max-separation-seconds 0
