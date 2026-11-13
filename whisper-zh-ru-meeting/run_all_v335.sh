#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HOME/whisper-ru/output/meeting_v33_two_phase}"
exec caffeinate -i python3 "$DIR/meeting_v335.py" all --output-dir "$OUT" --threads 2 --whisper-threads 2 --max-translations 20 --max-separation-seconds 0
