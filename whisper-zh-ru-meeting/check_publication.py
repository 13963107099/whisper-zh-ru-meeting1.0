#!/usr/bin/env python3
"""Read-only publication preflight on source tree. Not a complete secret scanner."""
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parent
IGNORED = {'.git', '__pycache__', '.venv', 'venv'}
BAD_SUFFIX = {'.mp3','.m4a','.wav','.flac','.mp4','.mov','.pdf','.docx','.zip','.pt','.pth','.bin','.ckpt','.safetensors'}
BLOCK_BASENAMES={'all_clean.txt','render_meta.json','speaker_raw.json','speaker_diarization.json','manual_overrides.json'}
SECRET_PATTERNS = [
  re.compile(rb'hf_[A-Za-z0-9]{16,}'),
  re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
  re.compile(rb'(?i)\b(?:api[_-]?key|hf_token|access[_-]?token)\s*[:=]\s*["\'][A-Za-z0-9_\-]{18,}["\']'),
]
issues=[]
count=0
for p in ROOT.rglob('*'):
    if not p.is_file() or any(part in IGNORED for part in p.relative_to(ROOT).parts):continue
    count+=1
    rel=str(p.relative_to(ROOT))
    if p.suffix.lower() in BAD_SUFFIX or p.name in BLOCK_BASENAMES:
        issues.append(f'Forbidden file: {rel}')
    if p.stat().st_size>10*1024*1024:
        issues.append(f'File > 10 MiB: {rel}')
    if p.suffix.lower() in {'.py','.sh','.md','.txt','.yml','.yaml','.example'}:
        content=p.read_bytes()
        if any(pat.search(content) for pat in SECRET_PATTERNS):
            issues.append(f'Possible embedded credential in: {rel}')
print(f'Scanned {count} files. Security preflight is heuristic; inspect all staged files manually.')
for item in issues:print('BLOCK:',item)
raise SystemExit(1 if issues else 0)
