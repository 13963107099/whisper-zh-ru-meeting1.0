#!/usr/bin/env python3
"""Read-only publication preflight on source tree. Not a complete secret scanner."""
from pathlib import Path
import re
PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT.parent if (PROJECT_ROOT.parent/'.github').is_dir() else PROJECT_ROOT
IGNORED = {'.git', '__pycache__', '.venv', 'venv'}
BAD_SUFFIX = {
  '.mp3','.m4a','.wav','.flac','.aac','.ogg','.opus','.aiff','.wma','.mp4','.mov','.mkv','.webm','.avi',
  '.srt','.vtt','.pdf','.docx','.rtf','.odt','.xlsx','.xls','.pptx','.pages','.keynote','.csv','.tsv','.jsonl',
  '.eml','.msg','.sqlite','.db','.sql','.zip','.tar','.gz','.tgz','.7z','.rar',
  '.pt','.pth','.bin','.ckpt','.safetensors','.gguf','.ggml','.onnx','.npy','.npz',
  '.jpg','.jpeg','.png','.heic',
}
BLOCK_BASENAMES={
  'all_clean.txt','render_meta.json','chunk_plan.json','audio_manifest.json','asr_manifest.json',
  'speaker_manifest.json','speaker_raw.json','speaker_diarization.json','speaker_names.json',
  'manual_overrides.json','job.json',
}
PRIVATE_DIRS={
  'output','outputs','results','transcripts','cache','asr_cache','separation','review_clips',
  'logs','models','checkpoints','data','private','secrets',
}
SECRET_PATTERNS = [
  re.compile(rb'hf_[A-Za-z0-9]{16,}'),
  re.compile(rb'github_pat_[A-Za-z0-9_]{20,}'),
  re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}'),
  re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{20,}'),
  re.compile(rb'AKIA[0-9A-Z]{16}'),
  re.compile(rb'AIza[0-9A-Za-z_-]{30,}'),
  re.compile(rb'xox[baprs]-[A-Za-z0-9-]{10,}'),
  re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
  re.compile(rb'(?i)\b(?:api[_-]?key|hf_token|access[_-]?token)\s*[:=]\s*["\'][A-Za-z0-9_\-]{18,}["\']'),
]
PRIVATE_PATH_PATTERNS = [
  re.compile(rb'/Users/[A-Za-z0-9._-]+/'),
  re.compile(rb'/home/[A-Za-z0-9._-]+/'),
  re.compile(rb'[A-Za-z]:\\\\Users\\\\[^\\\\\r\n]+'),
]
issues=[]
count=0
for p in ROOT.rglob('*'):
    if p.is_symlink():
        issues.append(f'Symlink requires manual review: {p.relative_to(ROOT)}')
        continue
    if not p.is_file() or any(part in IGNORED for part in p.relative_to(ROOT).parts):continue
    count+=1
    relative=p.relative_to(ROOT)
    rel=str(relative)
    if (p.suffix.lower() in BAD_SUFFIX or p.name in BLOCK_BASENAMES
            or any(part.lower() in PRIVATE_DIRS for part in relative.parts)
            or (p.name.startswith('v335_') and p.suffix.lower()=='.json')
            or p.name.endswith(('request.json','response.json'))):
        issues.append(f'Forbidden file: {rel}')
    if p.stat().st_size>10*1024*1024:
        issues.append(f'File > 10 MiB: {rel}')
    if p.suffix.lower() in {'.py','.sh','.md','.txt','.yml','.yaml','.example','.json'}:
        content=p.read_bytes()
        if any(pat.search(content) for pat in SECRET_PATTERNS):
            issues.append(f'Possible embedded credential in: {rel}')
        if any(pat.search(content) for pat in PRIVATE_PATH_PATTERNS):
            issues.append(f'Possible personal absolute path in: {rel}')
print(f'Scanned {count} files. Security preflight is heuristic; inspect all staged files manually.')
for item in issues:print('BLOCK:',item)
raise SystemExit(1 if issues else 0)
