#!/usr/bin/env python3
"""Static attribution inventory check; does not certify license compliance."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent
records=json.loads((root/'third_party_models.json').read_text(encoding='utf-8'))['models']
notices=(root/'THIRD_PARTY_NOTICES.md').read_text(encoding='utf-8')
readme=(root/'README.md').read_text(encoding='utf-8')
expected={
    'speechbrain/sepformer-whamr16k': 'meeting_model_worker.py',
    'speechbrain/spkrec-ecapa-voxceleb': 'meeting_model_worker.py',
    'pyannote/segmentation-3.0':'overlap_only_backend.py',
    'facebook/m2m100_418M': 'meeting_model_worker.py',
    'ggml-large-v3-turbo-q5_0.bin': 'whisper_zh_ru_v3.py',
    'ggml-silero-v6.2.0.bin': 'whisper_zh_ru_v3.py',
}
issues=[]
for entry in records:
    for field in ('id','name','owner','purpose','license','source','status','code'):
        if not str(entry.get(field,'')).strip(): issues.append(f'Missing {field}: {entry.get("id")}')
    if entry.get('source','') not in notices: issues.append(f'Model URL absent in NOTICE: {entry.get("id")}')
for model,source in expected.items():
    if model not in (root/source).read_text(encoding='utf-8'):
        issues.append(f'Expected model identifier not found in code: {model}')
if 'THIRD_PARTY_NOTICES.md' not in readme: issues.append('README missing third-party link')
if len(records)<9: issues.append('Fewer than 9 upstream model entries')
for issue in issues: print('FAIL:',issue)
if issues: raise SystemExit(1)
print('OK: 9 upstream model entries listed and linked; known model IDs found in code.')
print('NOTE: This is a static provenance test, not legal or license compliance certification.')
