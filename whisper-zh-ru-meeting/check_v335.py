#!/usr/bin/env python3
"""Non-destructive version/dependency check; no model downloads or inference."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import meeting_v335 as v

p=argparse.ArgumentParser()
p.add_argument('--output-dir',default=v.DEFAULT_FOLDER)
a=p.parse_args()
folder=Path(a.output_dir).expanduser().absolute()
try:
 meta,docs=v.cache_info(folder)
 print(f'Whisper 缓存：完整，{len(docs)} 个窗口；音频 {meta["duration"]:.1f} 秒。')
except Exception as e:
 print('Whisper 缓存：',e)
for cmd in ('ffmpeg','ffprobe'):
 print(f'{cmd}: {shutil.which(cmd) or "未找到"}')
for label,path,packages in [
 ('重叠检测',Path.home()/'whisper-ru/diarization-venv/bin/python',['torch','pyannote.audio']),
 ('翻译及分离',Path.home()/'whisper-ru/enhancement-venv/bin/python',['torch','speechbrain','transformers','sentencepiece'])]:
 if not path.is_file():
  print(f'{label} 环境：尚未安装：{path}')
  continue
 script='import importlib.util;mods='+repr(packages)+';print(", ".join(m+(":有" if importlib.util.find_spec(m) else ":缺") for m in mods))'
 # find_spec('pyannote.audio') may import pyannote parent, but not neural networks.
 result=subprocess.run([str(path),'-c',script],capture_output=True,text=True,timeout=30)
 print(f'{label} 环境：'+(result.stdout.strip() or result.stderr.strip()))
print('检查结束：未启动Whisper、pyannote推理、SepFormer或翻译模型。')
