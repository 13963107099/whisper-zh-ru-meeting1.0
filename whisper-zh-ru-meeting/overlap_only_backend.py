#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlap-only detection: no diarization, no speaker embeddings, no identity matching.

Uses pyannote/segmentation-3.0 -> OverlappedSpeechDetection, processing bounded WAV
windows with atomic checkpoints so an interrupted Intel Mac session can resume.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import wave

from privacy_defaults import DOWNLOAD_OPT_IN, MODEL_DOWNLOADS_ALLOWED


def atomic_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def merge_windows(windows, duration, min_duration=.15, max_gap=.12):
    normalized=[]
    for r in windows:
        a=max(0.0,float(r['start'])); b=min(duration,float(r['end']))
        if b>a:
            normalized.append({'start':a,'end':b})
    normalized.sort(key=lambda x:(x['start'],x['end']))
    merged=[]
    for x in normalized:
        if merged and x['start'] <= merged[-1]['end']+max_gap:
            merged[-1]['end']=max(x['end'],merged[-1]['end'])
        else:
            merged.append(dict(x))
    return [{'start':round(x['start'],3),'end':round(x['end'],3),
             'evidence':'pyannote_segmentation_overlap_candidate',
             'status':'detected_needs_audio_verification'}
            for x in merged if x['end']-x['start']>=min_duration]


def plan(duration, window):
    if duration <= 0 or window < 10:
        raise ValueError('无效录音长度或窗口大小（窗口应 >=10 秒）')
    result=[]; first=0.0
    while first < duration-1e-5:
        end=min(duration,first+window)
        result.append({'i':len(result),'start':round(first,4),'end':round(end,4)})
        first=end
    return result


def read_slice(handle,start,end,rate,frames):
    import numpy as np
    a=max(0,min(frames,int(round(start*rate))))
    b=max(a,min(frames,int(round(end*rate))))
    handle.setpos(a)
    raw=handle.readframes(b-a)
    return np.frombuffer(raw,dtype='<i2').astype('float32')/32768.0


def run(args):
    import numpy as np
    import torch
    from pyannote.audio import Model
    from pyannote.audio.pipelines import OverlappedSpeechDetection
    torch.set_num_threads(args.threads)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    wav=Path(args.wav).expanduser().absolute()
    if not wav.is_file():raise FileNotFoundError(wav)
    with wave.open(str(wav),'rb') as inp:
        if (inp.getframerate(),inp.getnchannels(),inp.getsampwidth(),inp.getcomptype())!=(16000,1,2,'NONE'):
            raise RuntimeError('需要16kHz、16bit、单声道PCM WAV')
        rate=inp.getframerate(); frames=inp.getnframes()
    duration=frames/rate
    if args.limit_seconds > 0: duration=min(duration,args.limit_seconds)
    output=Path(args.output).expanduser().absolute()
    signature={'wav':str(wav),'bytes':wav.stat().st_size,'mtime_ns':wav.stat().st_mtime_ns,
               'model':args.model,'window':args.window,'context':args.context,
               'min_duration':args.min_duration,'duration':duration}
    run_id=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
    root=output.parent/'v335_osd_chunks'/run_id; root.mkdir(exist_ok=True,parents=True)
    atomic_json(root/'manifest.json',signature)

    windows=plan(duration,args.window)
    missing=[ch for ch in windows if not (root/f"chunk_{ch['i']:05d}.json").is_file()]
    if missing:
        print('只加载重叠语音检测网络；不会加载全场声纹/聚类模型。',flush=True)
        token=os.environ.get('HF_TOKEN') or None
        model=Model.from_pretrained(args.model,use_auth_token=token)
        if model is None:raise RuntimeError('重叠检测模型未加载。检查 pyannote/segmentation-3.0 许可及 HF_TOKEN')
        model.to(torch.device('cpu'))
        detector=OverlappedSpeechDetection(segmentation=model)
        detector.instantiate({'min_duration_on':max(0.,args.min_duration/2),
                              'min_duration_off':0.06})
        detector.to(torch.device('cpu'))
    else:
        detector=None
        print('检测窗口全部已缓存，不重新加载模型。',flush=True)
    began=time.monotonic()
    with wave.open(str(wav),'rb') as handle:
        for k,ch in enumerate(windows,1):
            path=root/f"chunk_{ch['i']:05d}.json"
            if path.is_file():
                print(f'重叠检测 {k}/{len(windows)} 已缓存',flush=True)
                continue
            a=max(0.,ch['start']-args.context)
            b=min(duration,ch['end']+args.context)
            data=read_slice(handle,a,b,rate,frames)
            if len(data)<rate*10:
                data=np.pad(data,(0,rate*10-len(data)))
            waveform=torch.from_numpy(data.copy()).unsqueeze(0)
            with torch.inference_mode():
                annotation=detector({'waveform':waveform,'sample_rate':rate})
            spans=[]
            for interval in annotation.get_timeline().support():
                st=max(ch['start'],a+float(interval.start))
                en=min(ch['end'],a+float(interval.end))
                if en>st:
                    spans.append({'start':round(st,3),'end':round(en,3)})
            atomic_json(path,{'window':ch,'overlaps':spans,'elapsed_seconds':round(time.monotonic()-began,2)})
            print(f'重叠检测 {k}/{len(windows)}：本窗口发现 {len(spans)} 处；累计 {time.monotonic()-began:.1f}s',flush=True)
    raw=[]
    for ch in windows:
        doc=json.loads((root/f"chunk_{ch['i']:05d}.json").read_text(encoding='utf-8'))
        raw.extend(doc['overlaps'])
    detected=merge_windows(raw,duration,args.min_duration)
    atomic_json(output,detected)
    atomic_json(output.with_name(output.stem+'_meta.json'),
                {'signature':signature,'coverage_seconds':duration,
                 'count':len(detected),'cache_directory':str(root)})
    print(f'重叠检测完成：{len(detected)} 处候选区间。没有执行人物身份识别。',flush=True)
    return 0


def cli(argv=None):
    parser=argparse.ArgumentParser(description='只检测同时说话的时间，不识别任何人的身份')
    parser.add_argument('--wav',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--model',default='pyannote/segmentation-3.0')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--window',type=float,default=20.)
    parser.add_argument('--context',type=float,default=1.)
    parser.add_argument('--min-duration',type=float,default=.18)
    parser.add_argument('--limit-seconds',type=float,default=0.)
    args=parser.parse_args(argv)
    if not 1<=args.threads<=16 or not 10<=args.window<=120 or args.context<0 or args.min_duration<0:
        parser.error('线程数/窗口参数不合法')
    return run(args)


if __name__=='__main__':
    try:sys.exit(cli())
    except KeyboardInterrupt:
        print('扫描已中断，已完成的窗口不会丢失。',file=sys.stderr);sys.exit(130)
    except Exception as exc:
        hint = ''
        if not MODEL_DOWNLOADS_ALLOWED:
            hint = (f' 默认离线；若模型尚未缓存，确认访问条款后可临时设置 '
                    f'{DOWNLOAD_OPT_IN}=1。')
        print(f'重叠扫描失败：{type(exc).__name__}: {exc}{hint}',file=sys.stderr);sys.exit(2)
