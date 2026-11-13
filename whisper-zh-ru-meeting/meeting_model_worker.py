#!/usr/bin/env python3
"""Optional LOCAL model workers. One model load per batch, never cloud inference.

Run in enhancement-venv (Python 3.11), not the main Python 3.14 environment.
Official pre-trained model weights are downloaded on first use unless cached.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def finite_audio(samples):
    import numpy as np
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError('模型输出音频非一维或含非有限值')
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.
    # Never boost a nearly-silent track into apparent speech.
    return samples / max(1., peak / .98)


def best_assignment(scores, speakers, minimum=.55, margin=.10):
    """Two-stream bijection. Uncalibrated cosine scores, NOT probabilities."""
    if len(scores) != 2 or len(speakers) != 2:
        return [None, None]
    same = scores[0][0] + scores[1][1]
    swap = scores[0][1] + scores[1][0]
    indices = [0, 1] if same >= swap else [1, 0]
    if abs(same-swap) < 2*margin:
        return [None, None]
    answer = []
    for stream, index in enumerate(indices):
        val = scores[stream][index]
        gap = val-scores[stream][1-index]
        answer.append(speakers[index] if val >= minimum and gap >= margin else None)
    return answer


def denoise(request):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from df.enhance import init_df, enhance
    torch.set_num_threads(request.get('threads', 4))
    initialized = init_df(model_base_dir=request.get('model', 'DeepFilterNet3'),
                          post_filter=False, log_level='WARNING')
    model, state = initialized[0], initialized[1]
    rate = int(state.sr())
    input_path, output_path = Path(request['input']), Path(request['output'])
    temp = output_path.with_suffix('.partial.wav')
    info = sf.info(str(input_path))
    if info.samplerate != 16000 or info.channels != 1:
        raise ValueError('DeepFilter worker 输入必须是16kHz单声道')
    frames = info.frames
    block, context = 30*16000, 16000
    with sf.SoundFile(str(input_path)) as source, sf.SoundFile(
            str(temp), mode='w', samplerate=16000, channels=1, subtype='PCM_16') as dest:
        for first in range(0, frames, block):
            last = min(frames, first+block)
            a, b = max(0, first-context), min(frames, last+context)
            source.seek(a)
            raw = torch.from_numpy(source.read(b-a, dtype='float32')).unsqueeze(0)
            audio = AF.resample(raw, 16000, rate)
            with torch.inference_mode():
                # pad=True compensates STFT/model latency. Attenuation is limited.
                cleaned = enhance(model, state, audio, pad=True, atten_lim_db=12.)
            cleaned = AF.resample(cleaned.cpu(), rate, 16000).squeeze(0).numpy()
            if len(cleaned) < b-a:
                raise RuntimeError('DeepFilter 输出变短；拒绝填零或移动时间戳')
            result = cleaned[first-a:last-a]
            dest.write(finite_audio(result))
            print(f'DeepFilter: {last/16000:.1f}/{frames/16000:.1f}s', flush=True)
    if sf.info(str(temp)).frames != frames:
        raise RuntimeError('DeepFilter 输出长度与原音频不一致')
    temp.replace(output_path)
    return {'status': 'ok', 'path': str(output_path), 'model': request.get('model', 'DeepFilterNet3'),
            'attenuation_limit_db': 12, 'latency_compensated': True,
            'note': '分块降噪为候选轨；没有静音门控；尚需真人语音比较。'}


def separate(request):
    import numpy as np
    import soundfile as sf
    import torch
    from speechbrain.inference.separation import SepformerSeparation
    torch.set_num_threads(request.get('threads', 4))
    cache = Path(request['model_cache']); cache.mkdir(parents=True, exist_ok=True)
    model = SepformerSeparation.from_hparams(
        source=request.get('model', 'speechbrain/sepformer-whamr16k'),
        savedir=str(cache/'sepformer-whamr16k'), run_opts={'device': 'cpu'})
    if int(model.hparams.sample_rate) != 16000 or int(model.hparams.num_spks) != 2:
        raise ValueError('本接口要求16kHz、两路输出模型')
    embeddings = {}; encoder = None; warnings = []
    if request.get('references'):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            encoder = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                savedir=str(cache/'ecapa-voxceleb'), run_opts={'device': 'cpu'})
            for speaker, paths in request['references'].items():
                pieces = []
                for path in paths:
                    raw, rate = sf.read(path, dtype='float32')
                    if rate != 16000 or raw.ndim != 1:
                        raise ValueError('声纹参考音频格式错误')
                    with torch.inference_mode():
                        value = encoder.encode_batch(torch.from_numpy(raw).unsqueeze(0)).flatten()
                    pieces.append(torch.nn.functional.normalize(value, dim=0))
                if pieces:
                    embeddings[speaker] = torch.nn.functional.normalize(torch.stack(pieces).mean(0), dim=0)
        except Exception as exc:
            warnings.append('声纹匹配不可用，分离音轨将保持待定：'+str(exc))
            encoder = None; embeddings = {}
    results = []
    out = Path(request['output_dir']); out.mkdir(parents=True, exist_ok=True)
    for job in request['jobs']:
        if job.get('status') != 'ready':
            results.append(dict(job)); continue
        result_file = out/(job['id']+'.json')
        if result_file.exists():
            cached = json.loads(result_file.read_text(encoding='utf-8'))
            if cached.get('status') == 'separated' and all(Path(t['path']).is_file() for t in cached.get('tracks', [])):
                results.append(cached); continue
        item = dict(job)
        try:
            raw, rate = sf.read(job['input_path'], dtype='float32')
            if rate != 16000 or raw.ndim != 1 or not len(raw):
                raise ValueError('分离输入格式错误')
            with torch.inference_mode():
                estimated = model.separate_batch(torch.from_numpy(raw).unsqueeze(0)).cpu()[0]
            if tuple(estimated.shape) != (len(raw), 2):
                raise RuntimeError('分离模型没有返回等长的两路音频')
            scores = []; suggested = [None, None]
            tracks = []
            for i in range(2):
                samples = finite_audio(estimated[:, i].numpy())
                target = out/f'{job["id"]}_track{i+1}.wav'
                tmp = target.with_suffix('.partial.wav')
                sf.write(str(tmp), samples, 16000, subtype='PCM_16'); tmp.replace(target)
                rms = float(np.sqrt(np.mean(samples*samples)))
                track = {'stream': i+1, 'path': str(target), 'rms': rms,
                         'speaker_suggestion': None, 'speaker_status': 'unverified',
                         'note': '轨道序号仅在本片段有效；不是全场说话人编号。'}
                tracks.append(track)
                if encoder is not None and all(s in embeddings for s in job['speakers']) and rms > 1e-4:
                    with torch.inference_mode():
                        e = encoder.encode_batch(torch.from_numpy(samples).unsqueeze(0)).flatten()
                    e = torch.nn.functional.normalize(e, dim=0)
                    scores.append([float(torch.dot(e, embeddings[s])) for s in job['speakers']])
            if len(scores) == 2:
                suggested = best_assignment(scores, job['speakers'])
                for i, track in enumerate(tracks):
                    track['speaker_suggestion'] = suggested[i]
                    track['similarity_scores'] = dict(zip(job['speakers'], scores[i]))
                    track['speaker_status'] = 'heuristic_candidate' if suggested[i] else 'unverified'
            item.update(status='separated', tracks=tracks, model=request.get('model'),
                        model_domain='WHAMR English; Chinese/Russian meeting accuracy not measured')
        except Exception as exc:
            item.update(status='error', error=f'{type(exc).__name__}: {exc}')
        atomic_json(result_file, item); results.append(item)
        print(f'Separation {job["id"]}: {item["status"]}', flush=True)
    return {'results': results, 'warnings': warnings}


def split_translation_text(text, max_chars=600):
    """Sentence-aware bounded units; never silently truncate a long source."""
    import re
    if not text.strip():
        return []
    parts = re.split(r'(?<=[。！？.!?;；])\s*', text)
    units = []
    for part in parts:
        part = part.strip()
        while len(part) > max_chars:
            # Prefer whitespace for Russian, character boundary for Chinese.
            cut = part.rfind(' ', 0, max_chars+1)
            if cut < max_chars//2:
                cut = max_chars
            units.append(part[:cut]); part = part[cut:].lstrip()
        if part:
            units.append(part)
    return units


def translate(request):
    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    torch.set_num_threads(request.get('threads', 4))
    model_name = request.get('model', 'facebook/m2m100_418M')
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name).to('cpu').eval()
    out = Path(request['output_dir']); out.mkdir(parents=True, exist_ok=True)
    results = []
    for job in request['jobs']:
        record_file = out/(job['id']+'.json')
        if record_file.exists():
            cached = json.loads(record_file.read_text(encoding='utf-8'))
            if (cached.get('input_hash') == job['input_hash'] and cached.get('model') == model_name
                    and cached.get('status') == 'translated'):
                results.append(cached); continue
        record = dict(job)
        try:
            if {job['source_language'], job['target_language']} != {'zh', 'ru'}:
                raise ValueError('只支持中文和俄文之间互译')
            tokenizer.src_lang = job['source_language']
            units = split_translation_text(job['source_text']); translations = []
            for text in units:
                encoded = tokenizer(text, return_tensors='pt', truncation=False)
                if encoded['input_ids'].shape[1] > 900:
                    raise ValueError('源句超过模型长度，请分段；没有截断原文')
                with torch.inference_mode():
                    generated = model.generate(**encoded,
                        forced_bos_token_id=tokenizer.get_lang_id(job['target_language']),
                        num_beams=4, do_sample=False, max_new_tokens=768)
                sequence = generated[0].tolist()
                if len(sequence) >= 768 and sequence[-1] != tokenizer.eos_token_id:
                    raise ValueError('译文达到长度上限，拒绝把截断内容当完整译文')
                translations.append(tokenizer.decode(generated[0], skip_special_tokens=True).strip())
            translated = (' ' if job['target_language']=='ru' else '').join(translations)
            if not translated:
                raise ValueError('模型返回空译文')
            record.update(status='translated', translation=translated, model=model_name,
                          evidence_type='machine_translation_not_spoken_transcript')
        except Exception as exc:
            record.update(status='error', error=f'{type(exc).__name__}: {exc}', model=model_name)
        atomic_json(record_file, record); results.append(record)
        print(f'Translation {job["id"]}: {record["status"]}', flush=True)
    return {'results': results, 'model': model_name}


def main(argv=None):
    p = argparse.ArgumentParser(description='V3.2 本地可选模型批处理')
    p.add_argument('action', choices=['denoise', 'separate', 'translate'])
    p.add_argument('--request', required=True); p.add_argument('--response', required=True)
    args = p.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding='utf-8'))
    began = time.monotonic()
    try:
        response = {'denoise': denoise, 'separate': separate, 'translate': translate}[args.action](request)
        response['elapsed_seconds'] = round(time.monotonic()-began, 3)
        response['versions'] = {}
        for name in ('torch','torchaudio','speechbrain','transformers','deepfilternet'):
            try: response['versions'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: pass
        atomic_json(Path(args.response), response)
        return 0
    except Exception as exc:
        atomic_json(Path(args.response), {'status':'error','error':f'{type(exc).__name__}: {exc}'})
        print(f'模型运行失败：{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
