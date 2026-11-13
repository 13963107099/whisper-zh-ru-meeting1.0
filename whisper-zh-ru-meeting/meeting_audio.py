#!/usr/bin/env python3
"""Non-destructive meeting-audio preprocessing. No network operations."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import wave
from typing import Any

RATE = 16000


def run_checked(command: list[str], log: Path | None = None) -> None:
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, errors='replace', check=False)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as stream:
            stream.write(repr(command) + '\n' + result.stdout + result.stderr + '\n')
    if result.returncode:
        raise RuntimeError(f'{Path(command[0]).name} 失败 ({result.returncode})：'
                           + result.stderr[-1800:])


def require_tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f'缺少 {name}。macOS 可使用已有包管理器安装 ffmpeg；详见 README_V3.2.md。')
    return found


def probe(audio: Path) -> dict[str, Any]:
    cmd = [require_tool('ffprobe'), '-v', 'error', '-select_streams', 'a:0',
           '-show_entries', 'stream=channels,sample_rate,codec_name:format=duration',
           '-of', 'json', str(audio)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    if not data.get('streams'):
        raise ValueError('输入文件没有可解码的音轨')
    return data


def wav_info(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), 'rb') as source:
        return source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()


def prepare_audio(audio: Path, out: Path, limit: float = 0,
                  keep_channels: bool = True) -> dict[str, Any]:
    """Original is read only. Analysis mono and per-channel derivatives are separate."""
    ffmpeg = require_tool('ffmpeg')
    info = probe(audio)
    out.mkdir(parents=True, exist_ok=True)
    prefix = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(audio)]
    if limit > 0:
        prefix += ['-t', str(limit)]
    mono = out / 'original_16k_mono.wav'
    if not mono.exists():
        temp = mono.with_name('original_16k_mono.partial.wav')
        run_checked(prefix + ['-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(RATE),
                              '-c:a', 'pcm_s16le', str(temp)], out / 'audio_processing.log')
        temp.replace(mono)
    channels = int(info['streams'][0]['channels'])
    channel_files = []
    if keep_channels and channels > 1:
        folder = out / 'original_channels'; folder.mkdir(exist_ok=True)
        for index in range(channels):
            target = folder / f'channel_{index + 1:02d}.wav'
            if not target.exists():
                temp = target.with_suffix('.partial.wav')
                run_checked(prefix + ['-map', '0:a:0', '-vn', '-af', f'pan=mono|c0=c{index}',
                                     '-ar', str(RATE), '-c:a', 'pcm_s16le', str(temp)],
                            out / 'audio_processing.log')
                temp.replace(target)
            channel_files.append(str(target))
    ch, width, rate, frames = wav_info(mono)
    if (ch, width, rate) != (1, 2, RATE) or frames == 0:
        raise ValueError('音频转换结果异常')
    return {'original_path': str(audio.resolve()), 'source_info': info, 'mono': str(mono),
            'channels': channel_files, 'duration': frames / rate,
            'channel_policy': '声道仅保留备查，不自动等同于发言人；主流程分析混合轨。'}


def denoise_light(original: Path, target: Path, declick: bool = False) -> Path:
    """Gentle spectral suppression; does not delete time or use a silence gate."""
    if target.exists():
        return target
    filters = ['highpass=f=65']
    if declick:
        # High threshold deliberately avoids treating every transient as noise.
        filters.append('adeclick=t=10:o=50')
    filters.append('afftdn=nr=6:nf=-45:tn=1')
    temp = target.with_suffix('.partial.wav')
    run_checked([require_tool('ffmpeg'), '-nostdin', '-hide_banner', '-loglevel', 'error',
                 '-y', '-i', str(original), '-af', ','.join(filters),
                 '-ar', str(RATE), '-ac', '1', '-c:a', 'pcm_s16le', str(temp)],
                target.parent / 'audio_processing.log')
    if wav_info(temp) != wav_info(original):
        temp.unlink(missing_ok=True)
        raise RuntimeError('降噪改变了采样数或音频格式；拒绝使用，以免时间戳错位。')
    temp.replace(target)
    return target


def overlap_regions(turns: list[dict], duration: float) -> list[dict]:
    """Exact event sweep on regular (not exclusive) diarization annotations."""
    events: dict[float, list[tuple[str, int]]] = {}
    for turn in turns:
        a, b = max(0., float(turn['start'])), min(duration, float(turn['end']))
        if b <= a:
            continue
        events.setdefault(a, []).append((str(turn['speaker']), 1))
        events.setdefault(b, []).append((str(turn['speaker']), -1))
    times = sorted(events)
    active: dict[str, int] = {}
    out = []
    for index, a in enumerate(times[:-1]):
        for speaker, delta in events[a]:
            active[speaker] = active.get(speaker, 0) + delta
            if active[speaker] <= 0:
                active.pop(speaker, None)
        b = times[index + 1]
        names = sorted(active)
        if len(names) < 2 or b <= a:
            continue
        if out and out[-1]['end'] == a and out[-1]['speakers'] == names:
            out[-1]['end'] = b
        else:
            out.append({'start': a, 'end': b, 'speakers': names})
    return out


def plan_separation(turns: list[dict], duration: float, budget: float = 120.,
                    max_core: float = 10., context: float = .5) -> list[dict]:
    jobs = []; used = 0.
    for region in overlap_regions(turns, duration):
        start = region['start']
        if region['end'] - start < .20:
            jobs.append({**region, 'id': f'ov{len(jobs):05d}', 'status': 'review_short_overlap'})
            continue
        while start < region['end'] - .001:
            end = min(region['end'], start + max_core)
            status = 'ready'
            if len(region['speakers']) != 2:
                status = 'review_more_than_two_speakers'
            elif budget > 0 and used + end - start > budget:
                status = 'deferred_budget'
            # A third voice in padding must not be fed to a two-source separator.
            audio_start, audio_end = max(0., start-context), min(duration, end+context)
            nearby = {t['speaker'] for t in turns if t['end'] > audio_start and t['start'] < audio_end}
            if nearby - set(region['speakers']):
                audio_start, audio_end = start, end
            jobs.append({'id': f'ov{len(jobs):05d}', 'start': start, 'end': end,
                         'audio_start': audio_start, 'audio_end': audio_end,
                         'speakers': region['speakers'], 'status': status})
            if status == 'ready':
                used += end-start
            start = end
    return jobs


def select_reference_turns(turns: list[dict], duration: float) -> dict[str, list[dict]]:
    # Derive genuinely single-speaker spans, never exclusive diarization that hides overlap.
    from speaker_review import _nonoverlap_from_raw
    isolated = _nonoverlap_from_raw(turns)
    result: dict[str, list[dict]] = {}
    for turn in sorted(isolated, key=lambda t: -(t['end']-t['start'])):
        if turn['end']-turn['start'] < 2.0:
            continue
        samples = result.setdefault(turn['speaker'], [])
        if len(samples) < 3:
            samples.append({'start': turn['start'], 'end': min(turn['start']+8., turn['end'], duration)})
    return result
