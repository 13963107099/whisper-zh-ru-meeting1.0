#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic diarization/ASR reconciliation and human-review tools.

Only standard-library imports. Speaker ids are within-one-recording identities,
NOT names/biometric identification. Confidence thresholds are review heuristics,
not calibrated probabilities.
"""
from collections import defaultdict
import json
from pathlib import Path
import math
import re
import transcript_guard as guard

SCHEMA = 1


def stamp(seconds):
    n = max(0, int(round(float(seconds) * 100)))
    return f"{n // 360000:02d}:{(n // 6000) % 60:02d}:{(n // 100) % 60:02d}.{n % 100:02d}"


def parse_stamp(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    t = str(value).strip()
    parts = t.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h)*3600 + int(m)*60 + float(s)
    return float(t)


def intersection(a, b, c, d):
    return max(0.0, min(float(b), float(d)) - max(float(a), float(c)))


def _sort_spans(spans, duration):
    result = []
    for s in spans:
        a = max(0.0, float(s['start']))
        b = min(float(duration), float(s['end']))
        sp = str(s['speaker'])
        if b > a and sp:
            result.append({'start': round(a, 3), 'end': round(b, 3), 'speaker': sp})
    return sorted(result, key=lambda s: (s['start'], s['end'], s['speaker']))


def _nonoverlap_from_raw(raw):
    """Linear-time sweep after sorting; never invent identity for overlaps."""
    if not raw:
        return []
    events=[]
    for s in raw:
        events.append((s['start'],1,s['speaker']))
        events.append((s['end'],-1,s['speaker']))
    # Process all events at a time boundary before writing the next interval.
    events.sort(key=lambda x:x[0])
    active=defaultdict(int)
    out=[]
    i=0
    while i<len(events):
        t=events[i][0]
        while i<len(events) and events[i][0]==t:
            _,delta,speaker=events[i]
            active[speaker]+=delta
            if active[speaker]<=0:
                del active[speaker]
            i+=1
        if i==len(events):
            break
        b=events[i][0]
        if b-t<.005 or len(active)!=1:
            continue
        speaker=next(iter(active))
        if out and out[-1]['speaker']==speaker and t<=out[-1]['end']+.005:
            out[-1]['end']=round(b,3)
        else:
            out.append({'start':round(t,3),'end':round(b,3),'speaker':speaker})
    return out


def normalize_diarization(doc, duration):
    """Map model-specific speaker labels to S01, S02 in first-appearance order."""
    raw = _sort_spans(doc.get('turns', []), duration)
    exclusive = _sort_spans(doc.get('exclusive_turns', []), duration)
    first = {}
    for s in sorted(raw + exclusive, key=lambda x: (x['start'], x['end'], x['speaker'])):
        first.setdefault(s['speaker'], s['start'])
    ids = {old: f'S{i+1:02d}' for i, (old, _) in enumerate(sorted(first.items(), key=lambda x: (x[1], x[0])))}
    for s in raw + exclusive:
        s['speaker'] = ids[s['speaker']]
    if not exclusive:
        exclusive = _nonoverlap_from_raw(raw)
    return {
        'schema': SCHEMA, 'engine': doc.get('engine', 'unknown'),
        'audio_duration': round(duration, 3),
        'speaker_ids': list(ids.values()), 'original_speaker_ids': ids,
        'turns': raw, 'exclusive_turns': exclusive,
        'config': doc.get('config', {}),
    }


def overlapping_speakers(data, start, end, min_seconds=.1):
    amounts = defaultdict(float)
    for turn in data.get('turns', []):
        ov = intersection(start, end, turn['start'], turn['end'])
        if ov > 0:
            amounts[turn['speaker']] += ov
    return {k: round(v, 3) for k,v in amounts.items() if v >= min_seconds}


def has_real_overlap(data, start, end, min_seconds=.13):
    raw = data.get('turns', [])
    relevant = [r for r in raw if r['end'] > start and r['start'] < end]
    for i, a in enumerate(relevant):
        for b in relevant[i+1:]:
            if a['speaker'] == b['speaker']:
                continue
            if intersection(max(start,a['start']), min(end,a['end']), b['start'], b['end']) >= min_seconds:
                return True
    return False


def attribute(data, start, end, suggested=None):
    """Align a text interval to non-overlapping speaker turns; flag uncertainty.

    Amounts are *time intersections*, not pyannote calibrated probabilities.
    """
    raw = overlapping_speakers(data, start, end, min_seconds=.06)
    active = defaultdict(float)
    for s in data.get('exclusive_turns', []):
        ov = intersection(start, end, s['start'], s['end'])
        if ov > 0:
            active[s['speaker']] += ov
    ranked = sorted(active.items(), key=lambda x: (-x[1], x[0]))
    candidates = [s for s, v in sorted(raw.items(), key=lambda x: -x[1]) if v >= .13]
    total = sum(active.values())
    mixed = (len(ranked) >= 2 and ranked[1][1] >= max(.25, total*.24))
    overlap = has_real_overlap(data, start, end)
    short = end-start < .90
    flags = []
    speaker = None
    if suggested and suggested in data.get('speaker_ids', []):
        speaker = suggested
        if raw.get(suggested,0) < .1:
            flags.append('模型归属不一致')
    elif ranked and total >= .24 and not mixed and ranked[0][1] / max(total,.001) >= .76:
        # For an ASR interval spanning a long silence, require less coverage;
        # but don't attribute a 30-second transcript based on a 0.5s voice.
        if ranked[0][1] >= min(1.0, (end-start)*.18):
            speaker = ranked[0][0]
    if mixed:
        flags.append('多位发言/时间戳跨人')
        if suggested is None:
            speaker = None
    if overlap:
        flags.append('重叠发言')
        # Do not pretend one text stream can separate two simultaneous voices.
        speaker = None
    if short and (len(raw) > 1 or not speaker or (end-start)<.65):
        flags.append('短句归属不清')
    if not speaker and not flags:
        flags.append('说话人未确定')
    return {'speaker': speaker, 'candidates': candidates, 'speaker_flags': flags}


def annotate(data, segments):
    result = []
    for s in segments:
        copy = dict(s)
        a = attribute(data, copy['start'], copy['end'], suggested=copy.get('speaker'))
        if not copy.get('speaker') or a['speaker'] is None:
            copy['speaker'] = a['speaker']
        copy['speaker_candidates'] = a['candidates']
        copy['speaker_flags'] = a['speaker_flags']
        result.append(copy)
    return result


def turns_in_range(data, start, end, join_gap=.18):
    """Non-overlap turns for isolated short-turn decoding (clip to chunk core)."""
    relevant = []
    for t in data.get('exclusive_turns', []):
        a, b = max(t['start'], start), min(t['end'], end)
        if b-a >= .1:
            relevant.append({'start':a,'end':b,'speaker':t['speaker']})
    relevant.sort(key=lambda r: (r['start'], r['end']))
    merged=[]
    for r in relevant:
        if merged and merged[-1]['speaker']==r['speaker'] and r['start']-merged[-1]['end']<=join_gap:
            merged[-1]['end'] = max(merged[-1]['end'], r['end'])
        else:
            merged.append(dict(r))
    return merged


def needs_turn_refine(data, segments, chunk_start, chunk_end, mode='smart'):
    if mode == 'none' or not segments:
        return False
    turns=turns_in_range(data,chunk_start,chunk_end)
    if len({t['speaker'] for t in turns if t['end']-t['start']>=.7})<2:
        return False
    if mode=='all':
        return True
    for seg in segments:
        parts=overlapping_speakers(data,seg['start'],seg['end'],min_seconds=.35)
        if len(parts)>=2:
            return True
    return False


def init_speaker_files(job_dir, data, atomic_json):
    path=Path(job_dir)/'speaker_names.json'
    names=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if not isinstance(names,dict):
        raise ValueError('speaker_names.json 不是有效的说话人命名对象')
    updated=dict(names)
    for speaker in data.get('speaker_ids',[]):
        updated.setdefault(speaker,f'发言人{int(speaker[1:]):02d}')
    if not path.exists() or updated != names:
        atomic_json(path,updated)
    overrides=Path(job_dir)/'manual_overrides.json'
    if not overrides.exists():
        atomic_json(overrides, [])


def _load_overrides(path):
    if not path.exists():
        return []
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,list):
        raise ValueError('manual_overrides.json 必须是 JSON 数组')
    validated=[]
    for n, entry in enumerate(data, 1):
        a,b=parse_stamp(entry['start']),parse_stamp(entry['end'])
        if not (0 <= a < b):
            raise ValueError(f'第 {n} 条人工修改起止时间错误')
        sp=entry.get('speaker')
        if sp is not None and (not isinstance(sp,str) or not sp.strip()):
            raise ValueError(f'第 {n} 条人工修改的 speaker 必须是非空字符串')
        t=entry.get('text')
        if t is not None and not isinstance(t,str):
            raise ValueError(f'第 {n} 条人工修改的 text 必须是字符串')
        validated.append((a,b,sp,t))
    return validated


def apply_overrides(segments, overrides):
    ret=[]
    for item in segments:
        s=dict(item)
        center=(s['start']+s['end'])/2
        for a,b,sp,txt in overrides:
            if a-1e-6 <= center <= b+1e-6 or intersection(a,b,s['start'],s['end']) >= .6*max(.01,s['end']-s['start']):
                if sp is not None:
                    s['speaker']=sp
                    s['speaker_flags']=[f for f in s.get('speaker_flags',[]) if f not in ('说话人未确定', '多位发言/时间戳跨人', '短句归属不清','模型归属不一致')]
                    s['manual_speaker']=True
                if txt is not None:
                    s['text']=txt
                    s['manual_text']=True
        ret.append(s)
    return ret


def _create_review_clips(job_dir, wav_path, queue, duration, save_clip):
    base=Path(job_dir)/'review_clips'
    base.mkdir(exist_ok=True)
    for i, entry in enumerate(queue,1):
        a=float(entry['start_seconds']); b=float(entry['end_seconds'])
        if b-a > 24:
            # Long clips expensive; capture context around the suspicious midpoint.
            m=(a+b)/2
            a=max(0,m-12); b=min(duration,m+12)
        a=max(0,a-.35); b=min(duration,b+.35)
        if b-a<.05:
            continue
        target=base/f'review_{i:04d}_{int(a*100):08d}.wav'
        if not target.exists():
            save_clip(wav_path,target,a,b)
        entry['audio_clip']=str(target)


def choose_display_labels(records, diar, names):
    """Prefer voice-cluster IDs; fallback is a turn/segment ID, NOT a person ID.

    Preserve authoritative speaker in `speaker` separately. Candidate presentation
    never promotes uncertainty to a verified identity. Multiple candidates stay
    explicit rather than arbitrarily selecting one speaker.
    """
    known=set(diar.get('speaker_ids', []))
    unknown_count=0
    for r in records:
        if r.get('excluded_reason'):
            r['display_speaker']='已过滤片段'
            r['display_speaker_basis']='excluded_user_reported_text'
            continue
        speaker=r.get('speaker')
        candidates=list(dict.fromkeys(x for x in r.get('speaker_candidates',[]) if x in known))
        if speaker:
            label=names.get(speaker, f'发言人{speaker[1:]}') if speaker in known else speaker
            basis='manual' if r.get('manual_speaker') else 'acoustic_cluster'
        elif len(candidates)==1:
            label=names.get(candidates[0], '发言人'+candidates[0][1:])
            basis='candidate_unverified'
        elif len(candidates)>1:
            label='/'.join(names.get(c, '发言人'+c[1:]) for c in candidates)
            basis='multiple_candidate_speakers_not_separated'
        else:
            unknown_count+=1
            label=f'发言段{unknown_count:03d}'
            basis='asr_segment_only_no_voice_identity'
        r['display_speaker']=label
        r['display_speaker_basis']=basis
    return records


def build_output(job_dir, duration, clean, clock, dedup, atomic_text, atomic_json,
                 save_clip=None, wav_path=None, export_clips=False):
    """Read saved chunks; produce exactly one main TXT + JSON review/corrections."""
    job_dir=Path(job_dir)
    diag_path=job_dir/'speaker_diarization.json'
    diar=json.loads(diag_path.read_text(encoding='utf-8')) if diag_path.exists() else {'speaker_ids':[], 'turns':[], 'exclusive_turns':[]}
    init_speaker_files(job_dir,diar,atomic_json)
    names=json.loads((job_dir/'speaker_names.json').read_text(encoding='utf-8'))
    overrides=_load_overrides(job_dir/'manual_overrides.json')
    if not isinstance(names,dict):
        raise ValueError('speaker_names.json 必须是 {"S01":"某人"} 形式的 JSON 对象')
    originals=[]
    review_windows=0
    for path in sorted((job_dir/'cache').glob('chunk_*.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        review_windows += int(bool(record.get('review')))
        for entry in record.get('segments',[]):
            s=dict(entry)
            s['asr_review']=bool(record.get('review'))
            s['refine_review']=record.get('speaker_refine_note') in {
                'too_many_turns_need_review','overlapping_speech_need_review',
                'short_interjection_need_review','turn_recognition_error',
                'turn_recognition_empty','transcript_difference_need_review',
                'no_refined_text'}
            originals.append(s)
    originals=dedup(originals)
    if diar.get('speaker_ids'):
        records=annotate(diar,originals)
    else:
        records=[{**s,'speaker':None,'speaker_flags':['未启用说话人识别'], 'speaker_candidates':[]} for s in originals]
    for r in records:
        r['asr_text']=r.get('text','')  # Original machine transcription before human overrides.
    records=apply_overrides(records,overrides)
    # Keep exactly one human-facing TXT. Full original wording and edits are
    # always retained in machine-readable JSON for verification/re-rendering.
    cleanup_audit=[]
    for r in records:
        original_text=r.get('text','')
        # User-verified corrections are authoritative; do not silently edit them.
        cleaned=(original_text,[]) if r.get('manual_text') else clean(original_text)
        if isinstance(cleaned,tuple):
            clean_text, edits=cleaned
        else:
            clean_text, edits=cleaned, []
        r['raw_text']=original_text
        r['cleaned_text_before_filter']=clean_text
        final_text, reason=guard.filter_display_text(clean_text, r.get('manual_text',False))
        r['cleaned_text']=final_text
        r['excluded_reason']=reason if not final_text and clean_text else None
        r['hallucination_review_reason']=reason if reason and final_text else None
        r['cleanup_edits']=edits
        if edits:
            cleanup_audit.append({
                'start':r['start'], 'end':r['end'],
                'speaker':r.get('speaker'),
                'raw_text':original_text,'cleaned_text':clean_text,
                'edits':edits,
            })
    choose_display_labels(records,diar,names)
    excluded=[{'start':r['start'],'end':r['end'],'display_speaker':r['display_speaker'],
               'rule':r['excluded_reason'], 'removed_from_main_txt':r['cleaned_text_before_filter'],
               'raw_asr':r['asr_text'], 'not_audio_verified_by_software':True}
              for r in records if r.get('excluded_reason')]
    atomic_json(job_dir/'excluded_segments.json',excluded)
    # Inspectable machine-readable version of the final assignment, including
    # flags, manual overrides, original speech, and cleanup edits.
    atomic_json(job_dir/'attributed_segments.json',records)
    atomic_json(job_dir/'cleanup_audit.json',cleanup_audit)

    chapters=max(1,math.ceil(duration/1800))
    sections=defaultdict(list)
    queue=[]
    for r in records:
        if r.get('excluded_reason'):
            # Excluded, user-reported irrelevant wording is auditable in its own JSON.
            continue
        mid=(r['start']+r['end'])/2
        idx=min(chapters-1,max(0,int(mid//1800)))
        sections[idx].append(r)
        reasons=list(dict.fromkeys((['转录内容需复听'] if r.get('asr_review') and not r.get('manual_text') else [])
                                    + (['多人细分失败需复听'] if r.get('refine_review') else [])
                                    + r.get('speaker_flags',[])
                                    + ([r['hallucination_review_reason']] if r.get('hallucination_review_reason') else [])
                                    + (['仅轮次编号，尚未区分声纹'] if r.get('display_speaker_basis')=='asr_segment_only_no_voice_identity' else [])
                                    + (['说话人候选归属未确认'] if r.get('display_speaker_basis') in ('candidate_unverified','multiple_candidate_speakers_not_separated') else [])))
        if reasons:
            queue.append({'start':stamp(r['start']),'end':stamp(r['end']),
                          'start_seconds':r['start'],'end_seconds':r['end'],
                          'speaker':r.get('speaker'),'candidate_speakers':r.get('speaker_candidates',[]),
                          'reasons':reasons,'text':r.get('text','')})
    # Catch speaker turns absent from all ASR outputs. Cannot claim they were silent.
    for t in turns_in_range(diar,0,duration):
        dur=t['end']-t['start']
        if dur < .30:
            continue
        if not any(intersection(t['start'],t['end'],r['start'],r['end']) >= min(.2,dur*.30) for r in records):
            reason='短插话可能漏识别' if dur < .75 else '检测到发言但转录可能漏句'
            queue.append({'start':stamp(t['start']),'end':stamp(t['end']),
                          'start_seconds':round(t['start'],3),'end_seconds':round(t['end'],3),
                          'speaker':t['speaker'],'candidate_speakers':[t['speaker']],
                          'reasons':[reason],'text':''})
    queue.sort(key=lambda q:(q['start_seconds'],q['end_seconds']))
    if export_clips and wav_path is not None and save_clip:
        _create_review_clips(job_dir,wav_path,queue,duration,save_clip)
    atomic_json(job_dir/'review_queue.json',queue)
    has_voice_cluster=bool(diar.get('speaker_ids'))
    lines=[
        '中俄双语多人会议转录',
        ('说明：发言人01/02 等为声学聚类的临时人物编号，同一编号代表模型估计为同一发言人；可能需要人工修正。'
         if has_voice_cluster else
         '说明：尚未完成声纹聚类；发言段001/002 等是顺序段号，不能用来判断是否同一个人。'),
        '说明：正文不显示逐句复核标签；不确定归属与识别疑点保存在 review_queue.json。',
        '说明：已排除的用户确认无关文字保存在 excluded_segments.json，原始识别未删除。', '',
    ]
    if has_voice_cluster:
        lines.append('发言人索引：')
        for speaker in diar.get('speaker_ids',[]):
            label=names.get(speaker,'发言人'+speaker[1:])
            lines.append(f'  {speaker} = {label}（同一段录音中的临时声学编号）')
        lines.append('')
    for n in range(chapters):
        a=n*1800; b=min(duration,a+1800)
        lines.extend([f'第{n+1}部分｜{clock(a)}–{clock(b)}',''])
        for r in sections[n]:
            text=r['cleaned_text']
            if not text:
                continue
            # Review flags are deliberately kept out of the readable TXT.
            # Label provenance is kept in `attributed_segments.json`.
            who=r['display_speaker']
            lines.append(f"[{stamp(r['start'])}–{stamp(r['end'])}] {who}：{text}")
        lines.append('')
    lines.append('— 完 —')
    out=job_dir/'all_clean.txt'
    atomic_text(out,'\n'.join(lines)+'\n')
    return out, len(records), review_windows, len(queue)
