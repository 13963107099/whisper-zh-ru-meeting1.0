#!/usr/bin/env python3
"""Auditable bilingual support. Translations never replace acoustic ASR evidence.

Temporal proximity produces candidate links, not verified translation alignment.
Manual source/pair confirmations are invalidated when the linked texts change.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def language_of(text: str) -> str:
    zh = len(re.findall(r'[\u4e00-\u9fff]', text))
    ru = len(re.findall(r'[\u0400-\u052f]', text))
    if zh >= 2 and ru <= max(1, zh*.15):
        return 'zh'
    if ru >= 3 and zh <= max(1, ru*.15):
        return 'ru'
    return 'unknown'


def segment_id(record: dict) -> str:
    return 's_' + digest([round(record['start'], 3), round(record['end'], 3),
                          record.get('speaker'), record.get('asr_text', record.get('text', ''))])[:16]


def sensitive_warnings(text: str) -> list[str]:
    warnings = []
    if re.search(r'\d|[零〇一二三四五六七八九十百千万亿两]|\b(?:один|два|три|четыре|пять|десять|сто|тысяч\w*|миллион\w*|миллиард\w*)\b', text, re.I):
        warnings.append('含数字、金额或数量词：需核对数值、单位与币种，不自动按另一语言改数。')
    if re.search(r'不|没|否|无|\b(?:не|нет|нельзя|без)\b', text, re.I):
        warnings.append('含否定：需复听，不能把否定改成肯定。')
    if re.search(r'承诺|同意|确认|愿意|可能|计划|希望|готов\w*|обяз\w*|соглас\w*|возмож\w*|планир\w*', text, re.I):
        warnings.append('含意愿或承诺表述：核对情态强度，不能把意向写成已承诺。')
    return warnings


def group_turns(records: list[dict]) -> list[dict]:
    groups = []
    for r in sorted(records, key=lambda x: (x['start'], x['end'])):
        text = r.get('raw_text', r.get('text', '')).strip()
        language = language_of(text)
        flags = r.get('speaker_flags', [])
        risky = (bool(r.get('asr_review')) or bool(r.get('refine_review'))
                 or '重叠发言' in flags or '多位发言/时间戳跨人' in flags
                 or float(r.get('avg_logprob', -9)) < -.80
                 or not text or bool(re.search(r'听不清|无法辨认|неразборчив', text, re.I)))
        if r.get('manual_text') and text:
            risky = False
        good_source = (not risky and language in {'zh', 'ru'} and len(text) >= 6
                       and (bool(r.get('manual_text')) or float(r.get('avg_logprob', -9)) >= -.65))
        element = {'start': r['start'], 'end': r['end'], 'speaker': r.get('speaker'),
                   'language': language, 'text': text, 'segment_ids': [segment_id(r)],
                   'uncertain': risky, 'source_eligible': good_source,
                   'manual_text': bool(r.get('manual_text'))}
        # Combine contiguous pieces of one turn, retaining every original segment id.
        if (groups and r.get('speaker') and groups[-1]['speaker'] == r.get('speaker')
                and groups[-1]['language'] == language
                and 0 <= r['start'] - groups[-1]['end'] <= 1.0
                and r['end']-groups[-1]['start'] <= 90):
            g = groups[-1]
            g['end'] = r['end']; g['text'] += (' ' if language == 'ru' else '')+text
            g['segment_ids'] += element['segment_ids']
            g['uncertain'] |= risky
            g['source_eligible'] &= good_source
            g['manual_text'] &= element['manual_text']
        else:
            groups.append(element)
    return groups


def propose_pairs(records: list[dict], previous: list[dict] | None = None,
                  interpreter_speakers: set[str] | None = None,
                  window: float = 120.) -> list[dict]:
    groups = group_turns(records)
    previous_by_id = {p.get('id'): p for p in (previous or [])}
    result = []
    for target in groups:
        if not target['uncertain']:
            continue
        candidates = []
        for source in groups:
            if source is target or not source['source_eligible']:
                continue
            if target['language'] not in {'zh','ru'}:
                # A completely unrecognised language is not inferred solely by timing.
                continue
            if source['language'] == target['language']:
                continue
            distance = max(0., source['start']-target['end'], target['start']-source['end'])
            if distance > window:
                continue
            if interpreter_speakers and not ({source['speaker'], target['speaker']} & interpreter_speakers):
                continue
            # There is no automatic claim of semantic equivalence here.
            candidates.append((distance, source))
        candidates.sort(key=lambda pair: (pair[0], pair[1]['start']))
        for distance, source in candidates[:2]:
            signature = digest([source, target])
            ident = 'pair_'+signature[:16]
            prior = previous_by_id.get(ident, {})
            src_lang = source['language']; dst_lang = target['language']
            warnings = sensitive_warnings(source['text']+' '+target['text'])
            if src_lang == 'zh':
                warnings.append('中文可能本身是口译；回译俄文会继承口译遗漏和错误，不是恢复俄方逐字原话。')
            if len(candidates) > 1:
                warnings.append('同一疑难片段存在多个相邻候选，须确认哪一段确实对应。')
            result.append({'id': ident, 'input_hash': signature,
                           'source_ids': source['segment_ids'], 'target_ids': target['segment_ids'],
                           'source_start': source['start'], 'source_end': source['end'],
                           'target_start': target['start'], 'target_end': target['end'],
                           'source_speaker': source['speaker'], 'target_speaker': target['speaker'],
                           'source_language': src_lang, 'target_language': dst_lang,
                           'source_text': source['text'], 'target_asr_text': target['text'],
                           'source_verified': prior.get('source_verified') is True,
                           'pair_confirmed': prior.get('pair_confirmed') is True,
                           'disabled': prior.get('disabled') is True,
                           'pairing_method': 'time_neighbourhood_candidate_not_semantic_verification',
                           'gap_seconds': round(distance, 3),
                           'status': 'candidate', 'warnings': warnings})
    return result


def refresh_pairs(directory: Path, records: list[dict], interpreter_speakers=None,
                  window=120.) -> list[dict]:
    path = directory/'bilingual_pairs.json'
    prior = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    if not isinstance(prior, list):
        raise ValueError('bilingual_pairs.json 必须是数组')
    pairs = propose_pairs(records, prior, interpreter_speakers, window)
    from whisper_zh_ru_v3 import atomic_json
    atomic_json(path, pairs)
    stale = [p for p in prior if p.get('id') not in {q['id'] for q in pairs}]
    if stale:
        # Never lose past decisions; do not reuse a translation after a source edit.
        archive_path = directory/'bilingual_stale_pairs.json'
        archived = json.loads(archive_path.read_text(encoding='utf-8')) if archive_path.exists() else []
        seen = {p.get('id') for p in archived}
        archived.extend(p for p in stale if p.get('id') not in seen)
        atomic_json(archive_path, archived)
    return pairs


def select_translation_jobs(pairs: list[dict], max_pairs=20, verified_only=False) -> tuple[list[dict], list[dict]]:
    ready = []; deferred = []
    ordered = sorted(pairs, key=lambda p: (not (p['source_verified'] and p['pair_confirmed']),
                                            p['target_start'], p['gap_seconds']))
    for pair in ordered:
        if pair.get('disabled'):
            deferred.append({**pair, 'status': 'disabled'}); continue
        if verified_only and not (pair['source_verified'] and pair['pair_confirmed']):
            deferred.append({**pair, 'status': 'waiting_for_source_and_pair_confirmation'}); continue
        if max_pairs and len(ready) >= max_pairs:
            deferred.append({**pair, 'status': 'deferred_translation_budget'}); continue
        ready.append(dict(pair))
    return ready, deferred


def append_support(directory: Path, bilingual_results: list[dict], overlap_results: list[dict],
                   status: str = 'completed') -> None:
    """Append explicitly separated evidence layers to ONE main TXT; preserve base ASR."""
    from whisper_zh_ru_v3 import atomic_text, atomic_json
    from speaker_review import stamp
    base = directory/'all_clean.txt'
    body = base.read_text(encoding='utf-8')
    body = ('Whisper V3.2｜会议增强｜状态：'+status+'\n'
            '正文是声学转写；后附互译及分离候选不覆盖正文。\n\n') + body
    lines = [body]
    queue_path = directory/'review_queue.json'
    queue = json.loads(queue_path.read_text(encoding='utf-8')) if queue_path.exists() else []
    if bilingual_results:
        lines += ['\n=== 中俄互译辅助（不是对应发言人的逐字原话） ===']
        for item in sorted(bilingual_results, key=lambda p: (p['target_start'], p.get('gap_seconds', 0))):
            if item.get('disabled') or item.get('status') == 'disabled':
                continue
            label = ('据俄文片段机器翻译为中文' if item['source_language']=='ru'
                     else '据中文片段机器回译俄文，非俄方原话')
            confirmed = item.get('source_verified') and item.get('pair_confirmed')
            lines += [f"[{stamp(item['target_start'])}–{stamp(item['target_end'])}] {label}",
                      f"关联：{item['id']}；{'来源及对应关系已由用户确认，译文本身仍待核验' if confirmed else '来源准确性及片段对应关系待确认'}",
                      f"来源 [{stamp(item['source_start'])}–{stamp(item['source_end'])}]：{item['source_text']}",
                      '目标录音原转写：'+item['target_asr_text']]
            if item.get('translation') and item.get('status')=='translated':
                lines.append('辅助译文：'+item['translation'])
            else:
                lines.append('未生成译文：'+str(item.get('status', 'candidate')))
            lines.extend('核验：'+warning for warning in item.get('warnings', []))
            lines.append('')
            queue.append({'type': 'bilingual_support', 'id': item['id'],
                          'start_seconds': item['target_start'], 'end_seconds': item['target_end'],
                          'reasons': ['核对源文、口译对应关系及译文；不覆盖原始转写'],
                          'status': item.get('status')})
    if overlap_results:
        lines += ['\n=== 同时发言分离候选（按原混音复听后采用） ===']
        for item in overlap_results:
            lines.append(f"[{stamp(item['start'])}–{stamp(item['end'])}] {item['id']}｜{item['status']}")
            for track in item.get('tracks', []):
                suggested = track.get('speaker_suggestion') or '待定'
                lines.append(f"本片段音轨{track['stream']}｜说话人候选 {suggested}（未核实）")
                for seg in track.get('segments', []):
                    lines.append(f"  [{stamp(seg['start'])}–{stamp(seg['end'])}] {seg['text']}")
                if not track.get('segments'):
                    lines.append('  未获得可用文字；并不自动认定该音轨没有发言。')
                if track.get('duplicate_other_track'):
                    lines.append('  警告：两路识别文字相近，可能存在串音；不能视为已分离成功。')
                lines.append('  候选音频：'+track['path'])
            queue.append({'type': 'overlap_separation', 'id': item['id'],
                          'start_seconds': item['start'], 'end_seconds': item['end'],
                          'reasons': ['复听分离语音和原混音；轨道序号不代表全场身份'],
                          'status': item['status']})
    atomic_text(base, '\n'.join(lines)+'\n')
    atomic_json(queue_path, queue)


def load_manual_links(directory: Path, records: list[dict]) -> list[dict]:
    """Explicit user-linked spans also support a completely inaudible counterpart.
    Source is always assembled from the current ASR/manual-corrected transcript,
    never from fabricated replacement source_text in a link record.
    """
    from whisper_zh_ru_v3 import atomic_json
    path=directory/'bilingual_manual_links.json'
    if not path.exists():
        atomic_json(path,[]);return []
    links=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(links,list): raise ValueError('bilingual_manual_links.json 必须是数组')
    pairs=[]
    for index, link in enumerate(links):
        a,b=float(link['source_start']),float(link['source_end'])
        c,d=float(link['target_start']),float(link['target_end'])
        source_language,target_language=link['source_language'],link['target_language']
        if not (0<=a<b and 0<=c<d) or {source_language,target_language}!={'zh','ru'}:
            raise ValueError(f'人工双语关联第{index+1}项时间或语言无效')
        def selected(start,end):
            return [r for r in records if start<=(r['start']+r['end'])/2<end]
        source=selected(a,b); target=selected(c,d)
        source_text=(' ' if source_language=='ru' else '').join(r.get('raw_text',r.get('text','')) for r in source)
        if not source_text.strip() or language_of(source_text)!=source_language:
            raise ValueError(f'人工双语关联第{index+1}项没有对应语言的源文字；先校对源转写')
        current_source_hash=digest(source_text)
        # Confirmation is bound to a specific source text; editing source invalidates it.
        confirmed_hash=link.get('confirmed_source_hash')
        verified=(link.get('source_verified') is True and confirmed_hash==current_source_hash)
        target_text=(' ' if target_language=='ru' else '').join(r.get('raw_text',r.get('text','')) for r in target)
        signature=digest([a,b,c,d,source_text,target_text,source_language,target_language])
        pairs.append({'id':'manual_'+signature[:16],'input_hash':signature,
            'source_ids':[segment_id(r) for r in source],'target_ids':[segment_id(r) for r in target],
            'source_start':a,'source_end':b,'target_start':c,'target_end':d,
            'source_speaker':source[0].get('speaker'),'target_speaker':link.get('target_speaker'),
            'source_language':source_language,'target_language':target_language,
            'source_text':source_text,'target_asr_text':target_text or '[该片段未识别出文字]',
            'source_verified':verified,'pair_confirmed':link.get('pair_confirmed') is True,
            'disabled':link.get('disabled') is True,'pairing_method':'user_linked_times',
            'gap_seconds':max(0,a-d,c-b),'status':'candidate',
            'current_source_hash':current_source_hash,
            'warnings':sensitive_warnings(source_text)+(['中文回译俄文不是俄方原话。'] if source_language=='zh' else [])})
    atomic_json(directory/'bilingual_manual_pair_preview.json',pairs)
    return pairs
