#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3.2 meeting pipeline: original-first ASR, candidate denoising/separation,
progressive one-TXT output, auditable Chinese/Russian translation support.
Heavy dependencies are isolated in an optional Python 3.11 environment.
"""
from __future__ import annotations
from contextlib import ExitStack
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import whisper_zh_ru_v3 as core
import speaker_review as speakers
import meeting_audio as audio_tools
import bilingual_support as bilingual

VERSION = '3.3.0'


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default


def parser():
    p = core.build_parser()
    p.description = 'V3.3 双阶段会议转写：快速完整初稿 + 缓存精校疑难片段'
    p.add_argument('--phase', choices=['fast','refine','full'], default='full',
                   help='fast首轮快速全量转写和说话人；refine只复核疑难并复用首轮缓存；full一次完成两阶段')
    p.add_argument('--denoise', choices=['off','light','deepfilter'], default='light')
    p.add_argument('--declick', action='store_true', help='轻降噪中增加保守去脉冲声；不是去除所有翻纸声')
    p.add_argument('--no-keep-channels', action='store_true', help='不额外导出各声道；原文件仍不会更改')
    p.add_argument('--no-vad-union', action='store_true', help='只使用原声 VAD，不与降噪轨语音区间取并集')
    p.add_argument('--enhancement-python', default=str(Path.home()/'whisper-ru/enhancement-venv/bin/python'))
    p.add_argument('--enhancement-threads', type=int, default=4)
    p.add_argument('--separate-overlap', choices=['off','sepformer'], default='off')
    p.add_argument('--separation-max-seconds', type=float, default=120.,
                   help='两人抢话核心时长预算，默认120秒；0全部；未处理片段明确保留复核状态')
    p.add_argument('--bilingual', choices=['off','suggest','translate'], default='suggest',
                   help='suggest只列对应候选；translate用本地M2M100生成双向译文候选')
    p.add_argument('--translation-model', default='facebook/m2m100_418M')
    p.add_argument('--translation-max-pairs', type=int, default=20, help='默认最多20组；0不限；只影响翻译')
    p.add_argument('--translation-verified-only', action='store_true', help='只翻译用户已确认源文和对应关系的组')
    p.add_argument('--interpreter-speakers', default='', help='已确认的翻译员编号，如 S02,S04；仅帮助筛选候选')
    p.add_argument('--pair-window', type=float, default=120., help='相邻双语候选最大间隔秒数，非语义验证')
    p.add_argument('--draft-every', type=int, default=15, help='每N个片段更新同一个 all_clean.txt')
    return p


def run_worker(action, request, directory, args):
    py = Path(args.enhancement_python).expanduser().absolute()
    if not py.is_file():
        raise RuntimeError(f'缺少可选模型环境：{py}。先执行 bash setup_meeting_enhancement.sh；'
                           '已完成的原始转写保留。')
    request_file = directory/f'{action}_request.json'
    response_file = directory/f'{action}_response.json'
    core.atomic_json(request_file, request)
    response_file.unlink(missing_ok=True)
    env = dict(os.environ)
    env.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    env.setdefault('PYANNOTE_METRICS_ENABLED', '0')
    with (directory/f'{action}_worker.log').open('a', encoding='utf-8') as log:
        completed = subprocess.run([str(py), str(Path(__file__).with_name('meeting_model_worker.py')),
                                    action, '--request', str(request_file), '--response', str(response_file)],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
    response = read_json(response_file, {})
    if completed.returncode or response.get('status')=='error':
        raise RuntimeError(f'{action} 模型执行失败：{response.get("error", "请查看日志")}；'
                           f'日志：{directory / (action+"_worker.log")}')
    return response


def merge_spans(spans, duration):
    out = []
    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        a, b = max(0., span.start), min(duration, span.end)
        if b<=a: continue
        if out and a <= out[-1].end+.04:
            out[-1] = core.Span(out[-1].start, max(out[-1].end, b))
        else: out.append(core.Span(a,b))
    return out


def get_spans(wav, duration, folder, args):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'spans.json'
    if path.exists():
        return [core.Span(**value) for value in read_json(path)]
    spans = core.detect_speech(Path(args.vad_binary).expanduser(), Path(args.vad_model).expanduser(),
                              wav, duration, args.threads, folder)
    core.atomic_json(path, [s.__dict__ for s in spans])
    return spans


def safe_deduplicate(segments, diar):
    """Do not remove simultaneous repeated agreements from unknown speakers."""
    result=[]; ordinary=[]
    for segment in sorted(segments,key=lambda r:(r['start'],r['end'])):
        if speakers.has_real_overlap(diar,segment['start'],segment['end']):
            result.extend(core.deduplicate_anchored(ordinary));ordinary=[]
            result.append(segment)
        else:
            ordinary.append(segment)
    result.extend(core.deduplicate_anchored(ordinary))
    return result


def render_base(directory, duration, wav, args, stage='处理中', finalize=False):
    # Drafts must not export hundreds of premature review clips.
    diar=read_json(directory/'speaker_diarization.json',{'turns':[]})
    result = speakers.build_output(directory,duration,
        lambda text:core.disfluency.clean_text(text,args.cleanup_mode),core.clock,
        lambda segments:safe_deduplicate(segments,diar),core.atomic_text,core.atomic_json,
        save_clip=core.save_wav_clip,wav_path=wav,
        export_clips=args.export_review_clips and finalize)
    core.atomic_json(directory/'pipeline_status.json', {'version': VERSION, 'stage':stage,
                                                       'duration': duration, 'updated_at_unix': time.time()})
    path = result[0]
    core.atomic_text(path, f'V3.3 状态：{stage}。\n\n'+path.read_text(encoding='utf-8'))
    return result


def asr_record(server, wav, enhanced, chunk, spans, duration, args, *,
               first=None, fast=False):
    """Single pass or cached second pass. Never re-run the first pass if `first` is given."""
    started = time.monotonic()
    reused = first is not None
    if first is None:
        first = core.transcribe_chunk(server, wav, chunk, args.beam_size, args.prompt, args.language)
    # Avoid modifying the first-pass evidence when selecting a later candidate.
    first = json.loads(json.dumps(first, ensure_ascii=False))
    chosen, method, rerun = first, 'original_first_pass', False
    denoised = None
    if not fast:
        # Stage 2: only uncertain first-pass results need a noise-model comparison.
        if enhanced is not None and core.is_uncertain(first):
            denoised = core.transcribe_chunk(server, enhanced, chunk, args.beam_size,
                                            args.prompt, args.language)
            if core.text_for(denoised) and core.quality(denoised) >= core.quality(first)+.25:
                chosen, method = denoised, 'denoised_candidate_selected'
            rerun = True
        if args.recheck!='none' and (args.recheck=='all' or core.is_uncertain(chosen)):
            selected, second_method, again = core.choose_second_pass(
                server, enhanced if chosen is denoised else wav, chunk, chosen, spans, duration, args)
            chosen = selected
            method += '+'+second_method
            rerun |= again
    for segment in chosen.get('segments', []):
        segment['language'] = bilingual.language_of(segment['text'])
        segment['audio_evidence'] = 'enhanced_candidate' if method.startswith('denoised') else 'original'
    return {'version': VERSION, 'chunk': chunk.index,
            'core': [chunk.core_start, chunk.core_end], 'audio': [chunk.audio_start, chunk.audio_end],
            'method': method, 'rerun': rerun,
            'asr_pass': 'fast' if fast else 'refined',
            'first_pass_reused': reused,
            'review': bool(rerun or core.is_uncertain(chosen)),
            'first_language': first.get('language'), 'original_asr': first,
            'denoised_asr': denoised, 'original_segments': first.get('segments', []),
            'segments': chosen.get('segments', []),
            'elapsed_seconds': round(time.monotonic()-started,3)}


def reuse_and_refine_cached(record, get_server, wav, enhanced, chunk, spans, duration, args):
    """Only decode again if the stored first pass was uncertain or `recheck=all`."""
    if record.get('asr_pass') != 'fast':
        return record, False
    first = record.get('original_asr')
    if not isinstance(first, dict):
        raise ValueError(f'首轮 chunk {chunk.index} 缺少原始ASR证据；不能安全复用。')
    should_retry = (core.is_uncertain(first) and (enhanced is not None or args.recheck != 'none')
                    or args.recheck == 'all')
    if should_retry:
        result = asr_record(get_server(), wav, enhanced, chunk, spans, duration, args,
                            first=first, fast=False)
        return result, True
    updated = dict(record)
    updated['asr_pass'] = 'refined'
    updated['first_pass_reused'] = True
    return updated, False


def refine_nonoverlap(server, wav, chunk, record, diar, args, duration):
    """If overlap occurs, refine other independent ASR segments, not the mixture."""
    doc = {'segments':record.get('segments', []), 'language':record.get('first_language')}
    if not speakers.has_real_overlap(diar, chunk.core_start, chunk.core_end):
        result, changed, note = core.refine_speaker_turns(server, wav, chunk, doc, diar, args, duration)
        return result, changed, note
    changed = False; revised = []
    for segment in doc['segments']:
        a,b=segment['start'],segment['end']
        if speakers.has_real_overlap(diar,a,b) or b-a > 27:
            revised.append(segment); continue
        sub = core.Chunk(-1,a,b,max(0,a-.15),min(duration,b+.15))
        result, accepted, _ = core.refine_speaker_turns(server,wav,sub,
                    {'segments':[segment]},diar,args,duration)
        revised.extend(result['segments'] if accepted else [segment])
        changed |= accepted
    return {'segments': revised}, changed, ('partial_nonoverlap_refined' if changed else 'overlapping_speech_need_review')


def missing_spans(diar, records, minimum=.30):
    # Genuine single-speaker coverage only. Overlap remains a separate queue.
    turns = speakers._nonoverlap_from_raw(diar.get('turns', []))
    covered = sorted((r['start'],r['end']) for r in records if r.get('text','').strip())
    missing=[]
    for t in turns:
        intervals=[(t['start'],t['end'])]
        for a,b in covered:
            if b <= t['start']: continue
            if a >= t['end']: break
            remaining=[]
            for x,y in intervals:
                if b <= x or a >= y: remaining.append((x,y)); continue
                if a > x: remaining.append((x,min(y,a)))
                if b < y: remaining.append((max(x,b),y))
            intervals=remaining
        for a,b in intervals:
            while b-a >= minimum:
                end=min(b,a+25.)
                missing.append({'start':a,'end':end,'speaker':t['speaker']})
                a=end
    return missing


def process_separation(wav, diar, duration, directory, args, get_server):
    if args.separate_overlap=='off':
        return [{**r, 'id':f'ov{i:05d}', 'status':'detected_not_separated'}
                for i,r in enumerate(audio_tools.overlap_regions(diar.get('turns', []),duration))]
    jobs=audio_tools.plan_separation(diar.get('turns', []),duration,args.separation_max_seconds)
    if not jobs: return []
    cache_id=bilingual.digest([core.fingerprint(wav),diar,VERSION])[:16]
    folder=directory/'separation'/cache_id; folder.mkdir(parents=True,exist_ok=True)
    for job in jobs:
        if job['status']=='ready':
            clip=folder/(job['id']+'_mix.wav')
            if not clip.exists(): core.save_wav_clip(wav,clip,job['audio_start'],job['audio_end'])
            job['input_path']=str(clip)
    if not any(j['status']=='ready' for j in jobs): return jobs
    refs={}
    for speaker, turns in audio_tools.select_reference_turns(diar.get('turns', []), duration).items():
        refs[speaker]=[]
        for i,t in enumerate(turns):
            clip=folder/f'reference_{speaker}_{i}.wav'
            if not clip.exists(): core.save_wav_clip(wav,clip,t['start'],t['end'])
            refs[speaker].append(str(clip))
    known={}; pending=[]
    for job in jobs:
        cached=read_json(folder/(job['id']+'.json'),{})
        if job['status']!='ready':
            known[job['id']]=job
        elif (cached.get('status')=='separated' and len(cached.get('tracks',[]))==2
              and all(Path(t['path']).is_file() for t in cached['tracks'])):
            known[job['id']]=cached
        else:
            pending.append(job)
    response={'results':[],'warnings':[]}
    if pending:
        response=run_worker('separate',{'jobs':pending,'references':refs,
            'model':'speechbrain/sepformer-whamr16k','output_dir':str(folder),
            'model_cache':str(Path.home()/'whisper-ru/enhancement-models'),
            'threads':args.enhancement_threads},directory,args)
        known.update({r['id']:r for r in response['results']})
    results=[known[j['id']] for j in jobs]
    for result in results:
        if result.get('status')!='separated': continue
        for track in result.get('tracks', []):
            trans_cache=Path(track['path']).with_suffix('.asr.json')
            if trans_cache.exists():
                track.update(read_json(trans_cache)); continue
            if track.get('rms',0) < 1e-4:
                track['segments']=[]; track['asr_status']='near_silent_track_review'
            else:
                length=core.wav_duration(Path(track['path']))
                relative=core.Chunk(-1, result['start']-result['audio_start'],
                                   min(length,result['end']-result['audio_start']),0.,length)
                try:
                    text=core.transcribe_chunk(get_server(),Path(track['path']),relative,
                                               args.beam_size,args.prompt,args.language)
                    for s in text['segments']:
                        s['start']=round(s['start']+result['audio_start'],3)
                        s['end']=round(s['end']+result['audio_start'],3)
                        s['evidence_type']='separated_audio_candidate'
                    track['segments']=text['segments']; track['asr_status']='candidate_needs_listening'
                except Exception as exc:
                    track['segments']=[]; track['asr_status']='error'; track['error']=str(exc)
            core.atomic_json(trans_cache,{'segments':track['segments'],'asr_status':track['asr_status']})
        tracks=result.get('tracks',[])
        if len(tracks)==2:
            texts=[''.join(s['text'] for s in t.get('segments',[])) for t in tracks]
            if all(texts) and difflib.SequenceMatcher(None,*texts).ratio()>.90:
                for t in tracks: t['duplicate_other_track']=True
    core.atomic_json(directory/'separation_warnings.json',response.get('warnings',[]))
    return results


def process_bilingual(directory,args):
    if args.bilingual=='off': return []
    records=read_json(directory/'attributed_segments.json',[])
    interpreters={s.strip() for s in args.interpreter_speakers.split(',') if s.strip()}
    pairs=bilingual.refresh_pairs(directory,records,interpreters,args.pair_window)
    pairs += bilingual.load_manual_links(directory, records)
    if args.bilingual=='suggest':
        return [{**p,'status':'candidate_not_translated'} for p in pairs]
    jobs,deferred=bilingual.select_translation_jobs(pairs,args.translation_max_pairs,args.translation_verified_only)
    if not jobs: return deferred
    folder=directory/'translation_cache'; folder.mkdir(exist_ok=True)
    # No model load if all exact sources have already been translated with this model.
    cached=[]; missing=[]
    for job in jobs:
        old=read_json(folder/(job['id']+'.json'),{})
        if (old.get('status')=='translated' and old.get('input_hash')==job['input_hash']
                and old.get('model')==args.translation_model):
            cached.append({**old, **job, 'status':'translated', 'translation':old['translation'],
                           'model':old['model']})
        else: missing.append(job)
    if missing:
        response=run_worker('translate',{'jobs':missing,'model':args.translation_model,
                           'output_dir':str(folder),'threads':args.enhancement_threads},directory,args)
        cached += response['results']
    return cached+deferred


def check_manifest(directory,name,value):
    path=directory/name
    if path.exists() and read_json(path)!=value:
        raise ValueError(f'{name} 对应的源文件或关键参数已变化。请换 --output-dir；已有缓存未被改写。')
    if not path.exists(): core.atomic_json(path,value)


def main(argv=None):
    p=parser(); args=p.parse_args(argv)
    try:
        args.prompt = core.load_domain_prompt(args.prompt, args.glossary_file)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    if min(args.threads,args.speaker_threads,args.enhancement_threads,args.beam_size,args.draft_every)<=0:
        p.error('线程数、beam-size、draft-every 必须为正数')
    if not 4<=args.max_core<=27 or not 0<=args.overlap<=1 or not 1<=args.port<=65535:
        p.error('max-core / overlap / port 参数无效')
    if args.max_refine_turns<2 or not .4<=args.min_turn_duration<=2.5:
        p.error('max-refine-turns或min-turn-duration参数无效')
    if args.limit_seconds<0 or args.separation_max_seconds<0 or args.translation_max_pairs<0 or args.pair_window<=0:
        p.error('时长预算和数量不能为负；pair-window必须为正')
    if min(args.num_speakers,args.min_speakers,args.max_speakers)<0:
        p.error('发言人数不能为负')
    if args.num_speakers and (args.min_speakers or args.max_speakers):
        p.error('固定人数与人数范围不能同时使用')
    if args.max_speakers and args.min_speakers>args.max_speakers:
        p.error('min-speakers不能大于max-speakers')
    if args.diarization=='json' and not args.speaker_json:
        p.error('需要--speaker-json')
    if args.declick and args.denoise!='light':
        p.error('--declick仅用于--denoise light；DeepFilter使用自己的增强算法')
    audio=Path(args.audio).expanduser().resolve()
    if not audio.is_file(): p.error(f'找不到录音：{audio}')
    name=re.sub(r'[^\w.\-]','_',audio.stem)[:60] or 'meeting'
    if args.limit_seconds>0: name+=f'_test_{args.limit_seconds:g}s'
    directory=(Path(args.output_dir).expanduser().resolve() if args.output_dir
               else Path.home()/'whisper-ru/output'/(name+'_v33'))
    directory.mkdir(parents=True,exist_ok=True); (directory/'cache').mkdir(exist_ok=True)
    metrics={'version':VERSION,'phase':args.phase,'status':'running','stages_seconds':{},'warnings':[]}
    began=time.monotonic()
    def timed(name, call):
        start=time.monotonic(); value=call()
        metrics['stages_seconds'][name]=round(time.monotonic()-start,3)
        core.atomic_json(directory/'performance.json',metrics)
        return value
    try:
        if args.render_only:
            meta=read_json(directory/'render_meta.json')
            if not meta: raise ValueError('没有render_meta.json，请先完成一次转写')
            if Path(meta['audio']).resolve()!=audio: raise ValueError('录音路径与缓存不一致')
            wav=Path(meta['wav']); duration=float(meta['duration'])
            if args.export_review_clips and not wav.exists(): raise ValueError('原始WAV不存在，不能导出复听切片')
            render_base(directory,duration,wav if wav.exists() else None,args,'重新生成文字',finalize=True)
            translations=timed('bilingual',lambda:process_bilingual(directory,args))
            core.atomic_json(directory/'bilingual_results.json',translations)
            separated=read_json(directory/'separation_results.json',[])
            bilingual.append_support(directory,translations,separated,'文字已重新生成；候选仍需核验')
            metrics['status']='rendered'
            return 0
        if args.phase == 'refine' and not (directory/'fast_stage_complete.json').exists():
            raise ValueError('当前目录尚未完成 --phase fast；请先运行第一阶段，不能凭空精校。')
        if args.phase == 'fast' and (directory/'refine_stage_complete.json').exists():
            print('该目录已经完成精校；为保护终稿，不重新以快速初稿覆盖。',flush=True)
            metrics['status']='already_refined'; return 0
        for label,filename,minimum in [('Whisper模型',args.model,1000000),('VAD模型',args.vad_model,10000),
                                        ('Whisper服务器',args.server_binary,1),('VAD程序',args.vad_binary,1)]:
            if args.plan_only and label in ('Whisper模型','Whisper服务器'): continue
            file=Path(filename).expanduser()
            if not file.is_file() or file.stat().st_size<minimum:
                raise ValueError(f'缺少{label}：{file}；沿用V3.1安装位置或显式指定路径')
        audio_sig={'v':VERSION,'audio':core.fingerprint(audio),'limit':args.limit_seconds,
                   'keep_channels':not args.no_keep_channels}
        check_manifest(directory,'audio_manifest.json',audio_sig)
        info=timed('audio_preparation',lambda:audio_tools.prepare_audio(audio,directory,args.limit_seconds,
                                                                   not args.no_keep_channels))
        core.atomic_json(directory/'audio_inventory.json',info)
        wav=Path(info['mono']); duration=info['duration']; enhanced=None
        if args.denoise=='light':
            enhanced=timed('denoise',lambda:audio_tools.denoise_light(wav,directory/('denoised_declick.wav' if args.declick else 'denoised_light.wav'),args.declick))
        elif args.denoise=='deepfilter':
            enhanced=directory/'denoised_deepfilter.wav'
            if not enhanced.exists():
                timed('denoise',lambda:run_worker('denoise',{'input':str(wav),'output':str(enhanced),
                     'model':'DeepFilterNet3','threads':args.enhancement_threads},directory,args))
            if audio_tools.wav_info(enhanced)!=audio_tools.wav_info(wav):
                raise ValueError('DeepFilter采样数不匹配，拒绝使用')
        core.atomic_json(directory/'render_meta.json',{'audio':str(audio),'wav':str(wav),'duration':duration,
                                                       'enhanced_wav':str(enhanced) if enhanced else None})
        sig={**audio_sig,'model':core.fingerprint(Path(args.model).expanduser()) if not args.plan_only else None,
             'vad_model':core.fingerprint(Path(args.vad_model).expanduser()),
             'vad_binary':core.fingerprint(Path(args.vad_binary).expanduser()),
             'server':core.fingerprint(Path(args.server_binary).expanduser()) if not args.plan_only else None,
             'language':args.language,'beam':args.beam_size,'prompt':args.prompt,
             'max_core':args.max_core,'overlap':args.overlap,'denoise':args.denoise,'declick':args.declick,
             'no_vad_union':args.no_vad_union}
        if not args.plan_only: check_manifest(directory,'asr_manifest.json',sig)
        start=time.monotonic()
        spans=get_spans(wav,duration,directory/'vad_original',args)
        if enhanced is not None and not args.no_vad_union:
            spans=merge_spans(spans+get_spans(enhanced,duration,directory/('vad_'+enhanced.stem),args),duration)
        metrics['stages_seconds']['vad']=round(time.monotonic()-start,3)
        chunks=core.plan_chunks(spans,duration,args.max_core,args.overlap)
        core.atomic_json(directory/'chunk_plan.json',[ch.__dict__ for ch in chunks])
        print(f'录音 {core.clock(duration)}；{len(chunks)} 个首轮任务；输出 {directory}',flush=True)
        if args.plan_only:
            print('仅预处理和VAD；没有执行ASR、分离或翻译。'); metrics['status']='planned';return 0
        if not chunks: metrics['warnings'].append('VAD没有检测到语音；仍尝试通过说话人检测找回漏段。')
        raw_cache=directory/'asr_cache'; raw_cache.mkdir(exist_ok=True)
        with ExitStack() as stack:
            server=[None]
            def get_server():
                if server[0] is None:
                    server[0]=stack.enter_context(core.LocalWhisperServer(
                        Path(args.server_binary).expanduser(),Path(args.model).expanduser(),
                        directory/'whisper-server.log',args.port,args.threads))
                return server[0]
            start=time.monotonic()
            retry_calls = 0
            skipped_clear = 0
            cache_hits = 0
            for i,ch in enumerate(chunks):
                raw_file=raw_cache/f'chunk_{ch.index:05d}.json'
                render_file=directory/'cache'/raw_file.name
                if raw_file.exists():
                    record=read_json(raw_file)
                    cache_hits += 1
                    if args.phase != 'fast' and record.get('asr_pass') == 'fast':
                        record, did_retry = reuse_and_refine_cached(record, get_server,
                            wav, enhanced, ch, spans, duration, args)
                        retry_calls += int(did_retry)
                        skipped_clear += int(not did_retry)
                        core.atomic_json(raw_file, record)
                        core.atomic_json(render_file, record)
                else:
                    if args.phase == 'refine':
                        raise ValueError(f'缺少第一阶段缓存 {raw_file.name}；请先重新执行 --phase fast')
                    print(f'首轮 {i+1}/{len(chunks)} {core.clock(ch.core_start)}',flush=True)
                    record=asr_record(get_server(),wav,enhanced,ch,spans,duration,args,
                                      fast=(args.phase=='fast'))
                    core.atomic_json(raw_file,record)
                if not render_file.exists(): core.atomic_json(render_file,record)
                # Fast stage writes regular previews; refine previews get rebuilt once at the end.
                if args.phase=='fast' and (i==0 or (i+1)%args.draft_every==0 or i+1==len(chunks)):
                    render_base(directory,duration,wav,args,
                                f'快速初稿 {i+1}/{len(chunks)}；说话人处理中',finalize=False)
                if args.phase=='full' and (i==0 or (i+1)%args.draft_every==0 or i+1==len(chunks)):
                    render_base(directory,duration,wav,args,
                                f'全流程首轮 {i+1}/{len(chunks)}；待说话人和精校',finalize=False)
            metrics['stages_seconds']['first_asr_and_cached_recheck']=round(time.monotonic()-start,3)
            metrics['cache_stats']={'first_pass_cache_hits':cache_hits,
                                    'uncertain_rechecked_chunks':retry_calls,
                                    'clear_skipped_chunks':skipped_clear,
                                    'total_chunks':len(chunks)}
            if args.phase == 'fast':
                # The fast stage has finished ASR. Free the large Whisper model
                # before pyannote allocates its own model and speaker embeddings.
                stack.close()
                server[0] = None
            diar_sig={'provider':args.diarization,'num':args.num_speakers,'min':args.min_speakers,'max':args.max_speakers,
                      'model_path':args.speaker_model_path,'speaker_python':args.speaker_python,
                      'speaker_json':core.fingerprint(Path(args.speaker_json).expanduser()) if args.diarization=='json' else None}
            check_manifest(directory,'speaker_manifest.json',diar_sig)
            os.environ.setdefault('PYANNOTE_METRICS_ENABLED','0')
            diar=timed('diarization',lambda:core.run_speaker_stage(wav,duration,directory,args))
            if args.phase == 'fast':
                # Stage one is complete, even if optional neural dependencies are unavailable.
                render_base(directory,duration,wav,args,'第一阶段已完成：快速初稿及说话人编号；待疑难精校',finalize=False)
                suggest_args = args
                # Never load the translation model during the fast stage.
                original_bilingual = suggest_args.bilingual
                try:
                    if suggest_args.bilingual == 'translate': suggest_args.bilingual = 'suggest'
                    candidates = process_bilingual(directory,suggest_args)
                finally:
                    suggest_args.bilingual = original_bilingual
                core.atomic_json(directory/'bilingual_results.json',candidates)
                overlaps = [{**r, 'id':f'ov{i:05d}', 'status':'detected_not_separated'}
                            for i,r in enumerate(audio_tools.overlap_regions(diar.get('turns', []),duration))]
                core.atomic_json(directory/'separation_results.json',overlaps)
                bilingual.append_support(directory,candidates,overlaps,
                   '第一阶段初稿已完成；双语与抢话仅标记候选，尚未精校')
                core.atomic_json(directory/'fast_stage_complete.json',
                    {'version':VERSION, 'chunks':len(chunks), 'audio':core.fingerprint(audio),
                     'completed_at_unix':time.time()})
                metrics['status']='fast_completed'
                print(f'快速初稿已完成：{directory / "all_clean.txt"}。精校请用同一目录执行 --phase refine',flush=True)
                return 0
            if args.phase == 'refine':
                check_manifest(directory,'refinement_manifest.json',
                    {'version':VERSION, 'recheck':args.recheck,'turn_refine':args.turn_refine,
                     'min_turn_duration':args.min_turn_duration,
                     'max_refine_turns':args.max_refine_turns,'prompt':args.prompt,
                     'model':core.fingerprint(Path(args.model).expanduser())})
            refine_sig=bilingual.digest([diar,args.turn_refine,args.min_turn_duration,args.max_refine_turns,VERSION])
            start=time.monotonic()
            for ch in chunks:
                file=directory/'cache'/f'chunk_{ch.index:05d}.json'; record=read_json(file)
                if record.get('refine_signature')==refine_sig: continue
                original=read_json(raw_cache/file.name)
                doc={'segments':original['segments']}
                needs=(args.turn_refine!='none' and speakers.needs_turn_refine(
                    diar,original['segments'],ch.core_start,ch.core_end,args.turn_refine))
                if needs:
                    doc,accepted,note=refine_nonoverlap(get_server(),wav,ch,original,diar,args,duration)
                else: accepted,note=False,'no_refine_needed'
                result={**original,'segments':doc['segments'],'refine_signature':refine_sig,
                        'speaker_refined':accepted,'speaker_refine_note':note}
                core.atomic_json(file,result)
            # Recover original VAD omissions, not just mark them as missing.
            records=[s for file in (directory/'cache').glob('chunk_*.json') for s in read_json(file).get('segments',[])]
            rescue=missing_spans(diar,records)
            for index,turn in enumerate(rescue):
                ident='rescue_'+bilingual.digest(turn)[:16]
                file=directory/'cache'/('chunk_'+ident+'.json')
                if file.exists(): continue
                ch=core.Chunk(900000+index,turn['start'],turn['end'],max(0,turn['start']-.20),min(duration,turn['end']+.20))
                result=asr_record(get_server(),wav,enhanced,ch,spans,duration,args)
                result['review']=True; result['method']='diarization_vad_rescue+'+result['method']
                for s in result['segments']: s['speaker']=turn['speaker']
                core.atomic_json(file,result)
            metrics['stages_seconds']['refine_and_rescue']=round(time.monotonic()-start,3)
            render_base(directory,duration,wav,args,'主转录及说话人处理完成；正在处理补充候选',finalize=True)
            if args.separate_overlap == 'sepformer':
                # SepFormer and Whisper together can exceed Intel Mac RAM.
                # SepFormer runs first; the Whisper server is lazily relaunched
                # afterwards only if separated tracks actually need recognition.
                stack.close()
                server[0] = None
            separated=timed('overlap_separation',lambda:process_separation(wav,diar,duration,directory,args,get_server))
            core.atomic_json(directory/'separation_results.json',separated)
        # Release Whisper before loading translation weights.
        translations=timed('bilingual',lambda:process_bilingual(directory,args))
        core.atomic_json(directory/'bilingual_results.json',translations)
        enhancement_errors=sum(r.get('status')=='error' for r in translations+separated)
        final_stage=('主转录完成；部分增强项失败，请查看复核清单' if enhancement_errors
                     else '已完成本次所选处理；请核验标记内容')
        bilingual.append_support(directory,translations,separated,final_stage)
        metrics['status']='completed_with_enhancement_errors' if enhancement_errors else 'completed'
        if args.phase == 'refine':
            core.atomic_json(directory/'refine_stage_complete.json',
                {'version':VERSION,'audio':core.fingerprint(audio),'completed_at_unix':time.time(),
                 'enhancement_errors':enhancement_errors})
        metrics['enhancement_errors']=enhancement_errors
        metrics['unprocessed_overlap_regions']=sum(r.get('status')!='separated' for r in separated)
        print(f'完成：{directory / "all_clean.txt"}',flush=True)
        print(f'原文与修改记录：attributed_segments.json；性能：performance.json；候选：review_queue.json',flush=True)
        return 0
    except KeyboardInterrupt:
        metrics['status']='interrupted';print('已保留缓存和初稿；重复相同命令可续跑。',file=sys.stderr);return 130
    except Exception as exc:
        metrics['status']='failed';metrics['error']=f'{type(exc).__name__}: {exc}'
        print(f'执行失败：{exc}\n已完成的初稿与缓存不会删除：{directory}',file=sys.stderr)
        return 2
    finally:
        metrics['total_elapsed_seconds']=round(time.monotonic()-began,3)
        core.atomic_json(directory/'performance.json',metrics)
        if args.phase in ('fast','refine'):
            core.atomic_json(directory/f'performance_{args.phase}.json',metrics)
        core.atomic_json(directory/'pipeline_status.json',{'version':VERSION,'stage':metrics['status'],
                                                          'updated_at_unix':time.time()})


if __name__=='__main__':
    sys.exit(main())
