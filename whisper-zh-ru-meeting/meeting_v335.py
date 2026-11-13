#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3.3.5: sequential S labels, bilingual assistance, overlap-only detection/separation.

Never calls full speaker diarization or embedding/clustering. Reuses V3.3 ASR caches.
Sxx are ASR segment numbers, not recurring speaker identities.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import bilingual_support as bilingual
import disfluency_cleanup as cleanup
import meeting_audio
import speaker_review as review
import transcript_guard as guard
import whisper_zh_ru_v3 as core
import whisper_zh_ru_v33 as old_pipeline

VERSION='3.3.5'
DEFAULT_FOLDER='~/whisper-ru/output/meeting_v33_two_phase'


def read(path, default=None):
    p=Path(path)
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else default


def backup_original(folder):
    for name in ('all_clean.txt','attributed_segments.json','review_queue.json'):
        original=folder/name
        backup=folder/(name+'.before_v335')
        if original.is_file() and not backup.exists():
            shutil.copy2(original,backup)


def cache_info(folder):
    meta=read(folder/'render_meta.json')
    plan=read(folder/'chunk_plan.json')
    if not isinstance(meta,dict) or not isinstance(plan,list) or not plan:
        raise RuntimeError('缺少 render_meta.json 或 chunk_plan.json；必须先有原来的完整Whisper缓存')
    wav=Path(meta.get('wav','')).expanduser()
    audio=Path(meta.get('audio','')).expanduser()
    if not wav.is_file() or not audio.is_file():
        raise RuntimeError('原录音或16kHz WAV不存在；请检查旧目录，不要用其他录音替换')
    am=read(folder/'audio_manifest.json',{})
    if am.get('audio') and am['audio']!=core.fingerprint(audio):
        raise RuntimeError('原录音文件与首轮缓存指纹不一致；停止以防混合不同录音')
    files=[]
    seen=set()
    for ch in plan:
        idx=int(ch['index'])
        if idx in seen:raise RuntimeError('chunk_plan 包含重复索引')
        seen.add(idx)
        entry=folder/'asr_cache'/f'chunk_{idx:05d}.json'
        if not entry.is_file():
            # Older caches may contain only /cache; still use them without rerunning ASR.
            entry=folder/'cache'/f'chunk_{idx:05d}.json'
        doc=read(entry)
        if not isinstance(doc,dict) or not isinstance(doc.get('segments'),list):
            raise RuntimeError(f'缺少首轮缓存：{entry}；不会用未完成数据伪装完整转录')
        files.append((entry,doc))
    return meta,files


def render(folder, mode='smart'):
    meta,docs=cache_info(folder)
    backup_original(folder)
    originals=[]
    for _,doc in docs:
        for segment in doc['segments']:
            if not isinstance(segment,dict):continue
            item=dict(segment)
            if not isinstance(item.get('text'),str) or 'start' not in item or 'end' not in item:continue
            item['asr_review']=bool(doc.get('review'))
            item['refine_review']=False
            # The cached speaker label is deliberately discarded. It was never
            # a reliable person identity for this no-voiceprint workflow.
            item.pop('speaker',None)
            originals.append(item)
    originals=core.deduplicate_anchored(originals)
    overrides=[]
    for a,b,sp,text in review._load_overrides(folder/'manual_overrides.json'):
        # Keep manually corrected WORDING, not old speaker identities.
        overrides.append((a,b,None,text))
    records=review.apply_overrides(originals,overrides)
    records.sort(key=lambda r:(float(r['start']),float(r['end'])))
    out=[];excluded=[];audit=[];queue=[]
    for item in records:
        a=float(item['start']); b=float(item['end'])
        if not (0<=a<b<=float(meta['duration'])+.01):continue
        raw=str(item.get('text','')).strip()
        clean,edits=(raw,[]) if item.get('manual_text') else cleanup.clean_text(raw,mode)
        checked,reason=guard.filter_display_text(clean,bool(item.get('manual_text')))
        if edits:
            audit.append({'start':a,'end':b,'original':raw,'cleaned':clean,'edits':edits})
        if not checked.strip():
            if reason:
                excluded.append({'start':a,'end':b,'raw_text':raw,'reason':reason})
            continue
        copy={**item,'start':a,'end':b,'speaker':None,'asr_text':raw,
              'raw_text':raw,'cleaned_text':checked,'cleanup_edits':edits,
              'speaker_candidates':[],'speaker_flags':[],
              'display_speaker':f'S{len(out)+1:02d}',
              'display_speaker_basis':'sequential_asr_segment_not_a_person'}
        out.append(copy)
        notes=[]
        if copy.get('asr_review'):notes.append('首轮识别标记疑点')
        if float(copy.get('avg_logprob',-1.2))<-.9:notes.append('ASR平均对数概率较低')
        if reason:notes.append(reason)
        if notes:
            queue.append({'start_seconds':a,'end_seconds':b,'id':copy['display_speaker'],
                          'text':checked,'reasons':notes})
    duration=float(meta['duration'])
    lines=['中俄双语会议转录｜Whisper V3.3.5',
           '说明：S01、S02……是转写片段顺序编号，不是人物身份或声纹聚类。',
           '说明：正文为原声识别并作轻度去磕绊；互译和抢话分离候选附于文末，不冒充原话。',
           '说明：移除内容与清理记录单独保存，可追溯原始识别。','']
    for n in range(max(1,int((duration+1799)//1800))):
        start=n*1800;end=min((n+1)*1800,duration)
        lines.extend([f'第{n+1}部分 | {core.clock(start)}—{core.clock(end)}',''])
        for r in out:
            if int((r['start']+r['end'])/2//1800)!=n:continue
            lines.append(f"[{review.stamp(r['start'])}–{review.stamp(r['end'])}] {r['display_speaker']}：{r['cleaned_text']}")
        lines.append('')
    lines.append('— 会议文字主体结束 —')
    core.atomic_json(folder/'v335_attributed_segments.json',out)
    core.atomic_json(folder/'v335_excluded_segments.json',excluded)
    core.atomic_json(folder/'v335_cleanup_audit.json',audit)
    core.atomic_json(folder/'v335_review_queue.json',queue)
    core.atomic_text(folder/'all_clean.txt','\n'.join(lines)+'\n')
    print(f'未执行 Whisper / 未执行声纹识别：复用 {len(docs)} 个首轮窗口，输出 {len(out)} 个顺序编号片段。',flush=True)
    return out


def make_translation_jobs(folder, records, limit):
    # Prioritize user-confirmed correspondences, then temporal candidate links,
    # then standalone translations of clear segments in both languages.
    pair_docs=bilingual.propose_pairs(records,[],window=90.)
    pair_docs += bilingual.load_manual_links(folder,records)
    pair_docs.sort(key=lambda x:(not bool(x.get('source_verified') and x.get('pair_confirmed')),
                                 x.get('gap_seconds',9999),x['target_start']))
    jobs=[];used=set()
    for p in pair_docs:
        if p.get('disabled') or not p.get('source_text','').strip():continue
        key=(p['source_text'],p['source_language'],p['target_language'])
        if key in used:continue
        used.add(key)
        jobs.append(dict(p))
    for r in records:
        text=r.get('raw_text','').strip(); src=bilingual.language_of(text)
        if src not in ('zh','ru') or len(text)<5:continue
        if r.get('asr_review') or float(r.get('avg_logprob',-2))<-.95:continue
        dst='ru' if src=='zh' else 'zh'
        key=(text,src,dst)
        if key in used:continue
        used.add(key)
        signature=bilingual.digest([r['start'],r['end'],text,src,dst])
        jobs.append({'id':'v335_tr_'+signature[:16],'input_hash':signature,
                     'source_start':r['start'],'source_end':r['end'],
                     'target_start':r['start'],'target_end':r['end'],
                     'source_speaker':None,'target_speaker':None,
                     'source_language':src,'target_language':dst,
                     'source_text':text,'target_asr_text':'',
                     'source_ids':[r['display_speaker']],'target_ids':[],
                     'source_verified':bool(r.get('manual_text')),
                     'pair_confirmed':False,
                     'pairing_method':'standalone_translation_not_spoken_counterpart',
                     'status':'candidate', 'gap_seconds':0,
                     'warnings':bilingual.sensitive_warnings(text)})
    if limit>0:
        selected=jobs[:limit]; deferred=[{**item,'status':'deferred_translation_budget'} for item in jobs[limit:]]
    else:
        selected=jobs;deferred=[]
    return selected,deferred


def do_translate(args,folder,records):
    selected,deferred=make_translation_jobs(folder,records,args.max_translations)
    cache=folder/'v335_translation_cache';cache.mkdir(exist_ok=True)
    done=[];pending=[]
    for job in selected:
        item=read(cache/(job['id']+'.json'))
        if (isinstance(item,dict) and item.get('input_hash')==job['input_hash']
                and item.get('status')=='translated' and item.get('model')=='facebook/m2m100_418M'):
            done.append(item)
        else:pending.append(job)
    print(f'双语互译：已缓存 {len(done)}，需要新翻译 {len(pending)}；延期 {len(deferred)}。',flush=True)
    if pending:
        request={'jobs':pending,'model':'facebook/m2m100_418M',
                 'output_dir':str(cache),'threads':args.threads}
        proxy=old_pipeline.parser().parse_args([str(Path(read(folder/'render_meta.json')['audio']))])
        proxy.enhancement_python=str(Path(args.enhancement_python).expanduser().absolute())
        proxy.enhancement_threads=args.threads
        response=old_pipeline.run_worker('translate',request,folder,proxy)
        done.extend(response['results'])
    result=done+deferred
    core.atomic_json(folder/'v335_translation_results.json',result)
    return result


def do_detect(args,folder):
    meta,_=cache_info(folder)
    wav=Path(meta['wav']).expanduser().absolute()
    command=[str(Path(args.speaker_python).expanduser().absolute()),'-u',
             str(Path(__file__).with_name('overlap_only_backend.py')),
             '--wav',str(wav),'--output',str(folder/'v335_overlap_detected.json'),
             '--threads',str(args.threads),'--window',str(args.scan_window),
             '--min-duration',str(args.min_overlap)]
    if args.limit_scan_seconds>0:command+=['--limit-seconds',str(args.limit_scan_seconds)]
    logpath=folder/'v335_overlap_scan.log'
    print('开始单独检测多人同时讲话。不会运行说话人聚类或声纹提取。',flush=True)
    with logpath.open('a',encoding='utf-8') as log:
        completed=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,encoding='utf-8',errors='replace')
        assert completed.stdout is not None
        for line in completed.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        rc=completed.wait()
    if rc:
        raise RuntimeError(f'重叠检测退出码 {rc}，详情见 {logpath}')
    return read(folder/'v335_overlap_detected.json',[])


def validate_manual_overlaps(folder,duration):
    path=folder/'v335_manual_overlap.json'
    if not path.exists():core.atomic_json(path,[])
    data=read(path,[])
    if not isinstance(data,list):raise ValueError('v335_manual_overlap.json 必须是数组')
    out=[]
    for i,x in enumerate(data):
        a=float(x['start']);b=float(x['end'])
        if x.get('confirmed') is not True:continue
        if not (0<=a<b<=duration):raise ValueError(f'人工抢话区间 {i+1} 超出录音范围')
        out.append({'start':a,'end':b,'evidence':'manual_confirmed_overlap','status':'manual_confirmed'})
    return out


def no_identity_diar(segments):
    # Internal two-stream routing labels only. They are not voice IDs, S IDs,
    # or real-world speaker identities.
    turns=[]
    for span in segments:
        turns.append({'start':span['start'],'end':span['end'],'speaker':'TEMP_TRACK_A'})
        turns.append({'start':span['start'],'end':span['end'],'speaker':'TEMP_TRACK_B'})
    return {'turns':turns,'speaker_ids':[],'engine':'overlap_only_no_voice_id'}


def do_separate(args,folder):
    meta,_=cache_info(folder)
    duration=float(meta['duration'])
    detected=read(folder/'v335_overlap_detected.json',[])
    if not isinstance(detected,list):raise RuntimeError('重叠检测文件格式错误')
    manual=validate_manual_overlaps(folder,duration)
    regions=[]
    for s in detected+manual:
        if s.get('status','').startswith('error'):continue
        a=max(0,float(s['start']));b=min(duration,float(s['end']))
        if b-a>=args.min_overlap:regions.append({'start':a,'end':b})
    regions=meeting_audio.overlap_regions(no_identity_diar(regions)['turns'],duration)
    if not regions:
        print('未发现重叠语音候选；可在 v335_manual_overlap.json 中补充确知的抢话时间。',flush=True)
        core.atomic_json(folder/'v335_separation_results.json',[])
        return []
    if args.max_separation_seconds:
        print(f'两人抢话分离：最多处理 {args.max_separation_seconds:g} 秒；0 表示不设预算。',flush=True)
    else:print('两人抢话分离：处理所有检测到的候选区间（CPU 耗时可能较长）。',flush=True)
    diar=no_identity_diar(regions)
    wav=Path(meta['wav'])
    proxy=old_pipeline.parser().parse_args([str(meta['audio'])])
    proxy.enhancement_python=str(Path(args.enhancement_python).expanduser().absolute())
    proxy.enhancement_threads=args.threads
    proxy.separation_max_seconds=args.max_separation_seconds
    proxy.separate_overlap='sepformer'
    proxy.beam_size=5
    proxy.language='auto'
    proxy.prompt=read(folder/'asr_manifest.json',{}).get('prompt','')
    proxy.threads=args.whisper_threads
    server=[None]
    with ExitStack() as stack:
        def get_server():
            if server[0] is None:
                server[0]=stack.enter_context(core.LocalWhisperServer(
                    Path(proxy.server_binary).expanduser(),Path(proxy.model).expanduser(),
                    folder/'whisper-server-v335.log',proxy.port,proxy.threads))
            return server[0]
        results=old_pipeline.process_separation(wav,diar,duration,folder,proxy,get_server)
    for item in results:
        item['speaker_identity_note']='不执行声纹识别；音轨1、音轨2仅在本抢话区间内有效'
    core.atomic_json(folder/'v335_separation_results.json',results)
    return results


def assemble(folder,records,translations=None,separations=None):
    if translations is None:translations=read(folder/'v335_translation_results.json',[])
    if separations is None:separations=read(folder/'v335_separation_results.json',[])
    text=(folder/'all_clean.txt').read_text(encoding='utf-8')
    # always called immediately after `render`, so output is idempotent
    sections=[text.rstrip(),'']
    if translations:
        sections.extend(['=== 中俄双向辅助翻译｜非会议原声逐字稿 ===',''])
        for item in sorted(translations,key=lambda x:(x.get('source_start',0),x.get('id',''))):
            src=item['source_language'];dst=item['target_language']
            where=f"{review.stamp(item['source_start'])}–{review.stamp(item['source_end'])}"
            sections.append(f'[{where}] {src} → {dst}｜{item.get("status","unknown")}')
            sections.append('源文：'+item.get('source_text',''))
            if item.get('status')=='translated':sections.append('辅助译文：'+item.get('translation',''))
            if item.get('pairing_method')!='standalone_translation_not_spoken_counterpart':
                sections.append('关联不清楚的发言区间：'
                     +f"{review.stamp(item['target_start'])}–{review.stamp(item['target_end'])}")
                sections.append('注意：相邻片段的语义关系未经自动验证；译文不代替对方原话。')
            sections.append('')
    if separations:
        sections.extend(['=== 同时讲话分离转写｜双音轨候选，需复听 ===',''])
        for item in separations:
            if 'start' not in item:continue
            where=f"{review.stamp(item['start'])}–{review.stamp(item['end'])}"
            sections.append(f'[{where}] {item.get("status","pending")}')
            for track in item.get('tracks',[]):
                sections.append(f"  音轨{track.get('stream')}（不是S编号，也不是身份）：")
                for seg in track.get('segments',[]):
                    sections.append(f"    [{review.stamp(seg['start'])}–{review.stamp(seg['end'])}] {seg['text']}")
                if track.get('duplicate_other_track'):sections.append('    注意：两音轨文字高度相似，可能未分开。')
                sections.append('    分离音频：'+track.get('path',''))
            if item.get('status')!='separated':sections.append('  此区间尚未取得双路转写；保留复听任务。')
            sections.append('')
    core.atomic_text(folder/'all_clean.txt','\n'.join(sections).rstrip()+'\n')
    core.atomic_json(folder/'v335_summary.json',{'version':VERSION,'ASR_segments':len(records),
                    'translation_items':len(translations),'separation_items':len(separations),
                    'voiceprint_clustering':False,'identity_detection':False,
                    'last_updated_unix':time.time()})
    print('完成：'+str(folder/'all_clean.txt'),flush=True)


def show_status(folder):
    meta,docs=cache_info(folder)
    display=read(folder/'v335_attributed_segments.json',[])
    detected=read(folder/'v335_overlap_detected.json',[])
    separated=read(folder/'v335_separation_results.json',[])
    translated=read(folder/'v335_translation_results.json',[])
    print(json.dumps({'首轮Whisper窗口数':len(docs),'录音秒数':meta.get('duration'),
                      '顺序S编号段数':len(display),'自动发现疑似抢话':len(detected),
                      '抢话分离结果':len(separated),'辅助译文数':len(translated),
                      '执行人物声纹聚类':False,'输出目录':str(folder)},ensure_ascii=False,indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description='V3.3.5 不做声纹：S段编号 + 中俄翻译 + 同时说话检测/分离')
    parser.add_argument('mode',choices=('status','render','translate','detect','separate','all'),
                        help='render仅整理95段缓存；detect独立检测抢话；all执行全部选定增强')
    parser.add_argument('--output-dir',default=DEFAULT_FOLDER)
    parser.add_argument('--speaker-python',default='~/whisper-ru/diarization-venv/bin/python',
                        help='只用于重叠语音检测pyannote segmentation；不会执行声纹聚类')
    parser.add_argument('--enhancement-python',default='~/whisper-ru/enhancement-venv/bin/python')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--whisper-threads',type=int,default=2)
    parser.add_argument('--cleanup-mode',choices=('off','light','smart'),default='smart')
    parser.add_argument('--max-translations',type=int,default=20,help='默认20项；0表示全部可用候选')
    parser.add_argument('--max-separation-seconds',type=float,default=0.,help='默认0=所有检测到的抢话区间')
    parser.add_argument('--min-overlap',type=float,default=.18)
    parser.add_argument('--scan-window',type=float,default=20.,help='默认每20秒单独扫描并缓存')
    parser.add_argument('--limit-scan-seconds',type=float,default=0.,help='仅调试使用，0扫描全段录音')
    parser.add_argument('--manual-only',action='store_true',help='仅使用已确认的抢话时间，不运行自动重叠检测')
    parser.add_argument('--skip-translation',action='store_true')
    parser.add_argument('--skip-separation',action='store_true')
    parser.add_argument('--strict',action='store_true',help='可选增强报错时停止；否则保留已有文稿')
    args=parser.parse_args(argv)
    if args.threads<1 or args.whisper_threads<1 or args.max_translations<0 or args.max_separation_seconds<0:
        parser.error('线程数、翻译预算或分离预算不合法')
    folder=Path(args.output_dir).expanduser().absolute()
    try:
        if args.mode=='status':show_status(folder);return 0
        records=render(folder,args.cleanup_mode)
        if args.mode=='render':assemble(folder,records);return 0
        problems=[]
        if args.mode in ('translate','all') and not args.skip_translation:
            try:do_translate(args,folder,records)
            except Exception as exc:
                if args.strict:raise
                problems.append(f'互译未完成：{type(exc).__name__}: {exc}')
        if args.mode in ('detect','all','separate') and not args.skip_separation:
            # Automatic overlap detection is independent of speaker identity.
            # Manual-only mode can process known intervals without OSD inference.
            previous=read(folder/'v335_overlap_detected_meta.json',{})
            expected=float(read(folder/'render_meta.json')['duration'])
            if args.limit_scan_seconds>0:expected=min(expected,args.limit_scan_seconds)
            sig=previous.get('signature',{}) if isinstance(previous,dict) else {}
            already_scanned=((folder/'v335_overlap_detected.json').is_file()
                             and abs(float(previous.get('coverage_seconds',-1))-expected)<.01
                             and sig.get('window')==args.scan_window
                             and sig.get('min_duration')==args.min_overlap)
            if not already_scanned and not args.manual_only:
                try:do_detect(args,folder)
                except Exception as exc:
                    if args.strict:raise
                    problems.append(f'抢话检测未完成：{type(exc).__name__}: {exc}')
            if args.mode in ('all','separate'):
                try:do_separate(args,folder)
                except Exception as exc:
                    if args.strict:raise
                    problems.append(f'抢话分离未完成：{type(exc).__name__}: {exc}')
        assemble(folder,records)
        if problems:
            core.atomic_json(folder/'v335_processing_issues.json',problems)
            print('警告：增强尚未全部完成：'+'；'.join(problems),file=sys.stderr)
            return 1
        return 0
    except KeyboardInterrupt:
        print('已中断。首轮转录和已完成的增强缓存仍保留。',file=sys.stderr);return 130
    except Exception as exc:
        print(f'V3.3.5 失败：{type(exc).__name__}: {exc}',file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
