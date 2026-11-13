import json
import os
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0,str(Path(__file__).parent))
import meeting_v335 as a
import overlap_only_backend as b
import whisper_zh_ru_v3 as core
import meeting_audio
import privacy_defaults


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.mp3=self.folder/'audio.mp3';self.mp3.write_bytes(b'stub')
        self.wav=self.folder/'original_16k_mono.wav'
        with wave.open(str(self.wav),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
            w.writeframes(b'\0\0'*(16000*14))
        core.atomic_json(self.folder/'render_meta.json',{'audio':str(self.mp3),'wav':str(self.wav),'duration':14.})
        core.atomic_json(self.folder/'chunk_plan.json',[{'index':0},{'index':1}])
        (self.folder/'asr_cache').mkdir()
        (self.folder/'cache').mkdir()
        doc1={'segments':[{'start':.2,'end':1.8,'text':'我我觉得，我们需要确认','avg_logprob':-.3},
                          {'start':2.2,'end':5.1,'text':'请不吝点赞 订阅 转发 打赏支持明镜与点点栏目','avg_logprob':-.4}],
              'review':False,'original_asr':{'segments':[]}}
        doc2={'segments':[{'start':6.,'end':9.,'text':'Мы должны обсудить сроки.','avg_logprob':-.3},
                          {'start':10.,'end':12.,'text':'好的，继续讨论。','avg_logprob':-.2}],
              'review':False,'original_asr':{'segments':[]}}
        core.atomic_json(self.folder/'asr_cache'/'chunk_00000.json',doc1)
        core.atomic_json(self.folder/'asr_cache'/'chunk_00001.json',doc2)
        core.atomic_text(self.folder/'all_clean.txt','Previous draft\n')
        core.atomic_json(self.folder/'speaker_diarization.json',{'speaker_ids':['S01','S02'],'turns':[{'start':0,'end':13,'speaker':'S01'}]})
        core.atomic_json(self.folder/'manual_overrides.json',[])
    def test_render_without_voice_clustering(self):
        records=a.render(self.folder)
        self.assertEqual([x['display_speaker'] for x in records],['S01','S02','S03'])
        self.assertTrue(all(x.get('speaker') is None for x in records))
        txt=(self.folder/'all_clean.txt').read_text()
        self.assertIn('S01：',txt)
        self.assertNotIn('明镜与点点',txt)
        self.assertIn('不是人物身份',txt)
        self.assertIn('Previous draft',(self.folder/'all_clean.txt.before_v335').read_text())
        self.assertEqual(len(a.read(self.folder/'v335_excluded_segments.json')),1)
    def test_translation_jobs_from_clear_zh_ru(self):
        records=a.render(self.folder)
        jobs,delayed=a.make_translation_jobs(self.folder,records,0)
        self.assertTrue(any(x['source_language']=='zh' and x['target_language']=='ru' for x in jobs))
        self.assertTrue(any(x['source_language']=='ru' and x['target_language']=='zh' for x in jobs))
        self.assertFalse(delayed)
    def test_original_cache_not_modified(self):
        old=(self.folder/'asr_cache'/'chunk_00000.json').read_bytes()
        a.render(self.folder)
        self.assertEqual(old,(self.folder/'asr_cache'/'chunk_00000.json').read_bytes())
    def test_display_recovers_translation(self):
        records=a.render(self.folder)
        result=[{'id':'x','source_language':'zh','target_language':'ru',
                 'source_start':.2,'source_end':1.8,'source_text':'我们需要确认',
                 'target_start':.2,'target_end':1.8,
                 'pairing_method':'standalone_translation_not_spoken_counterpart',
                 'status':'translated','translation':'Нужно подтвердить.'}]
        core.atomic_json(self.folder/'v335_translation_results.json',result)
        a.assemble(self.folder,records)
        text=(self.folder/'all_clean.txt').read_text()
        self.assertIn('Нужно подтвердить.',text)
        self.assertIn('非会议原声逐字稿',text)
        # idempotent when re-rendering
        a.render(self.folder);a.assemble(self.folder,records)
        self.assertEqual((self.folder/'all_clean.txt').read_text().count('Нужно подтвердить.'),1)
    def test_no_identity_turns_for_overlap(self):
        doc=a.no_identity_diar([{'start':2,'end':4}])
        self.assertFalse(doc['speaker_ids'])
        self.assertEqual(len(doc['turns']),2)
        windows=meeting_audio.overlap_regions(doc['turns'],14.)
        self.assertEqual(windows[0]['start'],2)
        self.assertEqual(windows[0]['end'],4)
    def test_merge_overlaps(self):
        windows=b.merge_windows([{'start':1,'end':2},{'start':2.05,'end':2.5},
                                 {'start':8,'end':8.05}],14.)
        self.assertEqual(len(windows),1)
        self.assertEqual(windows[0]['end'],2.5)
    def test_manual_overlap(self):
        core.atomic_json(self.folder/'v335_manual_overlap.json',
                         [{'start':2.5,'end':5.0,'confirmed':True}])
        result=a.validate_manual_overlaps(self.folder,14)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['evidence'],'manual_confirmed_overlap')
    def test_separate_uses_no_voice_embeddings(self):
        core.atomic_json(self.folder/'v335_overlap_detected.json',[
          {'start':4.0,'end':5.5,'status':'detected_needs_audio_verification'}])
        args=type('Cfg',(),{
            'min_overlap':.18,'max_separation_seconds':0.,'enhancement_python':'~/whisper-ru/enhancement-venv/bin/python',
            'threads':2,'whisper_threads':2})()
        from unittest.mock import MagicMock
        def fake_process(wav,diar,duration,folder,proxy,get_server):
            self.assertEqual(diar['speaker_ids'],[])
            self.assertEqual(diar['engine'],'overlap_only_no_voice_id')
            self.assertEqual({t['speaker'] for t in diar['turns']},{'TEMP_TRACK_A','TEMP_TRACK_B'})
            return [{'start':4.,'end':5.5,'id':'ov00000','status':'detected_not_separated'}]
        with patch.object(a.old_pipeline,'process_separation',side_effect=fake_process):
            result=a.do_separate(args,self.folder)
        self.assertEqual(len(result),1)
        self.assertIn('不会',result[0]['speaker_identity_note']) if '不会' in result[0]['speaker_identity_note'] else self.assertIn('不执行声纹',result[0]['speaker_identity_note'])

    def test_skip_full_speaker_backend(self):
        # Existing on-disk full diarization must never be consulted to produce S IDs.
        with patch.object(core,'run_speaker_stage',side_effect=AssertionError('no speaker pipeline')):
            records=a.render(self.folder)
            self.assertTrue(all(r['speaker'] is None for r in records))

    def test_status(self):
        a.render(self.folder)
        self.assertEqual(a.main(['status','--output-dir',str(self.folder)]),0)
    def test_no_whisper_on_render(self):
        with patch.object(core,'LocalWhisperServer',side_effect=AssertionError('should not load whisper')):
            self.assertEqual(a.main(['render','--output-dir',str(self.folder)]),0)
    def test_plan(self):
        windows=b.plan(61,20)
        self.assertEqual(len(windows),4)
        self.assertEqual(windows[-1]['end'],61)

    def test_privacy_defaults_disable_telemetry_and_network(self):
        self.assertEqual(os.environ['DO_NOT_TRACK'],'1')
        self.assertEqual(os.environ['HF_HUB_DISABLE_TELEMETRY'],'1')
        self.assertEqual(os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'],'1')
        self.assertEqual(os.environ['HF_HUB_DISABLE_UPDATE_CHECK'],'1')
        self.assertEqual(os.environ['PYANNOTE_METRICS_ENABLED'],'0')
        self.assertEqual(os.environ['HF_HUB_OFFLINE'],'1')
        self.assertEqual(os.environ['TRANSFORMERS_OFFLINE'],'1')
        with patch.dict(os.environ,{privacy_defaults.DOWNLOAD_OPT_IN:'1'}):
            self.assertTrue(privacy_defaults.model_downloads_allowed())

if __name__=='__main__':unittest.main(verbosity=2)
