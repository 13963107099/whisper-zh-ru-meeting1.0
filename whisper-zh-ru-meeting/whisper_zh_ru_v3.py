#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bilingual Chinese/Russian multi-speaker transcription, optimized for Intel macOS.

Architecture:
  1. macOS afconvert -> mono 16 kHz PCM WAV.
  2. whisper.cpp 'whisper-vad-speech-segments' -> real speech boundaries.
  3. Keep one CPU-only whisper-server alive for the entire job.
  4. Short VAD-aligned overlapped windows, language:auto per window.
  5. Selective re-recognition for uncertain/cross-language spans.
  6. Separate pyannote diarization process, stable temporary speaker identities.
  7. Selective speaker-turn re-ASR and conservative uncertainty markers.
  8. Atomic JSON checkpoints; one main TXT plus human-editable JSON review queue.

The program does not train or fine-tune Whisper; it orchestrates local inference.
Main script uses stdlib; speaker backend uses a separate Python 3.11 venv.
One-time Hugging Face login/model downloads required for diarization;
actual audio inference is local and does not need a cloud API.

Offline unit/integration-tested against mocks. Real diarization/ASR must be tested on Intel Mac.
"""
import argparse
from dataclasses import dataclass
import difflib
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import wave

import speaker_review as speakers
import disfluency_cleanup as disfluency

SAMPLE_RATE = 16000
CHAPTER_SECONDS = 1800
VERSION = "3.0.0"
SPEECH_PATTERN = re.compile(
    r"Speech segment\s+\d+:\s+start\s*=\s*([0-9.]+),\s*end\s*=\s*([0-9.]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Span:
    start: float
    end: float


@dataclass(frozen=True)
class Chunk:
    index: int
    core_start: float
    core_end: float
    audio_start: float
    audio_end: float


def clock(seconds):
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int(seconds // 60) % 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def fingerprint(path):
    p = Path(path)
    st = p.stat()
    return {"path": str(p.resolve()), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def atomic_text(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def convert_audio(audio, outdir):
    converted = outdir / "audio_16k_mono.wav"
    try:
        with wave.open(str(audio), "rb") as wf:
            if (wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getcomptype()) == (1, 2, SAMPLE_RATE, "NONE"):
                return audio
    except (wave.Error, EOFError):
        pass
    if not converted.exists():
        executable = shutil.which("afconvert")
        if not executable:
            raise RuntimeError("缺少 macOS 自带的 afconvert 命令")
        command = [executable, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(audio), str(converted)]
        print("音频转换为 PCM WAV …", flush=True)
        subprocess.run(command, check=True)
    with wave.open(str(converted), "rb") as wf:
        params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getcomptype())
        if params != (1, 2, SAMPLE_RATE, "NONE"):
            raise RuntimeError(f"WAV 格式错误：{params}，应为 16kHz/单声道/PCM16")
    return converted


def wav_duration(wav_path):
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def save_wav_clip(wav_file, destination, start, end):
    """Random-access bounded WAV clip; avoids loading full meeting audio into RAM."""
    with wave.open(str(wav_file), "rb") as src:
        sr = src.getframerate()
        first = max(0, min(src.getnframes(), int(round(start * sr))))
        last = max(first, min(src.getnframes(), int(round(end * sr))))
        src.setpos(first)
        raw = src.readframes(last - first)
    if not raw:
        raise ValueError(f"空音频窗口 {start}..{end}")
    with wave.open(str(destination), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(SAMPLE_RATE)
        dst.writeframes(raw)


def parse_vad_output(output, duration):
    spans = []
    for x, y in SPEECH_PATTERN.findall(output):
        start = max(0.0, float(x))
        end = min(duration, float(y))
        if end > start:
            spans.append(Span(start, end))
    spans.sort(key=lambda x: (x.start, x.end))
    merged = []
    for s in spans:
        if merged and s.start <= merged[-1].end + 0.08:
            old = merged[-1]
            merged[-1] = Span(old.start, max(old.end, s.end))
        else:
            merged.append(s)
    return merged


def detect_speech(vad_binary, vad_model, wav_file, duration, threads, output_dir):
    """Execute Silero VAD once, record raw diagnostics and preserve absolute spans."""
    command = [str(vad_binary), "--vad-model", str(vad_model), "--file", str(wav_file),
               "--threads", str(threads), "--vad-min-silence-duration-ms", "350",
               "--vad-min-speech-duration-ms", "220",
               "--vad-speech-pad-ms", "160", "--no-prints"]
    print("Silero VAD 正在检测语音边界 …", flush=True)
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
    raw = (result.stdout or "") + "\n" + (result.stderr or "")
    (output_dir / "vad_detection.log").write_text(raw, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError("VAD 检测失败，请查看 vad_detection.log：\n" + "\n".join(raw.splitlines()[-15:]))
    spans = parse_vad_output(raw, duration)
    if not spans and not re.search(r"Detected\s+0\s+speech\s+segments", raw, re.I):
        raise RuntimeError("VAD 未返回可解析的语音边界，已停止以防遗漏内容。请查看 vad_detection.log")
    return spans


def plan_chunks(spans, duration, max_core=22.0, overlap=0.45, max_gap=0.7):
    """Build <=~23s clips at VAD boundaries, overlap only as extra context.

    Core spans are disjoint; output ownership is decided by timestamps in core.
    """
    if not (4 <= max_core <= 27):
        raise ValueError("--max-core 必须在 4–27 秒范围")
    if not (0 <= overlap <= 1):
        raise ValueError("--overlap 必须在 0–1 秒范围")
    cores = []
    current_start = current_end = None
    for s in spans:
        t = s.start
        while t < s.end - 1e-6:
            take = min(s.end, t + max_core)
            if current_start is None:
                current_start, current_end = t, take
            elif t - current_end <= max_gap and take - current_start <= max_core + 1e-6:
                current_end = take
            else:
                cores.append(Span(current_start, current_end))
                current_start, current_end = t, take
            t = take
    if current_start is not None:
        cores.append(Span(current_start, current_end))
    return [Chunk(i, c.start, c.end, max(0.0, c.start - overlap), min(duration, c.end + overlap))
            for i, c in enumerate(cores)]


def guess_lang_from_name(v):
    s = str(v or "").strip().lower()
    if s in {"zh", "chinese", "chinese (simplified)", "mandarin"} or "chinese" in s:
        return "zh"
    if s in {"ru", "russian"} or "russian" in s:
        return "ru"
    return s or "unknown"


def parse_asr(data, audio_start, core_start, core_end):
    """Normalize verbose_json and select only timestamps owned by this core."""
    if not isinstance(data, dict):
        raise ValueError("服务器没有返回 JSON 对象")
    doc = {"language": guess_lang_from_name(data.get("language") or data.get("detected_language")),
           "language_probability": data.get("detected_language_probability"),
           "language_probabilities": data.get("language_probabilities") or {},
           "segments": [], "raw_text": data.get("text", "")}
    spans = data.get("segments", [])
    if not isinstance(spans, list):
        raise ValueError("服务器返回的 segments 不是列表")
    for seg in spans:
        txt = str(seg.get("text") or "").strip()
        if not txt:
            continue
        st = audio_start + max(0.0, float(seg.get("start") or 0))
        en = audio_start + max(0.0, float(seg.get("end") or 0))
        if en <= st:
            en = st + 0.01
        center = (st + en) / 2
        if center < core_start - 0.001 or center > core_end + 0.001:
            continue
        avg_raw = seg.get("avg_logprob")
        nos_raw = seg.get("no_speech_prob")
        avg = float(avg_raw) if avg_raw is not None else -1.3
        if not math.isfinite(avg):
            avg = -1.3
        nos = float(nos_raw) if nos_raw is not None else 0.0
        if not math.isfinite(nos):
            nos = 0.0
        doc["segments"].append({
            "start": round(st, 3), "end": round(en, 3), "text": txt,
            "avg_logprob": avg, "no_speech_prob": nos
        })
    # Older response versions might omit segments; do not silently drop text.
    if not doc["segments"] and not spans and doc["raw_text"].strip():
        doc["segments"] = [{"start": core_start, "end": core_end,
                            "text": doc["raw_text"].strip(), "avg_logprob": -1.2,
                            "no_speech_prob": 0.0}]
    return doc


def repeated_hallucination(text):
    # Long repeated same word, or 3+ repeated phrases, tend to indicate a decode loop.
    return (bool(re.search(r"(?iu)(\b[^\W_]{3,}\b)(?:[\s,，。]+\1){3,}", text))
            or bool(re.search(r"([\u4e00-\u9fff]{2,5})\1{3,}", text)))


def script_statistics(text):
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    russian = len(re.findall(r"[\u0400-\u052f]", text))
    return chinese, russian


def text_for(doc):
    return " ".join(s["text"] for s in doc.get("segments", [])).strip()


def logprob(doc):
    segs = doc.get("segments", [])
    if not segs:
        return -3.0
    return sum(s.get("avg_logprob", -1.5) for s in segs) / len(segs)


def quality(doc):
    text = text_for(doc)
    if not text:
        return -10.0
    score = logprob(doc)
    zh, ru = script_statistics(text)
    total = zh + ru
    language = doc.get("language")
    if total:
        if language == "ru":
            score -= 0.8 * zh / total
        elif language == "zh":
            score -= 0.8 * ru / total
    if repeated_hallucination(text):
        score -= 1.8
    if len(text) <= 2:
        score -= 0.8
    return score


def is_uncertain(doc):
    text = text_for(doc)
    zh, ru = script_statistics(text)
    probs = doc.get("language_probabilities", {})
    pzh = float(probs.get("zh", 0) or 0)
    pru = float(probs.get("ru", 0) or 0)
    probability = doc.get("language_probability")
    return (not text or repeated_hallucination(text)
            or doc.get("language") not in {"zh", "ru"}
            or (probability is not None and float(probability) < 0.68)
            or (pzh + pru > .35 and abs(pzh - pru) < .17)
            or (zh >= 3 and ru >= 3)
            or (doc.get("language") == "zh" and ru >= 4 and ru > zh / 2)
            or (doc.get("language") == "ru" and zh >= 4 and zh > ru / 2)
            or (text and logprob(doc) < -1.15))


def conservative_clean(text):
    """Backward-compatible mild cleanup; original ASR text is preserved in JSON."""
    return disfluency.clean_text(text, 'light')[0]


class LocalWhisperServer:
    """Private local server, started once and shut down after job completion."""
    def __init__(self, executable, model, log_path, port, threads):
        self.executable = executable
        self.model = model
        self.log_path = log_path
        self.port = port
        self.threads = threads
        self.proc = None
        self.handle = None

    def __enter__(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"本地端口 {self.port} 已被占用。可指定 --port 8180")
        self.handle = open(self.log_path, "a", encoding="utf-8")
        cmd = [str(self.executable), "-m", str(self.model), "-ng", "-nf", "-sns",
               "-t", str(self.threads), "-mc", "0", "-l", "auto",
               "--host", "127.0.0.1", "--port", str(self.port)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                     stdout=self.handle, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.__exit__(None, None, None)
                raise RuntimeError(f"Whisper server 启动失败，请查看 {self.log_path}")
            try:
                con = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                con.request("GET", "/health")
                response = con.getresponse()
                response.read()
                con.close()
                if response.status == 200:
                    print("Whisper 模型已加载，服务器保持常驻。", flush=True)
                    return self
            except (OSError, TimeoutError):
                pass
            time.sleep(0.5)
        self.__exit__(None, None, None)
        raise RuntimeError(f"Whisper server 初始化超时。请查看 {self.log_path}")

    def __exit__(self, exc_type, exc, tb):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        if self.handle:
            self.handle.close()

    def recognize(self, audio_path, language, beam_size=5, prompt=""):
        boundary = "----local-whisper-" + uuid.uuid4().hex
        def field(name, value):
            return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode("utf-8")
        data = bytearray()
        data.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode())
        data.extend(Path(audio_path).read_bytes())
        data.extend(b"\r\n")
        args = {"response_format": "verbose_json", "language": language,
                "beam_size": str(beam_size), "temperature": "0.0", "temperature_inc": "0.0",
                "token_timestamps": "false", "no_language_probabilities": "false",
                "suppress_non_speech": "true"}
        if prompt:
            args["prompt"] = prompt
        for name, value in args.items():
            data.extend(field(name, value))
        data.extend(f"--{boundary}--\r\n".encode())
        con = http.client.HTTPConnection("127.0.0.1", self.port, timeout=900)
        try:
            con.request("POST", "/inference", body=bytes(data),
                        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                 "Content-Length": str(len(data))})
            res = con.getresponse()
            body = res.read()
            if res.status != 200:
                raise RuntimeError(f"服务器 HTTP {res.status}: {body[:500]!r}")
            data = json.loads(body.decode("utf-8", errors="strict"))
            if not isinstance(data, dict) or "segments" not in data:
                raise RuntimeError("Whisper server 没有返回 verbose_json 结果；请检查编译版本")
            return data
        finally:
            con.close()


def transcribe_chunk(server, wav_path, chunk, beam_size, prompt, language="auto"):
    with tempfile.TemporaryDirectory(prefix="local_asr_") as td:
        clip = Path(td) / "clip.wav"
        save_wav_clip(wav_path, clip, chunk.audio_start, chunk.audio_end)
        result = server.recognize(clip, language, beam_size, prompt)
    return parse_asr(result, chunk.audio_start, chunk.core_start, chunk.core_end)


def split_at_vad_pause(chunk, speeches, duration, overlap):
    """Select a real VAD silence inside the core, not an arbitrary midpoint."""
    possible = []
    in_core = [s for s in speeches if s.end > chunk.core_start and s.start < chunk.core_end]
    for a, b in zip(in_core, in_core[1:]):
        gap_start = max(a.end, chunk.core_start)
        gap_end = min(b.start, chunk.core_end)
        if gap_end - gap_start >= .3:
            cut = (gap_start + gap_end) / 2
            if cut - chunk.core_start >= 3.5 and chunk.core_end - cut >= 3.5:
                possible.append(cut)
    if not possible:
        return None
    middle = (chunk.core_start + chunk.core_end) / 2
    cut = min(possible, key=lambda c: abs(c - middle))
    return (
        Chunk(-1, chunk.core_start, cut, max(0, chunk.core_start-overlap), min(duration, cut+overlap)),
        Chunk(-2, cut, chunk.core_end, max(0, cut-overlap), min(duration, chunk.core_end+overlap)),
    )


def choose_second_pass(server, wav, chunk, first, speeches, duration, args):
    """Only accept an alternate when the measured heuristic margin is material.

    Whisper probability alone is *not* a verified transcript accuracy score.
    Every rechecked span remains tagged for possible listening review.
    """
    if args.language != "auto" or args.recheck == "none":
        return first, "first_pass", False
    if not (args.recheck == "all" or is_uncertain(first)):
        return first, "first_pass", False
    candidates = [("first_pass", first)]
    split = split_at_vad_pause(chunk, speeches, duration, args.overlap)
    # With a genuine switch of scripts but no VAD silence, a fallback midpoint
    # split is tested only as a tentative alternative and always flagged for review.
    zh_chars, ru_chars = script_statistics(text_for(first))
    if split is None and zh_chars >= 3 and ru_chars >= 3 and chunk.core_end - chunk.core_start >= 10:
        cut = (chunk.core_start + chunk.core_end) / 2
        split = (
            Chunk(-1, chunk.core_start, cut, max(0, chunk.core_start - args.overlap), min(duration, cut + args.overlap)),
            Chunk(-2, cut, chunk.core_end, max(0, cut - args.overlap), min(duration, chunk.core_end + args.overlap)),
        )
    if split:
        try:
            left = transcribe_chunk(server, wav, split[0], args.beam_size, args.prompt, "auto")
            right = transcribe_chunk(server, wav, split[1], args.beam_size, args.prompt, "auto")
            merged = {"language": "mixed", "segments": left["segments"] + right["segments"],
                      "raw_text": left.get("raw_text", "") + right.get("raw_text", "")}
            # Both halves are independently scored; prefer split decoding only when supported.
            if quality(merged) >= quality(first) + 0.20 and text_for(merged):
                return merged, "vad_pause_split_auto", True
            candidates.append(("vad_pause_split_auto", merged))
        except Exception as e:
            print(f"警告：细分重识别失败：{e}", file=sys.stderr)
    else:
        # No reliable silence boundary exists inside the chunk. Force candidates on
        # the same audio rather than splitting a word at an arbitrary midpoint.
        for lang in ("zh", "ru"):
            try:
                cand = transcribe_chunk(server, wav, chunk, args.beam_size, args.prompt, lang)
                candidates.append((f"forced_{lang}", cand))
            except Exception as e:
                print(f"警告：{lang} 重识别失败：{e}", file=sys.stderr)
    candidates.sort(key=lambda item: quality(item[1]), reverse=True)
    best_mode, best = candidates[0]
    if best_mode != "first_pass" and quality(best) >= quality(first) + 0.28:
        return best, best_mode, True
    return first, "first_pass_needs_review", True


def deduplicate_anchored(segments):
    """Deduplicate only close, highly similar overlapping ASR segments."""
    result = []
    for x in sorted(segments, key=lambda z: (z["start"], z["end"])):
        if not result:
            result.append(x)
            continue
        prev = result[-1]
        common = max(0.0, min(prev["end"], x["end"]) - max(prev["start"], x["start"]))
        short = max(0.01, min(prev["end"] - prev["start"], x["end"] - x["start"]))
        a = re.sub(r"\s+", "", prev["text"])
        b = re.sub(r"\s+", "", x["text"])
        if (not (prev.get("speaker") and x.get("speaker") and prev["speaker"] != x["speaker"])
                and common / short > 0.5 and difflib.SequenceMatcher(None, a, b).ratio() > 0.92):
            # Keep the one with better logprob.
            if x.get("avg_logprob", -2) > prev.get("avg_logprob", -2):
                result[-1] = x
        else:
            result.append(x)
    return result


def run_speaker_stage(wav, duration, directory, args):
    """Independent local pyannote pass. Results cached before Whisper starts.

    This keeps speaker clustering global over the entire recording (speaker IDs
    cannot be compared across recordings) and preserves one Whisper-server load.
    """
    speaker_file = directory / "speaker_diarization.json"
    if args.diarization == "off":
        if not speaker_file.exists():
            atomic_json(speaker_file, {"schema":1, "engine":"off", "speaker_ids":[],
                                       "turns":[], "exclusive_turns":[], "audio_duration":duration})
        return json.loads(speaker_file.read_text(encoding="utf-8"))
    if speaker_file.exists():
        print("已有说话人分离缓存，直接复用。", flush=True)
        return json.loads(speaker_file.read_text(encoding="utf-8"))
    if args.diarization == "json":
        raw_path = Path(args.speaker_json).expanduser().resolve()
        if not raw_path.is_file():
            raise RuntimeError("--diarization json 需要用 --speaker-json 指定现有时间区间 JSON")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        speaker_python = Path(args.speaker_python).expanduser().absolute()
        if not speaker_python.is_file():
            raise RuntimeError(f"找不到说话人分离 Python 环境：{speaker_python}\n"
                               "请先执行：bash setup_speaker_diarization.sh，或使用 --diarization off")
        backend = Path(__file__).with_name('speaker_diarization_backend.py')
        if not backend.is_file():
            raise RuntimeError(f'缺少 {backend.name}，请保持 ZIP 中三个 Python 文件放在同一个目录')
        raw_path = directory / 'speaker_raw.json'
        cmd = [str(speaker_python), str(backend), '--audio', str(wav),
               '--output', str(raw_path), '--threads', str(args.speaker_threads)]
        if args.num_speakers:
            cmd += ['--num-speakers',str(args.num_speakers)]
        else:
            if args.min_speakers:
                cmd += ['--min-speakers',str(args.min_speakers)]
            if args.max_speakers:
                cmd += ['--max-speakers',str(args.max_speakers)]
        if args.speaker_model_path:
            cmd += ['--model-path',str(Path(args.speaker_model_path).expanduser().resolve())]
        print('正在对整个录音执行本地声纹聚类与发言轮次检测 …', flush=True)
        with (directory / 'speaker_diarization.log').open('a',encoding='utf-8') as log:
            result = subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,
                                    stderr=subprocess.STDOUT,check=False)
        if result.returncode != 0 or not raw_path.is_file():
            raise RuntimeError(f'说话人模型执行失败（退出码 {result.returncode}）；请查看 '
                               f'{directory / "speaker_diarization.log"}。\n'
                               '常见原因是未接受 Hugging Face 模型许可或缺少 HF_TOKEN。')
        raw = json.loads(raw_path.read_text(encoding='utf-8'))
    diar = speakers.normalize_diarization(raw, duration)
    diar['speaker_settings']={'provider':args.diarization,'num':args.num_speakers,
                             'min':args.min_speakers,'max':args.max_speakers}
    atomic_json(speaker_file, diar)
    speakers.init_speaker_files(directory,diar,atomic_json)
    print(f'说话人聚类完成：{len(diar["speaker_ids"])} 个临时发言人编号，'
          f'{len(diar["turns"])} 个语音轮次。', flush=True)
    return diar


def _letters(text):
    return len(re.findall(r'[\w\u0400-\u052f\u4e00-\u9fff]',text,flags=re.UNICODE))


def refine_speaker_turns(server, wav, chunk, doc, diar, args, duration):
    """Rerun only chunks with plausible multi-speaker ASR ambiguity.

    We NEVER assign one sentence to a different speaker merely on language.
    Original text is kept in the JSON cache even when refinement is accepted.
    """
    if args.turn_refine == 'none' or not diar.get('speaker_ids'):
        return doc, False, 'not_requested'
    if not speakers.needs_turn_refine(diar,doc['segments'],chunk.core_start,chunk.core_end,args.turn_refine):
        return doc, False, 'no_multi_speaker_asr_segment'
    turns=speakers.turns_in_range(diar,chunk.core_start,chunk.core_end)
    if len(turns) > args.max_refine_turns:
        return doc, False, 'too_many_turns_need_review'
    if speakers.has_real_overlap(diar,chunk.core_start,chunk.core_end,.14):
        # Simultaneous voices cannot be isolated simply by cropping time.
        return doc, False, 'overlapping_speech_need_review'
    if any(t['end']-t['start'] < args.min_turn_duration for t in turns):
        # Short interjections are better flagged than silently erased.
        return doc, False, 'short_interjection_need_review'
    new=[]
    for turn in turns:
        a,b=turn['start'],turn['end']
        clip=Chunk(-1,a,b,max(0.,a-.22),min(duration,b+.22))
        try:
            r=transcribe_chunk(server,wav,clip,args.beam_size,args.prompt,args.language)
        except Exception as e:
            print(f'说话人细分重识别失败，保留主识别：{e}',file=sys.stderr)
            return doc, False, 'turn_recognition_error'
        if not r.get('segments'):
            return doc, False, 'turn_recognition_empty'
        for entry in r['segments']:
            s=dict(entry)
            s['speaker']=turn['speaker']
            s['turn_refined']=True
            new.append(s)
    oldsize=sum(_letters(s['text']) for s in doc['segments'])
    newsize=sum(_letters(s['text']) for s in new)
    if oldsize >= 5 and not (.55 <= newsize/max(oldsize,1) <= 1.65):
        return doc, False, 'transcript_difference_need_review'
    if not new:
        return doc, False, 'no_refined_text'
    refined=dict(doc)
    refined['segments']=sorted(new,key=lambda x:(x['start'],x['end']))
    return refined, True, 'speaker_turn_reasr'


def render_final(job_dir, duration, wav_path=None, export_clips=False, cleanup_mode='smart'):
    # Text processing is after ASR/diarization, and can be re-run from cache.
    return speakers.build_output(job_dir,duration,
                                 lambda text: disfluency.clean_text(text, cleanup_mode),
                                 clock,deduplicate_anchored,atomic_text,atomic_json,
                                 save_clip=save_wav_clip,wav_path=wav_path,
                                 export_clips=export_clips)


def build_parser():
    p = argparse.ArgumentParser(description="Whisper V3：中俄 VAD+模型常驻+多人说话人分离+人工复核")
    p.add_argument("audio", help="输入 MP3/M4A/WAV 文件")
    p.add_argument("--model", default=str(Path.home()/"whisper-ru/models/ggml-large-v3-turbo-q5_0.bin"))
    p.add_argument("--server-binary", default=str(Path.home()/"whisper-ru/whisper.cpp-server-v1.9.4/build/bin/whisper-server"))
    p.add_argument("--vad-binary", default=str(Path.home()/"whisper-ru/whisper.cpp-server-v1.9.4/build/bin/whisper-vad-speech-segments"))
    p.add_argument("--vad-model", default=str(Path.home()/"whisper-ru/models/ggml-silero-v6.2.0.bin"))
    p.add_argument("--output-dir", default="")
    p.add_argument("--language", choices=["auto", "zh", "ru"], default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--max-core", type=float, default=22, help="VAD 语音分段最长的无重叠主体秒数")
    p.add_argument("--overlap", type=float, default=0.45, help="片段两侧上下文重叠，秒")
    p.add_argument("--recheck", choices=["none", "suspicious", "all"], default="suspicious")
    p.add_argument("--prompt", default="", help="人工核实的术语提示；可能带来偏差")
    p.add_argument("--port", type=int, default=8179, help="localhost 的 Whisper HTTP 服务端口")
    p.add_argument("--limit-seconds", type=float, default=0, help="仅用于小样本测试，例如 90；0 表示全长")
    p.add_argument("--plan-only", action="store_true", help="仅运行 VAD、列出分段，不执行 Whisper")
    p.add_argument('--diarization', choices=['pyannote','json','off'], default='pyannote',
                   help='默认本地 pyannote；json 导入说话人轮次；off 不区分说话人')
    p.add_argument('--speaker-python',default=str(Path.home()/'whisper-ru/diarization-venv/bin/python'),
                   help='独立 PyTorch 虚拟环境解释器')
    p.add_argument('--speaker-json',default='',help='当 --diarization json 时导入 JSON')
    p.add_argument('--speaker-model-path',default='',help='本地下载的 pyannote pipeline 路径（可选）')
    p.add_argument('--speaker-threads',type=int,default=4)
    p.add_argument('--num-speakers',type=int,default=0,help='若确知发言人数可指定，0 为自动估计')
    p.add_argument('--min-speakers',type=int,default=0)
    p.add_argument('--max-speakers',type=int,default=0)
    p.add_argument('--turn-refine',choices=['none','smart','all'],default='smart',
                   help='对含多位发言人的 ASR 长片段额外按说话轮次重新识别；默认 smart')
    p.add_argument('--min-turn-duration',type=float,default=.75,
                   help='短于此值的轮次不单独重识别，自动标复核')
    p.add_argument('--max-refine-turns',type=int,default=18)
    p.add_argument('--cleanup-mode', choices=['off','light','smart'], default='smart',
                   help='去除口吃重复：off 不清理；light 清短重复；smart 加清短句重复起头（默认）')
    p.add_argument('--render-only',action='store_true',help='仅依据缓存及人工 JSON 修改重建 TXT')
    p.add_argument('--export-review-clips',action='store_true',help='将待复听的时间段另存 WAV')
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    if args.threads <= 0 or args.beam_size <= 0:
        p.error("--threads 与 --beam-size 必须大于 0")
    if not 1 <= args.port <= 65535:
        p.error("--port 超出合法范围")
    if not 4 <= args.max_core <= 27 or not 0 <= args.overlap <= 1:
        p.error("--max-core 应为 4–27，--overlap 应为 0–1")
    if (args.speaker_threads<=0 or args.num_speakers<0 or args.min_speakers<0 or args.max_speakers<0
            or args.max_refine_turns<2 or not .4<=args.min_turn_duration<=2.5):
        p.error('说话人参数无效')
    if args.num_speakers and (args.min_speakers or args.max_speakers):
        p.error('--num-speakers 不能与 --min-speakers 或 --max-speakers 同时使用')
    if args.max_speakers and args.min_speakers>args.max_speakers:
        p.error('--min-speakers 不能大于 --max-speakers')
    if args.diarization=='json' and not args.speaker_json:
        p.error('--diarization json 必须同时指定 --speaker-json')
    audio = Path(args.audio).expanduser().resolve()
    model = Path(args.model).expanduser().resolve()
    vad_model = Path(args.vad_model).expanduser().resolve()
    vad_binary = Path(args.vad_binary).expanduser().resolve()
    server_binary = Path(args.server_binary).expanduser().resolve()
    if not audio.is_file():
        p.error(f"找不到音频：{audio}")
    if not args.render_only and (not vad_model.is_file() or vad_model.stat().st_size < 10000):
        p.error(f"找不到 Silero VAD 模型：{vad_model}；请先执行安装脚本")
    if not args.render_only and not vad_binary.is_file():
        p.error(f"找不到 VAD 可执行文件：{vad_binary}；请先执行安装脚本")
    if not args.plan_only and not args.render_only:
        if not model.is_file() or model.stat().st_size < 1000000:
            p.error(f"找不到有效 Whisper GGML 模型：{model}")
        if not server_binary.is_file():
            p.error(f"找不到 whisper-server：{server_binary}；请先执行安装脚本")
    name = re.sub(r"[^\w.\-]", "_", audio.stem, flags=re.UNICODE)[:60] or "audio"
    if args.limit_seconds > 0:
        name += f"_test_{int(args.limit_seconds)}s"
    if args.plan_only:
        name += "_planning"
    directory = (Path(args.output_dir).expanduser().resolve() if args.output_dir
                 else Path.home()/"whisper-ru/output"/(name + "_v3"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cache").mkdir(exist_ok=True)
    if args.render_only:
        meta=directory/'render_meta.json'
        if not meta.exists():
            p.error('找不到先前缓存 render_meta.json，无法仅重建文字；请先完成一次 V3 识别')
        data=json.loads(meta.read_text(encoding='utf-8'))
        if Path(data['audio']).resolve()!=audio:
            p.error('音频文件与已有缓存来源不一致')
        cached_wav=Path(data['wav'])
        if args.export_review_clips and not cached_wav.is_file():
            p.error('复听切片需要音频：缓存 WAV 不存在')
        final, count, reviews, undecided = render_final(directory,float(data['duration']),
                                            cached_wav if cached_wav.exists() else None,args.export_review_clips,
                                            args.cleanup_mode)
        print(f'仅重新生成：{final}；片段 {count}；需人工复核条目 {undecided}')
        return 0
    signature = {"v": VERSION, "audio": fingerprint(audio),
                 "model": fingerprint(model) if model.is_file() else None,
                 "vad_model": fingerprint(vad_model), "vad_binary": fingerprint(vad_binary),
                 "server_binary": fingerprint(server_binary) if server_binary.is_file() else None,
                 "language": args.language, "threads": args.threads, "beam_size": args.beam_size,
                 "max_core": args.max_core, "overlap": args.overlap,
                 "prompt": args.prompt, "recheck": args.recheck, "limit_seconds": args.limit_seconds,
                 "diarization":args.diarization,
                 "speaker_python":args.speaker_python if args.diarization=='pyannote' else '',
                 "speaker_json":fingerprint(args.speaker_json) if args.diarization=='json' else None,
                 "speaker_model_path":args.speaker_model_path,
                 "speaker_threads":args.speaker_threads,"num_speakers":args.num_speakers,
                 "min_speakers":args.min_speakers,"max_speakers":args.max_speakers,
                 "turn_refine":args.turn_refine,"min_turn_duration":args.min_turn_duration,
                 "max_refine_turns":args.max_refine_turns}
    manifest = directory / "job.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != signature:
            p.error(f"该输出目录的配置已改变，请使用新的 --output-dir：{directory}")
    else:
        atomic_json(manifest, signature)
    print(f"录音：{audio}")
    print(f"输出：{directory}")
    wav = convert_audio(audio, directory)
    duration = wav_duration(wav)
    vad_input = wav
    if args.limit_seconds > 0 and args.limit_seconds < duration:
        duration = args.limit_seconds
        vad_input = directory / "test_excerpt.wav"
        if not vad_input.exists():
            save_wav_clip(wav, vad_input, 0.0, duration)
    vad_json = directory / "vad_spans.json"
    if vad_json.exists():
        spans = [Span(x["start"], x["end"]) for x in json.loads(vad_json.read_text(encoding="utf-8"))]
    else:
        spans = detect_speech(vad_binary, vad_model, vad_input, duration, args.threads, directory)
        spans = [Span(s.start, min(s.end, duration)) for s in spans if s.start < duration]
        atomic_json(vad_json, [s.__dict__ for s in spans])
    atomic_json(directory/'render_meta.json',{'audio':str(audio),'wav':str(vad_input),'duration':duration})
    chunks = plan_chunks(spans, duration, args.max_core, args.overlap)
    print(f"总时长：{clock(duration)}；VAD 语音段：{len(spans)}；识别任务：{len(chunks)}；30 分钟章节：{max(1,math.ceil(duration/1800))}")
    if args.plan_only:
        for ch in chunks[:12]:
            print(f"  {ch.index+1:03d} 核心 {ch.core_start:.2f}-{ch.core_end:.2f}s | 识别音频 {ch.audio_start:.2f}-{ch.audio_end:.2f}s")
        print("仅规划模式，无模型推理")
        return 0
    try:
        diar = run_speaker_stage(vad_input,duration,directory,args)
    except Exception as e:
        print(f'说话人识别环节失败：{e}',file=sys.stderr)
        print('已有 ASR 缓存不会被删除；如确需只转文字，请在新的输出目录加 --diarization off。',file=sys.stderr)
        return 3
    if not chunks:
        print('警告：VAD 未识别任何语音。请试听原始音频并检查 vad_detection.log。')
        final, _, _, _ = render_final(directory, duration, vad_input,args.export_review_clips,args.cleanup_mode)
        print(f'已创建空白章节文稿供检查：{final}')
        return 0
    expected = [directory/"cache"/f"chunk_{c.index:05d}.json" for c in chunks]
    if all(x.exists() for x in expected):
        print("所有小段均已缓存，直接重建最终文稿，无需加载模型")
    else:
        with LocalWhisperServer(server_binary, model, directory/"whisper-server.log", args.port, args.threads) as server:
            for i, chunk in enumerate(chunks):
                state_path = expected[i]
                if state_path.exists():
                    print(f"[{i+1}/{len(chunks)}] 已完成，跳过 {clock(chunk.core_start)}")
                    continue
                print(f"[{i+1}/{len(chunks)}] 识别 {clock(chunk.core_start)}–{clock(chunk.core_end)} …", flush=True)
                try:
                    first = transcribe_chunk(server, wav, chunk, args.beam_size, args.prompt, args.language)
                    final_doc, method, rerun = choose_second_pass(server, wav, chunk, first, spans, duration, args)
                    original_segments = [dict(x) for x in final_doc['segments']]
                    # Uncertainty is based on the original bilingual ASR pass.
                    review = (rerun or is_uncertain(final_doc) or method == 'first_pass_needs_review')
                    final_doc, refined, refine_reason = refine_speaker_turns(
                        server,wav,chunk,final_doc,diar,args,duration)
                    if refined:
                        print(f'  已将混合发言重新分段：{len(final_doc["segments"])} 句',flush=True)
                    atomic_json(state_path, {"version": VERSION, "chunk": chunk.index,
                                             "core": [chunk.core_start, chunk.core_end],
                                             "audio": [chunk.audio_start, chunk.audio_end],
                                             "first_language": first.get("language"),
                                             "first_probability": first.get("language_probability"),
                                             "method": method, "rerun": rerun, "review": review,
                                             "speaker_refined":refined,"speaker_refine_note":refine_reason,
                                             "original_segments":original_segments,
                                             "segments": final_doc["segments"]})
                except KeyboardInterrupt:
                    print("用户中断，已完成结果保存在 cache/；同样命令可续跑", file=sys.stderr)
                    return 130
                except Exception as e:
                    print(f"第 {i+1} 段处理失败：{e}\n已完成的缓存保留，可重试。", file=sys.stderr)
                    return 2
    final, count, reviews, undecided = render_final(directory, duration, vad_input, args.export_review_clips,args.cleanup_mode)
    print(f'全部完成：{final}')
    print(f'文本片段：{count}；语言疑点窗口：{reviews}；待人工复核条目：{undecided}。')
    print(f'说话人命名：{directory / "speaker_names.json"}')
    print(f'人工更正：{directory / "manual_overrides.json"}')
    print(f'人工复核清单：{directory / "review_queue.json"}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
