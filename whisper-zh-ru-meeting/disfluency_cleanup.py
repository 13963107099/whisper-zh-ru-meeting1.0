#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conservative, auditable Chinese/Russian speech-disfluency cleanup.

Operate on ASR text *after* recognition and speaker assignment; never modify
ASR caches or audio. Never attempt to infer which speaker said a word.
Avoid deleting negation, affirmation, money, dates or numerical repeats.
"""
import re

# Common restarted Chinese utterances that continue into a longer statement.
# Restrict to grammatical lead-ins, rather than arbitrary names/numbers.
ZH_RESTART = re.compile(
    r'(?P<phrase>(?:我们|咱们|你们|他们|她们|这个|那个|接下来|关于|首先|我|你|他|她|就是|然后|因为|所以|如果|希望|需要|现在)'
    r'[\u4e00-\u9fff]{0,12})\s*[，,、]\s*(?P=phrase)(?=[\u4e00-\u9fff])'
)
ZH_TOKEN = re.compile(
    r'(?P<token>我们|咱们|你们|他们|她们|这个|那个|就是|然后|因为|所以|如果|我|你|他|她)'
    r'(?:\s*[，,、]?\s*(?P=token))+'
)
RU_RESTART = re.compile(
    r'(?i)(?<!\w)(?P<phrase>(?:мы|я|это|ну|как бы)(?:\s+[а-яё]{2,}){1,4})'
    r'\s*[,，]\s*(?P=phrase)\b(?=\s+[а-яё])'
)
RU_TOKEN = re.compile(
    r'(?i)(?<!\w)(?P<token>мы|я|это|ну)(?:[\s,，、-]+(?P=token)\b)+'
)
ZH_FILLER = re.compile(
    r'(?P<leading>^|[，,。！？!?；;:]\s*)(?P<filler>呃+|额+)(?:\s*[,，、]\s*|\s+)(?=\S)'
)
RU_FILLER = re.compile(
    r'(?i)(?P<leading>^|[,，。.！!?？；;:]\s*)(?P<filler>ээ+|эм+)(?:\s*[,，]\s*|\s+)(?=\S)'
)


def clean_text(text, mode='smart'):
    """Return (cleaned_text, audit_edits).

    off: exact ASR text, no editing.
    light: remove clear isolated filler and repeated short starter words.
    smart: light + limited immediate short-clause restarts.
    """
    if mode not in {'off', 'light', 'smart'}:
        raise ValueError('cleanup mode must be off, light or smart')
    raw = str(text or '')
    if mode == 'off' or not raw.strip():
        return raw, []

    output = raw.strip()
    edits = []

    def apply(pattern, rule, transform):
        nonlocal output
        def replace(m):
            replacement = transform(m)
            if replacement != m.group(0):
                edits.append({'rule': rule, 'before': m.group(0), 'after': replacement})
            return replacement
        # Re-run only a few rounds for adjacent/tripled stutters.
        for _ in range(3):
            rewritten, count = pattern.subn(replace, output)
            if not count or rewritten == output:
                break
            output = rewritten

    if mode == 'smart':
        apply(ZH_RESTART, 'zh_restart', lambda m: m.group('phrase'))
        apply(RU_RESTART, 'ru_restart', lambda m: m.group('phrase'))
    apply(ZH_TOKEN, 'zh_short_repeat', lambda m: m.group('token'))
    apply(RU_TOKEN, 'ru_short_repeat', lambda m: m.group('token'))
    apply(ZH_FILLER, 'zh_filler', lambda m: m.group('leading'))
    apply(RU_FILLER, 'ru_filler', lambda m: m.group('leading'))

    if edits:
        output = re.sub(r'\s+([，。！？；,.!?;:])', r'\1', output)
        output = re.sub(r'(?:(?:\s*[，,、]){2,})', '，', output)
        output = re.sub(r'[ \t]{2,}', ' ', output).strip(' ，,、\n\t')
        if not output:
            # Never erase a whole utterance solely via heuristic cleanup.
            return raw, []
    return output, edits
