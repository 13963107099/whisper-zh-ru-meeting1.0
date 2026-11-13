#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auditable suppression for *explicitly user-reported* non-audio ASR phrases.

This is not a generic truth detector. Do not silently erase plausible speech.
No third-party dependencies. Never rewrite original ASR cache.
"""
from __future__ import annotations
import re

# User reported this recurring Chinese media-outro text was not in the recording.
# Require all distinctive parts to be present. Variants in spacing/punctuation
# and short regional characters are handled by normalization.
_CONFIRMED_OUTRO = re.compile(r"请不吝点赞订阅转发打赏支持明镜与点点栏目")
_SUSPICIOUS = (
    re.compile(r"(?:点赞.{0,10}订阅|订阅.{0,10}点赞).{0,26}(?:频道|节目|栏目)"),
    re.compile(r"(?:感谢观看|感谢收看).{0,20}(?:下期再见|订阅)"),
)
_TRANS = str.maketrans({'讚':'赞','請':'请','訂':'订','閱':'阅','轉':'转','發':'发','賞':'赏','鏡':'镜','與':'与','點':'点','欄':'栏','闔':'合'})


def compact(text):
    value = str(text or '').translate(_TRANS)
    return ''.join(c for c in value if c.isalnum())


def classify(text, manual_verified=False):
    """Return (suppress, reason) for text. Flag plausible outros for review only."""
    if manual_verified or not text:
        return False, ''
    c = compact(text)
    if _CONFIRMED_OUTRO.search(c):
        return True, 'user_reported_non_audio_outro'
    if any(p.search(c) for p in _SUSPICIOUS):
        return False, 'possible_media_outro_verify_against_audio'
    return False, ''


def filter_display_text(text, manual_verified=False):
    suppress, reason = classify(text, manual_verified)
    return ('' if suppress else text), reason
