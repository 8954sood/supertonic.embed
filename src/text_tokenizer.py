# -*- coding: utf-8 -*-
"""Turn text into the token ids and mask SupertonicTTS expects.

Taken from the SupertonicTTS release (MIT), unchanged in behaviour: the model
was trained on exactly this normalization, so any edit here changes what the
model hears. Only the parts the extractor uses are kept.
"""
import json
import re
from unicodedata import normalize

import numpy as np

LANGS = ["en", "ko", "es", "pt", "fr"]

_EMOJI = re.compile(
    "[\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "☀-⛿"
    "✀-➿"
    "\U0001f1e6-\U0001f1ff]+",
    flags=re.UNICODE,
)
_REPLACE = {
    "–": "-", "‑": "-", "—": "-", "_": " ",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "´": "'", "`": "'",
    "[": " ", "]": " ", "|": " ", "/": " ", "#": " ", "→": " ", "←": " ",
}
_EXPAND = {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}
_TIGHTEN = [" ,", " .", " !", " ?", " ;", " :", " '"]
_ENDS_SENTENCE = re.compile(r"[.!?;:,'\"')\]}…。」』】〉》›»]$")


def normalize_for_tts(text, lang):
    """The release's own text cleanup, then the language tags it expects."""
    if lang not in LANGS:
        raise ValueError(f"unsupported language {lang!r}; expected one of {LANGS}")
    text = _EMOJI.sub("", normalize("NFKD", text))
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    text = re.sub(r"[♥☆♡©\\]", "", text)
    for a, b in _EXPAND.items():
        text = text.replace(a, b)
    for a in _TIGHTEN:                       # a space before punctuation is not spoken
        text = text.replace(a, a[1])
    for doubled in ('""', "''", "``"):
        while doubled in text:
            text = text.replace(doubled, doubled[0])
    text = re.sub(r"\s+", " ", text).strip()
    if not _ENDS_SENTENCE.search(text):      # the model expects a terminated sentence
        text += "."
    return f"<{lang}>{text}</{lang}>"


class TextTokenizer:
    """Unicode code points to the model's token ids, via the released indexer."""

    def __init__(self, indexer_path):
        with open(indexer_path, encoding="utf-8") as f:
            self.indexer = json.load(f)

    def __call__(self, text, lang="en"):
        """-> (ids [1,T] int64, mask [1,1,T] float32)"""
        text = normalize_for_tts(text, lang)
        ids = np.array([[self.indexer[ord(c)] for c in text]], dtype=np.int64)
        mask = np.ones((1, 1, len(text)), dtype=np.float32)
        return ids, mask
