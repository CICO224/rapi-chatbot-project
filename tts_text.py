"""Text preparation for the TTS engine: sanitising and sentence chunking.

Pure-Python and dependency-free so it can be tested without the audio stack.
"""
import re

_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # emoji, pictographs, flags
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U00002B00-\U00002BFF"  # stars, arrows
    "\U00002190-\U000021FF"  # arrows
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U000E0020-\U000E007F"  # tag characters
    "‍⃣"           # zero-width joiner, keycap
    "]+"
)

# Words the TTS should keep in capitals (it spells them out, which is correct).
_KEEP_CAPS = {"AI", "OK", "HQ", "ID", "TV"}


def _lower_shouting(match):
    word = match.group(0)
    return word if word in _KEEP_CAPS else word.lower()


def sanitize(text):
    """Turn a chat reply into plain speakable text.

    Removes roleplay actions (*sighs*), emojis, markdown and stage directions,
    and normalises punctuation so the TTS pauses where the text does.
    """
    t = text

    # Markdown blocks and inline code
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`([^`]*)`", r"\1", t)

    # Images, links, bare URLs
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"(?:https?://|www\.)\S+", " ", t)

    # **bold** / __bold__ keep their words; *actions* are dropped entirely
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.S)
    t = re.sub(r"__(.+?)__", r"\1", t, flags=re.S)
    t = re.sub(r"\*[^*]*\*", " ", t)
    t = re.sub(r"\*[^*]*$", " ", t)  # an action the model never closed

    # Stage directions in (parentheses) or [brackets]
    t = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", t)

    # Line-level markdown: headings, block quotes, bullets, table pipes
    t = re.sub(r"^\s*(?:#{1,6}|>+|[-+])\s+", "", t, flags=re.M)
    t = re.sub(r"[*_~#|<>\\]", "", t)

    t = _EMOJI.sub("", t)

    # Dialogue quotes are not spoken (apostrophes are kept)
    t = re.sub(r'["“”«»]', "", t)

    # Punctuation the TTS can pause on
    t = t.replace("…", "...")
    t = re.sub(r"\.{2,}", "...", t)
    t = re.sub(r"\.\.\.(?=\w)", "... ", t)
    t = re.sub(r"\s*[—–]+\s*|\s+--+\s+", ", ", t)
    t = re.sub(r"([,;:!?])(?=[A-Za-z])", r"\1 ", t)

    # SHOUTED words get read letter-by-letter by some voices
    t = re.sub(r"\b[A-Z]{2,}(?:'[A-Z]+)?\b", _lower_shouting, t)

    t = re.sub(r"\bNikkes?\b", lambda m: "Nik kehs" if m.group(0).lower().endswith("s") else "Nik keh", t, flags=re.I)

    # Whitespace and leftover punctuation tidy-up
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"(?:,\s*){2,}", ", ", t)
    t = re.sub(r"^[\s,.;:]+", "", t)
    return t.strip()


def has_speech(text):
    return bool(re.search(r"[A-Za-z0-9]", text))


# A sentence end: closing punctuation, optional closing quote/bracket, then whitespace.
_BOUNDARY = re.compile(r"[.!?…]+[\"'”’)\]]*\s+|\n+")


class SentenceChunker:
    """Splits a token stream into speakable chunks at sentence boundaries.

    Chunks are returned already sanitised. A boundary is only used when it is
    not inside an unfinished *action*, and when the chunk is long enough to be
    worth a TTS round trip (the first chunk may be shorter so audio starts fast).
    """

    def __init__(self, first_min=20, min_chars=50):
        self.first_min = first_min
        self.min_chars = min_chars
        self._buf = ""
        self._emitted = False

    def feed(self, token):
        self._buf += token
        out = []
        while True:
            chunk = self._next_chunk()
            if chunk is None:
                break
            out.append(chunk)
        return out

    def flush(self):
        """Call once the stream has ended to get whatever is left."""
        rest = sanitize(self._buf)
        self._buf = ""
        return [rest] if has_speech(rest) else []

    def _next_chunk(self):
        need = self.min_chars if self._emitted else self.first_min
        for m in _BOUNDARY.finditer(self._buf):
            raw = self._buf[: m.end()]
            if raw.count("*") % 2:
                continue  # boundary sits inside an unfinished *action*
            clean = sanitize(raw)
            if len(clean) < need or not has_speech(clean):
                continue
            self._buf = self._buf[m.end():]
            self._emitted = True
            return clean
        return None
