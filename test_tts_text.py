"""Run with:  python -m pytest test_tts_text.py   (or just: python test_tts_text.py)"""
from tts_text import SentenceChunker, sanitize


def test_strips_actions_emoji_and_markdown():
    raw = '*sighs softly* "Commander... you\'re **late** again." 😒 (she looks away) [link](http://x.y)'
    out = sanitize(raw)
    assert out == "Commander... you're late again. link"


def test_unclosed_action_is_dropped():
    assert sanitize('"Fine." *turns away and') == "Fine."


def test_punctuation_and_shouting():
    out = sanitize('"I said NO — not now...okay?"')
    assert out == "I said no, not now... okay?"


def test_nikke_pronunciation():
    assert sanitize("A Nikke among Nikkes.") == "A Nik keh among Nik kehs."


def test_chunker_waits_for_action_to_close():
    c = SentenceChunker(first_min=5, min_chars=5)
    assert c.feed('"Understood, Commander." *nods. Then ') == ["Understood, Commander."]
    assert c.feed('salutes* "Moving out now." ') == ["Moving out now."]
    assert c.flush() == []


def test_chunker_merges_short_fragments_and_flushes():
    c = SentenceChunker(first_min=20, min_chars=50)
    out = []
    for tok in ['"Hm... ', "Commander... ", "I was waiting for you. ", "Coffee", '?"']:
        out += c.feed(tok)
    assert out == ["Hm... Commander... I was waiting for you."]
    assert c.flush() == ["Coffee?"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
