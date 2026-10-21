"""Text-reuse timing rule: the server's self-check keeps reuse only if detections are identical with it on and off."""

import numpy as np

from mapmad_recog.models import RawDet, memo_forward, text_reuse_check


class Fake:
    """A recogniser whose 'text encoder' is a counted function; `broken` makes reuse change the result."""

    def __init__(self, broken=False):
        self.reuse_text, self.text_caches, self.text_reuse = True, [], "fake encoder memoised"
        self.calls, self.broken = 0, broken
        self.encoder = type("Enc", (), {"forward": lambda _self, p: (self._count(), len(p))[1]})()
        memo_forward(self.encoder, self)

    def _count(self):
        self.calls += 1

    def clear_text_caches(self):
        for c in self.text_caches:
            c.clear()

    def detect(self, rgb, prompt):
        n = self.encoder.forward(prompt)
        score = 0.5 + (0.1 if (self.broken and self.reuse_text) else 0.0)
        return [RawDet(score, box=np.array([0.0, 0.0, n, n]))]


def test_identical_keeps_reuse_and_encodes_once_per_prompt():
    m = Fake()
    rec = text_reuse_check(m, [np.zeros((2, 2, 3))], ["chair", "tv", "chair"])
    assert rec["reuse"] and rec["identical"] and m.reuse_text
    m.calls = 0
    for p in ["chair", "tv", "chair", "tv"]:
        m.detect(None, p)
    assert m.calls == 2  # each prompt's text encoded once


def test_different_detections_switch_reuse_off():
    m = Fake(broken=True)
    rec = text_reuse_check(m, [np.zeros((2, 2, 3))], ["chair", "tv"])
    assert rec["reuse"] is False and not m.reuse_text


def test_no_text_encoder():
    m = Fake()
    m.text_reuse = "none (no text encoder found)"
    assert text_reuse_check(m, [np.zeros((2, 2, 3))], ["chair"])["reuse"] is False
