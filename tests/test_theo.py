"""Kleine Tests, die sicherstellen, dass die Bausteine funktionieren."""

import torch

from theo.data import ZeichenDaten
from theo.model import GPT, GPTConfig


def test_encode_decode_ist_umkehrbar():
    daten = ZeichenDaten("abcabc\nxyz")
    text = "abc\nxyz"
    assert daten.decode(daten.encode(text)) == text


def test_modell_liefert_passende_form_und_loss():
    cfg = GPTConfig(vocab_size=20, block_size=8, n_layer=2, n_head=2, n_embd=16)
    model = GPT(cfg)
    x = torch.randint(0, 20, (4, 8))
    y = torch.randint(0, 20, (4, 8))
    logits, loss = model(x, y)
    assert logits.shape == (4, 8, 20)
    assert loss.item() > 0


def test_generieren_haengt_zeichen_an():
    cfg = GPTConfig(vocab_size=20, block_size=8, n_layer=2, n_head=2, n_embd=16)
    model = GPT(cfg)
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=10)
    assert out.shape == (1, 11)


def test_batch_formen_stimmen():
    daten = ZeichenDaten("abcdefghij" * 50)
    x, y = daten.batch("train", batch_size=3, block_size=5, device="cpu")
    assert x.shape == (3, 5)
    assert y.shape == (3, 5)
