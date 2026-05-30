"""Mit dem trainierten Modell neuen Text schreiben."""

from __future__ import annotations

from pathlib import Path

import torch

from .data import ZeichenDaten
from .model import GPT, GPTConfig


def lade_modell(out_dir: str | Path = "ausgabe"):
    """Lädt das trainierte Modell und sein Wörterbuch."""
    out_dir = Path(out_dir)
    ckpt = torch.load(out_dir / "modell.pt", map_location="cpu")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    vocab = ZeichenDaten.load_vocab(out_dir / "vocab.json")
    stoi = {z: i for i, z in enumerate(vocab)}
    itos = {i: z for i, z in enumerate(vocab)}
    return model, cfg, stoi, itos


def schreibe(
    start: str = "\n",
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int | None = 40,
    out_dir: str | Path = "ausgabe",
) -> str:
    """Lässt Theo ab ``start`` weiterschreiben und gibt den Text zurück."""
    model, cfg, stoi, itos = lade_modell(out_dir)
    # unbekannte Startzeichen ignorieren; leerer Start -> Zeilenumbruch
    ids = [stoi[c] for c in start if c in stoi] or [stoi.get("\n", 0)]
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return "".join(itos[int(i)] for i in out[0].tolist())


if __name__ == "__main__":
    print(schreibe("Eduard ", max_new_tokens=400))
