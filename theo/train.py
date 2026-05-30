"""Das Modell trainieren.

Trainieren heißt: dem Modell tausende Textausschnitte zeigen, es das nächste
Zeichen raten lassen, messen wie falsch es lag (der "loss"), und die internen
Stellschrauben ein winziges bisschen in die richtige Richtung drehen. Wiederholt
man das oft genug, sinkt der loss und das Modell schreibt immer besseren Text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .corpus import korpus_laden
from .data import ZeichenDaten
from .model import GPT, GPTConfig


@dataclass
class TrainConfig:
    max_iters: int = 3000      # wie viele Lernschritte
    eval_interval: int = 250   # wie oft wir den Lernfortschritt prüfen
    eval_iters: int = 50       # wie viele Häppchen pro Prüfung
    batch_size: int = 32       # wie viele Ausschnitte gleichzeitig
    learning_rate: float = 3e-4
    out_dir: str = "ausgabe"   # hier landet das fertige Modell


@torch.no_grad()
def _bewerte(model: GPT, daten: ZeichenDaten, cfg: TrainConfig, block_size, device):
    """Misst den loss auf Trainings- und Validierungsdaten."""
    model.eval()
    ergebnis = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = daten.batch(split, cfg.batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        ergebnis[split] = losses.mean().item()
    model.train()
    return ergebnis


def train(model_cfg: GPTConfig | None = None, train_cfg: TrainConfig | None = None):
    train_cfg = train_cfg or TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)

    print(f"Gerät: {device}")
    text = korpus_laden()
    daten = ZeichenDaten(text)
    print(f"Korpus: {len(text):,} Zeichen, {daten.vocab_size} verschiedene Zeichen")

    model_cfg = model_cfg or GPTConfig(vocab_size=daten.vocab_size)
    model_cfg.vocab_size = daten.vocab_size  # immer zum Korpus passend
    model = GPT(model_cfg).to(device)
    print(f"Modellgröße: {model.num_params()/1e6:.2f} Mio. Parameter")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(exist_ok=True)
    daten.save_vocab(out_dir / "vocab.json")

    start = time.time()
    best_val = float("inf")
    for it in range(train_cfg.max_iters + 1):
        if it % train_cfg.eval_interval == 0 or it == train_cfg.max_iters:
            l = _bewerte(model, daten, train_cfg, model_cfg.block_size, device)
            dt = time.time() - start
            print(
                f"Schritt {it:>5} | train {l['train']:.3f} | val {l['val']:.3f} "
                f"| {dt:.0f}s"
            )
            if l["val"] < best_val:
                best_val = l["val"]
                torch.save(
                    {"model": model.state_dict(), "config": model_cfg.__dict__},
                    out_dir / "modell.pt",
                )

        x, y = daten.batch("train", train_cfg.batch_size, model_cfg.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Fertig. Bestes val-loss: {best_val:.3f}. Modell in {out_dir}/modell.pt")
    return model, daten


if __name__ == "__main__":
    train()
