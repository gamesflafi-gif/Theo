"""Daten für das Modell vorbereiten.

Ein neuronales Netz versteht keine Buchstaben, nur Zahlen. Wir bauen also ein
kleines "Wörterbuch": jedem vorkommenden Zeichen geben wir eine Nummer
(zeichenweise / "character-level"). Das ist die einfachste Art, Text in Zahlen
zu verwandeln – und perfekt zum Lernen.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


class ZeichenDaten:
    """Verwaltet das Zeichen-Wörterbuch und liefert Trainings-Häppchen."""

    def __init__(self, text: str):
        # alle einzigartigen Zeichen, sortiert -> stabile Reihenfolge
        zeichen = sorted(set(text))
        self.vocab = zeichen
        self.vocab_size = len(zeichen)
        self.stoi = {z: i for i, z in enumerate(zeichen)}  # Zeichen -> Nummer
        self.itos = {i: z for i, z in enumerate(zeichen)}  # Nummer -> Zeichen

        daten = torch.tensor(self.encode(text), dtype=torch.long)
        # 90 % zum Lernen, 10 % zum ehrlichen Prüfen (das Modell sieht sie beim
        # Training nicht – so merken wir, ob es wirklich versteht oder nur
        # auswendig lernt).
        n = int(0.9 * len(daten))
        self.train_daten = daten[:n]
        self.val_daten = daten[n:]

    def encode(self, s: str) -> list[int]:
        """Text -> Liste von Zahlen (unbekannte Zeichen werden übersprungen)."""
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        """Liste von Zahlen -> Text."""
        return "".join(self.itos[int(i)] for i in ids)

    def batch(self, split: str, batch_size: int, block_size: int, device: str):
        """Zieht zufällige Text-Ausschnitte fürs Training.

        x = Eingabe (block_size Zeichen), y = die jeweils um eins
        verschobene Eingabe (das "richtige nächste Zeichen").
        """
        daten = self.train_daten if split == "train" else self.val_daten
        ix = torch.randint(len(daten) - block_size, (batch_size,))
        x = torch.stack([daten[i : i + block_size] for i in ix])
        y = torch.stack([daten[i + 1 : i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)

    def save_vocab(self, pfad: str | Path) -> None:
        """Speichert das Wörterbuch (damit Generieren später passt)."""
        Path(pfad).write_text(
            json.dumps({"vocab": self.vocab}, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def load_vocab(pfad: str | Path) -> list[str]:
        return json.loads(Path(pfad).read_text(encoding="utf-8"))["vocab"]
