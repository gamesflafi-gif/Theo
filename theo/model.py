"""Die Architektur unseres Sprachmodells: ein GPT-Transformer.

Ein "GPT" (Generative Pretrained Transformer) lernt eine einzige Aufgabe:
"Welches Zeichen kommt als nächstes?" Wenn ein Modell das richtig gut kann,
entsteht daraus die Fähigkeit, ganze Texte zu schreiben.

Der Aufbau (von unten nach oben):
1. Embedding   – jedes Zeichen wird zu einem Vektor (einer Liste von Zahlen)
2. Self-Attention – jedes Zeichen "schaut" auf die vorherigen Zeichen
3. Feed-Forward – ein kleines neuronales Netz denkt darüber nach
4. (2 und 3 mehrfach gestapelt = "Blöcke")
5. Ausgabe     – eine Wahrscheinlichkeit für jedes mögliche nächste Zeichen

Das ist bewusst kompakt gehalten, aber es ist ein *echter* Transformer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Alle Einstellschrauben unseres Modells an einem Ort."""

    vocab_size: int          # wie viele verschiedene Zeichen es gibt
    block_size: int = 128    # wie viele Zeichen das Modell auf einmal sieht (Kontext)
    n_layer: int = 4         # wie viele Transformer-Blöcke gestapelt werden
    n_head: int = 4          # wie viele "Aufmerksamkeits-Köpfe" pro Block
    n_embd: int = 128        # Größe der Vektoren (mehr = klüger, aber langsamer)
    dropout: float = 0.1     # gegen "Auswendiglernen" (Overfitting)


class CausalSelfAttention(nn.Module):
    """Self-Attention: jedes Zeichen sammelt Infos von den vorherigen Zeichen.

    "Causal" heißt: ein Zeichen darf nur in die Vergangenheit schauen, nie in
    die Zukunft – sonst wäre das "nächste Zeichen vorhersagen" geschummelt.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # eine Projektion, die gleichzeitig Query, Key und Value erzeugt
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Dreiecks-Maske: verhindert das Schauen in die Zukunft
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch, Zeit (Zeichen), Kanäle (n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # in mehrere Köpfe aufteilen
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # wie stark passt jede Query zu jedem Key?
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # gewichtete Summe der Values
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Ein kleines Feed-Forward-Netz – hier "denkt" jeder Block nach."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Ein Transformer-Block = Attention + Feed-Forward (mit Restverbindungen)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))   # "+" = Restverbindung: hilft beim Lernen
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Das ganze Modell: Embeddings -> mehrere Blöcke -> Ausgabe."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Anzahl der lernbaren Parameter (die "Gehirnzellen-Verbindungen")."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Eingabe länger als der Kontext"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            # Wie falsch lag das Modell? (Cross-Entropy = Standardmaß)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Schreibt Zeichen für Zeichen neuen Text.

        - temperature: <1 = vorsichtiger/braver, >1 = kreativer/wilder
        - top_k: nur aus den k wahrscheinlichsten Zeichen wählen (oder None)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # nur die letzten block_size Zeichen als Kontext nehmen
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
