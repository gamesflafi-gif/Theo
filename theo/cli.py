"""Kommandozeile für Theo.

Beispiele:
    python -m theo.cli train                 # Modell trainieren
    python -m theo.cli train --max-iters 500 # kurzes Training (Schnelltest)
    python -m theo.cli schreibe --start "Eduard " --laenge 400
"""

from __future__ import annotations

import argparse

from .model import GPTConfig
from .train import TrainConfig, train


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Theo – unser eigenes Sprachmodell")
    sub = p.add_subparsers(dest="befehl", required=True)

    t = sub.add_parser("train", help="das Modell trainieren")
    t.add_argument("--max-iters", type=int, default=3000)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--n-layer", type=int, default=4)
    t.add_argument("--n-head", type=int, default=4)
    t.add_argument("--n-embd", type=int, default=128)
    t.add_argument("--block-size", type=int, default=128)

    g = sub.add_parser("schreibe", help="mit dem Modell Text schreiben")
    g.add_argument("--start", default="\n")
    g.add_argument("--laenge", type=int, default=500)
    g.add_argument("--temperatur", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=40)

    args = p.parse_args(argv)

    if args.befehl == "train":
        model_cfg = GPTConfig(
            vocab_size=1,  # wird im Training an den Korpus angepasst
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
        )
        train(model_cfg, TrainConfig(max_iters=args.max_iters, batch_size=args.batch_size))
    elif args.befehl == "schreibe":
        from .generate import schreibe

        print(
            schreibe(
                start=args.start,
                max_new_tokens=args.laenge,
                temperature=args.temperatur,
                top_k=args.top_k,
            )
        )


if __name__ == "__main__":
    main()
