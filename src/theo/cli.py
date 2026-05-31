"""Kommandozeilen-Interface für Theo.

Beispiele:
    theo ask "Was ist ein Touchdown?"
    theo chat
    theo analyze spiel.mp4
    theo topics
"""

from __future__ import annotations

import argparse
import sys

from theo.qa import QAEngine
from theo.qa import llm as _llm


def _print_answer(answer, *, show_sources: bool) -> None:
    print(answer.text)
    if show_sources and answer.sources:
        print()
        print("Quellen:")
        for s in answer.sources:
            print(f"  - {s.section.heading_path}  ({s.section.doc})")
    backend = "Claude API" if answer.used_llm else "lokale Wissensbasis"
    print(f"\n[Backend: {backend}]")


def cmd_ask(args: argparse.Namespace) -> int:
    engine = QAEngine(mode=args.mode)
    answer = engine.ask(args.question, top_k=args.top_k)
    _print_answer(answer, show_sources=not args.no_sources)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    engine = QAEngine(mode=args.mode)
    backend = "Claude API" if _llm.is_available() and args.mode != "extractive" \
        else "lokale Wissensbasis"
    print(f"Theo – Football-Chat ({backend}). 'exit' zum Beenden.\n")
    while True:
        try:
            question = input("Du> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "ende"}:
            break
        answer = engine.ask(question, top_k=args.top_k)
        print(f"\nTheo> {answer.text}\n")
    return 0


def _save_keyframes(result, out_dir: str) -> None:
    import cv2  # type: ignore
    from pathlib import Path

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if not result.keyframes:
        print(f"(Keine Keyframes erzeugt – nichts in {out_dir} gespeichert.)")
        return
    for i, kf in enumerate(result.keyframes, 1):
        path = Path(out_dir) / f"keyframe_{i:02d}_{kf.time_s:.1f}s.png"
        cv2.imwrite(str(path), kf.image)
        print(f"  gespeichert: {path}")


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        # --save-frames impliziert die Detektions-Pipeline.
        if getattr(args, "save_frames", None):
            args.detect = True
        if args.save_video:
            from theo.video import render_annotated_video

            written = render_annotated_video(
                args.path, args.save_video, detector=args.detector,
                max_seconds=args.max_seconds)
            print(f"Annotiertes Video geschrieben: {args.save_video} "
                  f"({written} Frames)")
            return 0

        if args.detect:
            from theo.video import VideoPipeline

            pipeline = VideoPipeline(detector=args.detector)
            result = pipeline.process(
                args.path,
                max_seconds=args.max_seconds,
                annotate=bool(args.save_frames),
            )
            print(result.summary())
            if args.save_frames:
                _save_keyframes(result, args.save_frames)
            return 0

        from theo.video import analyze_video

        result = analyze_video(args.path)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1
    print(result.summary())
    if args.show_roadmap:
        print("\nGeplante Analysen (Stufe 2+):")
        for feat in result.planned_features:
            print(f"  - {feat}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Fehler: Das Web-Backend benötigt `pip install theo[web]`.",
              file=sys.stderr)
        return 1
    print(f"Theo-Weboberfläche läuft auf http://{args.host}:{args.port}")
    uvicorn.run("theo.web.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_topics(_args: argparse.Namespace) -> int:
    from theo.knowledge import load_sections

    current_doc = None
    for sec in load_sections():
        if sec.doc != current_doc:
            current_doc = sec.doc
            print(f"\n# {sec.doc}")
        indent = "  " * (sec.heading_path.count(">"))
        print(f"{indent}- {sec.title}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="theo",
        description="Theo – die KI rund um American Football.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common_mode = dict(
        choices=["auto", "llm", "extractive"],
        default="auto",
        help="Antwort-Backend (auto = Claude wenn verfügbar, sonst lokal).",
    )

    p_ask = sub.add_parser("ask", help="Eine Football-Frage stellen.")
    p_ask.add_argument("question", help="Die Frage.")
    p_ask.add_argument("--mode", **common_mode)
    p_ask.add_argument("--top-k", type=int, default=4, help="Anzahl Kontext-Abschnitte.")
    p_ask.add_argument("--no-sources", action="store_true", help="Quellen ausblenden.")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interaktiver Football-Chat.")
    p_chat.add_argument("--mode", **common_mode)
    p_chat.add_argument("--top-k", type=int, default=4)
    p_chat.set_defaults(func=cmd_chat)

    p_an = sub.add_parser("analyze", help="Ein Spiel-/Trainingsvideo analysieren.")
    p_an.add_argument("path", help="Pfad zur Videodatei.")
    p_an.add_argument("--detect", action="store_true",
                      help="Volle CV-Pipeline: Spielererkennung, Tracking, "
                           "Formation & Spielzug-Schätzung.")
    p_an.add_argument("--detector", choices=["hog", "yolo"], default="hog",
                      help="Detektor für --detect (yolo benötigt theo[video-yolo]).")
    p_an.add_argument("--max-seconds", type=float, default=30.0,
                      help="Maximale analysierte Videolänge (Sekunden).")
    p_an.add_argument("--save-frames", metavar="DIR",
                      help="Annotierte Keyframes (Boxen/LOS) als PNG in DIR speichern "
                           "(impliziert --detect).")
    p_an.add_argument("--save-video", metavar="FILE",
                      help="Annotiertes Video (Boxen je Frame) als MP4 speichern.")
    p_an.add_argument("--show-roadmap", action="store_true",
                      help="Geplante CV-Analysen anzeigen (Basisanalyse).")
    p_an.set_defaults(func=cmd_analyze)

    p_top = sub.add_parser("topics", help="Inhalt der Wissensbasis auflisten.")
    p_top.set_defaults(func=cmd_topics)

    p_srv = sub.add_parser("serve", help="Weboberfläche starten (Q&A + Upload).")
    p_srv.add_argument("--host", default="127.0.0.1", help="Host-Adresse.")
    p_srv.add_argument("--port", type=int, default=8000, help="Port.")
    p_srv.add_argument("--reload", action="store_true", help="Auto-Reload (Dev).")
    p_srv.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        # Ausgabe wurde abgeschnitten (z. B. `theo topics | head`) – sauber beenden.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
