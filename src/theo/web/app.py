"""Web-Backend für Theo (FastAPI): Q&A + Video-Upload-Analyse.

Startet eine kleine Single-Page-Oberfläche, über die man Football-Fragen stellen
und Spiel-/Trainingsvideos hochladen kann.

Starten:
    theo serve            # oder: uvicorn theo.web.app:app
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from theo.qa import QAEngine
from theo.simulation import (
    DEFENSE_LIBRARY,
    OFFENSE_LIBRARY,
    OFFENSE_SLOTS,
    ROUTE_NAMES,
    Simulator,
    make_offense_play,
    rank_defenses,
    rank_offenses,
)

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import (
        FileResponse,
        HTMLResponse,
        JSONResponse,
        Response,
    )
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    from starlette.background import BackgroundTask
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Das Web-Backend benötigt FastAPI. Installiere es mit "
        "`pip install theo[web]`."
    ) from exc


class SimRequest(BaseModel):
    offense_id: str | None = None
    defense_id: str = "cover2_zone"
    routes: dict[str, str] | None = None
    kind: str = "pass"
    seed: int = 0
    n: int = 100


class AdvisorRequest(BaseModel):
    mode: str = "best_defense"      # best_defense | best_offense
    offense_id: str | None = None
    defense_id: str = "cover2_zone"
    routes: dict[str, str] | None = None
    kind: str = "pass"
    n: int = 60


def _resolve_plays(req: "SimRequest"):
    deff = DEFENSE_LIBRARY.get(req.defense_id)
    if deff is None:
        raise HTTPException(status_code=400, detail="Unbekannte Defense.")
    if req.routes:
        off = make_offense_play(req.routes, kind=req.kind)
    elif req.offense_id in OFFENSE_LIBRARY:
        off = OFFENSE_LIBRARY[req.offense_id]
    else:
        raise HTTPException(status_code=400, detail="Unbekannte Offense.")
    return off, deff

_STATIC_DIR = Path(__file__).parent / "static"

# Erlaubte Video-Endungen für den Upload.
_ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
# Obergrenze für die analysierte Videolänge (Sekunden) – per Env konfigurierbar.
_MAX_ANALYZE_SECONDS = float(os.environ.get("THEO_MAX_ANALYZE_SECONDS", "30"))
# Maximale Upload-Größe in MB (Schutz vor zu großen Dateien).
_MAX_UPLOAD_MB = float(os.environ.get("THEO_MAX_UPLOAD_MB", "200"))
# Standard-Detektor für die Weboberfläche (hog | yolo).
_DEFAULT_DETECTOR = os.environ.get("THEO_DEFAULT_DETECTOR", "hog")


def _save_upload_limited(src, dest: Path, max_bytes: int) -> None:
    """Schreibt einen Upload streamend und bricht bei Überschreitung des Limits ab."""
    written = 0
    with dest.open("wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Video zu groß (Limit {_MAX_UPLOAD_MB:.0f} MB).",
                )
            out.write(chunk)


def create_app() -> "FastAPI":
    app = FastAPI(title="Theo – Football-KI", version="0.1.0")
    # Engine einmalig aufbauen (Wissensbasis-Index wird gecacht).
    engine = QAEngine()

    # Statische Assets (PWA-Icons etc.).
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/manifest.webmanifest")
    def manifest() -> Response:
        return Response(_MANIFEST, media_type="application/manifest+json")

    @app.get("/sw.js")
    def service_worker() -> Response:
        # Muss im Root-Scope liegen, um die ganze App zu kontrollieren.
        return Response(_SERVICE_WORKER, media_type="application/javascript")

    @app.get("/api/plays")
    def plays() -> dict:
        return {
            "offense": [p.to_dict() for p in OFFENSE_LIBRARY.values()],
            "defense": [p.to_dict() for p in DEFENSE_LIBRARY.values()],
            "routes": list(ROUTE_NAMES) + ["block"],
            "slots": [{"id": s.id, "role": s.role}
                      for s in OFFENSE_SLOTS if s.id != "QB"],
        }

    @app.post("/api/simulate")
    def simulate(req: SimRequest) -> dict:
        off, deff = _resolve_plays(req)
        return Simulator().simulate(off, deff, seed=req.seed).to_dict()

    @app.post("/api/simulate/batch")
    def simulate_batch(req: SimRequest) -> dict:
        off, deff = _resolve_plays(req)
        n = max(1, min(500, req.n))
        return Simulator().simulate_many(
            off, deff, n=n, base_seed=req.seed).to_dict()

    @app.post("/api/advisor")
    def advisor(req: AdvisorRequest) -> dict:
        n = max(10, min(200, req.n))
        if req.mode == "best_offense":
            deff = DEFENSE_LIBRARY.get(req.defense_id)
            if deff is None:
                raise HTTPException(status_code=400, detail="Unbekannte Defense.")
            return {"mode": "best_offense", "vs": deff.name,
                    "rows": rank_offenses(deff, n=n)}
        # best_defense (Standard): Offense auflösen.
        if req.routes:
            off = make_offense_play(req.routes, kind=req.kind)
        elif req.offense_id in OFFENSE_LIBRARY:
            off = OFFENSE_LIBRARY[req.offense_id]
        else:
            raise HTTPException(status_code=400, detail="Unbekannte Offense.")
        return {"mode": "best_defense", "vs": off.name,
                "rows": rank_defenses(off, n=n)}

    @app.get("/api/health")
    def health() -> dict:
        from theo.qa import llm

        return {"status": "ok", "llm_backend": llm.is_available()}

    @app.post("/api/ask")
    def ask(question: str = Form(...)) -> JSONResponse:
        question = question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Leere Frage.")
        answer = engine.ask(question)
        return JSONResponse(
            {
                "question": answer.question,
                "answer": answer.text,
                "sources": answer.source_titles,
                "backend": "Claude API" if answer.used_llm else "lokale Wissensbasis",
            }
        )

    @app.post("/api/analyze")
    def analyze(
        file: UploadFile = File(...),
        detector: str = Form(_DEFAULT_DETECTOR),
        detect: bool = Form(True),
    ) -> JSONResponse:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Nicht unterstütztes Format: {suffix or '(keins)'}.",
            )

        tmp_dir = Path(tempfile.mkdtemp(prefix="theo_upload_"))
        tmp_path = tmp_dir / f"upload{suffix}"
        try:
            _save_upload_limited(file.file, tmp_path, int(_MAX_UPLOAD_MB * 1024 * 1024))
            payload = _run_analysis(tmp_path, detector=detector, detect=detect)
            return JSONResponse(payload)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @app.post("/api/analyze/video")
    def analyze_video_download(
        file: UploadFile = File(...),
        detector: str = Form(_DEFAULT_DETECTOR),
    ):
        from theo.video import render_annotated_video

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise HTTPException(status_code=400,
                                detail=f"Nicht unterstütztes Format: {suffix or '(keins)'}.")
        tmp_dir = Path(tempfile.mkdtemp(prefix="theo_video_"))
        in_path = tmp_dir / f"in{suffix}"
        out_path = tmp_dir / "theo_annotated.mp4"
        cleanup = BackgroundTask(shutil.rmtree, str(tmp_dir), ignore_errors=True)
        try:
            _save_upload_limited(file.file, in_path,
                                 int(_MAX_UPLOAD_MB * 1024 * 1024))
            written = render_annotated_video(
                in_path, out_path, detector=detector,
                max_seconds=_MAX_ANALYZE_SECONDS)
            if written == 0 or not out_path.exists():
                raise HTTPException(status_code=400,
                                    detail="Keine Frames – Video nicht lesbar.")
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return FileResponse(str(out_path), media_type="video/mp4",
                            filename="theo_annotated.mp4", background=cleanup)

    return app


def _run_analysis(path: Path, *, detector: str, detect: bool) -> dict:
    if detect:
        from theo.video import VideoPipeline

        result = VideoPipeline(detector=detector).process(
            path, max_seconds=_MAX_ANALYZE_SECONDS, annotate=True
        )
        data = {
            "mode": "detect",
            "summary": result.summary(),
            "detector": result.detector_name,
            "avg_players": round(result.avg_players, 1),
            "max_players": result.max_players,
            "active_segments": len(result.active_segments),
            "segments": [{"start": round(s.start_s, 1), "end": round(s.end_s, 1)}
                         for s in result.active_segments[:6]],
            "keyframes": [
                {"label": kf.label, "time_s": round(kf.time_s, 1),
                 "image": kf.to_data_url()}
                for kf in result.keyframes
            ],
        }
        if result.formation:
            data["formation"] = result.formation.descriptor
        if result.play:
            data["play"] = {
                "type": result.play.play_type,
                "confidence": result.play.confidence,
            }
        return data

    from theo.video import analyze_video

    result = analyze_video(path)
    return {"mode": "basic", "summary": result.summary()}


# Single-Page-Oberfläche (bewusst abhängigkeitsfrei, reines HTML/JS).
_INDEX_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Theo – Football-KI</title>
<link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="#0f1115">
<link rel="apple-touch-icon" href="/static/icons/icon-192.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Theo">
<link rel="icon" type="image/png" href="/static/icons/icon-192.png">
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto;
         padding: 0 1rem; background: #0f1115; color: #e8e8e8; }
  h1 { font-size: 1.8rem; } h2 { margin-top: 2rem; font-size: 1.2rem; }
  .card { background: #181b22; border: 1px solid #262a33; border-radius: 10px;
          padding: 1rem 1.2rem; margin: 1rem 0; }
  input, textarea, select, button { font: inherit; }
  textarea, input[type=text] { width: 100%; padding: .6rem; border-radius: 8px;
          border: 1px solid #333; background: #0f1115; color: #e8e8e8; box-sizing: border-box; }
  button { background: #2d6cdf; color: #fff; border: 0; padding: .6rem 1.1rem;
           border-radius: 8px; cursor: pointer; margin-top: .6rem; }
  button:disabled { opacity: .6; cursor: progress; }
  pre { white-space: pre-wrap; word-wrap: break-word; background: #0f1115;
        padding: .8rem; border-radius: 8px; border: 1px solid #262a33; }
  .muted { color: #8a90a0; font-size: .85rem; }
  .src { color: #8a90a0; font-size: .8rem; margin-top: .4rem; }

  /* --- Polierter Look (überschreibt Basis) --- */
  body { max-width: 820px; background:
         radial-gradient(1200px 600px at 50% -10%, #16203a 0%, #0d0f14 55%) fixed;
         line-height: 1.5; padding-bottom: 3rem; }
  .hero { text-align: center; padding: 1.4rem 0 .4rem; }
  .hero h1 { font-size: 2.1rem; margin: 0;
             background: linear-gradient(90deg,#7aa2ff,#9b7bff,#56c596);
             -webkit-background-clip: text; background-clip: text; color: transparent; }
  .hero p { margin: .35rem 0 0; }
  nav { position: sticky; top: 0; z-index: 5; padding: .5rem 0;
        backdrop-filter: blur(8px); background: rgba(13,15,20,.7);
        justify-content: center; }
  nav a { transition: .15s; font-size: .9rem; }
  nav a:hover { background:#222a3a !important; color:#cfe0ff !important; }
  .card { border-radius: 16px; padding: 1.1rem 1.3rem;
          box-shadow: 0 6px 24px rgba(0,0,0,.25);
          background: linear-gradient(180deg,#1a1e27,#15181f); }
  .card h2 { display:flex; align-items:center; gap:.5rem; }
  button { background: linear-gradient(180deg,#3b7bff,#2a5fe0); font-weight:600;
           box-shadow: 0 2px 8px rgba(40,90,220,.3); transition: transform .08s, filter .15s; }
  button:hover { filter: brightness(1.08); }
  button:active { transform: translateY(1px); }
  select { padding:.45rem; border-radius:8px; background:#0f1115; color:#e8e8e8;
           border:1px solid #333; }
  a { color:#8ab4ff; }
  /* Ergebnis-Badges */
  .badge { display:inline-block; padding:.15rem .6rem; border-radius:999px;
           font-size:.8rem; font-weight:600; }
  .badge.ok { background:#16442a; color:#7be0a3; }
  .badge.bad { background:#4a1f24; color:#ff9aa6; }
  table { font-size:.9rem; }
  @media (max-width:600px){ .hero h1{font-size:1.7rem} .card{padding:.9rem 1rem} }
</style>
</head>
<body>
  <div class="hero">
    <h1>🏈 Theo</h1>
    <p class="muted">Deine KI rund um American Football – fragen, Videos
    analysieren, Spielzüge simulieren.</p>
  </div>
  <button id="installBtn" style="display:none;background:#1f8a4c" onclick="installApp()">
    📲 Als App installieren</button>

  <nav style="margin:.6rem 0;display:flex;gap:.5rem;flex-wrap:wrap">
    <a href="#ask" style="color:#9db4e8;text-decoration:none;background:#181b22;padding:.35rem .7rem;border-radius:999px;border:1px solid #262a33">💬 Fragen</a>
    <a href="#video" style="color:#9db4e8;text-decoration:none;background:#181b22;padding:.35rem .7rem;border-radius:999px;border:1px solid #262a33">🎬 Video</a>
    <a href="#sim" style="color:#9db4e8;text-decoration:none;background:#181b22;padding:.35rem .7rem;border-radius:999px;border:1px solid #262a33">🏈 Simulator</a>
  </nav>

  <div class="card" id="ask">
    <h2>Frage stellen</h2>
    <textarea id="q" rows="2" placeholder="z. B. Was macht der Quarterback?"></textarea>
    <button id="askBtn" onclick="ask()">Fragen</button>
    <div id="examples" style="margin-top:.5rem;display:flex;gap:.35rem;flex-wrap:wrap"></div>
    <div id="answer"></div>
  </div>

  <div class="card" id="video">
    <h2>Video analysieren</h2>
    <input type="file" id="file" accept="video/*">
    <div style="margin:.5rem 0">
      <label class="muted">Detektor:
        <select id="detector">
          <option value="hog">HOG (schnell, ohne Extra)</option>
          <option value="yolo">YOLO (genauer, benötigt theo[video-yolo])</option>
        </select>
      </label>
    </div>
    <button id="anBtn" onclick="analyze()">Analysieren</button>
    <button id="vidBtn" style="background:#0d7a8c" onclick="downloadAnnotated()">⬇️ Annotiertes Video</button>
    <p class="muted">Es werden max. 30 s analysiert. Das Video wird nach der
    Analyse serverseitig gelöscht.</p>
    <div id="analysis"></div>
  </div>

  <div class="card" id="sim">
    <h2>Spielzug-Simulator</h2>
    <p class="muted">Offense-Play gegen Defense-Play antreten lassen, den Verlauf
    ansehen und mögliche Ausgänge berechnen. (Vereinfachtes Modell.)</p>
    <div style="display:flex;gap:.8rem;flex-wrap:wrap">
      <label class="muted">Offense<br><select id="offSel"></select></label>
      <label class="muted">Defense<br><select id="defSel"></select></label>
    </div>
    <details style="margin:.6rem 0">
      <summary class="muted" style="cursor:pointer">Routen anpassen & eigenen Spielzug speichern</summary>
      <div id="routeEditor" style="margin-top:.5rem"></div>
      <div style="margin-top:.4rem">
        <input type="text" id="playName" placeholder="Name für eigenen Spielzug"
               style="max-width:220px;display:inline-block;width:auto">
        <button style="background:#1f8a4c" onclick="savePlay()">💾 Speichern</button>
      </div>
      <div id="playbook" class="src" style="margin-top:.4rem"></div>
    </details>
    <div>
      <button onclick="simulateOne()">▶ Simulieren</button>
      <button style="background:#6b46c1" onclick="simulateBatch()">📊 100×</button>
      <button style="background:#b45309" onclick="advise('best_defense')">🧠 Beste Defense</button>
      <button style="background:#b45309" onclick="advise('best_offense')">🧠 Beste Offense</button>
    </div>
    <canvas id="field" width="320" height="470"
      style="width:100%;max-width:340px;display:block;background:#14401f;border-radius:8px;margin-top:.6rem;border:1px solid #262a33"></canvas>
    <div id="simOut"></div>
  </div>

<script>
const EXAMPLE_QS = [
  'Was macht der Quarterback?',
  'Wie viele Punkte gibt ein Touchdown?',
  'Was ist ein Blitz?',
  'Welche Teams sind in der AFC West?',
  'Was ist die GFL?',
  'Erkläre Cover 2.',
];
function renderExamples() {
  const el = document.getElementById('examples');
  if (!el) return;
  el.innerHTML = EXAMPLE_QS.map(q =>
    '<button type="button" style="background:#222733;font-size:.8rem;margin:0" ' +
    'onclick="askExample(this.dataset.q)" data-q="' + q.replace(/"/g, '&quot;') + '">' +
    escapeHtml(q) + '</button>').join('');
}
function askExample(q) {
  document.getElementById('q').value = q;
  ask();
  document.getElementById('ask').scrollIntoView({ behavior: 'smooth' });
}
async function ask() {
  const q = document.getElementById('q').value.trim();
  const out = document.getElementById('answer');
  const btn = document.getElementById('askBtn');
  if (!q) { out.innerHTML = '<p class="muted">Bitte eine Frage eingeben.</p>'; return; }
  btn.disabled = true; out.innerHTML = '<p class="muted">Theo denkt nach…</p>';
  try {
    const fd = new FormData(); fd.append('question', q);
    const r = await fetch('/api/ask', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Fehler');
    let html = '<pre>' + escapeHtml(d.answer) + '</pre>';
    if (d.sources && d.sources.length)
      html += '<div class="src">Quellen: ' + d.sources.map(escapeHtml).join(' · ') + '</div>';
    html += '<div class="src">Backend: ' + escapeHtml(d.backend) + '</div>';
    out.innerHTML = html;
  } catch (e) { out.innerHTML = '<p style="color:#e06">' + escapeHtml(e.message) + '</p>'; }
  finally { btn.disabled = false; }
}
async function downloadAnnotated() {
  const f = document.getElementById('file').files[0];
  const out = document.getElementById('analysis');
  const btn = document.getElementById('vidBtn');
  if (!f) { out.innerHTML = '<p class="muted">Bitte ein Video auswählen.</p>'; return; }
  btn.disabled = true; out.innerHTML = '<p class="muted">Erstelle annotiertes Video… das dauert.</p>';
  try {
    const fd = new FormData();
    fd.append('file', f);
    fd.append('detector', document.getElementById('detector').value);
    const r = await fetch('/api/analyze/video', { method: 'POST', body: fd });
    if (!r.ok) { const d = await r.json(); throw new Error(d.detail || 'Fehler'); }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'theo_annotated.mp4'; a.click();
    URL.revokeObjectURL(url);
    out.innerHTML = '<p class="muted">Annotiertes Video heruntergeladen (theo_annotated.mp4).</p>';
  } catch (e) { out.innerHTML = '<p style="color:#e06">' + escapeHtml(e.message) + '</p>'; }
  finally { btn.disabled = false; }
}
async function analyze() {
  const f = document.getElementById('file').files[0];
  const out = document.getElementById('analysis');
  const btn = document.getElementById('anBtn');
  if (!f) { out.innerHTML = '<p class="muted">Bitte ein Video auswählen.</p>'; return; }
  btn.disabled = true; out.innerHTML = '<p class="muted">Analysiere Video… das kann etwas dauern.</p>';
  try {
    const fd = new FormData();
    fd.append('file', f);
    fd.append('detector', document.getElementById('detector').value);
    fd.append('detect', 'true');
    const r = await fetch('/api/analyze', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Fehler');
    let html = '<pre>' + escapeHtml(d.summary) + '</pre>';
    if (d.keyframes && d.keyframes.length) {
      html += '<p class="muted">Annotierte Schlüsselbilder:</p>';
      for (const kf of d.keyframes) {
        html += '<figure style="margin:.5rem 0">' +
          '<img src="' + kf.image + '" alt="' + escapeHtml(kf.label) +
          '" style="width:100%;border-radius:8px;border:1px solid #262a33">' +
          '<figcaption class="src">' + escapeHtml(kf.label) +
          ' (' + kf.time_s + 's)</figcaption></figure>';
      }
    }
    out.innerHTML = html;
  } catch (e) { out.innerHTML = '<p style="color:#e06">' + escapeHtml(e.message) + '</p>'; }
  finally { btn.disabled = false; }
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// --- Spielzug-Simulator ---
let PLAYS = null, animTimer = null;
const YMIN = -9, YMAX = 46, FW = 53.3;
async function loadPlays() {
  try { PLAYS = await (await fetch('/api/plays')).json(); }
  catch (e) { return; }
  const off = document.getElementById('offSel'), def = document.getElementById('defSel');
  off.innerHTML = PLAYS.offense.map(p => '<option value="' + p.id + '">' + escapeHtml(p.name) + '</option>').join('');
  def.innerHTML = PLAYS.defense.map(p => '<option value="' + p.id + '">' + escapeHtml(p.name) + '</option>').join('');
  off.onchange = fillRoutes; fillRoutes();
  renderPlaybook();
  drawField();
}
function curOffense() { return PLAYS.offense.find(p => p.id === document.getElementById('offSel').value); }
function fillRoutes() {
  const cur = curOffense(), ed = document.getElementById('routeEditor');
  ed.innerHTML = PLAYS.slots.map(s => {
    const opts = PLAYS.routes.map(r => '<option value="' + r + '"' + (cur.routes[s.id] === r ? ' selected' : '') + '>' + r + '</option>').join('');
    return '<label class="muted" style="display:inline-block;margin:.2rem .5rem .2rem 0">' + s.id +
      '<br><select data-slot="' + s.id + '">' + opts + '</select></label>';
  }).join('');
}
function curRequest(extra) {
  const routes = {};
  document.querySelectorAll('#routeEditor select').forEach(s => routes[s.dataset.slot] = s.value);
  return Object.assign({ routes: routes, kind: curOffense().kind,
    defense_id: document.getElementById('defSel').value }, extra);
}
function field() { return document.getElementById('field'); }
function sx(x) { return x / FW * field().width; }
function sy(y) { return field().height - (y - YMIN) / (YMAX - YMIN) * field().height; }
function drawField() {
  const cv = field(), ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = '#14401f'; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = 1;
  ctx.fillStyle = 'rgba(255,255,255,0.5)'; ctx.font = '9px sans-serif';
  for (let y = -5; y <= 45; y += 5) {
    ctx.beginPath(); ctx.moveTo(0, sy(y)); ctx.lineTo(cv.width, sy(y)); ctx.stroke();
  }
  // Line of Scrimmage betonen.
  ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(0, sy(0)); ctx.lineTo(cv.width, sy(0)); ctx.stroke();
}
function drawFrame(players, ball) {
  drawField();
  const ctx = field().getContext('2d');
  for (const p of players) {
    if (p.role === 'OL') continue; // Linemen ausblenden für Übersicht
    ctx.beginPath();
    ctx.fillStyle = p.team === 'offense' ? '#3b82f6' : '#ef4444';
    ctx.arc(sx(p.x), sy(p.y), 6, 0, 6.29); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = '8px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(p.role, sx(p.x), sy(p.y) - 8);
  }
  if (ball) {
    ctx.beginPath(); ctx.fillStyle = '#c98a3a';
    ctx.arc(sx(ball.x), sy(ball.y), 4, 0, 6.29); ctx.fill();
  }
}
function animate(d) {
  clearInterval(animTimer);
  let i = 0;
  animTimer = setInterval(() => {
    if (i >= d.frames.length) { clearInterval(animTimer); return; }
    drawFrame(d.frames[i], d.ball[i]); i++;
  }, Math.max(40, d.dt * 1000));
}
async function postJSON(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body) });
  const d = await r.json();
  if (!r.ok) throw new Error(d.detail || 'Fehler');
  return d;
}
async function simulateOne() {
  const out = document.getElementById('simOut');
  try {
    const d = await postJSON('/api/simulate', curRequest({ seed: Math.floor(Math.random() * 1e6) }));
    let txt = d.summary;
    if (d.notes && d.notes.length) txt += '\\n' + d.notes.join('\\n');
    out.innerHTML = '<pre>' + escapeHtml(txt) + '</pre>';
    animate(d);
  } catch (e) { out.innerHTML = '<p style="color:#e06">' + escapeHtml(e.message) + '</p>'; }
}
async function simulateBatch() {
  const out = document.getElementById('simOut');
  out.innerHTML = '<p class="muted">Simuliere 100 Spielzüge…</p>';
  try {
    const d = await postJSON('/api/simulate/batch', curRequest({ n: 100, seed: Math.floor(Math.random() * 1e5) }));
    let html = '<pre>Ø ' + d.mean_yards + ' Yards (Median ' + d.median_yards +
      ', best ' + d.best_yards + ', worst ' + d.worst_yards + ')</pre>';
    const labels = { complete: 'Komplett', incomplete: 'Inkomplett', sack: 'Sack',
      interception: 'Interception', run: 'Lauf' };
    for (const k of Object.keys(d.outcome_pct)) {
      const pct = d.outcome_pct[k];
      html += '<div style="margin:2px 0"><span class="muted" style="display:inline-block;width:90px">' +
        (labels[k] || k) + '</span>' +
        '<span style="display:inline-block;height:12px;background:#2d6cdf;width:' + (pct * 1.5) +
        'px;border-radius:3px;vertical-align:middle"></span> ' + pct + '%</div>';
    }
    out.innerHTML = html;
  } catch (e) { out.innerHTML = '<p style="color:#e06">' + escapeHtml(e.message) + '</p>'; }
}
async function advise(mode) {
  const out = document.getElementById('simOut');
  out.innerHTML = '<p class="muted">Berechne Matchups…</p>';
  try {
    const d = await postJSON('/api/advisor', curRequest({ mode: mode, n: 60 }));
    const title = mode === 'best_defense'
      ? 'Beste Defense gegen ' + escapeHtml(d.vs) + ' (wenigste Yards zuerst):'
      : 'Beste Offense gegen ' + escapeHtml(d.vs) + ' (meiste Yards zuerst):';
    let html = '<p class="muted">' + title + '</p><table style="width:100%;border-collapse:collapse">';
    html += '<tr class="src"><th align="left">Spielzug</th><th>Ø Yd</th><th>Compl</th><th>Sack</th></tr>';
    d.rows.forEach((r, i) => {
      const compl = r.outcome_pct.complete || 0, sack = r.outcome_pct.sack || 0;
      html += '<tr style="border-top:1px solid #262a33">' +
        '<td>' + (i === 0 ? '⭐ ' : '') + escapeHtml(r.name) + '</td>' +
        '<td align="center">' + r.mean_yards + '</td>' +
        '<td align="center" class="src">' + compl + '%</td>' +
        '<td align="center" class="src">' + sack + '%</td></tr>';
    });
    out.innerHTML = html + '</table>';
  } catch (e) { out.innerHTML = '<p style="color:#e06">' + escapeHtml(e.message) + '</p>'; }
}

// --- Playbook (eigene Spielzüge im Browser speichern) ---
function loadBook() { try { return JSON.parse(localStorage.getItem('theo_playbook') || '[]'); } catch (e) { return []; } }
function saveBook(b) { localStorage.setItem('theo_playbook', JSON.stringify(b)); }
function savePlay() {
  const name = (document.getElementById('playName').value || '').trim();
  if (!name) { alert('Bitte einen Namen eingeben.'); return; }
  const req = curRequest({});
  const book = loadBook().filter(p => p.name !== name);
  book.push({ name: name, routes: req.routes, kind: req.kind });
  saveBook(book); document.getElementById('playName').value = ''; renderPlaybook();
}
function loadPlay(name) {
  const p = loadBook().find(x => x.name === name); if (!p) return;
  document.querySelectorAll('#routeEditor select').forEach(s => {
    if (p.routes[s.dataset.slot]) s.value = p.routes[s.dataset.slot];
  });
}
function deletePlay(name) { saveBook(loadBook().filter(p => p.name !== name)); renderPlaybook(); }
function renderPlaybook() {
  const el = document.getElementById('playbook'); const book = loadBook();
  if (!book.length) { el.innerHTML = 'Noch keine eigenen Spielzüge gespeichert.'; return; }
  el.innerHTML = 'Eigene Spielzüge: ' + book.map(p =>
    '<span style="display:inline-block;margin:2px 4px">' +
    '<a href="#" onclick="loadPlay(\\'' + p.name.replace(/'/g, "") + '\\');return false">' + escapeHtml(p.name) + '</a> ' +
    '<a href="#" onclick="deletePlay(\\'' + p.name.replace(/'/g, "") + '\\');return false" style="color:#e06">✕</a></span>').join('');
}

window.addEventListener('load', () => { renderExamples(); loadPlays(); });

// PWA: Service Worker registrieren.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () =>
    navigator.serviceWorker.register('/sw.js').catch(() => {}));
}
// PWA: Installations-Button anzeigen, wenn installierbar.
let deferredPrompt = null;
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredPrompt = e;
  document.getElementById('installBtn').style.display = 'inline-block';
});
async function installApp() {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  await deferredPrompt.userChoice;
  deferredPrompt = null;
  document.getElementById('installBtn').style.display = 'none';
}
window.addEventListener('appinstalled', () => {
  document.getElementById('installBtn').style.display = 'none';
});
</script>
</body>
</html>"""

_MANIFEST = '''{
  "name": "Theo – Football-KI",
  "short_name": "Theo",
  "description": "Fragen rund um American Football beantworten und Spiel-/Trainingsvideos analysieren.",
  "start_url": "/",
  "scope": "/",
  "display": "standalone",
  "orientation": "portrait",
  "background_color": "#0f1115",
  "theme_color": "#0f1115",
  "lang": "de",
  "categories": ["sports", "education"],
  "icons": [
    {"src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png"},
    {"src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png"},
    {"src": "/static/icons/icon-512-maskable.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"}
  ]
}'''

_SERVICE_WORKER = '''const CACHE = 'theo-v1';
const SHELL = [
  '/',
  '/manifest.webmanifest',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png'
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Network-first mit Cache-Fallback; API-Aufrufe nicht abfangen.
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.pathname.startsWith('/api/')) return;
  e.respondWith(
    fetch(req)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(req, copy));
        return res;
      })
      .catch(() => caches.match(req).then((r) => r || caches.match('/')))
  );
});'''


# Modul-Level-App für `uvicorn theo.web.app:app`.
app = create_app()

__all__ = ["app", "create_app"]
