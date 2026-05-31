"""Web-Backend für Theo (FastAPI): Q&A + Video-Upload-Analyse.

Startet eine kleine Single-Page-Oberfläche, über die man Football-Fragen stellen
und Spiel-/Trainingsvideos hochladen kann.

Starten:
    theo serve            # oder: uvicorn theo.web.app:app
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from theo.qa import QAEngine

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Das Web-Backend benötigt FastAPI. Installiere es mit "
        "`pip install theo[web]`."
    ) from exc

_STATIC_DIR = Path(__file__).parent / "static"

# Erlaubte Video-Endungen für den Upload.
_ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
# Obergrenze für die analysierte Videolänge (Sekunden), um Laufzeit zu begrenzen.
_MAX_ANALYZE_SECONDS = 30.0


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
        detector: str = Form("hog"),
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
            with tmp_path.open("wb") as out:
                shutil.copyfileobj(file.file, out)

            payload = _run_analysis(tmp_path, detector=detector, detect=detect)
            return JSONResponse(payload)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return app


def _run_analysis(path: Path, *, detector: str, detect: bool) -> dict:
    if detect:
        from theo.video import VideoPipeline

        result = VideoPipeline(detector=detector).process(
            path, max_seconds=_MAX_ANALYZE_SECONDS
        )
        data = {
            "mode": "detect",
            "summary": result.summary(),
            "detector": result.detector_name,
            "avg_players": round(result.avg_players, 1),
            "max_players": result.max_players,
            "active_segments": len(result.active_segments),
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
</style>
</head>
<body>
  <h1>🏈 Theo – die Football-KI</h1>
  <p class="muted">Stelle Fragen rund um American Football oder lade ein Spiel-/
  Trainingsvideo zur Analyse hoch.</p>
  <button id="installBtn" style="display:none;background:#1f8a4c" onclick="installApp()">
    📲 Als App installieren</button>

  <div class="card">
    <h2>Frage stellen</h2>
    <textarea id="q" rows="2" placeholder="z. B. Was macht der Quarterback?"></textarea>
    <button id="askBtn" onclick="ask()">Fragen</button>
    <div id="answer"></div>
  </div>

  <div class="card">
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
    <p class="muted">Es werden max. 30 s analysiert. Das Video wird nach der
    Analyse serverseitig gelöscht.</p>
    <div id="analysis"></div>
  </div>

<script>
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
    out.innerHTML = '<pre>' + escapeHtml(d.summary) + '</pre>';
  } catch (e) { out.innerHTML = '<p style="color:#e06">' + escapeHtml(e.message) + '</p>'; }
  finally { btn.disabled = false; }
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

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
