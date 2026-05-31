# Theo auf einem STRATO V-Server hosten

Diese Anleitung bringt Theo als installierbare App (PWA) mit **automatischem
HTTPS** auf einen STRATO V-Server (Linux, root). Mit HTTPS lässt sich Theo dann
auf Handy und Desktop als App installieren.

---

## 1. Was du bei STRATO bestellen musst

### a) V-Server (VPS mit root-Zugang)
Im STRATO-Kundenbereich einen **V-Server** ("Linux V-Server" / VPS) bestellen –
**nicht** das normale "Webhosting" (das kann kein Python/Docker).

**Empfohlene Ausstattung:**

| Nutzung | vCPU | RAM | SSD | Hinweis |
|---|---|---|---|---|
| **Nur Q&A + HOG-Video** (Standard) | 2 | 4 GB | 50 GB | reicht gut |
| **Mit YOLO-Detektor** (genaue Video-Analyse) | 4 | 8 GB | 100 GB | PyTorch braucht RAM/Disk |

- **Betriebssystem:** Ubuntu 22.04 LTS (oder Debian 12) wählen.
- Du bekommst von STRATO eine **feste IPv4-Adresse** und SSH-Zugangsdaten
  (Root-Passwort bzw. SSH-Key).

### b) Domain
Eine **Domain oder Subdomain** ist für HTTPS nötig (z. B. `theo.deinedomain.de`).
- Entweder eine vorhandene Domain nutzen oder bei STRATO eine registrieren.
- **DNS einrichten:** Im Domain-Verwaltungsbereich einen **A-Record** anlegen,
  der auf die **IPv4 deines V-Servers** zeigt.
  - Name: `theo` (für `theo.deinedomain.de`) oder `@` (für die Hauptdomain)
  - Typ: `A`, Wert: die Server-IP
- Die DNS-Änderung kann bis zu einige Stunden dauern.

> Ohne Domain geht es nur über die IP – dann gibt es **kein gültiges
> HTTPS-Zertifikat** und die App ist auf dem Handy nicht installierbar.

---

## 2. Server vorbereiten (einmalig)

Per SSH einloggen (Daten aus dem STRATO-Kundenbereich):

```bash
ssh root@DEINE_SERVER_IP
```

Docker + Compose installieren:

```bash
apt update && apt upgrade -y
curl -fsSL https://get.docker.com | sh
```

Firewall öffnen (Ports 80 und 443 für HTTP/HTTPS, 22 für SSH):

```bash
ufw allow 22 && ufw allow 80 && ufw allow 443 && ufw --force enable
```

---

## 3. Theo holen und konfigurieren

```bash
git clone <DEIN_REPO_URL> theo && cd theo
git checkout claude/american-football-model-fit6N

cp .env.example .env
nano .env
```

In `.env` mindestens setzen:
```ini
DOMAIN=theo.deinedomain.de
ACME_EMAIL=du@deinedomain.de
```
Optional: `ANTHROPIC_API_KEY=...` (bessere Antworten via Claude),
`INSTALL_YOLO=true` + `THEO_DEFAULT_DETECTOR=yolo` (genaue Video-Analyse).

---

## 4. Starten

```bash
docker compose up -d --build
```

Beim ersten Aufruf von `https://theo.deinedomain.de` holt Caddy automatisch ein
Let's-Encrypt-Zertifikat. Danach ist Theo erreichbar – fertig.

**Logs ansehen / Status:**
```bash
docker compose logs -f
docker compose ps
```

**Aktualisieren (nach Code-Änderungen):**
```bash
git pull && docker compose up -d --build
```

**Stoppen:**
```bash
docker compose down
```

---

## 5. Als App installieren

`https://theo.deinedomain.de` im Browser öffnen:
- **Android (Chrome):** Menü → „App installieren" / „Zum Startbildschirm".
- **iPhone (Safari):** Teilen → „Zum Home-Bildschirm".
- **Desktop (Chrome/Edge):** Install-Symbol in der Adressleiste oder der Button
  „📲 Als App installieren".

---

## Troubleshooting

- **Kein Zertifikat / HTTPS-Fehler:** DNS-A-Record korrekt auf die Server-IP?
  Ports 80/443 in der Firewall offen? `docker compose logs caddy` prüfen.
- **Video-Analyse langsam/Out-of-Memory:** RAM zu klein für YOLO →
  `INSTALL_YOLO=false` (HOG) oder größeren V-Server wählen.
- **Upload abgelehnt (413):** `THEO_MAX_UPLOAD_MB` in `.env` erhöhen und
  `docker compose up -d` erneut ausführen.
- **Port 80 belegt:** läuft schon ein anderer Webserver? Stoppen
  (`systemctl stop apache2` o. ä.).
