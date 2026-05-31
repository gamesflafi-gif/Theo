"""Erinnerungen berechnen und als Push-Nachricht verschicken.

Zwei Teile:
1. *Welche* Erinnerung ist gerade fällig? (reine Logik, gut testbar)
2. *Wie* wird sie zugestellt? – über austauschbare "Melder":
   - Konsole  : einfach ausgeben (immer verfügbar)
   - Desktop  : System-Benachrichtigung (Linux: notify-send)
   - Telegram : echte Push aufs Handy (kostenlos, braucht Bot-Token)

Alles läuft lokal; nur der Telegram-Melder schickt – auf deinen Wunsch – die
reine Erinnerungs-Nachricht an deinen eigenen Telegram-Chat.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime

from .planer import Planeintrag


def faellige_erinnerungen(
    eintraege: list[Planeintrag],
    jetzt: datetime | None = None,
    fenster_min: int = 1,
) -> list[Planeintrag]:
    """Gibt die Einträge zurück, deren Erinnerung *jetzt* fällig ist.

    Fällig = aktuelle Uhrzeit liegt im Fenster
    [Start − Vorlauf, Start − Vorlauf + ``fenster_min``).
    Das ``fenster_min`` passt zum Prüf-Intervall des Dienstes (z. B. jede Minute).
    """
    jetzt = jetzt or datetime.now()
    jetzt_min = jetzt.hour * 60 + jetzt.minute
    faellig = []
    for e in eintraege:
        ausloeser = e.minute - e.erinnerung_min
        if ausloeser <= jetzt_min < ausloeser + fenster_min:
            faellig.append(e)
    return faellig


def nachricht(e: Planeintrag) -> tuple[str, str]:
    """Baut Titel und Text einer Erinnerung."""
    vorlauf = f"in {e.erinnerung_min} Min" if e.erinnerung_min > 0 else "jetzt"
    return (f"Theo erinnert: {e.titel}", f"Um {e.uhrzeit} ({vorlauf}): {e.titel}")


# ---------------------------------------------------------------- Melder ----

class KonsolenMelder:
    """Gibt die Erinnerung auf der Konsole aus (immer verfügbar)."""

    def sende(self, titel: str, text: str) -> bool:
        print(f"\n🔔 {titel}\n   {text}")
        return True


class DesktopMelder:
    """System-Benachrichtigung über ``notify-send`` (Linux-Desktop)."""

    def verfuegbar(self) -> bool:
        return shutil.which("notify-send") is not None

    def sende(self, titel: str, text: str) -> bool:
        if not self.verfuegbar():
            return False
        try:
            subprocess.run(["notify-send", titel, text], check=True, timeout=10)
            return True
        except Exception:
            return False


class TelegramMelder:
    """Echte Push aufs Handy über einen Telegram-Bot.

    Einrichtung (einmalig):
    1. In Telegram dem @BotFather schreiben -> /newbot -> Token erhalten.
    2. Dem eigenen Bot eine Nachricht schicken, damit ein Chat existiert.
    3. Chat-ID herausfinden (z. B. über die getUpdates-URL des Bots).
    4. Token & Chat-ID als Umgebungsvariablen setzen:
         export THEO_TG_TOKEN="123456:ABC..."
         export THEO_TG_CHAT="987654321"
    """

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self.token = token or os.environ.get("THEO_TG_TOKEN")
        self.chat_id = chat_id or os.environ.get("THEO_TG_CHAT")

    def verfuegbar(self) -> bool:
        return bool(self.token and self.chat_id)

    def sende(self, titel: str, text: str) -> bool:
        if not self.verfuegbar():
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        daten = urllib.parse.urlencode(
            {"chat_id": self.chat_id, "text": f"🔔 {titel}\n{text}"}
        ).encode()
        try:
            with urllib.request.urlopen(url, data=daten, timeout=15) as r:
                return r.status == 200
        except Exception:
            return False


def waehle_melder() -> object:
    """Wählt automatisch den besten verfügbaren Melder."""
    tg = TelegramMelder()
    if tg.verfuegbar():
        return tg
    desktop = DesktopMelder()
    if desktop.verfuegbar():
        return desktop
    return KonsolenMelder()


def melde_faellige(eintraege: list[Planeintrag], melder=None, jetzt=None,
                   fenster_min: int = 1) -> int:
    """Verschickt alle gerade fälligen Erinnerungen. Gibt die Anzahl zurück."""
    melder = melder or waehle_melder()
    anzahl = 0
    for e in faellige_erinnerungen(eintraege, jetzt, fenster_min):
        titel, text = nachricht(e)
        if melder.sende(titel, text):
            anzahl += 1
    return anzahl
