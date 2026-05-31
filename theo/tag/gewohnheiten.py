"""Gewohnheiten aus dem Aktivitäts-Verlauf erkennen.

Idee: Wenn du dieselbe Tätigkeit immer wieder am selben Wochentag zur ähnlichen
Uhrzeit machst, ist das eine Gewohnheit. Theo findet solche Muster automatisch
und kann sie dann in deinen Tagesplan einbauen und dich daran erinnern.

Das Verfahren ist bewusst einfach und nachvollziehbar (keine Blackbox):
1. Aktivitäten nach (Tätigkeit, Wochentag) gruppieren.
2. Kommt eine Kombination an genügend *verschiedenen* Tagen vor -> Gewohnheit.
3. Die typische Uhrzeit ist der Median der beobachteten Zeiten.
4. Je enger die Zeiten beieinander liegen, desto höher die Konfidenz.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from .modell import WOCHENTAGE, Aktivitaet


@dataclass
class Gewohnheit:
    name: str
    wochentag: int        # 0=Mo ... 6=So
    uhrzeit: str          # typische Uhrzeit "HH:MM"
    vorkommen: int        # an wie vielen verschiedenen Tagen beobachtet
    konfidenz: float      # 0..1, wie verlässlich das Muster ist

    def __str__(self) -> str:
        proz = int(self.konfidenz * 100)
        return (f"{WOCHENTAGE[self.wochentag]} {self.uhrzeit}  {self.name} "
                f"(gesehen an {self.vorkommen} Tagen, Konfidenz {proz}%)")


def _hhmm(minuten: int) -> str:
    return f"{minuten // 60:02d}:{minuten % 60:02d}"


def erkenne_gewohnheiten(
    verlauf: list[Aktivitaet],
    min_vorkommen: int = 2,
    max_streuung_min: int = 90,
) -> list[Gewohnheit]:
    """Findet Gewohnheiten im Verlauf.

    - ``min_vorkommen``: an so vielen verschiedenen Tagen muss es vorkommen
    - ``max_streuung_min``: streuen die Uhrzeiten weiter, sinkt die Konfidenz
    """
    # (name, wochentag) -> { datum: [minuten_des_tages, ...] }
    gruppen: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for a in verlauf:
        z = a.zeit
        schluessel = (a.name.lower().strip(), z.weekday())
        minute = z.hour * 60 + z.minute
        gruppen[schluessel][z.date().isoformat()].append(minute)

    gewohnheiten: list[Gewohnheit] = []
    for (name, wochentag), tage in gruppen.items():
        if len(tage) < min_vorkommen:
            continue
        # pro Tag die früheste Zeit nehmen (eine Gewohnheit je Tag)
        zeiten = [min(minuten) for minuten in tage.values()]
        median = int(statistics.median(zeiten))
        streuung = statistics.pstdev(zeiten) if len(zeiten) > 1 else 0.0
        # Konfidenz: viele Tage gut, große Streuung schlecht
        konf_zeit = max(0.0, 1.0 - streuung / max_streuung_min)
        konf_menge = min(1.0, len(tage) / 5.0)
        konfidenz = round(0.5 * konf_zeit + 0.5 * konf_menge, 2)
        gewohnheiten.append(
            Gewohnheit(name, wochentag, _hhmm(median), len(tage), konfidenz)
        )

    gewohnheiten.sort(key=lambda g: (g.wochentag, g.uhrzeit))
    return gewohnheiten
