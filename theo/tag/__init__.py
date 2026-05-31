"""Theo Tag – dein persönlicher, lokaler Tagesplaner-Assistent.

Theo Tag plant deinen Tag, erkennt deine Gewohnheiten aus dem, was du tust, und
erinnert dich rechtzeitig – per Konsole, Desktop-Benachrichtigung oder Telegram.

Wichtig: **Alle Daten bleiben lokal** auf deinem Gerät (eine einfache
JSON-Datei). Ein Assistent, der deinen ganzen Tagesablauf kennt, sollte deine
Daten nicht in die Cloud schicken – das ist hier eingebaut.

Bausteine:
- ``modell``       : Datentypen (Termin, Aufgabe, Aktivität) + lokale Speicherung
- ``gewohnheiten`` : erkennt wiederkehrende Muster aus deinem Aktivitäts-Verlauf
- ``planer``       : baut aus Terminen, Gewohnheiten & Aufgaben den Tagesplan
- ``erinnerung``   : welche Erinnerungen sind fällig + Versandkanäle (Push)
- ``cli``          : Kommandozeile
"""
