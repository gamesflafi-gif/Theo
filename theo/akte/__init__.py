"""Theo Akte – der lokale, datenschutzsichere Dokumenten-Assistent.

Idee: vertrauliche Dokumente (Verträge, Akten, Gutachten) blitzschnell
durchsuchen und passende Stellen finden – **komplett offline**. Kein Byte
verlässt den Rechner, daher DSGVO-konform und auch für Kanzleien, Praxen,
Steuerberater und Behörden nutzbar, die Cloud-KI nicht verwenden dürfen.

Bausteine:
- ``dokumente`` : Dateien einlesen und in durchsuchbare Häppchen schneiden
- ``suche``     : ein lokaler Suchindex (BM25), der die relevantesten Stellen findet
- ``cli``       : Kommandozeile (indizieren & fragen)
"""
