"""Theo — unser eigenes, von Grund auf gebautes Sprachmodell.

Theo ist ein kleiner GPT-Transformer (dieselbe Grundarchitektur wie ChatGPT),
den wir selbst trainieren. Module:

- ``model``    : die neuronale Netz-Architektur (der Transformer)
- ``data``     : Texte laden, in Zahlen umwandeln, Trainingshäppchen bauen
- ``train``    : das Modell trainieren
- ``generate`` : mit dem trainierten Modell neuen Text schreiben
"""

__version__ = "0.1.0"
