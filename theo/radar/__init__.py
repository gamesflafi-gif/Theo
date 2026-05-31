"""Theo Radar – der Signalverarbeitungs-Kern eines zivilen Radars.

Wir simulieren ein FMCW-Radar (Frequency Modulated Continuous Wave) – die
Bauart, die in Auto-, Drohnen- und Verkehrssensorik steckt – und verarbeiten
die Signale mit denselben Verfahren wie echte Hardware:

- ``szenario``     : Radar-Parameter und Ziele
- ``signal``       : aus Zielen ein realistisches Empfangssignal erzeugen (mit Rauschen)
- ``verarbeitung`` : Range-Doppler-Karte (2D-FFT) + CFAR-Zielerkennung
- ``tracking``     : Ziele über die Zeit verfolgen (Kalman-Filter)
- ``plot``         : Visualisierung
- ``cli``          : ein komplettes Szenario durchrechnen

Die Algorithmen sind identisch mit denen auf echter Radar-Hardware – nur die
Antenne/HF-Elektronik fehlt (und ist in DE frequenz-reguliert).
"""
