Python-Bibliotheken installieren:
Pillow: Bildverarbeitung und Export
OpenCV: Bildoptimierung und Objekterkennung
rembg: KI-basierte Hintergrundentfernung
numpy: Numerische Operationen

Der Workflow

Bild einlesen - beliebiges Format und Auflösung
Helligkeit/Sättigung optimieren - automatische Kontrastanpassung
Objekt erkennen und zentrieren - Kantenerkennung für Objektposition
Quadratischen Ausschnitt erstellen - um das Objekt herum
Hintergrund entfernen - mit KI-Modell
Weißen Hintergrund hinzufügen - RGB(255,255,255)
Als WebP exportieren - 1024x1024px

pip install loguru
```

### Logging-Features:

1. **Zweifache Ausgabe:**
   - Farbige Konsolen-Ausgabe mit Timestamps
   - Automatische Log-Dateien: `product_processor_YYYY-MM-DD.log`

2. **Log-Levels:**
   - `DEBUG`: Technische Details (Shape, Werte, Koordinaten)
   - `INFO`: Prozessschritte und wichtige Infos
   - `SUCCESS`: Erfolgreiche Abschlüsse
   - `WARNING`: Potenzielle Probleme (z.B. Bildrand)
   - `ERROR`: Fehler mit Details
   - `EXCEPTION`: Kritische Fehler mit Stacktrace

3. **Detaillierte Informationen pro Schritt:**
   - Bildauflösungen und Dateigrößen
   - Vor/Nach-Werte bei Optimierungen
   - Objektposition und -größe (absolut + prozentual)
   - Kontur- und Kantenstatistiken
   - Transparenzanalyse
   - Kompressionsraten

4. **Log-Rotation:**
   - Täglich neue Log-Datei um Mitternacht
   - Automatische Löschung nach 30 Tagen

### Beispiel-Ausgabe:
```
2024-12-09 15:30:45 | INFO     | Bild geladen - Auflösung: 3024x4032px
2024-12-09 15:30:46 | DEBUG    | Helligkeit Ø: 128.34→142.67
2024-12-09 15:30:47 | SUCCESS  | Objekt erkannt - Zentrum: (1512, 2016) [50.0%, 50.0%]