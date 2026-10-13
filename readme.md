# Foto_tools

Sammlung von Skripten und GUIs zur Bearbeitung von **Shop- und Produktbildern** (insbesondere Fisch-/Köderfotos): Hintergrundentfernung mit **rembg**, OpenCV-Optimierung, Größenanpassung, Umbenennung per Excel-Liste, URL-Download sowie Übersicht, welche Produkte noch fotografiert werden müssen.

## Voraussetzungen

```bash
pip install -r requirements.txt
```

**Hauptabhängigkeiten:** OpenCV, Pillow, rembg, numpy, pandas, openpyxl, requests, loguru, gradio, dearpygui, customtkinter, rawpy

---

## Übersicht nach Kategorien

| Kategorie | Dateien |
|-----------|---------|
| Produkt-Pipeline (Kern) | `complete_pipeline.py`, `pipeline_GUI.py` |
| Gradio-Oberflächen | `gradio_app.py`, `gradio_2_app.py`, `foto_gui.py`, `foto_gui_02.py` |
| Helios (rembg + WebP) | `Helios_BV.py`, `Helios_GUI.py`, `Helios_BV_Batch.py` |
| Hintergrund entfernen | `BGremove.py`, `bgremove_gui.py`, `remove_bg.py`, `reba_enha.py` |
| Download & Umbenennen | `downloader.py`, `foto_umbenennen.py`, `MatchingNumbers.py`, `umbennnen.py` |
| Foto-Übersicht | `produktfoto_tracker.py` |
| CustomTkinter-Einzeltools | `weiss_quadrat_webp.py`, `varianten_kopierer.py`, `massband_overlay.py`, `cr3_quadrat.py`, `bild_resizer.py` |
| Einzeloperationen | `foto_rotator.py`, `image_rotator.py`, `resizer.py`, `finale.py` |
| Setup & Referenz | `installer.bat`, `mermaid.md`, `poster`, `names_skins.py`, `output_names.py`, `axis_alignment.py` |

---

## Produkt-Pipeline (Kern)

### `complete_pipeline.py`

Zentrale **Kommandozeilen-Pipeline** für Produktbilder. Verarbeitet ein Bild in mehreren Schritten:

1. Helligkeit & Sättigung (CLAHE, HSV, Unsharp Mask)
2. Hintergrundentfernung via **rembg**
3. Schatten-Artefakte am Rand bereinigen
4. Objekt zentrieren, quadratisch zuschneiden (1024×1024 px)
5. Weißer Hintergrund, Export als **WebP**

Logging nach `product_processor_YYYY-MM-DD.log`.

```bash
python complete_pipeline.py
```

*(Dateinamen in der `__main__`-Sektion anpassen.)*

### `pipeline_GUI.py`

**DearPyGUI**-Oberfläche für dieselbe Pipeline. Eingabe-/Ausgabeordner wählen, Schritte einzeln aktivieren (Hintergrund, Farben, Schärfung, Auto-Korrektur, Resize) und Helligkeit/Sättigung per Slider justieren. Batch-Verarbeitung mit Fortschrittsanzeige.

```bash
python pipeline_GUI.py
```

---

## Gradio-Oberflächen

Alle Gradio-Apps starten einen lokalen Webserver (Standard: `http://127.0.0.1:7860`).

### `gradio_app.py`

Webbasierte UI für `complete_pipeline.py` mit zwei Tabs:

- **Batch-Verarbeitung:** Eingabe- → Ausgabeordner, optional Live-Vorschau
- **Einzelbild-Vorschau:** Ein Bild hochladen und Ergebnis prüfen

### `gradio_2_app.py`

Erweiterte Variante von `gradio_app.py` mit zusätzlichen Batch-Optionen (Vorschau-Intervall, Verzögerung zwischen Vorschau-Updates, Überschreiben vorhandener Dateien).

### `foto_gui.py`

Eigenständige **Gradio-Batch-App** mit eingebauter Pipeline (nicht nur Wrapper um `complete_pipeline.py`). Einstellbare Parameter:

- Rotation (0° / 90° / 180° / 270°)
- Helligkeit, Sättigung, Kontrast
- Alpha-Matting für rembg

Reihenfolge: Rotation → Farbanpassung → Hintergrundentfernung → Zentrierung → WebP.

### `foto_gui_02.py`

Erweiterte Version von `foto_gui.py` mit zusätzlichem Schritt **Schattenentfernung** vor der Farbanpassung (9 statt 8 Verarbeitungsschritte). Für Bilder mit grauen Rand-Artefakten nach der Freistellung.

---

## Helios-Familie

Helios-Skripte nutzen **rembg** + **PIL** für einen schlanken Workflow: Hintergrund entfernen → drehen → zentrieren → quadratisch → **WebP**.

### `Helios_BV.py`

Kernmodul und **CLI-Tool** (kein GUI). Verarbeitet ein einzelnes Bild:

- Hintergrund entfernen
- 90° gegen den Uhrzeigersinn drehen
- Objekt zentriert auf quadratischem Canvas
- Ausgabe als WebP (Standard: 1024×1024, Qualität 80)

```bash
python Helios_BV.py bild.jpg
python Helios_BV.py bild.jpg --output ergebnis.webp --size 1024 --quality 80
```

### `Helios_GUI.py`

Einfache **Tkinter-GUI** für Helios-Batch: Eingabe-/Ausgabeordner wählen, Größe und WebP-Qualität einstellen, alle Bilder im Ordner verarbeiten.

```bash
python Helios_GUI.py
```

### `Helios_BV_Batch.py`

Erweiterte **Tkinter-Batch-GUI** auf Basis der Helios-Logik. Zusätzliche Optionen gegenüber `Helios_GUI.py`:

- Drehung (keine / 90° / 180° / 270°)
- Optionale Helligkeits- und Sättigungsanpassung
- Fortschrittsbalken und Log-Ausgabe im Fenster

```bash
python Helios_BV_Batch.py
```

---

## Hintergrund entfernen

### `BGremove.py`

**Kommandozeilen-Tool** für hochwertige Hintergrundentfernung. Unterstützt Einzelbilder, mehrere Dateien oder ganze Ordner.

- Ausgabe: PNG mit Alpha-Kanal, separate Maske, Vorschau auf Schachbrettmuster
- Wählbare rembg-Modelle (`u2net`, `isnet-general-use`, `u2net_human_seg`, …)
- Alpha-Matting für saubere Kanten

```bash
python BGremove.py bild.jpg
python BGremove.py --ordner ./fotos --ausgabe ./ergebnisse --modell isnet-general-use
```

### `bgremove_gui.py`

**Tkinter-GUI** für `BGremove.py` mit Vorschau-Panels (Original vs. Ergebnis), Modellauswahl, Ordner-Batch und Fortschrittsanzeige.

```bash
python bgremove_gui.py
```

### `remove_bg.py`

Schlanke **CLI** nur für Hintergrundentfernung: JPG → transparentes **WebP** (lossless). Einzelbild oder ganzer Ordner.

```bash
python remove_bg.py bild.jpg
```

### `reba_enha.py`

Wie `remove_bg.py`, erweitert um **Helligkeit/Kontrast** (PIL ImageEnhance), Zentrierung und feste Ausgabegröße (Standard 1024×1024).

```bash
python reba_enha.py bild.jpg
```

---

## Foto-Übersicht

### `produktfoto_tracker.py`

**CustomTkinter + SQLite:** Übersicht, welche Produkte noch zu fotografieren sind, welche Fotos vorliegen und wo Original- bzw. bearbeitete Bilder liegen (Google Drive).

Namensaufbau: **Präfix** + variabler Teil (z. B. `12g`) + **Suffix**. Präfix, Suffix und Bezeichner nur beim ersten Produkt einer Serie, danach nur noch den variablen Teil eingeben. Der **Bezeichner** (max. 30 Zeichen) gilt für alle Produkte mit gleichem Suffix.

Weitere Felder: Produkt vorhanden, Foto erledigt (Kreuzchen). Die Spalte **Geändert** (`geaendert_am`) wird bei jeder Änderung automatisch gesetzt.

**Bilderordner einlesen:** Dateiname ohne Endung wird als Produktname eingetragen, **Foto vorhanden** gesetzt.

Datenbank: `produktfotos.db` (liegt in `.gitignore`).

```bash
python produktfoto_tracker.py
```

---

## CustomTkinter-Einzeltools

### `weiss_quadrat_webp.py`

**Batch-GUI:** Alle Bilder eines Ordners auf weißen Hintergrund legen, proportional auf ein Quadrat skalieren (Kantenlänge **1000–3000 px**, Standard 2000) und als **WebP** speichern (Qualität einstellbar, Standard **80 %**). Transparente Bereiche werden weiß; die längste Seite füllt das Quadrat, der Rest ist weißer Rand.

```bash
python weiss_quadrat_webp.py
```

### `varianten_kopierer.py`

Kopiert ein gewähltes Bild unter mehreren Namensvarianten (z. B. Stamm `Y-DSS-` und Varianten `5,7,12` → `Y-DSS-5`, `Y-DSS-7`, `Y-DSS-12`).

```bash
python varianten_kopierer.py
```

### `massband_overlay.py`

Legt ein Maßband/Koordinatensystem in X und Y an ein Produktbild. Objektbreite in cm eingeben; die Pixelbreite wird erkannt. Overlay im Randbereich, ohne das Objekt zu verdecken.

```bash
python massband_overlay.py
```

### `cr3_quadrat.py`

Canon **CR3** (und gängige Bildformate) einlesen, Objekt freistellen, senkrecht ausrichten und auf ein weißes Quadrat (1000 / 1500 / 2000 px) als WebP speichern. Nutzt `axis_alignment.py`.

```bash
python cr3_quadrat.py
```

### `bild_resizer.py`

Größenanpassung ohne Hintergrundentfernung: Quadrat (zentriert, weißer Rand) oder Skalieren bei erhaltenem Seitenverhältnis. Optional WebP, Unterordner, Objektzuschnitt.

```bash
python bild_resizer.py
```

### `output_names.py`

Hilfsmodul: liest Ausgabenamen aus einer `.md`-Datei im Eingabeordner und prüft, ob Anzahl Bilder und Namen übereinstimmen. Wird von der Pipeline-GUI genutzt.

---

## Download & Umbenennen

### `downloader.py`

**Tkinter-GUI:** Liest URLs aus der **ersten Spalte** einer Excel-Datei (`.xlsx`), lädt die Bilder per HTTP herunter und protokolliert Dateigröße sowie Pixelmaße. Log: `image_downloader_full_audit.log`.

```bash
python downloader.py
```

### `foto_umbenennen.py`

**Tkinter-GUI:** Verknüpft **Picture-IDs** aus einer Excel-Tabelle mit **Hersteller-Nummern** und kopiert die passenden Fotos aus einem Quellordner in einen Zielordner – umbenannt nach Hersteller-Nr. Sucht Dateien, deren Name auf die Picture-ID endet (z. B. `IMG_1042.jpg`).

### `MatchingNumbers.py`

Verbesserte Variante von `foto_umbenennen.py`. Baut vorab einen **Index** aller Fotos im Quellordner (erkennt führende und abschließende Ziffern in Dateinamen). Unterstützt mehr Formate (WebP, TIFF, PNG, RAW). Sicherere Dateinamen durch `safe_filename()`.

### `umbennnen.py`

Einfaches **Tkinter-Tool** zum Umbenennen von **WebP-Dateien** anhand einer Excel-/CSV-Tabelle: Spalte 1 = alte Bildnummer, Spalten 2–4 = Teile des neuen Dateinamens (`teil1_teil2_teil3.webp`).

---

## Einzeloperationen

### `foto_rotator.py` / `image_rotator.py`

**Tkinter-GUIs** zum Drehen von Bildern in einem Ordner (90° / 180° / 270°).

- `image_rotator.py` — einfache Variante („Bild-Rotator Pro“)
- `foto_rotator.py` — erweiterte Variante mit Qualitätseinstellung, EXIF-Erhaltung und verlustfreier Rotation für JPEG/WebP

### `resizer.py`

**Tkinter-GUI „Bild-Größen-Downloader & Fixer 2.0“:** Prüft Bilder in einem Quellordner und kopiert sie in einen Zielordner. Bilder kleiner als **251 px** werden auf mindestens 251×251 px vergrößert (weißer Hintergrund, zentriert). Größere Bilder werden mit Optimierung (Qualität 95) kopiert. Log: `bild_optimierung_detail.log`.

### `finale.py`

**Tkinter-GUI „PNG zu WebP Konverter“:** Einzelnes PNG-Bild auswählen → weißer Hintergrund, 90° drehen, zentriert auf 1024×1024 px, speichern als WebP (80 % Qualität). Ohne rembg (keine Hintergrundentfernung).

---

## Setup & Referenz

### `installer.bat`

Python-Setup-Skript für Kollegen: prüft Python, installiert Kernpakete (Pillow, numpy, opencv-python, rembg, loguru) und erzeugt eine `process_image.bat` zum Drag-and-Drop-Verarbeiten.

### `mermaid.md`

Flussdiagramm der **complete_pipeline**-Architektur (Preprocessing → KI-Verarbeitung → Postprocessing & Export) im Mermaid-Format.

### `poster`

SVG-Grafik mit derselben Pipeline-Architektur (technische Übersicht zum Ausdruck/Anzeigen).

### `names_skins.py`

Referenzliste mit **Skin-/Starterkit-Namen** (z. B. `brown_trout`, `rainbow_trout`, `Koi_carp`) – keine ausführbare Anwendung.

---

## Pipeline-Ablauf (Kurzüberblick)

```
Bild laden → Helligkeit/Sättigung → rembg (Hintergrund) → Schatten bereinigen
→ Objekt zentrieren → Quadrat 1024×1024 → Weißer Hintergrund → WebP-Export
```

Details siehe `mermaid.md` und `complete_pipeline.py`.

---

## Log-Dateien

| Log | Erzeugt von |
|-----|-------------|
| `product_processor_YYYY-MM-DD.log` | `complete_pipeline.py`, `pipeline_GUI.py`, `foto_gui*.py` |
| `image_downloader_full_audit.log` | `downloader.py` |
| `bild_optimierung_detail.log` | `resizer.py` |
| `bildbearbeitung.log` | `finale.py` |
| `~/foto_umbenenner.log` | `foto_umbenennen.py`, `MatchingNumbers.py` |
| `produktfoto_tracker_YYYY-MM-DD.log` | `produktfoto_tracker.py` |
| `weiss_quadrat_webp_YYYY-MM-DD.log` | `weiss_quadrat_webp.py` |
| `varianten_kopierer_YYYY-MM-DD.log` | `varianten_kopierer.py` |
| `massband_overlay_YYYY-MM-DD.log` | `massband_overlay.py` |
| `cr3_quadrat_YYYY-MM-DD.log` | `cr3_quadrat.py` |

---

## Welches Tool wofür?

| Aufgabe | Empfohlenes Tool |
|---------|------------------|
| Vollständige Produkt-Pipeline (CLI) | `complete_pipeline.py` |
| Pipeline mit Slidern & Vorschau (Desktop) | `pipeline_GUI.py` |
| Pipeline im Browser | `gradio_app.py` / `gradio_2_app.py` |
| Schnelle Helios-Verarbeitung (WebP) | `Helios_BV.py` (CLI) oder `Helios_BV_Batch.py` (GUI) |
| Nur Hintergrund entfernen (hohe Qualität) | `BGremove.py` / `bgremove_gui.py` |
| Bilder aus Excel-URLs laden | `downloader.py` |
| Fotos per Excel-Liste umbenennen | `MatchingNumbers.py` |
| Bilder unter 251 px vergrößern | `resizer.py` |
| Bilder drehen | `foto_rotator.py` |
| WebP-Dateinamen aus Tabelle | `umbennnen.py` |
| Produktfotos planen / Status / Drive-Ablage | `produktfoto_tracker.py` |
| Ordner: Weiß + Quadrat + WebP | `weiss_quadrat_webp.py` |
| Ein Bild unter Varianten speichern | `varianten_kopierer.py` |
| Maßband-Overlay | `massband_overlay.py` |
| CR3 auf weißes Quadrat | `cr3_quadrat.py` |
| Nur Größe / Quadrat ohne rembg | `bild_resizer.py` |
