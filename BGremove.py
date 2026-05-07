#!/usr/bin/env python3
"""
remove_background.py
====================
Entfernt den Hintergrund von JPG/PNG-Bildern in hoher Qualität.
Ausgabe: PNG mit Alpha-Kanal + editierbare Maske als separate PNG-Datei.

Abhängigkeiten installieren:
    pip install rembg onnxruntime pillow numpy

Verwendung:
    # Einzelnes Bild:
    python remove_background.py bild.jpg

    # Mehrere Bilder:
    python remove_background.py bild1.jpg bild2.png foto.JPG

    # Ganzen Ordner verarbeiten:
    python remove_background.py --ordner ./fotos

    # Ausgabe-Ordner angeben:
    python remove_background.py bild.jpg --ausgabe ./ergebnisse

    # Modell wählen (Standard: u2net – beste Qualität):
    python remove_background.py bild.jpg --modell isnet-general-use

Verfügbare Modelle:
    u2net              – Allgemein, gute Balance (Standard)
    u2net_human_seg    – Optimiert für Personen/Porträts
    isnet-general-use  – Hohe Qualität, etwas langsamer
    isnet-anime        – Anime/Zeichentrick-Bilder
    silueta            – Leicht, schnell

Ausgabe-Dateien pro Bild:
    bild_nobg.png      – Freigestelltes Bild mit transparentem Hintergrund
    bild_mask.png      – Editierbare Graustufenmaske (weiß = Vordergrund)
    bild_preview.png   – Vorschau auf grauem Schachbrettmuster
"""

import argparse
import sys
import time
from pathlib import Path

try:
    from rembg import remove, new_session
    from PIL import Image, ImageDraw
    import numpy as np
except ImportError as e:
    print(f"[FEHLER] Fehlende Abhängigkeit: {e}")
    print("\nBitte installieren mit:")
    print("    pip install rembg onnxruntime pillow numpy")
    sys.exit(1)


# ──────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────

def checkerboard_background(width: int, height: int, tile: int = 20) -> Image.Image:
    """Erstellt ein Schachbrettmuster als Hintergrund für die Vorschau."""
    bg = Image.new("RGB", (width, height), (200, 200, 200))
    draw = ImageDraw.Draw(bg)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if (x // tile + y // tile) % 2 == 0:
                draw.rectangle([x, y, x + tile - 1, y + tile - 1], fill=(160, 160, 160))
    return bg


def process_image(
    input_path: Path,
    output_dir: Path,
    session,
    create_preview: bool = True,
    alpha_matting: bool = True,
    alpha_matting_fg: int = 20,
    alpha_matting_bg: int = 10,
    alpha_matting_er: int = 15,
) -> dict:
    """
    Verarbeitet ein einzelnes Bild.

    Returns:
        dict mit Pfaden zu den erzeugten Dateien und Statistiken.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    print(f"\n  Lade: {input_path.name}")
    img = Image.open(input_path).convert("RGBA")
    w, h = img.size
    print(f"  Größe: {w} × {h} px")

    # ── Hintergrund entfernen ──────────────────
    t0 = time.time()
    result: Image.Image = remove(
        img,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=alpha_matting_fg,
        alpha_matting_background_threshold=alpha_matting_bg,
        alpha_matting_erode_size=alpha_matting_er,
        post_process_mask=True,           # Maske nachschärfen
    )
    elapsed = time.time() - t0
    print(f"  Verarbeitung: {elapsed:.1f} s")

    # ── Freigestelltes Bild speichern ──────────
    nobg_path = output_dir / f"{stem}_nobg.png"
    result.save(nobg_path, format="PNG", optimize=False)
    print(f"  ✓ Freigestellt: {nobg_path.name}")

    # ── Maske extrahieren & speichern ──────────
    # Alpha-Kanal → Graustufenbild (weiß = sichtbar, schwarz = transparent)
    alpha = result.split()[3]                          # PIL Image, mode "L"
    mask_path = output_dir / f"{stem}_mask.png"
    alpha.save(mask_path, format="PNG")
    print(f"  ✓ Maske:       {mask_path.name}")

    # ── Vorschau auf Schachbrett ───────────────
    preview_path = None
    if create_preview:
        checker = checkerboard_background(w, h)
        checker.paste(result, mask=alpha)
        preview_path = output_dir / f"{stem}_preview.png"
        checker.save(preview_path, format="PNG")
        print(f"  ✓ Vorschau:    {preview_path.name}")

    # ── Statistik ─────────────────────────────
    alpha_arr = np.array(alpha)
    fg_pct = float((alpha_arr > 128).sum()) / alpha_arr.size * 100

    return {
        "nobg": nobg_path,
        "mask": mask_path,
        "preview": preview_path,
        "width": w,
        "height": h,
        "foreground_pct": fg_pct,
        "elapsed_s": elapsed,
    }


# ──────────────────────────────────────────────
# Haupt-Logik
# ──────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

MODELLE = {
    "u2net": "u2net",
    "u2net_human_seg": "u2net_human_seg",
    "isnet-general-use": "isnet-general-use",
    "isnet-anime": "isnet-anime",
    "silueta": "silueta",
}


def collect_images(paths: list[Path]) -> list[Path]:
    images = []
    for p in paths:
        if p.is_dir():
            for ext in SUPPORTED_EXTS:
                images.extend(p.glob(f"*{ext}"))
                images.extend(p.glob(f"*{ext.upper()}"))
        elif p.suffix.lower() in SUPPORTED_EXTS:
            images.append(p)
        else:
            print(f"[WARNUNG] Überspringe (nicht unterstützt): {p}")
    return sorted(set(images))


def main():
    parser = argparse.ArgumentParser(
        description="Hintergrund von Bildern entfernen – hohe Qualität, editierbare Maske",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("bilder", nargs="*", type=Path, help="Bilddateien (JPG, PNG, …)")
    parser.add_argument("--ordner", "-d", type=Path, help="Ordner mit Bildern verarbeiten")
    parser.add_argument(
        "--ausgabe", "-o", type=Path, default=Path("./ergebnisse"),
        help="Ausgabe-Ordner (Standard: ./ergebnisse)"
    )
    parser.add_argument(
        "--modell", "-m", default="u2net",
        choices=list(MODELLE.keys()),
        help="KI-Modell (Standard: u2net)"
    )
    parser.add_argument(
        "--kein-matting", action="store_true",
        help="Alpha-Matting deaktivieren (schneller, aber weniger weiche Kanten)"
    )
    parser.add_argument(
        "--keine-vorschau", action="store_true",
        help="Keine Schachbrett-Vorschau erzeugen"
    )
    parser.add_argument(
        "--fg-schwelle", type=int, default=20,
        help="Alpha-Matting Vordergrund-Schwelle (Standard: 20)"
    )
    parser.add_argument(
        "--bg-schwelle", type=int, default=10,
        help="Alpha-Matting Hintergrund-Schwelle (Standard: 10)"
    )

    args = parser.parse_args()

    # Bilder sammeln
    input_paths: list[Path] = list(args.bilder or [])
    if args.ordner:
        input_paths.append(args.ordner)

    if not input_paths:
        parser.print_help()
        print("\n[FEHLER] Keine Bilder oder Ordner angegeben.")
        sys.exit(1)

    images = collect_images(input_paths)
    if not images:
        print("[FEHLER] Keine unterstützten Bilddateien gefunden.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Hintergrund-Entfernung  |  {len(images)} Bild(er)")
    print(f"  Modell: {args.modell}")
    print(f"  Alpha-Matting: {'Nein' if args.kein_matting else 'Ja'}")
    print(f"  Ausgabe: {args.ausgabe.resolve()}")
    print(f"{'='*55}")

    # Modell laden (einmalig – beschleunigt Batch-Verarbeitung)
    print(f"\n  Lade KI-Modell '{args.modell}' …")
    session = new_session(args.modell)
    print("  Modell bereit.")

    # Verarbeitung
    results = []
    errors = []
    t_total = time.time()

    for img_path in images:
        try:
            r = process_image(
                input_path=img_path,
                output_dir=args.ausgabe,
                session=session,
                create_preview=not args.keine_vorschau,
                alpha_matting=not args.kein_matting,
                alpha_matting_fg=args.fg_schwelle,
                alpha_matting_bg=args.bg_schwelle,
            )
            results.append((img_path, r))
        except Exception as e:
            print(f"  [FEHLER] {img_path.name}: {e}")
            errors.append((img_path, str(e)))

    # Zusammenfassung
    total = time.time() - t_total
    print(f"\n{'='*55}")
    print(f"  Fertig! {len(results)}/{len(images)} Bild(er) verarbeitet")
    print(f"  Gesamtzeit: {total:.1f} s")
    if results:
        avg = sum(r["elapsed_s"] for _, r in results) / len(results)
        print(f"  Ø Zeit/Bild: {avg:.1f} s")
    if errors:
        print(f"\n  Fehler ({len(errors)}):")
        for p, e in errors:
            print(f"    • {p.name}: {e}")
    print(f"\n  Dateien in: {args.ausgabe.resolve()}")
    print(f"{'='*55}\n")

    # Hinweis zur Masken-Bearbeitung
    print("  TIPP – Maske bearbeiten:")
    print("  • Photoshop: Maske als 'Ebenenmaske' laden")
    print("  • GIMP:      Bild öffnen → Ebene → Maske → Maske hinzufügen")
    print("  • Affinity:  PNG importieren, Alpha-Kanal direkt bearbeiten")
    print("  • Code:      mask.png als Graustufenbild laden und beliebig filtern\n")


if __name__ == "__main__":
    main()