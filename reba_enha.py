from rembg import remove
from PIL import Image, ImageEnhance
import sys
import os


def remove_background(input_path, output_path=None, brightness=1.2, contrast=1.3,
                      output_size=(1024, 1024), dpi=72):
    """
    Entfernt den Hintergrund von einem Fischfoto, verbessert Helligkeit/Kontrast,
    und speichert es zentriert mit definierter Größe als WebP mit Transparenz.

    Args:
        input_path: Pfad zum Eingabebild (JPG)
        output_path: Pfad zum Ausgabebild (WebP). Falls None, wird automatisch generiert.
        brightness: Helligkeitsfaktor (1.2 = +20%)
        contrast: Kontrastfaktor (1.3 = +30%)
        output_size: Ausgabegröße in Pixel (Breite, Höhe)
        dpi: DPI-Wert für das Ausgabebild
    """
    try:
        # Überprüfen ob die Eingabedatei existiert
        if not os.path.exists(input_path):
            print(f"Fehler: Datei '{input_path}' nicht gefunden!")
            return False

        print(f"Lade Bild: {input_path}")

        # Bild einlesen
        input_image = Image.open(input_path)

        print("Entferne Hintergrund...")

        # Hintergrund entfernen
        output_image = remove(input_image)

        print(f"Erhöhe Helligkeit um {int((brightness - 1) * 100)}% und Kontrast um {int((contrast - 1) * 100)}%...")

        # Helligkeit erhöhen
        enhancer = ImageEnhance.Brightness(output_image)
        output_image = enhancer.enhance(brightness)

        # Kontrast erhöhen
        enhancer = ImageEnhance.Contrast(output_image)
        output_image = enhancer.enhance(contrast)

        print(f"Skaliere und zentriere auf {output_size[0]}x{output_size[1]}px...")

        # Seitenverhältnis beibehalten und auf maximale Größe skalieren
        output_image.thumbnail((output_size[0], output_size[1]), Image.Resampling.LANCZOS)

        # Neues Bild mit transparentem Hintergrund erstellen
        final_image = Image.new('RGBA', output_size, (0, 0, 0, 0))

        # Bild zentrieren
        x = (output_size[0] - output_image.size[0]) // 2
        y = (output_size[1] - output_image.size[1]) // 2
        final_image.paste(output_image, (x, y), output_image)

        # Ausgabepfad generieren falls nicht angegeben
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_bearbeitet.webp"

        # Als WebP mit Transparenz und DPI speichern
        # quality: 0-100 (höher = bessere Qualität, größere Datei)
        # lossless=False für kleinere Dateien bei guter Qualität
        print(f"Speichere Bild: {output_path}")
        final_image.save(output_path, format='WEBP', quality=85, method=6, dpi=(dpi, dpi))

        print("Fertig!")
        print(f"Größe Original: {input_image.size}")
        print(f"Größe Ausgabe: {final_image.size}")

        return True

    except Exception as e:
        print(f"Fehler beim Verarbeiten des Bildes: {str(e)}")
        return False


def batch_process(input_folder, output_folder=None, brightness=1.2, contrast=1.3):
    """
    Verarbeitet alle JPG-Bilder in einem Ordner.

    Args:
        input_folder: Ordner mit Eingabebildern
        output_folder: Ordner für Ausgabebilder. Falls None, wird im Eingabeordner gespeichert.
        brightness: Helligkeitsfaktor (1.2 = +20%)
        contrast: Kontrastfaktor (1.3 = +30%)
    """
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    jpg_files = [f for f in os.listdir(input_folder)
                 if f.lower().endswith(('.jpg', '.jpeg'))]

    print(f"Gefunden: {len(jpg_files)} Bilder")

    for i, filename in enumerate(jpg_files, 1):
        print(f"\n[{i}/{len(jpg_files)}] Verarbeite: {filename}")
        input_path = os.path.join(input_folder, filename)

        if output_folder:
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{base_name}.webp")
        else:
            output_path = None

        remove_background(input_path, output_path, brightness, contrast)


if __name__ == "__main__":
    # Beispielverwendung
    if len(sys.argv) < 2:
        print("Verwendung:")
        print("  Einzelnes Bild: python script.py pfad/zum/bild.jpg")
        print("  Batch-Verarbeitung: python script.py pfad/zum/ordner/")
        print("\nOptional: python script.py eingabe.jpg ausgabe.webp")
        sys.exit(1)

    input_arg = sys.argv[1]

    # Prüfen ob es ein Ordner oder eine Datei ist
    if os.path.isdir(input_arg):
        output_folder = sys.argv[2] if len(sys.argv) > 2 else None
        batch_process(input_arg, output_folder)
    else:
        output_arg = sys.argv[2] if len(sys.argv) > 2 else None
        remove_background(input_arg, output_arg)