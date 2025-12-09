import cv2
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove
import os


def optimize_brightness_saturation(image):
    """Optimiert Helligkeit und Sättigung automatisch"""
    # Konvertiere zu HSV für bessere Kontrolle
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Histogramm-Ausgleich für Helligkeit
    v = cv2.equalizeHist(v)

    # Sättigung leicht erhöhen (Faktor 1.2)
    s = np.clip(s * 1.2, 0, 255).astype(np.uint8)

    # Wieder zusammensetzen
    hsv = cv2.merge([h, s, v])
    optimized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return optimized


def find_object_center(image):
    """Findet das Zentrum des Hauptobjekts durch Kantenerkennung"""
    # Zu Graustufen konvertieren
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Kantenerkennung
    edges = cv2.Canny(gray, 50, 150)

    # Morphologische Operation um Lücken zu schließen
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Konturen finden
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: Bildmitte
        h, w = image.shape[:2]
        return w // 2, h // 2, min(w, h) // 2

    # Größte Kontur nehmen (das Objekt)
    largest_contour = max(contours, key=cv2.contourArea)

    # Bounding Box berechnen
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Zentrum und Größe
    center_x = x + w // 2
    center_y = y + h // 2
    size = max(w, h)

    return center_x, center_y, size


def create_square_crop(image, center_x, center_y, size):
    """Erstellt einen quadratischen Ausschnitt mit Padding"""
    h, w = image.shape[:2]

    # Größe mit 20% Padding
    crop_size = int(size * 1.4)

    # Koordinaten berechnen
    half = crop_size // 2
    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(w, center_x + half)
    y2 = min(h, center_y + half)

    # Ausschneiden
    cropped = image[y1:y2, x1:x2]

    # Zu Quadrat machen (falls an Rand)
    h_crop, w_crop = cropped.shape[:2]
    if h_crop != w_crop:
        max_dim = max(h_crop, w_crop)
        square = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255
        y_offset = (max_dim - h_crop) // 2
        x_offset = (max_dim - w_crop) // 2
        square[y_offset:y_offset + h_crop, x_offset:x_offset + w_crop] = cropped
        return square

    return cropped


def process_product_image(input_path, output_path):
    """Hauptfunktion: Verarbeitet das Produktbild komplett"""

    print(f"📷 Lade Bild: {input_path}")

    # 1. Bild einlesen
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Bild konnte nicht geladen werden!")

    print(f"   Original: {image.shape[1]}x{image.shape[0]}px")

    # 2. Helligkeit und Sättigung optimieren
    print("✨ Optimiere Helligkeit und Sättigung...")
    optimized = optimize_brightness_saturation(image)

    # 3. Objekt finden und zentrieren
    print("🎯 Erkenne Objekt und zentriere...")
    center_x, center_y, obj_size = find_object_center(optimized)
    print(f"   Objekt bei ({center_x}, {center_y}), Größe: {obj_size}px")

    # 4. Quadratischen Ausschnitt erstellen
    print("✂️  Erstelle quadratischen Ausschnitt...")
    cropped = create_square_crop(optimized, center_x, center_y, obj_size)

    # 5. Zu PIL konvertieren für rembg
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cropped_rgb)

    # 6. Hintergrund entfernen
    print("🔮 Entferne Hintergrund (dauert etwas)...")
    no_bg = remove(pil_image)

    # 7. Weißen Hintergrund hinzufügen
    print("🎨 Füge weißen Hintergrund hinzu...")
    white_bg = Image.new('RGB', no_bg.size, (255, 255, 255))

    # Alpha-Channel für transparente Bereiche nutzen
    if no_bg.mode == 'RGBA':
        white_bg.paste(no_bg, mask=no_bg.split()[3])
    else:
        white_bg.paste(no_bg)

    # 8. Auf 1024x1024 skalieren
    print("📐 Skaliere auf 1024x1024px...")
    final = white_bg.resize((1024, 1024), Image.Resampling.LANCZOS)

    # 9. Als WebP exportieren
    print(f"💾 Exportiere als WebP: {output_path}")
    final.save(output_path, 'WEBP', quality=95)

    print("✅ Fertig!")
    return final


# Beispiel-Verwendung
if __name__ == "__main__":
    input_file = "gummifisch.jpg"  # Dein Eingabebild
    output_file = "gummifisch_final.webp"  # Ausgabedatei

    # Prüfen ob Datei existiert
    if not os.path.exists(input_file):
        print(f"❌ Fehler: '{input_file}' nicht gefunden!")
        print("   Bitte passe den Dateinamen in der letzten Zeile an.")
    else:
        try:
            result = process_product_image(input_file, output_file)
            print(f"\n🎉 Bild erfolgreich verarbeitet!")
            print(f"   Ausgabe: {output_file}")
        except Exception as e:
            print(f"❌ Fehler: {e}")