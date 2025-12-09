import cv2
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove
import os
import sys
from loguru import logger

# Loguru Konfiguration
logger.remove()  # Standard-Handler entfernen
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG"
)
logger.add(
    "product_processor_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)


def optimize_brightness_saturation(image):
    """Optimiert Helligkeit und Sättigung automatisch"""
    logger.info("Starte Helligkeits- und Sättigungsoptimierung")
    logger.debug(f"Eingabebild Shape: {image.shape}, Dtype: {image.dtype}")

    try:
        # Konvertiere zu HSV für bessere Kontrolle
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        logger.debug("Bild erfolgreich zu HSV konvertiert")

        h, s, v = cv2.split(hsv)
        logger.debug(f"HSV-Kanäle getrennt - H: {h.shape}, S: {s.shape}, V: {v.shape}")

        # Originalwerte loggen
        v_mean_before = np.mean(v)
        s_mean_before = np.mean(s)
        logger.debug(f"Ursprüngliche Werte - Helligkeit Ø: {v_mean_before:.2f}, Sättigung Ø: {s_mean_before:.2f}")

        # Histogramm-Ausgleich für Helligkeit
        v = cv2.equalizeHist(v)
        v_mean_after = np.mean(v)
        logger.debug(f"Histogramm-Ausgleich angewendet - Neue Helligkeit Ø: {v_mean_after:.2f}")

        # Sättigung leicht erhöhen (Faktor 1.2)
        s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
        s_mean_after = np.mean(s)
        logger.debug(f"Sättigung erhöht (Faktor 1.2) - Neue Sättigung Ø: {s_mean_after:.2f}")

        # Wieder zusammensetzen
        hsv = cv2.merge([h, s, v])
        optimized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        logger.success(
            f"Optimierung abgeschlossen - Helligkeit: {v_mean_before:.2f}→{v_mean_after:.2f}, Sättigung: {s_mean_before:.2f}→{s_mean_after:.2f}")
        return optimized

    except Exception as e:
        logger.error(f"Fehler bei Bildoptimierung: {e}")
        raise


def find_object_center(image):
    """Findet das Zentrum des Hauptobjekts durch Kantenerkennung"""
    logger.info("Starte Objekterkennung")
    logger.debug(f"Eingabebild für Objekterkennung: {image.shape}")

    try:
        # Zu Graustufen konvertieren
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug("Bild zu Graustufen konvertiert")

        # Kantenerkennung
        edges = cv2.Canny(gray, 50, 150)
        edge_count = np.count_nonzero(edges)
        logger.debug(f"Canny-Kantenerkennung durchgeführt - {edge_count} Kantenpixel gefunden")

        # Morphologische Operation um Lücken zu schließen
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        dilated_edge_count = np.count_nonzero(edges)
        logger.debug(f"Morphologische Dilatation angewendet - {dilated_edge_count} Kantenpixel nach Dilatation")

        # Konturen finden
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug(f"{len(contours)} Konturen gefunden")

        if not contours:
            # Fallback: Bildmitte
            h, w = image.shape[:2]
            logger.warning(f"Keine Konturen gefunden! Verwende Bildmitte als Fallback: ({w // 2}, {h // 2})")
            return w // 2, h // 2, min(w, h) // 2

        # Größte Kontur nehmen (das Objekt)
        contour_areas = [cv2.contourArea(c) for c in contours]
        largest_idx = np.argmax(contour_areas)
        largest_contour = contours[largest_idx]
        largest_area = contour_areas[largest_idx]

        logger.debug(f"Größte Kontur ausgewählt - Index: {largest_idx}, Fläche: {largest_area:.2f}px²")
        logger.debug(f"Top 3 Konturflächen: {sorted(contour_areas, reverse=True)[:3]}")

        # Bounding Box berechnen
        x, y, w, h = cv2.boundingRect(largest_contour)
        logger.debug(f"Bounding Box berechnet - Position: ({x}, {y}), Größe: {w}x{h}px")

        # Zentrum und Größe
        center_x = x + w // 2
        center_y = y + h // 2
        size = max(w, h)

        # Prozentuale Position im Bild
        img_h, img_w = image.shape[:2]
        pos_x_percent = (center_x / img_w) * 100
        pos_y_percent = (center_y / img_h) * 100
        size_percent = (size / max(img_w, img_h)) * 100

        logger.success(
            f"Objekt erkannt - Zentrum: ({center_x}, {center_y}) [{pos_x_percent:.1f}%, {pos_y_percent:.1f}%], Größe: {size}px ({size_percent:.1f}% der Bildgröße)")

        return center_x, center_y, size

    except Exception as e:
        logger.error(f"Fehler bei Objekterkennung: {e}")
        raise


def create_square_crop(image, center_x, center_y, size):
    """Erstellt einen quadratischen Ausschnitt mit Padding"""
    logger.info("Erstelle quadratischen Ausschnitt")
    h, w = image.shape[:2]
    logger.debug(f"Ursprüngliches Bild: {w}x{h}px, Objektgröße: {size}px")

    try:
        # Größe mit 20% Padding (Faktor 1.4)
        crop_size = int(size * 1.4)
        padding_added = crop_size - size
        logger.debug(f"Crop-Größe mit Padding: {crop_size}px (Padding: {padding_added}px, Faktor: 1.4)")

        # Koordinaten berechnen
        half = crop_size // 2
        x1 = max(0, center_x - half)
        y1 = max(0, center_y - half)
        x2 = min(w, center_x + half)
        y2 = min(h, center_y + half)

        clipped_left = (center_x - half) < 0
        clipped_top = (center_y - half) < 0
        clipped_right = (center_x + half) > w
        clipped_bottom = (center_y + half) > h

        if any([clipped_left, clipped_top, clipped_right, clipped_bottom]):
            clipped_sides = []
            if clipped_left: clipped_sides.append("links")
            if clipped_top: clipped_sides.append("oben")
            if clipped_right: clipped_sides.append("rechts")
            if clipped_bottom: clipped_sides.append("unten")
            logger.warning(f"Ausschnitt an Bildrand beschnitten: {', '.join(clipped_sides)}")

        logger.debug(f"Crop-Koordinaten: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        # Ausschneiden
        cropped = image[y1:y2, x1:x2]
        logger.debug(f"Ausschnitt erstellt: {cropped.shape[1]}x{cropped.shape[0]}px")

        # Zu Quadrat machen (falls an Rand)
        h_crop, w_crop = cropped.shape[:2]
        if h_crop != w_crop:
            max_dim = max(h_crop, w_crop)
            logger.info(f"Bild ist nicht quadratisch ({w_crop}x{h_crop}px) - erweitere auf {max_dim}x{max_dim}px")

            square = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255
            y_offset = (max_dim - h_crop) // 2
            x_offset = (max_dim - w_crop) // 2

            logger.debug(f"Offsets für Zentrierung: x={x_offset}px, y={y_offset}px")
            square[y_offset:y_offset + h_crop, x_offset:x_offset + w_crop] = cropped

            logger.success(f"Quadratischer Ausschnitt erstellt: {max_dim}x{max_dim}px")
            return square

        logger.success(f"Quadratischer Ausschnitt erstellt: {w_crop}x{h_crop}px")
        return cropped

    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Ausschnitts: {e}")
        raise


def process_product_image(input_path, output_path):
    """Hauptfunktion: Verarbeitet das Produktbild komplett"""

    logger.info("=" * 80)
    logger.info("STARTE PRODUKTBILD-VERARBEITUNG")
    logger.info("=" * 80)
    logger.info(f"Eingabedatei: {input_path}")
    logger.info(f"Ausgabedatei: {output_path}")

    # 1. Bild einlesen
    logger.info("SCHRITT 1/9: Lade Bild")

    if not os.path.exists(input_path):
        logger.error(f"Datei nicht gefunden: {input_path}")
        raise FileNotFoundError(f"Datei '{input_path}' existiert nicht!")

    file_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    logger.debug(f"Dateigröße: {file_size:.2f} MB")

    image = cv2.imread(input_path)
    if image is None:
        logger.error("Bild konnte nicht geladen werden - möglicherweise ungültiges Format")
        raise ValueError("Bild konnte nicht geladen werden!")

    orig_h, orig_w = image.shape[:2]
    logger.success(f"Bild geladen - Auflösung: {orig_w}x{orig_h}px, Kanäle: {image.shape[2]}")

    # 2. Helligkeit und Sättigung optimieren
    logger.info("SCHRITT 2/9: Optimiere Helligkeit und Sättigung")
    optimized = optimize_brightness_saturation(image)

    # 3. Objekt finden und zentrieren
    logger.info("SCHRITT 3/9: Erkenne Objekt und ermittle Position")
    center_x, center_y, obj_size = find_object_center(optimized)

    # 4. Quadratischen Ausschnitt erstellen
    logger.info("SCHRITT 4/9: Erstelle quadratischen Ausschnitt")
    cropped = create_square_crop(optimized, center_x, center_y, obj_size)

    # 5. Zu PIL konvertieren für rembg
    logger.info("SCHRITT 5/9: Konvertiere zu PIL-Format")
    try:
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)
        logger.success(f"Konvertierung erfolgreich - PIL-Modus: {pil_image.mode}, Größe: {pil_image.size}")
    except Exception as e:
        logger.error(f"Fehler bei PIL-Konvertierung: {e}")
        raise

    # 6. Hintergrund entfernen
    logger.info("SCHRITT 6/9: Entferne Hintergrund mit KI-Modell (kann etwas dauern)")
    logger.debug("Rufe rembg.remove() auf...")
    try:
        no_bg = remove(pil_image)
        logger.success(f"Hintergrund entfernt - Modus: {no_bg.mode}, Größe: {no_bg.size}")

        # Alpha-Channel Statistik
        if no_bg.mode == 'RGBA':
            alpha = np.array(no_bg.split()[3])
            transparent_pixels = np.sum(alpha == 0)
            total_pixels = alpha.size
            transparency_percent = (transparent_pixels / total_pixels) * 100
            logger.debug(
                f"Transparenz-Statistik: {transparent_pixels}/{total_pixels} Pixel transparent ({transparency_percent:.2f}%)")

    except Exception as e:
        logger.error(f"Fehler beim Entfernen des Hintergrunds: {e}")
        raise

    # 7. Weißen Hintergrund hinzufügen
    logger.info("SCHRITT 7/9: Füge weißen Hintergrund hinzu")
    try:
        white_bg = Image.new('RGB', no_bg.size, (255, 255, 255))
        logger.debug(f"Weißer Hintergrund erstellt - Größe: {white_bg.size}")

        # Alpha-Channel für transparente Bereiche nutzen
        if no_bg.mode == 'RGBA':
            white_bg.paste(no_bg, mask=no_bg.split()[3])
            logger.debug("Objekt mit Alpha-Maske auf weißen Hintergrund eingefügt")
        else:
            white_bg.paste(no_bg)
            logger.debug("Objekt direkt auf weißen Hintergrund eingefügt")

        logger.success("Weißer Hintergrund erfolgreich hinzugefügt")

    except Exception as e:
        logger.error(f"Fehler beim Hinzufügen des weißen Hintergrunds: {e}")
        raise

    # 8. Auf 1024x1024 skalieren
    logger.info("SCHRITT 8/9: Skaliere auf Zielgröße 1024x1024px")
    try:
        before_size = white_bg.size
        final = white_bg.resize((1024, 1024), Image.Resampling.LANCZOS)
        scale_factor = 1024 / max(before_size)
        logger.success(
            f"Skalierung erfolgreich - Von {before_size[0]}x{before_size[1]}px auf 1024x1024px (Faktor: {scale_factor:.3f})")
    except Exception as e:
        logger.error(f"Fehler beim Skalieren: {e}")
        raise

    # 9. Als WebP exportieren
    logger.info("SCHRITT 9/9: Exportiere als WebP")
    try:
        final.save(output_path, 'WEBP', quality=95)
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression_ratio = (file_size / output_size) if output_size > 0 else 0

        logger.success(f"WebP exportiert - Dateigröße: {output_size:.2f} MB (Kompression: {compression_ratio:.2f}x)")

    except Exception as e:
        logger.error(f"Fehler beim Exportieren: {e}")
        raise

    logger.info("=" * 80)
    logger.success("VERARBEITUNG ERFOLGREICH ABGESCHLOSSEN!")
    logger.info("=" * 80)
    logger.info(f"Ursprüngliche Auflösung: {orig_w}x{orig_h}px ({file_size:.2f} MB)")
    logger.info(f"Finale Auflösung: 1024x1024px ({output_size:.2f} MB)")
    logger.info(f"Ausgabedatei: {output_path}")
    logger.info("=" * 80)

    return final


# Beispiel-Verwendung
if __name__ == "__main__":
    logger.info("Produktbild-Prozessor gestartet")

    input_file = "gummifisch.jpg"  # Dein Eingabebild
    output_file = "gummifisch_final.webp"  # Ausgabedatei

    # Prüfen ob Datei existiert
    if not os.path.exists(input_file):
        logger.error(f"Eingabedatei '{input_file}' nicht gefunden!")
        logger.info("Bitte passe den Dateinamen in der letzten Zeile an.")
        sys.exit(1)
    else:
        try:
            result = process_product_image(input_file, output_file)
            logger.info("Programm erfolgreich beendet")
        except Exception as e:
            logger.exception(f"Kritischer Fehler bei der Verarbeitung: {e}")
            sys.exit(1)