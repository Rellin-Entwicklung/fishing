import dearpygui.dearpygui as dpg
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove
import os
import threading
from pathlib import Path
from loguru import logger
import sys

# Loguru Konfiguration
logger.remove()
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


class ProductImageProcessor:
    def __init__(self):
        self.input_folder = ""
        self.output_folder = ""
        self.last_processed_image = None
        self.stats = {"processed": 0, "successful": 0, "failed": 0}

        # Einstellungen
        self.settings = {
            "remove_background": True,
            "enhance_colors": True,
            "sharpen": True,
            "auto_correct": True,
            "resize": True,
            "saturation": 150,  # 150% = Faktor 1.5
            "brightness": 110  # 110% = Faktor 1.1
        }

    def optimize_brightness_saturation(self, image, brightness_factor, saturation_factor):
        """Optimiert Helligkeit und Sättigung mit variablen Faktoren"""
        logger.info(f"Optimiere mit Helligkeit={brightness_factor}, Sättigung={saturation_factor}")

        try:
            # Konvertiere zu LAB für bessere Helligkeitskontrolle
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # CLAHE für bessere Helligkeit
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            lab = cv2.merge([l, a, b])
            brightened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Zu HSV für Sättigung
            hsv = cv2.cvtColor(brightened, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            # Sättigung mit Faktor erhöhen
            s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)

            # Helligkeit mit Faktor anheben
            v = np.clip(v * brightness_factor, 0, 255).astype(np.uint8)

            hsv = cv2.merge([h, s, v])
            optimized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Schärfen mit Unsharp Mask
            if self.settings["sharpen"]:
                gaussian = cv2.GaussianBlur(optimized, (0, 0), 2.0)
                optimized = cv2.addWeighted(optimized, 1.5, gaussian, -0.5, 0)

            logger.success("Optimierung abgeschlossen")
            return optimized

        except Exception as e:
            logger.error(f"Fehler bei Bildoptimierung: {e}")
            raise

    def remove_shadow_artifacts(self, image_rgba):
        """Entfernt graue Schatten und Artefakte"""
        logger.info("Entferne Schatten-Artefakte")

        try:
            img_array = np.array(image_rgba)
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]

            very_transparent = alpha < 50
            alpha[very_transparent] = 0

            medium_transparent = (alpha >= 50) & (alpha < 200)

            if np.count_nonzero(medium_transparent) > 0:
                for i in range(img_array.shape[0]):
                    for j in range(img_array.shape[1]):
                        if medium_transparent[i, j]:
                            r, g, b = rgb[i, j]
                            color_range = max(r, g, b) - min(r, g, b)
                            brightness = (int(r) + int(g) + int(b)) / 3

                            if color_range < 30 and brightness < 180:
                                if alpha[i, j] < 120:
                                    alpha[i, j] = 0
                                else:
                                    factor = 255 / max(brightness, 1)
                                    rgb[i, j] = np.clip(rgb[i, j] * factor * 0.8 + 50, 0, 255).astype(np.uint8)

            kernel = np.ones((3, 3), np.uint8)
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

            result = np.dstack([rgb, alpha])
            cleaned = Image.fromarray(result.astype(np.uint8), 'RGBA')

            logger.success("Schatten-Artefakte entfernt")
            return cleaned

        except Exception as e:
            logger.warning(f"Fehler beim Entfernen der Schatten: {e}")
            return image_rgba

    def find_object_bounds_from_alpha(self, alpha_channel):
        """Findet die Bounding Box des Objekts"""
        logger.info("Ermittle Objektgrenzen")

        mask = alpha_channel > 10
        if np.count_nonzero(mask) == 0:
            logger.warning("Keine nicht-transparenten Pixel gefunden!")
            return None

        coords = np.argwhere(mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        return {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'width': x_max - x_min + 1,
            'height': y_max - y_min + 1,
            'center_x': (x_min + x_max) // 2,
            'center_y': (y_min + y_max) // 2
        }

    def create_centered_square_with_padding(self, image_pil, bounds, target_size=1024, padding_factor=1.3):
        """Erstellt ein zentriertes Quadrat mit Padding"""
        logger.info(f"Erstelle zentriertes Quadrat {target_size}x{target_size}px")

        if bounds is None:
            return image_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)

        obj_size = max(bounds['width'], bounds['height'])
        crop_size = int(obj_size * padding_factor)

        center_x = bounds['center_x']
        center_y = bounds['center_y']
        half_crop = crop_size // 2

        x1 = center_x - half_crop
        y1 = center_y - half_crop
        x2 = center_x + half_crop
        y2 = center_y + half_crop

        img_width, img_height = image_pil.size

        if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - img_width)
            pad_bottom = max(0, y2 - img_height)

            new_width = img_width + pad_left + pad_right
            new_height = img_height + pad_top + pad_bottom

            extended = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 255))
            extended.paste(image_pil, (pad_left, pad_top))

            x1 += pad_left
            y1 += pad_top
            x2 += pad_left
            y2 += pad_top

            image_pil = extended

        cropped = image_pil.crop((x1, y1, x2, y2))

        crop_w, crop_h = cropped.size
        if crop_w != crop_h:
            square_size = max(crop_w, crop_h)
            square = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 255))
            offset_x = (square_size - crop_w) // 2
            offset_y = (square_size - crop_h) // 2
            square.paste(cropped, (offset_x, offset_y))
            cropped = square

        if cropped.size[0] != target_size:
            cropped = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

        return cropped

    def process_single_image(self, input_path, output_path):
        """Verarbeitet ein einzelnes Bild"""
        logger.info("=" * 80)
        logger.info(f"Verarbeite: {input_path}")

        try:
            # 1. Bild laden
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError("Bild konnte nicht geladen werden!")

            # 2. Helligkeit und Sättigung optimieren
            if self.settings["enhance_colors"]:
                brightness_factor = self.settings["brightness"] / 100.0
                saturation_factor = self.settings["saturation"] / 100.0
                optimized = self.optimize_brightness_saturation(image, brightness_factor, saturation_factor)
            else:
                optimized = image

            # 3. Zu PIL konvertieren
            optimized_rgb = cv2.cvtColor(optimized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(optimized_rgb)

            # 4. Hintergrund entfernen
            if self.settings["remove_background"]:
                logger.info("Entferne Hintergrund...")
                no_bg = remove(pil_image)

                # 5. Schatten-Artefakte entfernen
                if no_bg.mode == 'RGBA':
                    no_bg = self.remove_shadow_artifacts(no_bg)

                # 6. Objektgrenzen ermitteln
                if no_bg.mode == 'RGBA':
                    alpha_array = np.array(no_bg.split()[3])
                    bounds = self.find_object_bounds_from_alpha(alpha_array)
                else:
                    bounds = None

                # 7. Zentriertes Quadrat erstellen
                if self.settings["resize"]:
                    final = self.create_centered_square_with_padding(no_bg, bounds, target_size=1024,
                                                                     padding_factor=1.3)
                else:
                    final = no_bg
            else:
                final = pil_image
                if self.settings["resize"]:
                    final = final.resize((1024, 1024), Image.Resampling.LANCZOS)

            # 8. Weißen Hintergrund hinzufügen und exportieren
            white_bg = Image.new('RGB', final.size, (255, 255, 255))

            if final.mode == 'RGBA':
                white_bg.paste(final, mask=final.split()[3])
            else:
                white_bg.paste(final)

            # Finale Verbesserungen
            if self.settings["auto_correct"]:
                final_array = np.array(white_bg)
                final_array = cv2.cvtColor(final_array, cv2.COLOR_RGB2BGR)
                final_array = cv2.fastNlMeansDenoisingColored(final_array, None, 5, 5, 7, 21)
                final_array = cv2.cvtColor(final_array, cv2.COLOR_BGR2RGB)
                final_pil = Image.fromarray(final_array)

                enhancer_sharp = ImageEnhance.Sharpness(final_pil)
                final_pil = enhancer_sharp.enhance(1.3)

                enhancer_contrast = ImageEnhance.Contrast(final_pil)
                final_pil = enhancer_contrast.enhance(1.1)
            else:
                final_pil = white_bg

            # Als WebP speichern
            final_pil.save(output_path, 'WEBP', quality=95)

            logger.success(f"Gespeichert: {output_path}")

            return final_pil

        except Exception as e:
            logger.error(f"Fehler bei {input_path}: {e}")
            raise

    def process_folder(self):
        """Verarbeitet alle Bilder im Eingabeordner"""
        if not self.input_folder or not self.output_folder:
            logger.error("Eingabe- oder Ausgabeordner nicht gesetzt!")
            return

        # Erstelle Ausgabeordner falls nicht vorhanden
        os.makedirs(self.output_folder, exist_ok=True)

        # Finde alle Bilddateien
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(Path(self.input_folder).glob(f'*{ext}'))
            image_files.extend(Path(self.input_folder).glob(f'*{ext.upper()}'))

        total = len(image_files)
        logger.info(f"Gefunden: {total} Bilder zum Verarbeiten")

        if total == 0:
            logger.warning("Keine Bilder gefunden!")
            return

        # Verarbeite jedes Bild
        for idx, img_path in enumerate(image_files, 1):
            try:
                # Ausgabedateiname
                output_name = img_path.stem + '_optimized.webp'
                output_path = os.path.join(self.output_folder, output_name)

                logger.info(f"[{idx}/{total}] Verarbeite {img_path.name}...")

                # Verarbeite Bild
                result = self.process_single_image(str(img_path), output_path)

                # Aktualisiere letztes Bild
                self.last_processed_image = result

                # Statistik aktualisieren
                self.stats["processed"] += 1
                self.stats["successful"] += 1

                # GUI aktualisieren
                self.update_gui()

            except Exception as e:
                logger.error(f"Fehler bei {img_path.name}: {e}")
                self.stats["processed"] += 1
                self.stats["failed"] += 1
                self.update_gui()

        logger.success(f"Verarbeitung abgeschlossen! {self.stats['successful']}/{total} erfolgreich")

    def update_gui(self):
        """Aktualisiert die GUI mit den aktuellen Werten"""
        if dpg.does_item_exist("stats_processed"):
            dpg.set_value("stats_processed", str(self.stats["processed"]))
        if dpg.does_item_exist("stats_successful"):
            dpg.set_value("stats_successful", str(self.stats["successful"]))
        if dpg.does_item_exist("stats_failed"):
            dpg.set_value("stats_failed", str(self.stats["failed"]))

        # Bildvorschau aktualisieren
        if self.last_processed_image:
            self.update_image_preview()

    def update_image_preview(self):
        """Aktualisiert die Bildvorschau"""
        if not self.last_processed_image:
            return

        try:
            # Bild für Vorschau vorbereiten (verkleinern)
            preview_size = (400, 400)
            preview_img = self.last_processed_image.copy()
            preview_img.thumbnail(preview_size, Image.Resampling.LANCZOS)

            # Zu numpy array konvertieren
            img_array = np.array(preview_img).astype(np.float32) / 255.0

            # Flatten für DearPyGUI
            flat_data = img_array.flatten()

            # Texture aktualisieren
            if dpg.does_item_exist("preview_texture"):
                dpg.set_value("preview_texture", flat_data)
        except Exception as e:
            logger.error(f"Fehler bei Bildvorschau: {e}")


# GUI-Funktionen
processor = ProductImageProcessor()


def select_input_folder(sender, app_data):
    processor.input_folder = app_data['file_path_name']
    dpg.set_value("input_folder_text", processor.input_folder)
    logger.info(f"Eingabeordner: {processor.input_folder}")


def select_output_folder(sender, app_data):
    processor.output_folder = app_data['file_path_name']
    dpg.set_value("output_folder_text", processor.output_folder)
    logger.info(f"Ausgabeordner: {processor.output_folder}")


def update_setting(setting_name, value):
    processor.settings[setting_name] = value
    logger.debug(f"Einstellung geändert: {setting_name} = {value}")


def start_processing():
    """Startet die Verarbeitung in einem separaten Thread"""
    if not processor.input_folder or not processor.output_folder:
        logger.warning("Bitte beide Ordner auswählen!")
        return

    # Statistik zurücksetzen
    processor.stats = {"processed": 0, "successful": 0, "failed": 0}
    processor.update_gui()

    # In separatem Thread ausführen
    thread = threading.Thread(target=processor.process_folder, daemon=True)
    thread.start()
    logger.info("Verarbeitung gestartet...")


# GUI erstellen
def create_gui():
    dpg.create_context()

    # File Dialog
    with dpg.file_dialog(directory_selector=True, show=False, callback=select_input_folder,
                         tag="input_folder_dialog", width=700, height=400):
        dpg.add_file_extension(".*")

    with dpg.file_dialog(directory_selector=True, show=False, callback=select_output_folder,
                         tag="output_folder_dialog", width=700, height=400):
        dpg.add_file_extension(".*")

    # Hauptfenster
    with dpg.window(label="Produktbild-Optimierer", tag="main_window"):
        # Header
        with dpg.group(horizontal=False):
            dpg.add_text("Produktbild-Optimierer", color=(100, 200, 255))
            dpg.add_text("Verbessern Sie Ihre Produktbilder automatisch", color=(150, 150, 150))
            dpg.add_separator()

        with dpg.group(horizontal=True):
            # Linke Spalte - Einstellungen
            with dpg.child_window(width=350, height=700):
                # Ordnerauswahl
                with dpg.collapsing_header(label="Ordnerauswahl", default_open=True):
                    dpg.add_text("Eingabeordner:")
                    dpg.add_button(label="Ordner wählen...", callback=lambda: dpg.show_item("input_folder_dialog"))
                    dpg.add_input_text(tag="input_folder_text", default_value="Nicht ausgewählt",
                                       width=300, readonly=True)

                    dpg.add_spacer(height=10)

                    dpg.add_text("Ausgabeordner:")
                    dpg.add_button(label="Ordner wählen...", callback=lambda: dpg.show_item("output_folder_dialog"))
                    dpg.add_input_text(tag="output_folder_text", default_value="Nicht ausgewählt",
                                       width=300, readonly=True)

                dpg.add_spacer(height=10)

                # Funktionen
                with dpg.collapsing_header(label="Funktionen", default_open=True):
                    dpg.add_checkbox(label="Hintergrund entfernen", default_value=True,
                                     callback=lambda s, v: update_setting("remove_background", v))
                    dpg.add_checkbox(label="Farben verbessern", default_value=True,
                                     callback=lambda s, v: update_setting("enhance_colors", v))
                    dpg.add_checkbox(label="Schärfen", default_value=True,
                                     callback=lambda s, v: update_setting("sharpen", v))
                    dpg.add_checkbox(label="Auto-Korrektur", default_value=True,
                                     callback=lambda s, v: update_setting("auto_correct", v))
                    dpg.add_checkbox(label="Größe anpassen (1024x1024)", default_value=True,
                                     callback=lambda s, v: update_setting("resize", v))

                dpg.add_spacer(height=10)

                # Anpassungen
                with dpg.collapsing_header(label="Anpassungen", default_open=True):
                    dpg.add_text("Sättigung:")
                    dpg.add_slider_int(label="##saturation", default_value=150, min_value=0, max_value=200,
                                       width=300, callback=lambda s, v: update_setting("saturation", v))

                    dpg.add_spacer(height=10)

                    dpg.add_text("Helligkeit:")
                    dpg.add_slider_int(label="##brightness", default_value=110, min_value=50, max_value=150,
                                       width=300, callback=lambda s, v: update_setting("brightness", v))

                dpg.add_spacer(height=20)

                # Verarbeiten Button
                dpg.add_button(label="Bilder verarbeiten", width=300, height=40, callback=start_processing)

            # Rechte Spalte - Vorschau
            with dpg.child_window(width=650, height=700):
                dpg.add_text("Letzte Bearbeitung", color=(100, 200, 255))
                dpg.add_separator()

                # Bildvorschau
                with dpg.texture_registry():
                    # Platzhalter-Textur erstellen
                    placeholder = np.ones((400, 400, 3), dtype=np.float32) * 0.2
                    dpg.add_raw_texture(width=400, height=400, default_value=placeholder.flatten(),
                                        format=dpg.mvFormat_Float_rgb, tag="preview_texture")

                dpg.add_image("preview_texture")

                dpg.add_spacer(height=20)

                # Statistiken
                with dpg.group(horizontal=True):
                    with dpg.child_window(width=200, height=80):
                        dpg.add_text("Verarbeitet", color=(150, 150, 150))
                        dpg.add_text("0", tag="stats_processed", color=(255, 255, 255))

                    with dpg.child_window(width=200, height=80):
                        dpg.add_text("Erfolgreich", color=(150, 150, 150))
                        dpg.add_text("0", tag="stats_successful", color=(100, 255, 100))

                    with dpg.child_window(width=200, height=80):
                        dpg.add_text("Fehler", color=(150, 150, 150))
                        dpg.add_text("0", tag="stats_failed", color=(255, 100, 100))

    dpg.create_viewport(title="Produktbild-Optimierer", width=1100, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    logger.info("Starte GUI...")
    create_gui()