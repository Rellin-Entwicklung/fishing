"""Helios - Batch Bildverarbeitung mit tkinter GUI."""

import io
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List
import threading

from PIL import Image
from rembg import remove


class ImageProcessorGUI:
    """GUI für Batch-Bildverarbeitung."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Helios Batch Bildverarbeitung")
        self.root.geometry("700x500")
        self.root.resizable(False, False)

        self.input_dir = None
        self.output_dir = None
        self.processing = False

        self._create_widgets()

    def _create_widgets(self):
        """Erstellt alle GUI-Elemente."""
        # Header
        header = tk.Label(
            self.root,
            text="Helios Batch Bildverarbeitung",
            font=("Arial", 16, "bold"),
            pady=10
        )
        header.pack()

        # Eingangsverzeichnis
        input_frame = tk.LabelFrame(self.root, text="Eingangsverzeichnis", padx=10, pady=10)
        input_frame.pack(fill="x", padx=20, pady=5)

        self.input_label = tk.Label(input_frame, text="Kein Verzeichnis ausgewählt", fg="gray")
        self.input_label.pack(side="left", fill="x", expand=True)

        input_btn = tk.Button(
            input_frame,
            text="Durchsuchen...",
            command=self._select_input_dir,
            width=15
        )
        input_btn.pack(side="right")

        # Ausgangsverzeichnis
        output_frame = tk.LabelFrame(self.root, text="Ausgangsverzeichnis", padx=10, pady=10)
        output_frame.pack(fill="x", padx=20, pady=5)

        self.output_label = tk.Label(output_frame, text="Kein Verzeichnis ausgewählt", fg="gray")
        self.output_label.pack(side="left", fill="x", expand=True)

        output_btn = tk.Button(
            output_frame,
            text="Durchsuchen...",
            command=self._select_output_dir,
            width=15
        )
        output_btn.pack(side="right")

        # Einstellungen
        settings_frame = tk.LabelFrame(self.root, text="Einstellungen", padx=10, pady=10)
        settings_frame.pack(fill="x", padx=20, pady=5)

        # Größe
        size_frame = tk.Frame(settings_frame)
        size_frame.pack(fill="x", pady=5)
        tk.Label(size_frame, text="Ausgabegröße (Pixel):").pack(side="left")
        self.size_var = tk.IntVar(value=1024)
        size_spinner = tk.Spinbox(
            size_frame,
            from_=256,
            to=4096,
            increment=128,
            textvariable=self.size_var,
            width=10
        )
        size_spinner.pack(side="right")

        # Qualität
        quality_frame = tk.Frame(settings_frame)
        quality_frame.pack(fill="x", pady=5)
        tk.Label(quality_frame, text="WebP-Qualität (0-100):").pack(side="left")
        self.quality_var = tk.IntVar(value=80)
        quality_spinner = tk.Spinbox(
            quality_frame,
            from_=1,
            to=100,
            increment=5,
            textvariable=self.quality_var,
            width=10
        )
        quality_spinner.pack(side="right")

        # Fortschrittsanzeige
        progress_frame = tk.LabelFrame(self.root, text="Fortschritt", padx=10, pady=10)
        progress_frame.pack(fill="both", expand=True, padx=20, pady=5)

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode="determinate",
            length=400
        )
        self.progress_bar.pack(pady=5)

        self.status_label = tk.Label(
            progress_frame,
            text="Bereit",
            font=("Arial", 10)
        )
        self.status_label.pack()

        # Log-Textfeld
        log_scroll = tk.Scrollbar(progress_frame)
        log_scroll.pack(side="right", fill="y")

        self.log_text = tk.Text(
            progress_frame,
            height=8,
            width=70,
            yscrollcommand=log_scroll.set,
            state="disabled"
        )
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.config(command=self.log_text.yview)

        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill="x", padx=20, pady=10)

        self.start_btn = tk.Button(
            button_frame,
            text="Verarbeitung starten",
            command=self._start_processing,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            width=20
        )
        self.start_btn.pack(side="left", padx=5)

        close_btn = tk.Button(
            button_frame,
            text="Schließen",
            command=self.root.quit,
            height=2,
            width=15
        )
        close_btn.pack(side="right", padx=5)

    def _select_input_dir(self):
        """Wählt das Eingangsverzeichnis aus."""
        directory = filedialog.askdirectory(title="Eingangsverzeichnis wählen")
        if directory:
            self.input_dir = Path(directory)
            self.input_label.config(text=str(self.input_dir), fg="black")
            self._log(f"Eingangsverzeichnis: {self.input_dir}")

    def _select_output_dir(self):
        """Wählt das Ausgangsverzeichnis aus."""
        directory = filedialog.askdirectory(title="Ausgangsverzeichnis wählen")
        if directory:
            self.output_dir = Path(directory)
            self.output_label.config(text=str(self.output_dir), fg="black")
            self._log(f"Ausgangsverzeichnis: {self.output_dir}")

    def _log(self, message: str):
        """Fügt eine Nachricht zum Log hinzu."""
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _start_processing(self):
        """Startet die Batch-Verarbeitung."""
        if self.processing:
            return

        if not self.input_dir:
            messagebox.showerror("Fehler", "Bitte wählen Sie ein Eingangsverzeichnis!")
            return

        if not self.output_dir:
            messagebox.showerror("Fehler", "Bitte wählen Sie ein Ausgangsverzeichnis!")
            return

        # Unterstützte Bildformate
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [
            f for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            messagebox.showwarning(
                "Keine Bilder gefunden",
                f"Im Verzeichnis {self.input_dir} wurden keine Bilddateien gefunden."
            )
            return

        # Verarbeitung in separatem Thread starten
        thread = threading.Thread(
            target=self._process_images,
            args=(image_files,),
            daemon=True
        )
        thread.start()

    def _process_images(self, image_files: List[Path]):
        """Verarbeitet alle Bilder."""
        self.processing = True
        self.start_btn.config(state="disabled")

        total = len(image_files)
        self.progress_bar["maximum"] = total
        self.progress_bar["value"] = 0

        self._log(f"\n{'=' * 50}")
        self._log(f"Starte Verarbeitung von {total} Bild(ern)...")
        self._log(f"{'=' * 50}\n")

        success_count = 0
        error_count = 0

        for idx, input_file in enumerate(image_files, 1):
            try:
                self.status_label.config(
                    text=f"Verarbeite {idx}/{total}: {input_file.name}"
                )

                output_file = self.output_dir / f"{input_file.stem}_processed.webp"

                self._log(f"[{idx}/{total}] Verarbeite: {input_file.name}")

                self._process_single_image(
                    input_file,
                    output_file,
                    self.size_var.get(),
                    self.quality_var.get()
                )

                self._log(f"  ✓ Gespeichert: {output_file.name}")
                success_count += 1

            except Exception as e:
                self._log(f"  ✗ Fehler: {str(e)}")
                error_count += 1

            self.progress_bar["value"] = idx
            self.root.update_idletasks()

        # Abschluss
        self._log(f"\n{'=' * 50}")
        self._log(f"Verarbeitung abgeschlossen!")
        self._log(f"Erfolgreich: {success_count}")
        self._log(f"Fehler: {error_count}")
        self._log(f"{'=' * 50}\n")

        self.status_label.config(text="Fertig!")
        self.start_btn.config(state="normal")
        self.processing = False

        messagebox.showinfo(
            "Fertig",
            f"Verarbeitung abgeschlossen!\n\n"
            f"Erfolgreich: {success_count}\n"
            f"Fehler: {error_count}"
        )

    def _process_single_image(
            self,
            input_path: Path,
            output_path: Path,
            size: int,
            quality: int
    ):
        """Verarbeitet ein einzelnes Bild."""
        # 1. Bild laden und Hintergrund entfernen
        with open(input_path, "rb") as f:
            input_data = f.read()
        img_data = remove(input_data)
        img = Image.open(io.BytesIO(img_data)).convert("RGBA")

        # 2. 90° gegen Uhrzeigersinn drehen
        img = img.transpose(Image.ROTATE_90)

        # 3. Begrenzungsrahmen des sichtbaren Objekts
        bbox = img.getbbox()
        if bbox is None:
            raise ValueError("Kein sichtbares Objekt nach Hintergrundentfernung gefunden.")

        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        side = max(w, h)

        # 4. Quadratisches Canvas, Objekt zentriert
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        paste_x = (side - w) // 2
        paste_y = (side - h) // 2
        canvas.paste(img.crop(bbox), (paste_x, paste_y))

        # 5. Auf Zielgröße skalieren und als WebP speichern
        canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
        canvas.save(output_path, "WEBP", quality=quality)


def main():
    """Startet die GUI-Anwendung."""
    root = tk.Tk()
    app = ImageProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()