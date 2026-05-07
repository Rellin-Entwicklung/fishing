import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
from threading import Thread
from loguru import logger

# Loguru Konfiguration
logger.remove()
logger.add(sys.stderr,
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
           level="INFO")
logger.add("bild_optimierung_detail.log", rotation="10 MB", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
           level="DEBUG")


class ImageResizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bild-Größen-Downloader & Fixer 2.0")
        self.root.geometry("600x420")

        self.src_path = tk.StringVar()
        self.dst_path = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        # UI (Analog zu den Vorversionen für einfache Bedienung)
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")

        tk.Label(main_frame, text="1. Quellverzeichnis:", font=("Arial", 10, "bold")).pack(anchor="w")
        src_f = tk.Frame(main_frame)
        src_f.pack(fill="x", pady=(0, 15))
        tk.Entry(src_f, textvariable=self.src_path, state="readonly").pack(side="left", expand=True, fill="x")
        tk.Button(src_f, text="Wählen", command=lambda: self.browse(self.src_path)).pack(side="right", padx=5)

        tk.Label(main_frame, text="2. Zielverzeichnis:", font=("Arial", 10, "bold")).pack(anchor="w")
        dst_f = tk.Frame(main_frame)
        dst_f.pack(fill="x", pady=(0, 15))
        tk.Entry(dst_f, textvariable=self.dst_path, state="readonly").pack(side="left", expand=True, fill="x")
        tk.Button(dst_f, text="Wählen", command=lambda: self.browse(self.dst_path)).pack(side="right", padx=5)

        self.progress_label = tk.Label(main_frame, text="Bereit für Analyse...")
        self.progress_label.pack()
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=500, mode="determinate")
        self.progress.pack(pady=10)

        self.start_btn = tk.Button(main_frame, text="Größenprüfung & Optimierung starten",
                                   command=self.start_task, bg="#4CAF50", fg="white",
                                   font=("Arial", 11, "bold"), height=2)
        self.start_btn.pack(pady=10, fill="x")

    def browse(self, var):
        folder = filedialog.askdirectory()
        if folder: var.set(folder)

    def format_bytes(self, size):
        # Hilfsfunktion zur Formatierung der Dateigröße
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024

    def start_task(self):
        Thread(target=self.process_images, daemon=True).start()

    @logger.catch
    def process_images(self):
        src, dst = self.src_path.get(), self.dst_path.get()
        if not src or not dst:
            messagebox.showwarning("Fehler", "Pfade fehlen!")
            return

        self.start_btn.config(state="disabled")
        supported = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
        files = [f for f in os.listdir(src) if f.lower().endswith(supported)]

        logger.info(f"Starte Batch-Verarbeitung: {len(files)} Dateien gefunden.")
        self.progress["maximum"] = len(files)

        for i, filename in enumerate(files):
            input_path = os.path.join(src, filename)
            output_path = os.path.join(dst, filename)

            try:
                # 1. Größe der Quelldatei ermitteln
                size_before = os.path.getsize(input_path)

                with Image.open(input_path) as img:
                    w, h = img.size
                    logger.debug(f"Verarbeite: {filename} | {w}x{h}px | {self.format_bytes(size_before)}")

                    # Logik-Check für die 251px Regel
                    if w < 251 or h < 251:
                        target_dim = max(w, h, 251)
                        logger.info(f"  -> {filename} zu klein. Erstelle Hintergrund: {target_dim}x{target_dim}px")

                        # Weißer Hintergrund (Modus RGB für Standard, wird bei Bedarf zu WebP)
                        new_img = Image.new("RGB", (target_dim, target_dim), (255, 255, 255))

                        # Zentrierung
                        x = (target_dim - w) // 2
                        y = (target_dim - h) // 2

                        # Transparenz-Handling (besonders wichtig für WebP/PNG)
                        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                            # Nutze das Bild selbst als Alpha-Maske
                            new_img.paste(img, (x, y), img.convert("RGBA"))
                        else:
                            new_img.paste(img, (x, y))

                        # Speichern (Format bleibt gleich)
                        new_img.save(output_path, quality=95, optimize=True)
                    else:
                        # Bild ist groß genug -> Kopie mit Optimierung
                        img.save(output_path, quality=95, optimize=True)

                # 2. Größe der Zieldatei ermitteln
                size_after = os.path.getsize(output_path)
                diff = size_after - size_before
                diff_str = f"{'+' if diff > 0 else ''}{self.format_bytes(diff)}"

                logger.success(f"  -> Fertig: {filename} | Neu: {self.format_bytes(size_after)} (Diff: {diff_str})")

            except Exception as e:
                logger.error(f"Fehler bei {filename}: {e}")

            self.progress["value"] = i + 1
            self.progress_label.config(text=f"Verarbeite: {i + 1}/{len(files)}")
            self.root.update_idletasks()

        logger.info("Verarbeitung aller Dateien abgeschlossen.")
        self.start_btn.config(state="normal")
        messagebox.showinfo("Erfolg", f"{len(files)} Bilder verarbeitet.\nDetails siehe Log-Datei.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageResizerApp(root)
    root.mainloop()