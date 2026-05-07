import os
import sys
import pandas as pd
import requests
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from threading import Thread
from loguru import logger
from PIL import Image  # Neu hinzugefügt für Pixel-Auslesung
import io

# Loguru Konfiguration
logger.remove()
logger.add(sys.stderr,
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
           level="INFO")
logger.add("image_downloader_full_audit.log", rotation="5 MB",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="DEBUG")


class ExcelDownloaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Excel Image Downloader Pro (Audit Mode)")
        self.root.geometry("600x420")

        self.xlsx_path = tk.StringVar()
        self.dest_path = tk.StringVar()

        logger.info("Downloader gestartet. Logging: Dateigröße + Pixelmaße.")
        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")

        tk.Label(main_frame, text="1. Excel-Datei (.xlsx):", font=("Arial", 10, "bold")).pack(anchor="w")
        xls_f = tk.Frame(main_frame)
        xls_f.pack(fill="x", pady=(0, 15))
        tk.Entry(xls_f, textvariable=self.xlsx_path, state="readonly").pack(side="left", expand=True, fill="x")
        tk.Button(xls_f, text="Wählen", command=self.browse_xlsx).pack(side="right", padx=5)

        tk.Label(main_frame, text="2. Zielordner für Bilder:", font=("Arial", 10, "bold")).pack(anchor="w")
        dst_f = tk.Frame(main_frame)
        dst_f.pack(fill="x", pady=(0, 15))
        tk.Entry(dst_f, textvariable=self.dest_path, state="readonly").pack(side="left", expand=True, fill="x")
        tk.Button(dst_f, text="Wählen", command=self.browse_dest).pack(side="right", padx=5)

        self.progress_label = tk.Label(main_frame, text="Warte auf Start...")
        self.progress_label.pack()
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=500, mode="determinate")
        self.progress.pack(pady=10)

        self.start_btn = tk.Button(main_frame, text="Download & Analyse starten",
                                   command=self.start_thread, bg="#2196F3", fg="white",
                                   font=("Arial", 11, "bold"), height=2)
        self.start_btn.pack(pady=10, fill="x")

    def browse_xlsx(self):
        file = filedialog.askopenfilename(filetypes=[("Excel Dateien", "*.xlsx")])
        if file: self.xlsx_path.set(file)

    def browse_dest(self):
        folder = filedialog.askdirectory()
        if folder: self.dest_path.set(folder)

    def format_bytes(self, size):
        for unit in ['B', 'KB', 'MB']:
            if size < 1024: return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} GB"

    def start_thread(self):
        Thread(target=self.run_downloads, daemon=True).start()

    @logger.catch
    def run_downloads(self):
        xlsx, dst = self.xlsx_path.get(), self.dest_path.get()
        if not xlsx or not dst:
            messagebox.showwarning("Fehler", "Pfade fehlen!")
            return

        self.start_btn.config(state="disabled")

        try:
            df = pd.read_excel(xlsx)
            urls = df.iloc[:, 0].dropna().astype(str).tolist()
            urls = [u.strip() for u in urls if u.strip().lower().startswith("http")]

            total = len(urls)
            self.progress["maximum"] = total
            success_count, fail_count = 0, 0

            for i, url in enumerate(urls):
                try:
                    filename = url.split("/")[-1].split("?")[0]
                    if not filename: filename = f"img_{i}.webp"
                    filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
                    target_file = os.path.join(dst, filename)

                    response = requests.get(url, stream=True, timeout=20)
                    if response.status_code == 200:
                        content = response.content  # Gesamten Inhalt laden für Analyse

                        # Bild im Speicher öffnen, um Pixel zu lesen
                        with Image.open(io.BytesIO(content)) as img:
                            w, h = img.size

                        # Datei speichern
                        with open(target_file, 'wb') as f:
                            f.write(content)

                        file_size = len(content)
                        logger.success(f"OK: {filename} | {w}x{h} px | {self.format_bytes(file_size)}")
                        success_count += 1
                    else:
                        logger.error(f"HTTP {response.status_code}: {url}")
                        fail_count += 1

                except Exception as e:
                    logger.error(f"Fehler bei {url}: {e}")
                    fail_count += 1

                self.progress["value"] = i + 1
                self.progress_label.config(text=f"Bild {i + 1} von {total}")
                self.root.update_idletasks()

            logger.info(f"Fertig. Erfolg: {success_count}, Fehler: {fail_count}")
            messagebox.showinfo("Abschluss", f"Download beendet.\nErfolgreich: {success_count}")

        except Exception as e:
            logger.critical(f"Excel-Fehler: {e}")
        finally:
            self.start_btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = ExcelDownloaderApp(root)
    root.mainloop()