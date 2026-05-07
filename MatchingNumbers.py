"""
Foto-Umbenenner Tool
Ablauf: Iteriere über jede Hersteller-Nr. in der Excel-Tabelle
        → schlage die zugehörige Picture-ID nach
        → finde das Foto im Quellordner
        → kopiere es mit der Hersteller-Nr. als neuem Namen in den Zielordner
"""

import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.expanduser("~"), "foto_umbenenner.log")
logger.add(LOG_FILE, rotation="5 MB", retention="10 days",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".webp", ".cr2", ".nef", ".arw", ".dng", ".rw2", ".tif", ".tiff", ".png"}


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def build_file_index(photo_folder: str) -> dict:
    """
    Scannt den Quellordner einmalig und baut ein Index-Dict:
        picture_id (str, ohne führende Nullen)  →  vollständiger Dateipfad

    Kameradateinamen wie IMG_1042.jpg, DSC_1042.CR2 usw. werden erkannt,
    indem die abschließende Ziffernfolge des Dateinamenstamms extrahiert wird.
    """
    index = {}
    for fname in os.listdir(photo_folder):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            continue
        # Führende Ziffern (z.B. "1_processed") UND abschließende Ziffern (z.B. "IMG_1042")
        # werden erkannt; der längere Treffer gewinnt.
        leading = ""
        for ch in stem:
            if ch.isdigit():
                leading += ch
            else:
                break
        trailing = ""
        for ch in reversed(stem):
            if ch.isdigit():
                trailing = ch + trailing
            else:
                break
        digits = leading if len(leading) >= len(trailing) else trailing
        if digits:
            key = str(int(digits))
            index[key] = os.path.join(photo_folder, fname)
            logger.debug(f"Indiziert: {fname}  →  ID {key}")
    logger.info(f"Foto-Index aufgebaut: {len(index)} Dateien gefunden")
    return index


def safe_filename(name: str) -> str:
    """Ersetzt im Dateinamen ungültige Zeichen durch Bindestrich."""
    for ch in r'/\:*?"<>|':
        name = name.replace(ch, "-")
    return name.strip()


def process_photos(df: pd.DataFrame, id_col: str, name_col: str,
                   photo_folder: str, dest_folder: str,
                   progress_var: tk.IntVar, status_var: tk.StringVar,
                   root: tk.Tk):
    """
    Hauptschleife – iteriert Zeile für Zeile über die Excel-Tabelle:
      1. Hersteller-Nr. lesen
      2. Zugehörige Picture-ID lesen
      3. Foto per Index nachschlagen
      4. Kopie mit Hersteller-Nr. als Dateiname anlegen
    """
    file_index = build_file_index(photo_folder)

    total = len(df)
    ok = skipped = 0

    for row_num, (_, row) in enumerate(df.iterrows(), start=1):

        # Hersteller-Nr. lesen
        hersteller_raw = row[name_col]
        if pd.isna(hersteller_raw) or str(hersteller_raw).strip() == "":
            logger.warning(f"Zeile {row_num}: Hersteller-Nr. leer – übersprungen")
            skipped += 1
            _update_progress(progress_var, status_var, root, row_num, total)
            continue
        hersteller = str(hersteller_raw).strip()

        # Picture-ID lesen
        try:
            pic_id = str(int(row[id_col]))
        except (ValueError, TypeError):
            logger.warning(f"Zeile {row_num} [{hersteller}]: "
                           f"Ungültige Picture-ID '{row[id_col]}' – übersprungen")
            skipped += 1
            _update_progress(progress_var, status_var, root, row_num, total)
            continue

        # Foto im Index nachschlagen
        src = file_index.get(pic_id)
        if src is None:
            logger.warning(f"Zeile {row_num} [{hersteller}]: "
                           f"Kein Foto für Picture-ID {pic_id} gefunden – übersprungen")
            skipped += 1
        else:
            ext = os.path.splitext(src)[1].lower()
            dst_name = safe_filename(hersteller) + ext
            dst = os.path.join(dest_folder, dst_name)

            # Bei Namenskollision Suffix anhängen
            counter = 1
            while os.path.exists(dst):
                dst = os.path.join(dest_folder,
                                   f"{safe_filename(hersteller)}_{counter}{ext}")
                counter += 1

            shutil.copy2(src, dst)
            logger.info(f"Zeile {row_num} [{hersteller}]: "
                        f"{os.path.basename(src)}  →  {os.path.basename(dst)}")
            ok += 1

        _update_progress(progress_var, status_var, root, row_num, total)

    summary = (f"Fertig!\n\n"
               f"✔ Erfolgreich kopiert: {ok}\n"
               f"⚠ Übersprungen: {skipped}")
    logger.info(f"Abgeschlossen – kopiert: {ok}, übersprungen: {skipped}")
    status_var.set("Fertig!")
    messagebox.showinfo("Ergebnis", summary)


def _update_progress(progress_var, status_var, root, current, total):
    progress_var.set(int(current / total * 100))
    status_var.set(f"Verarbeite {current}/{total} ...")
    root.update_idletasks()


# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Foto-Umbenenner")
        self.resizable(False, False)
        self.configure(padx=20, pady=20)

        self.excel_path = tk.StringVar()
        self.photo_folder = tk.StringVar()
        self.dest_folder = tk.StringVar()
        self.id_col = tk.StringVar()
        self.name_col = tk.StringVar()
        self.progress_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Bereit")
        self.df = None

        self._build_ui()
        logger.info("Anwendung gestartet")

    def _build_ui(self):
        pad = {"pady": 5}

        # 1. Excel
        tk.Label(self, text="1. Excel-Datei auswählen",
                 font=("", 10, "bold")).grid(row=0, column=0, columnspan=3,
                                              sticky="w", **pad)
        tk.Entry(self, textvariable=self.excel_path, width=55,
                 state="readonly").grid(row=1, column=0, columnspan=2, sticky="ew")
        tk.Button(self, text="Durchsuchen …",
                  command=self._pick_excel).grid(row=1, column=2, padx=(8, 0))

        # 2. Spalten
        tk.Label(self, text="2. Spalten zuordnen",
                 font=("", 10, "bold")).grid(row=2, column=0, columnspan=3,
                                              sticky="w", **pad)
        tk.Label(self, text="Picture-ID Spalte:").grid(row=3, column=0, sticky="w")
        self.id_combo = ttk.Combobox(self, textvariable=self.id_col,
                                     state="disabled", width=30)
        self.id_combo.grid(row=3, column=1, columnspan=2, sticky="ew", **pad)

        tk.Label(self, text="Hersteller-Nr. Spalte:").grid(row=4, column=0, sticky="w")
        self.name_combo = ttk.Combobox(self, textvariable=self.name_col,
                                       state="disabled", width=30)
        self.name_combo.grid(row=4, column=1, columnspan=2, sticky="ew", **pad)

        # 3. Ordner
        tk.Label(self, text="3. Ordner auswählen",
                 font=("", 10, "bold")).grid(row=5, column=0, columnspan=3,
                                              sticky="w", **pad)
        tk.Label(self, text="Foto-Quellordner:").grid(row=6, column=0, sticky="w")
        tk.Entry(self, textvariable=self.photo_folder, width=45,
                 state="readonly").grid(row=6, column=1, sticky="ew")
        tk.Button(self, text="Wählen …",
                  command=lambda: self._pick_folder(
                      self.photo_folder, "Quellordner")).grid(
            row=6, column=2, padx=(8, 0))

        tk.Label(self, text="Zielordner (Kopien):").grid(row=7, column=0, sticky="w")
        tk.Entry(self, textvariable=self.dest_folder, width=45,
                 state="readonly").grid(row=7, column=1, sticky="ew")
        tk.Button(self, text="Wählen …",
                  command=lambda: self._pick_folder(
                      self.dest_folder, "Zielordner")).grid(
            row=7, column=2, padx=(8, 0))

        # Fortschritt
        ttk.Separator(self, orient="horizontal").grid(
            row=8, column=0, columnspan=3, sticky="ew", pady=10)
        tk.Label(self, textvariable=self.status_var, fg="gray").grid(
            row=9, column=0, columnspan=3, sticky="w")
        ttk.Progressbar(self, variable=self.progress_var, maximum=100,
                        length=430).grid(row=10, column=0, columnspan=3,
                                         sticky="ew", pady=(0, 10))

        # Start
        self.start_btn = tk.Button(
            self, text="▶  Umbenennen starten", bg="#2563eb", fg="white",
            font=("", 10, "bold"), padx=10, pady=6,
            command=self._start, state="disabled")
        self.start_btn.grid(row=11, column=0, columnspan=3, sticky="ew")

        tk.Label(self, text=f"Log: {LOG_FILE}", fg="gray",
                 font=("", 8)).grid(row=12, column=0, columnspan=3,
                                    sticky="w", pady=(6, 0))
        self.columnconfigure(1, weight=1)

    def _pick_excel(self):
        path = filedialog.askopenfilename(
            title="Excel-Datei auswählen",
            filetypes=[("Excel-Dateien", "*.xlsx *.xls *.xlsm"),
                       ("Alle Dateien", "*.*")])
        if not path:
            return
        self.excel_path.set(path)
        logger.info(f"Excel-Datei gewählt: {path}")
        try:
            self.df = pd.read_excel(path)
            cols = list(self.df.columns)
            self.id_combo["values"] = cols
            self.name_combo["values"] = cols
            self.id_combo["state"] = "readonly"
            self.name_combo["state"] = "readonly"
            logger.info(f"Spalten: {cols}")
            self._check_ready()
        except Exception as e:
            logger.error(f"Excel-Lesefehler: {e}")
            messagebox.showerror("Fehler",
                                 f"Excel-Datei konnte nicht gelesen werden:\n{e}")

    def _pick_folder(self, var: tk.StringVar, title: str):
        folder = filedialog.askdirectory(title=title)
        if folder:
            var.set(folder)
            logger.info(f"{title}: {folder}")
            self._check_ready()

    def _check_ready(self, *_):
        ready = all([self.excel_path.get(), self.id_col.get(),
                     self.name_col.get(), self.photo_folder.get(),
                     self.dest_folder.get()])
        self.start_btn["state"] = "normal" if ready else "disabled"

    def _start(self):
        if self.id_col.get() == self.name_col.get():
            messagebox.showwarning("Warnung",
                                   "Bitte zwei verschiedene Spalten auswählen.")
            return
        if not os.path.isdir(self.photo_folder.get()):
            messagebox.showerror("Fehler", "Quellordner existiert nicht.")
            return
        os.makedirs(self.dest_folder.get(), exist_ok=True)

        self.start_btn["state"] = "disabled"
        self.progress_var.set(0)
        logger.info(f"Starte: {len(self.df)} Zeilen, "
                    f"Quelle: {self.photo_folder.get()}, "
                    f"Ziel: {self.dest_folder.get()}")

        process_photos(
            self.df,
            self.id_col.get(),
            self.name_col.get(),
            self.photo_folder.get(),
            self.dest_folder.get(),
            self.progress_var,
            self.status_var,
            self
        )
        self.start_btn["state"] = "normal"


def _main():
    app = App()
    app.id_combo.bind("<<ComboboxSelected>>", app._check_ready)
    app.name_combo.bind("<<ComboboxSelected>>", app._check_ready)
    app.mainloop()


if __name__ == "__main__":
    _main()