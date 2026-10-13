#!/usr/bin/env python3
"""
Produktfoto-Tracker: Namen erfassen, Foto-Status und Drive-Ablageorte im Blick behalten.

- customtkinter GUI, Speicherung in SQLite
- Namensaufbau: Präfix + variabler Teil + Suffix
- Bezeichner der Produktfamilie gilt für alle Produkte mit gleichem Suffix (max. 30 Zeichen)
- Übersicht gruppiert nach Suffix / Bezeichner
- Präfix, Suffix und Bezeichner nur beim ersten Produkt der Serie, danach nur noch der variable Teil
- Felder: Produkt vorhanden, Foto erledigt (Kreuzchen)
- geaendert_am wird bei jeder Änderung automatisch gesetzt
- Bilderordner einlesen: Dateiname ohne Endung = Produktname, Foto vorhanden
- Ablageorte Original und bearbeitet (Google Drive)
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import webbrowser
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import customtkinter as ctk
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "produktfotos.db"
BEZEICHNER_MAX = 30
GROUP_IID_PREFIX = "grp:"
EMPTY_SUFFIX_TOKEN = "#empty"
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".cr3", ".cr2", ".dng",
}

FILTER_ALL = "Alle"
FILTER_TODO = "Zu fotografieren"
FILTER_DONE = "Fotos vorhanden"
FILTER_MISSING = "Produkt fehlt"


logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
)
logger.add(
    str(PROJECT_ROOT / "produktfoto_tracker_{time:YYYY-MM-DD}.log"),
    rotation="00:00",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG",
)


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_geaendert(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "—"
    return value[:16]


def collect_image_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda p: p.name.casefold())
    return files


def split_imported_stem(stem: str, prefix: str, suffix: str) -> tuple[str, str, str]:
    stem = stem.strip()
    prefix = prefix.strip()
    suffix = suffix.strip()
    parsed_prefix, variable, parsed_suffix = "", stem, ""
    if prefix and stem.startswith(prefix):
        rest = stem[len(prefix) :]
        if suffix and rest.endswith(suffix) and len(rest) >= len(suffix):
            parsed_prefix, variable, parsed_suffix = prefix, rest[: len(rest) - len(suffix)], suffix
        elif not suffix:
            parsed_prefix, variable, parsed_suffix = prefix, rest, ""
    elif suffix and stem.endswith(suffix) and len(stem) > len(suffix):
        parsed_prefix, variable, parsed_suffix = "", stem[: -len(suffix)], suffix
    if build_full_name(parsed_prefix, variable, parsed_suffix) != stem:
        return "", stem, ""
    if not variable:
        return "", stem, ""
    return parsed_prefix, variable, parsed_suffix


def build_full_name(prefix: str, variable: str, suffix: str) -> str:
    return f"{prefix.strip()}{variable.strip()}{suffix.strip()}"


def natural_key(text: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", text)]


def suffix_group_label(suffix: str, bezeichner: str) -> str:
    family = bezeichner.strip() or "ohne Bezeichner"
    ending = suffix.strip() or "ohne Suffix"
    return f"{family}  ·  {ending}"


def group_iid(suffix: str) -> str:
    return GROUP_IID_PREFIX + (suffix if suffix else EMPTY_SUFFIX_TOKEN)


def suffix_from_group_iid(iid: str) -> str | None:
    if not iid.startswith(GROUP_IID_PREFIX):
        return None
    rest = iid[len(GROUP_IID_PREFIX) :]
    return "" if rest == EMPTY_SUFFIX_TOKEN else rest


def is_product_iid(iid: str) -> bool:
    return iid.isdigit()


def normalize_bezeichner(value: str) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) > BEZEICHNER_MAX:
        raise ValueError(f"Bezeichner darf höchstens {BEZEICHNER_MAX} Zeichen haben.")
    return cleaned


def shorten_path(path: str, max_len: int = 36) -> str:
    path = path.strip()
    if not path:
        return "—"
    if len(path) <= max_len:
        return path
    return "…" + path[-(max_len - 1) :]


def mark(value: bool) -> str:
    return "☑" if value else "☐"


@dataclass
class Product:
    id: int
    prefix: str
    variable_part: str
    suffix: str
    bezeichner: str
    full_name: str
    produkt_vorhanden: bool
    foto_erledigt: bool
    pfad_original: str
    pfad_bearbeitet: str
    created_at: str
    updated_at: str
    geaendert_am: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Product:
        keys = set(row.keys())
        geaendert = ""
        if "geaendert_am" in keys:
            geaendert = row["geaendert_am"] or ""
        if not geaendert:
            geaendert = row["updated_at"] or ""
        return cls(
            id=row["id"],
            prefix=row["prefix"],
            variable_part=row["variable_part"],
            suffix=row["suffix"],
            bezeichner=(row["bezeichner"] if "bezeichner" in keys else "") or "",
            full_name=row["full_name"],
            produkt_vorhanden=bool(row["produkt_vorhanden"]),
            foto_erledigt=bool(row["foto_erledigt"]),
            pfad_original=row["pfad_original"] or "",
            pfad_bearbeitet=row["pfad_bearbeitet"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            geaendert_am=geaendert,
        )


class ProductDB:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prefix TEXT NOT NULL,
                variable_part TEXT NOT NULL,
                suffix TEXT NOT NULL DEFAULT '',
                bezeichner TEXT NOT NULL DEFAULT '',
                full_name TEXT NOT NULL UNIQUE,
                produkt_vorhanden INTEGER NOT NULL DEFAULT 0,
                foto_erledigt INTEGER NOT NULL DEFAULT 0,
                pfad_original TEXT NOT NULL DEFAULT '',
                pfad_bearbeitet TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                geaendert_am TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_products_status
                ON products (foto_erledigt, produkt_vorhanden);
            """
        )
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(products)")}
        if "bezeichner" not in columns:
            self.conn.execute(
                "ALTER TABLE products ADD COLUMN bezeichner TEXT NOT NULL DEFAULT ''"
            )
            logger.info("Datenbank: Spalte bezeichner ergänzt")
        if "geaendert_am" not in columns:
            self.conn.execute(
                "ALTER TABLE products ADD COLUMN geaendert_am TEXT NOT NULL DEFAULT ''"
            )
            self.conn.execute(
                """
                UPDATE products
                SET geaendert_am = CASE
                    WHEN updated_at IS NOT NULL AND updated_at != '' THEN updated_at
                    ELSE strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')
                END
                WHERE geaendert_am = '' OR geaendert_am IS NULL
                """
            )
            logger.info("Datenbank: Spalte geaendert_am ergänzt")
        self.conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_products_geaendert_am
            AFTER UPDATE ON products
            FOR EACH ROW
            WHEN NEW.geaendert_am IS OLD.geaendert_am
            BEGIN
                UPDATE products
                SET geaendert_am = strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')
                WHERE id = NEW.id;
            END;
            """
        )
        self.conn.commit()
        self._normalize_bezeichner_by_suffix()

    def get_setting(self, key: str, default: str = "") -> str:
        row = self.conn.execute(
            "SELECT value FROM settings WHERE key = ?",
            (key,),
        ).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def add_product(
        self,
        prefix: str,
        variable: str,
        suffix: str,
        bezeichner: str,
        produkt_vorhanden: bool,
        foto_erledigt: bool,
        pfad_original: str,
        pfad_bearbeitet: str,
    ) -> Product:
        full_name = build_full_name(prefix, variable, suffix)
        suffix = suffix.strip()
        family = normalize_bezeichner(bezeichner) or self.get_bezeichner_for_suffix(suffix)
        now = now_stamp()
        try:
            cursor = self.conn.execute(
                """
                INSERT INTO products (
                    prefix, variable_part, suffix, bezeichner, full_name,
                    produkt_vorhanden, foto_erledigt,
                    pfad_original, pfad_bearbeitet, created_at, updated_at, geaendert_am
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prefix.strip(),
                    variable.strip(),
                    suffix,
                    family,
                    full_name,
                    int(produkt_vorhanden),
                    int(foto_erledigt),
                    pfad_original.strip(),
                    pfad_bearbeitet.strip(),
                    now,
                    now,
                    now,
                ),
            )
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError(f"Produktname existiert bereits: {full_name}") from e
        if family:
            self.apply_bezeichner_to_suffix(suffix, family)
        return self.get_product(cursor.lastrowid)

    def update_product(
        self,
        product_id: int,
        prefix: str,
        variable: str,
        suffix: str,
        bezeichner: str,
        produkt_vorhanden: bool,
        foto_erledigt: bool,
        pfad_original: str,
        pfad_bearbeitet: str,
    ) -> Product:
        full_name = build_full_name(prefix, variable, suffix)
        suffix = suffix.strip()
        family = normalize_bezeichner(bezeichner) or self.get_bezeichner_for_suffix(suffix)
        try:
            self.conn.execute(
                """
                UPDATE products SET
                    prefix = ?, variable_part = ?, suffix = ?, bezeichner = ?, full_name = ?,
                    produkt_vorhanden = ?, foto_erledigt = ?,
                    pfad_original = ?, pfad_bearbeitet = ?, updated_at = ?, geaendert_am = ?
                WHERE id = ?
                """,
                (
                    prefix.strip(),
                    variable.strip(),
                    suffix,
                    family,
                    full_name,
                    int(produkt_vorhanden),
                    int(foto_erledigt),
                    pfad_original.strip(),
                    pfad_bearbeitet.strip(),
                    now_stamp(),
                    now_stamp(),
                    product_id,
                ),
            )
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError(f"Produktname existiert bereits: {full_name}") from e
        if family:
            self.apply_bezeichner_to_suffix(suffix, family)
        return self.get_product(product_id)

    def set_flag(self, product_id: int, field: str, value: bool) -> None:
        if field not in {"produkt_vorhanden", "foto_erledigt"}:
            raise ValueError(f"Unbekanntes Feld: {field}")
        now = now_stamp()
        self.conn.execute(
            f"UPDATE products SET {field} = ?, updated_at = ?, geaendert_am = ? WHERE id = ?",
            (int(value), now, now, product_id),
        )
        self.conn.commit()

    def delete_product(self, product_id: int) -> None:
        self.conn.execute("DELETE FROM products WHERE id = ?", (product_id,))
        self.conn.commit()

    def get_product(self, product_id: int) -> Product:
        row = self.conn.execute(
            "SELECT * FROM products WHERE id = ?",
            (product_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Produkt {product_id} nicht gefunden.")
        return Product.from_row(row)

    def list_products(self) -> list[Product]:
        rows = self.conn.execute("SELECT * FROM products").fetchall()
        products = [Product.from_row(row) for row in rows]
        products.sort(key=lambda p: (natural_key(p.suffix), natural_key(p.bezeichner), natural_key(p.full_name)))
        return products

    def get_bezeichner_for_suffix(self, suffix: str) -> str:
        row = self.conn.execute(
            """
            SELECT bezeichner FROM products
            WHERE suffix = ? AND TRIM(bezeichner) != ''
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (suffix.strip(),),
        ).fetchone()
        return (row["bezeichner"] if row else "") or ""

    def apply_bezeichner_to_suffix(self, suffix: str, bezeichner: str) -> str:
        family = normalize_bezeichner(bezeichner)
        self.conn.execute(
            """
            UPDATE products
            SET bezeichner = ?, updated_at = ?, geaendert_am = ?
            WHERE suffix = ? AND bezeichner != ?
            """,
            (family, now_stamp(), now_stamp(), suffix.strip(), family),
        )
        self.conn.commit()
        return family

    def suffix_exists(self, suffix: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM products WHERE suffix = ? LIMIT 1",
            (suffix.strip(),),
        ).fetchone()
        return row is not None

    def get_product_by_full_name(self, full_name: str) -> Product | None:
        row = self.conn.execute(
            "SELECT * FROM products WHERE full_name = ?",
            (full_name,),
        ).fetchone()
        return Product.from_row(row) if row else None

    def mark_foto_vorhanden(self, product_id: int, pfad_original: str = "") -> Product:
        now = now_stamp()
        if pfad_original.strip():
            self.conn.execute(
                """
                UPDATE products
                SET foto_erledigt = 1, pfad_original = ?, updated_at = ?, geaendert_am = ?
                WHERE id = ?
                """,
                (pfad_original.strip(), now, now, product_id),
            )
        else:
            self.conn.execute(
                """
                UPDATE products
                SET foto_erledigt = 1, updated_at = ?, geaendert_am = ?
                WHERE id = ?
                """,
                (now, now, product_id),
            )
        self.conn.commit()
        return self.get_product(product_id)

    def import_from_image_folder(
        self,
        folder: str | Path,
        prefix_hint: str = "",
        suffix_hint: str = "",
        bezeichner_hint: str = "",
    ) -> tuple[int, int, int]:
        folder = Path(folder)
        files = collect_image_files(folder)
        added = 0
        updated = 0
        skipped = 0
        seen: set[str] = set()
        folder_text = str(folder)
        for path in files:
            stem = path.stem.strip()
            if not stem or stem in seen:
                skipped += 1
                continue
            seen.add(stem)
            existing = self.get_product_by_full_name(stem)
            if existing is not None:
                self.mark_foto_vorhanden(existing.id, folder_text)
                updated += 1
                logger.info(f"Import: Foto vorhanden für {stem}")
                continue
            prefix, variable, suffix = split_imported_stem(stem, prefix_hint, suffix_hint)
            family = self.get_bezeichner_for_suffix(suffix) if suffix else ""
            if not family and suffix:
                try:
                    family = normalize_bezeichner(bezeichner_hint)
                except ValueError:
                    family = ""
            self.add_product(
                prefix,
                variable,
                suffix,
                family,
                True,
                True,
                folder_text,
                "",
            )
            added += 1
            logger.info(f"Import: angelegt {stem}")
        return added, updated, skipped

    def _normalize_bezeichner_by_suffix(self) -> None:
        suffixes = self.conn.execute("SELECT DISTINCT suffix FROM products").fetchall()
        for row in suffixes:
            known = self.get_bezeichner_for_suffix(row["suffix"])
            if not known:
                continue
            self.conn.execute(
                "UPDATE products SET bezeichner = ? WHERE suffix = ? AND bezeichner != ?",
                (known, row["suffix"], known),
            )
        self.conn.commit()

    def list_suffix_groups(self) -> list[tuple[str, str]]:
        grouped: dict[str, str] = {}
        for product in self.list_products():
            current = grouped.get(product.suffix, "")
            if product.bezeichner and not current:
                grouped[product.suffix] = product.bezeichner
            elif product.suffix not in grouped:
                grouped[product.suffix] = product.bezeichner
        items = list(grouped.items())
        items.sort(key=lambda item: (natural_key(item[1]), natural_key(item[0])))
        return items


class ProduktfotoTrackerApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Produktfoto-Tracker")
        self.geometry("1360x800")
        self.minsize(1040, 640)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.db = ProductDB(DB_PATH)
        self.selected_id: int | None = None
        self.series_locked = False
        self._syncing_form = False
        self._tree_by_id: dict[str, str] = {}
        self._series_lookup: dict[str, str | None] = {FILTER_ALL: None}

        self.prefix_var = ctk.StringVar(value=self.db.get_setting("last_prefix"))
        self.variable_var = ctk.StringVar()
        self.suffix_var = ctk.StringVar(value=self.db.get_setting("last_suffix"))
        self.bezeichner_var = ctk.StringVar(value=self.db.get_setting("last_bezeichner"))
        self.bezeichner_count_var = ctk.StringVar()
        self.preview_var = ctk.StringVar(value="—")
        self.vorhanden_var = ctk.BooleanVar(value=True)
        self.foto_var = ctk.BooleanVar(value=False)
        self.original_var = ctk.StringVar(value=self.db.get_setting("last_pfad_original"))
        self.bearbeitet_var = ctk.StringVar(value=self.db.get_setting("last_pfad_bearbeitet"))
        self.filter_var = ctk.StringVar(value=FILTER_ALL)
        self.series_var = ctk.StringVar(value=FILTER_ALL)
        self.search_var = ctk.StringVar()
        self.stats_var = ctk.StringVar()
        self.status_var = ctk.StringVar(value="Bereit.")

        self._build_ui()
        self._bind_traces()
        if (
            self.prefix_var.get().strip()
            or self.suffix_var.get().strip()
            or self.bezeichner_var.get().strip()
        ):
            self._set_series_locked(True)
        self._refresh_series_menu()
        self._refresh_preview()
        self._refresh_table()
        self.after(80, lambda: self.variable_entry.focus_set())

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        logger.info(f"Datenbank: {DB_PATH}")

    def _build_ui(self) -> None:
        outer = ctk.CTkFrame(self, corner_radius=14)
        outer.pack(fill="both", expand=True, padx=18, pady=18)
        outer.grid_columnconfigure(0, weight=0)
        outer.grid_columnconfigure(1, weight=1)
        outer.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            outer,
            text="Produktfotos: Namen, Status, Ablageorte",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 4))

        ctk.CTkLabel(
            outer,
            textvariable=self.stats_var,
            anchor="w",
        ).grid(row=1, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 8))

        form = ctk.CTkFrame(outer)
        form.grid(row=2, column=0, sticky="nsw", padx=(18, 10), pady=(0, 8))
        form.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            form,
            text="Namenseingabe",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(14, 6))

        ctk.CTkLabel(form, text="Präfix").grid(row=1, column=0, sticky="w", padx=(14, 8), pady=6)
        self.prefix_entry = ctk.CTkEntry(form, textvariable=self.prefix_var, width=260)
        self.prefix_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 14), pady=6)

        ctk.CTkLabel(form, text="Variabler Teil").grid(row=2, column=0, sticky="w", padx=(14, 8), pady=6)
        self.variable_entry = ctk.CTkEntry(form, textvariable=self.variable_var, width=260)
        self.variable_entry.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(0, 14), pady=6)
        self.variable_entry.bind("<Return>", lambda _e: self._save())

        ctk.CTkLabel(form, text="Suffix").grid(row=3, column=0, sticky="w", padx=(14, 8), pady=6)
        self.suffix_entry = ctk.CTkEntry(form, textvariable=self.suffix_var, width=260)
        self.suffix_entry.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(0, 14), pady=6)

        ctk.CTkLabel(form, text="Bezeichner").grid(row=4, column=0, sticky="w", padx=(14, 8), pady=6)
        self.bezeichner_entry = ctk.CTkEntry(form, textvariable=self.bezeichner_var, width=260)
        self.bezeichner_entry.grid(row=4, column=1, sticky="ew", padx=(0, 8), pady=6)
        self.bezeichner_entry.bind("<Return>", lambda _e: self._save())
        ctk.CTkLabel(form, textvariable=self.bezeichner_count_var, width=50).grid(
            row=4, column=2, sticky="e", padx=(0, 14)
        )

        self.lock_hint = ctk.CTkLabel(
            form,
            text="Beim ersten Produkt Präfix, Suffix und Bezeichner setzen. Der Bezeichner gilt für alle gleichen Suffixe.",
            justify="left",
            wraplength=360,
        )
        self.lock_hint.grid(row=5, column=0, columnspan=3, sticky="w", padx=14, pady=(2, 4))

        ctk.CTkLabel(form, text="Vorschau").grid(row=6, column=0, sticky="w", padx=(14, 8), pady=6)
        ctk.CTkLabel(
            form,
            textvariable=self.preview_var,
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        ).grid(row=6, column=1, columnspan=2, sticky="ew", padx=(0, 14), pady=6)

        ctk.CTkLabel(
            form,
            text="Status",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=7, column=0, columnspan=3, sticky="w", padx=14, pady=(12, 6))

        checks = ctk.CTkFrame(form, fg_color="transparent")
        checks.grid(row=8, column=0, columnspan=3, sticky="w", padx=14, pady=(0, 6))
        ctk.CTkCheckBox(checks, text="Produkt vorhanden", variable=self.vorhanden_var).pack(
            side="left", padx=(0, 16)
        )
        ctk.CTkCheckBox(checks, text="Foto erledigt", variable=self.foto_var).pack(side="left")

        ctk.CTkLabel(
            form,
            text="Ablageorte (Google Drive)",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=9, column=0, columnspan=3, sticky="w", padx=14, pady=(12, 6))

        ctk.CTkLabel(form, text="Original").grid(row=10, column=0, sticky="w", padx=(14, 8), pady=6)
        ctk.CTkEntry(form, textvariable=self.original_var).grid(
            row=10, column=1, sticky="ew", padx=(0, 8), pady=6
        )
        orig_btns = ctk.CTkFrame(form, fg_color="transparent")
        orig_btns.grid(row=10, column=2, sticky="e", padx=(0, 14))
        ctk.CTkButton(orig_btns, text="Ordner", width=70, command=lambda: self._choose_folder(self.original_var)).pack(
            side="left", padx=(0, 4)
        )
        ctk.CTkButton(orig_btns, text="Öffnen", width=70, command=lambda: self._open_location(self.original_var.get())).pack(
            side="left"
        )

        ctk.CTkLabel(form, text="Bearbeitet").grid(row=11, column=0, sticky="w", padx=(14, 8), pady=6)
        ctk.CTkEntry(form, textvariable=self.bearbeitet_var).grid(
            row=11, column=1, sticky="ew", padx=(0, 8), pady=6
        )
        edit_btns = ctk.CTkFrame(form, fg_color="transparent")
        edit_btns.grid(row=11, column=2, sticky="e", padx=(0, 14))
        ctk.CTkButton(
            edit_btns, text="Ordner", width=70, command=lambda: self._choose_folder(self.bearbeitet_var)
        ).pack(side="left", padx=(0, 4))
        ctk.CTkButton(
            edit_btns, text="Öffnen", width=70, command=lambda: self._open_location(self.bearbeitet_var.get())
        ).pack(side="left")

        hint = (
            "Lokalen Drive-Ordner wählen oder Drive-Link einfügen.\n"
            "Enter im variablen Teil speichert und bereitet das nächste Produkt vor."
        )
        ctk.CTkLabel(form, text=hint, justify="left").grid(
            row=12, column=0, columnspan=3, sticky="w", padx=14, pady=(8, 8)
        )

        buttons = ctk.CTkFrame(form, fg_color="transparent")
        buttons.grid(row=13, column=0, columnspan=3, sticky="ew", padx=14, pady=(4, 4))
        buttons.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(buttons, text="Speichern", height=36, command=self._save).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ctk.CTkButton(buttons, text="Neu", height=36, command=self._prepare_next).grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )
        ctk.CTkButton(buttons, text="Neue Serie", height=36, command=self._new_series).grid(
            row=1, column=0, sticky="ew", padx=(0, 6), pady=(8, 0)
        )
        ctk.CTkButton(
            buttons,
            text="Löschen",
            height=36,
            fg_color="#8b2e2e",
            hover_color="#6f2424",
            command=self._delete_selected,
        ).grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(8, 0))
        ctk.CTkButton(
            buttons,
            text="Bilderordner einlesen",
            height=36,
            command=self._import_image_folder,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        ctk.CTkLabel(form, textvariable=self.status_var, anchor="w", wraplength=380).grid(
            row=14, column=0, columnspan=3, sticky="ew", padx=14, pady=(10, 16)
        )

        right = ctk.CTkFrame(outer)
        right.grid(row=2, column=1, sticky="nsew", padx=(0, 18), pady=(0, 8))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            right,
            text="Übersicht",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 6))

        toolbar = ctk.CTkFrame(right, fg_color="transparent")
        toolbar.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        toolbar.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(toolbar, text="Bezeichner").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.series_menu = ctk.CTkOptionMenu(
            toolbar,
            values=[FILTER_ALL],
            variable=self.series_var,
            command=lambda _v: self._refresh_table(),
            width=180,
        )
        self.series_menu.grid(row=0, column=1, sticky="w", padx=(0, 12))

        self.filter_bar = ctk.CTkSegmentedButton(
            toolbar,
            values=[FILTER_ALL, FILTER_TODO, FILTER_DONE, FILTER_MISSING],
            variable=self.filter_var,
            command=lambda _v: self._refresh_table(),
        )
        self.filter_bar.grid(row=0, column=2, sticky="ew", padx=(0, 12))

        ctk.CTkEntry(
            toolbar,
            textvariable=self.search_var,
            placeholder_text="Suche…",
            width=160,
        ).grid(row=0, column=3, sticky="e")

        tree_holder = tk.Frame(right, bg="#2b2b2b", highlightthickness=0)
        tree_holder.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 8))
        tree_holder.grid_columnconfigure(0, weight=1)
        tree_holder.grid_rowconfigure(0, weight=1)

        columns = ("suffix", "variable", "vorhanden", "foto", "geaendert", "original", "bearbeitet")
        self.tree = ttk.Treeview(
            tree_holder,
            columns=columns,
            show="tree headings",
            selectmode="browse",
        )
        self.tree.heading("#0", text="Bezeichner / Produkt")
        self.tree.column("#0", width=220, minwidth=140, stretch=True)
        headings = {
            "suffix": "Suffix",
            "variable": "Variabel",
            "vorhanden": "Vorhanden",
            "foto": "Foto",
            "geaendert": "Geändert",
            "original": "Original (Drive)",
            "bearbeitet": "Bearbeitet (Drive)",
        }
        widths = {
            "suffix": 90,
            "variable": 80,
            "vorhanden": 80,
            "foto": 70,
            "geaendert": 120,
            "original": 140,
            "bearbeitet": 140,
        }
        for key, label in headings.items():
            self.tree.heading(key, text=label)
            stretch = key in {"original", "bearbeitet", "geaendert"}
            self.tree.column(key, width=widths[key], minwidth=60, stretch=stretch)

        scrollbar = ttk.Scrollbar(tree_holder, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self._style_tree()
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Button-1>", self._on_tree_click)
        self.tree.bind("<Delete>", lambda _e: self._delete_selected())

        ctk.CTkLabel(
            right,
            text="Gruppiert nach Suffix: gleicher Suffix = gleicher Bezeichner. Klick auf ☑ / ☐ ändert den Status.",
            anchor="w",
        ).grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 14))

    def _style_tree(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#1f1f1f",
            foreground="#f2f2f2",
            fieldbackground="#1f1f1f",
            rowheight=28,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Treeview.Heading",
            background="#2f2f2f",
            foreground="#ffffff",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Treeview", background=[("selected", "#1f6aa5")], foreground=[("selected", "#ffffff")])
        style.map("Treeview.Heading", background=[("active", "#3a3a3a")])
        self.tree.tag_configure("todo", background="#3a2f1c")
        self.tree.tag_configure("done", background="#1c3324")
        self.tree.tag_configure("missing", background="#3a1f1f")
        self.tree.tag_configure("group", background="#2a3344")

    def _bind_traces(self) -> None:
        for var in (self.prefix_var, self.variable_var, self.suffix_var):
            var.trace_add("write", lambda *_: self._refresh_preview())
        self.suffix_var.trace_add("write", lambda *_: self._lookup_bezeichner_for_suffix())
        self.bezeichner_var.trace_add("write", lambda *_: self._limit_bezeichner())
        self.search_var.trace_add("write", lambda *_: self._refresh_table())
        self._limit_bezeichner()

    def _limit_bezeichner(self) -> None:
        value = self.bezeichner_var.get()
        if len(value) > BEZEICHNER_MAX:
            self.bezeichner_var.set(value[:BEZEICHNER_MAX])
            return
        self.bezeichner_count_var.set(f"{len(value)}/{BEZEICHNER_MAX}")

    def _lookup_bezeichner_for_suffix(self) -> None:
        if self._syncing_form:
            return
        known = self.db.get_bezeichner_for_suffix(self.suffix_var.get())
        if known:
            self.bezeichner_var.set(known)

    def _set_series_locked(self, locked: bool) -> None:
        self.series_locked = locked
        state = "disabled" if locked and self.selected_id is None else "normal"
        self.prefix_entry.configure(state=state)
        self.suffix_entry.configure(state=state)
        self.bezeichner_entry.configure(state="normal")
        if locked and self.selected_id is None:
            self.lock_hint.configure(
                text="Präfix und Suffix sind gesperrt. Bezeichner kann für dieses Suffix nachgetragen oder geändert werden."
            )
        elif self.selected_id is not None:
            self.lock_hint.configure(text="Ausgewähltes Produkt bearbeiten. Speichern übernimmt die Änderungen.")
        else:
            self.lock_hint.configure(
                text="Beim ersten Produkt Präfix, Suffix und Bezeichner setzen. Der Bezeichner gilt für alle gleichen Suffixe."
            )

    def _refresh_preview(self) -> None:
        name = build_full_name(self.prefix_var.get(), self.variable_var.get(), self.suffix_var.get())
        self.preview_var.set(name or "—")

    def _choose_folder(self, target: ctk.StringVar) -> None:
        initial = target.get().strip()
        if initial and Path(initial).exists():
            start = initial
        else:
            start = str(PROJECT_ROOT)
        path = filedialog.askdirectory(title="Google-Drive-Ordner wählen", initialdir=start)
        if path:
            target.set(path)

    def _import_image_folder(self) -> None:
        initial = self.db.get_setting("last_import_dir") or self.original_var.get().strip() or str(PROJECT_ROOT)
        folder = filedialog.askdirectory(title="Bilderordner wählen", initialdir=initial)
        if not folder:
            return
        files = collect_image_files(folder)
        if not files:
            messagebox.showinfo("Bilderordner", "Keine Bilddateien in diesem Ordner gefunden.")
            return
        if not messagebox.askyesno(
            "Bilderordner einlesen",
            f"{len(files)} Bild(er) gefunden.\n\n"
            "Dateiname ohne Endung wird als Produktname eingetragen.\n"
            "Feld „Foto vorhanden“ wird gesetzt.\n\nFortfahren?",
        ):
            return
        try:
            added, updated, skipped = self.db.import_from_image_folder(
                folder,
                self.prefix_var.get(),
                self.suffix_var.get(),
                self.bezeichner_var.get(),
            )
        except Exception as e:
            logger.exception(f"Import fehlgeschlagen: {e}")
            messagebox.showerror("Import", str(e))
            return
        self.db.set_setting("last_import_dir", folder)
        if not self.original_var.get().strip():
            self.original_var.set(folder)
        self._refresh_series_menu()
        self._refresh_table()
        summary = f"{added} neu, {updated} auf Foto vorhanden gesetzt"
        if skipped:
            summary += f", {skipped} übersprungen"
        self.status_var.set(f"Import {folder}: {summary}.")
        logger.info(f"Import aus {folder}: {summary}")
        messagebox.showinfo("Import fertig", summary)

    def _open_location(self, location: str) -> None:
        location = location.strip()
        if not location:
            messagebox.showinfo("Ablageort", "Kein Ablageort hinterlegt.")
            return
        if location.startswith(("http://", "https://")):
            webbrowser.open(location)
            return
        path = Path(location)
        if path.exists():
            os.startfile(path)  # type: ignore[attr-defined]
            return
        messagebox.showwarning("Ablageort", f"Pfad nicht gefunden:\n{location}")

    def _read_form(self) -> tuple[str, str, str, str]:
        prefix = self.prefix_var.get().strip()
        variable = self.variable_var.get().strip()
        suffix = self.suffix_var.get().strip()
        bezeichner = normalize_bezeichner(self.bezeichner_var.get())
        if not prefix:
            raise ValueError("Bitte ein Präfix eingeben.")
        if not variable:
            raise ValueError("Bitte den variablen Teil eingeben (z. B. 12g).")
        if not bezeichner:
            raise ValueError("Bitte einen Bezeichner für die Produktfamilie eingeben (max. 30 Zeichen).")
        if not build_full_name(prefix, variable, suffix):
            raise ValueError("Der Produktname darf nicht leer sein.")
        return prefix, variable, suffix, bezeichner

    def _persist_defaults(self, prefix: str, suffix: str, bezeichner: str) -> None:
        self.db.set_setting("last_prefix", prefix)
        self.db.set_setting("last_suffix", suffix)
        self.db.set_setting("last_bezeichner", bezeichner)
        self.db.set_setting("last_pfad_original", self.original_var.get().strip())
        self.db.set_setting("last_pfad_bearbeitet", self.bearbeitet_var.get().strip())

    def _save(self) -> None:
        suffix = self.suffix_var.get().strip()
        variable = self.variable_var.get().strip()
        try:
            bezeichner = normalize_bezeichner(self.bezeichner_var.get())
        except ValueError as e:
            messagebox.showerror("Eingabe", str(e))
            return

        if self.selected_id is None and not variable:
            if not bezeichner:
                messagebox.showerror("Eingabe", "Bitte einen Bezeichner eingeben (max. 30 Zeichen).")
                return
            if not self.db.suffix_exists(suffix):
                messagebox.showerror(
                    "Eingabe",
                    "Bitte den variablen Teil eingeben, um ein Produkt anzulegen.",
                )
                return
            self.db.apply_bezeichner_to_suffix(suffix, bezeichner)
            self._persist_defaults(self.prefix_var.get().strip(), suffix, bezeichner)
            self._refresh_series_menu()
            self._refresh_table()
            self.status_var.set(
                f"Bezeichner „{bezeichner}“ für Suffix {suffix or '(leer)'} übernommen."
            )
            logger.info(f"Bezeichner für Suffix {suffix!r}: {bezeichner}")
            return

        try:
            prefix, variable, suffix, bezeichner = self._read_form()
        except ValueError as e:
            messagebox.showerror("Eingabe", str(e))
            return

        try:
            if self.selected_id is None:
                product = self.db.add_product(
                    prefix,
                    variable,
                    suffix,
                    bezeichner,
                    self.vorhanden_var.get(),
                    self.foto_var.get(),
                    self.original_var.get(),
                    self.bearbeitet_var.get(),
                )
                logger.info(f"Angelegt: {product.full_name}")
                self.status_var.set(f"Angelegt: {product.full_name}")
                self._persist_defaults(prefix, suffix, bezeichner)
                self._refresh_series_menu()
                self._prepare_next(after_insert=True)
            else:
                product = self.db.update_product(
                    self.selected_id,
                    prefix,
                    variable,
                    suffix,
                    bezeichner,
                    self.vorhanden_var.get(),
                    self.foto_var.get(),
                    self.original_var.get(),
                    self.bearbeitet_var.get(),
                )
                logger.info(f"Aktualisiert: {product.full_name}")
                self.status_var.set(f"Aktualisiert: {product.full_name}")
                self._persist_defaults(prefix, suffix, bezeichner)
                self._refresh_series_menu()
        except ValueError as e:
            logger.warning(str(e))
            messagebox.showerror("Fehler", str(e))
            return

        self._refresh_table(select_id=self.selected_id)

    def _prepare_next(self, after_insert: bool = False) -> None:
        self.selected_id = None
        self.variable_var.set("")
        self.foto_var.set(False)
        self._set_series_locked(True)
        self.tree.selection_remove(self.tree.selection())
        self.variable_entry.focus_set()
        if after_insert:
            self.status_var.set(f"{self.status_var.get()} — nächster variabler Teil.")
        else:
            self.status_var.set("Neues Produkt: nur noch den variablen Teil eingeben.")
        self._refresh_preview()

    def _new_series(self) -> None:
        self.selected_id = None
        self._syncing_form = True
        self.prefix_var.set("")
        self.variable_var.set("")
        self.suffix_var.set("")
        self.bezeichner_var.set("")
        self.vorhanden_var.set(True)
        self.foto_var.set(False)
        self._syncing_form = False
        self.tree.selection_remove(self.tree.selection())
        self._set_series_locked(False)
        self.prefix_entry.focus_set()
        self.status_var.set("Neue Serie: Präfix, Suffix, Bezeichner und ersten variablen Teil eingeben.")
        self._refresh_preview()

    def _delete_selected(self) -> None:
        if self.selected_id is None:
            messagebox.showinfo("Löschen", "Bitte zuerst ein Produkt in der Übersicht auswählen.")
            return
        product = self.db.get_product(self.selected_id)
        if not messagebox.askyesno("Löschen", f"Produkt „{product.full_name}“ wirklich löschen?"):
            return
        self.db.delete_product(product.id)
        logger.info(f"Gelöscht: {product.full_name}")
        self.status_var.set(f"Gelöscht: {product.full_name}")
        self.selected_id = None
        self.variable_var.set("")
        self._refresh_series_menu()
        self._set_series_locked(
            bool(
                self.prefix_var.get().strip()
                or self.suffix_var.get().strip()
                or self.bezeichner_var.get().strip()
            )
        )
        self._refresh_table()

    def _refresh_series_menu(self) -> None:
        groups = self.db.list_suffix_groups()
        labels = [FILTER_ALL]
        self._series_lookup = {FILTER_ALL: None}
        for suffix, bezeichner in groups:
            label = suffix_group_label(suffix, bezeichner)
            key = label
            n = 2
            while key in self._series_lookup and self._series_lookup[key] != suffix:
                key = f"{label} ({n})"
                n += 1
            labels.append(key)
            self._series_lookup[key] = suffix
        current = self.series_var.get()
        self.series_menu.configure(values=labels)
        if current not in labels:
            self.series_var.set(FILTER_ALL)

    def _in_selected_series(self, product: Product) -> bool:
        if self.series_var.get() == FILTER_ALL:
            return True
        selected_suffix = self._series_lookup.get(self.series_var.get())
        if selected_suffix is None:
            return True
        return product.suffix == selected_suffix

    def _matches_status(self, product: Product) -> bool:
        mode = self.filter_var.get()
        if mode == FILTER_TODO:
            return not product.foto_erledigt
        if mode == FILTER_DONE:
            return product.foto_erledigt
        if mode == FILTER_MISSING:
            return not product.produkt_vorhanden
        return True

    def _row_tag(self, product: Product) -> str:
        if not product.produkt_vorhanden:
            return "missing"
        if product.foto_erledigt:
            return "done"
        return "todo"

    def _refresh_table(self, select_id: int | None = None) -> None:
        wanted = select_id if select_id is not None else self.selected_id
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._tree_by_id.clear()

        products = self.db.list_products()
        series_products = [p for p in products if self._in_selected_series(p)]
        grouped: dict[str, list[Product]] = defaultdict(list)
        for product in series_products:
            grouped[product.suffix].append(product)

        query = self.search_var.get().strip().casefold()
        shown = 0
        for suffix in sorted(grouped, key=natural_key):
            members = grouped[suffix]
            bezeichner = next((p.bezeichner for p in members if p.bezeichner), "")
            group_hit = bool(query) and (
                query in bezeichner.casefold() or query in suffix.casefold()
            )
            children: list[Product] = []
            for product in members:
                if query and not group_hit:
                    haystack = f"{product.full_name} {product.variable_part} {product.prefix}".casefold()
                    if query not in haystack:
                        continue
                if not self._matches_status(product):
                    continue
                children.append(product)
            if not children:
                continue

            total_n = len(members)
            vorh_n = sum(1 for p in members if p.produkt_vorhanden)
            done_n = sum(1 for p in members if p.foto_erledigt)
            gid = group_iid(suffix)
            latest = max((p.geaendert_am for p in members), default="")
            self.tree.insert(
                "",
                "end",
                iid=gid,
                text=bezeichner or "—",
                values=(
                    suffix or "—",
                    f"{total_n} Var.",
                    f"{vorh_n}/{total_n}",
                    f"{done_n}/{total_n}",
                    format_geaendert(latest),
                    "",
                    "",
                ),
                tags=("group",),
                open=True,
            )
            for product in children:
                item_id = self.tree.insert(
                    gid,
                    "end",
                    iid=str(product.id),
                    text=product.full_name,
                    values=(
                        "",
                        product.variable_part,
                        mark(product.produkt_vorhanden),
                        mark(product.foto_erledigt),
                        format_geaendert(product.geaendert_am),
                        shorten_path(product.pfad_original),
                        shorten_path(product.pfad_bearbeitet),
                    ),
                    tags=(self._row_tag(product),),
                )
                self._tree_by_id[str(product.id)] = item_id
                shown += 1

        if wanted is not None and str(wanted) in self._tree_by_id:
            self.tree.selection_set(str(wanted))
            self.tree.see(str(wanted))

        total = len(series_products)
        done = sum(1 for p in series_products if p.foto_erledigt)
        todo = sum(1 for p in series_products if not p.foto_erledigt)
        missing = sum(1 for p in series_products if not p.produkt_vorhanden)
        groups_n = len(grouped)
        self.stats_var.set(
            f"{done} von {total} fotografiert  ·  {todo} noch offen  ·  "
            f"{missing} Produkt(e) nicht vorhanden  ·  {groups_n} Suffix-Gruppe(n)  ·  angezeigt: {shown}"
        )

    def _on_tree_select(self, _event: tk.Event | None = None) -> None:
        if self._syncing_form:
            return
        selection = self.tree.selection()
        if not selection:
            return
        iid = selection[0]
        suffix = suffix_from_group_iid(iid)
        if suffix is not None:
            self._load_group(suffix)
            return
        if not is_product_iid(iid):
            return
        product = self.db.get_product(int(iid))
        self._load_product(product)

    def _load_group(self, suffix: str) -> None:
        members = [p for p in self.db.list_products() if p.suffix == suffix]
        bezeichner = self.db.get_bezeichner_for_suffix(suffix)
        prefixes = {p.prefix for p in members}
        self.selected_id = None
        self._syncing_form = True
        self.suffix_var.set(suffix)
        if bezeichner:
            self.bezeichner_var.set(bezeichner)
        if len(prefixes) == 1:
            self.prefix_var.set(next(iter(prefixes)))
        self.variable_var.set("")
        self.foto_var.set(False)
        self._syncing_form = False
        self._set_series_locked(True)
        if not bezeichner:
            self.bezeichner_entry.focus_set()
            self.status_var.set(
                f"Bezeichner für Suffix {suffix or '(leer)'} eintragen und Speichern."
            )
        else:
            self.variable_entry.focus_set()
            self.status_var.set(
                f"Gruppe „{bezeichner}“  ·  Suffix {suffix or '(leer)'} — nächsten variablen Teil eingeben."
            )
        self._refresh_preview()

    def _load_product(self, product: Product) -> None:
        self._syncing_form = True
        self.selected_id = product.id
        self.prefix_var.set(product.prefix)
        self.variable_var.set(product.variable_part)
        self.suffix_var.set(product.suffix)
        self.bezeichner_var.set(product.bezeichner)
        self.vorhanden_var.set(product.produkt_vorhanden)
        self.foto_var.set(product.foto_erledigt)
        self.original_var.set(product.pfad_original)
        self.bearbeitet_var.set(product.pfad_bearbeitet)
        self._syncing_form = False
        self._set_series_locked(False)
        self.status_var.set(
            f"Gewählt: {product.full_name}  ·  geändert {format_geaendert(product.geaendert_am)}"
        )
        self._refresh_preview()

    def _on_tree_click(self, event: tk.Event) -> str | None:
        if self.tree.identify_region(event.x, event.y) != "cell":
            return None
        row_id = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if not row_id or not is_product_iid(row_id):
            return None
        if column == "#3":
            self._toggle_flag(int(row_id), "produkt_vorhanden")
            return "break"
        if column == "#4":
            self._toggle_flag(int(row_id), "foto_erledigt")
            return "break"
        return None

    def _toggle_flag(self, product_id: int, field: str) -> None:
        product = self.db.get_product(product_id)
        new_value = not getattr(product, field)
        self.db.set_flag(product_id, field, new_value)
        logger.info(f"{product.full_name}: {field} = {new_value}")
        if self.selected_id == product_id:
            if field == "produkt_vorhanden":
                self.vorhanden_var.set(new_value)
            else:
                self.foto_var.set(new_value)
        self._refresh_table(select_id=self.selected_id)
        self.status_var.set(
            f"{product.full_name}: {'vorhanden' if field == 'produkt_vorhanden' else 'Foto'} "
            f"{'ja' if new_value else 'nein'}."
        )

    def _on_close(self) -> None:
        prefix = self.prefix_var.get().strip()
        suffix = self.suffix_var.get().strip()
        bezeichner = self.bezeichner_var.get().strip()
        if prefix:
            self._persist_defaults(prefix, suffix, bezeichner[:BEZEICHNER_MAX])
        else:
            self.db.set_setting("last_pfad_original", self.original_var.get().strip())
            self.db.set_setting("last_pfad_bearbeitet", self.bearbeitet_var.get().strip())
        self.db.close()
        logger.info("ProduktfotoTrackerApp beendet")
        self.destroy()


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Starte ProduktfotoTrackerApp")
    logger.info("=" * 70)
    app = ProduktfotoTrackerApp()
    app.mainloop()
