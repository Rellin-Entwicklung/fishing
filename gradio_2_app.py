
"""
Gradio UI for complete_pipeline.py

Run:
    pip install gradio rembg opencv-python pillow loguru numpy
    python gradio_app.py

Notes:
- Browser UIs can't always open a native "select folder" dialog on all setups.
  This UI uses text inputs for folder paths + a file explorer helper.
"""
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Optional

import gradio as gr

# Import your pipeline
# The file uploaded by the user is expected to sit next to this app, or on PYTHONPATH.
# If it's in the same folder, this import works:
try:
    from complete_pipeline import process_product_image  # type: ignore
except Exception as e:
    # Friendly message if import fails
    raise RuntimeError(
        "Konnte complete_pipeline.py nicht importieren. "
        "Lege gradio_app.py in den gleichen Ordner wie complete_pipeline.py "
        "oder stelle sicher, dass complete_pipeline.py im PYTHONPATH liegt.\n"
        f"Originalfehler: {e}"
    )

from loguru import logger

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _ensure_dir(p: str) -> Path:
    path = Path(p).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _list_images(input_dir: str) -> List[Path]:
    in_dir = Path(input_dir).expanduser().resolve()
    if not in_dir.exists():
        return []
    imgs = []
    for ext in IMAGE_EXTS:
        imgs.extend(in_dir.glob(f"*{ext}"))
        imgs.extend(in_dir.glob(f"*{ext.upper()}"))
    # stable ordering
    imgs = sorted(set(imgs), key=lambda p: p.name.lower())
    return imgs


class _LogCapture:
    """Capture loguru logs into an in-memory buffer (string)."""

    def __init__(self) -> None:
        self._lines: List[str] = []
        self._handler_id: Optional[int] = None

    def start(self) -> None:
        # Remove any previous handler we added
        self.stop()

        # Keep logs compact for UI
        fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"

        def sink(msg):
            # msg is a loguru "Message" object; str(msg) includes formatting/newline
            self._lines.append(str(msg).rstrip("\n"))

        self._handler_id = logger.add(sink, format=fmt, level="DEBUG", enqueue=False)

    def stop(self) -> None:
        if self._handler_id is not None:
            try:
                logger.remove(self._handler_id)
            except Exception:
                pass
            self._handler_id = None

    def text(self, tail: int = 4000) -> str:
        # Limit size for UI responsiveness
        t = "\n".join(self._lines)
        if len(t) > tail:
            return "…(gekürzt)…\n" + t[-tail:]
        return t


def _safe_output_name(src: Path) -> str:
    # Keep base name but strip weird chars
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", src.stem)
    return f"{stem}.webp"


def process_batch(
    input_dir: str,
    output_dir: str,
    overwrite: bool,
    show_live_previews: bool,
    limit: int,
    update_every: int,
    preview_delay_ms: int,
    progress=gr.Progress(track_tqdm=False),
):
    """
    Generator for streaming UI updates.
    Outputs: status_md, log_text, current_in_image, current_out_image, gallery
    """
    # Sanitize UI params
    try:
        update_every = int(update_every)
    except Exception:
        update_every = 1
    update_every = max(1, update_every)

    try:
        preview_delay_ms = int(preview_delay_ms)
    except Exception:
        preview_delay_ms = 0
    preview_delay_ms = max(0, preview_delay_ms)

    logcap = _LogCapture()
    logcap.start()

    try:
        # Validate dirs
        in_dir = Path(input_dir).expanduser().resolve()
        out_dir = _ensure_dir(output_dir)

        if not in_dir.exists() or not in_dir.is_dir():
            yield (
                f"❌ Eingabeordner nicht gefunden: `{in_dir}`",
                logcap.text(),
                None,
                None,
                [],
            )
            return

        images = _list_images(str(in_dir))
        if limit and limit > 0:
            images = images[:limit]

        if not images:
            yield (
                f"⚠️ Keine Bilder im Eingabeordner gefunden: `{in_dir}`",
                logcap.text(),
                None,
                None,
                [],
            )
            return

        gallery_items: List[Tuple[str, str]] = []
        total = len(images)

        # Keep last previews so we never "blink" to None between updates
        last_in: Optional[str] = None
        last_out: Optional[str] = None

        def _maybe_sleep():
            if show_live_previews and preview_delay_ms > 0:
                time.sleep(preview_delay_ms / 1000.0)

        start_time = time.time()
        for idx, src in enumerate(images, start=1):
            progress((idx - 1) / total, desc=f"Verarbeite {idx}/{total}: {src.name}")

            dst = out_dir / _safe_output_name(src)
            if dst.exists() and not overwrite:
                msg = f"⏭️ Übersprungen (existiert): `{dst.name}` ({idx}/{total})"
                last_in = str(src)
                last_out = str(dst)
                # Always show skips (force)
                yield (
                    msg,
                    logcap.text(),
                    (last_in if show_live_previews else None),
                    (last_out if show_live_previews else None),
                    gallery_items,
                )
                _maybe_sleep()
                continue

            # Live preview: show current input image before processing
            last_in = str(src)
            last_out = last_out  # keep previous output preview until new is ready
            # Show start only if throttling allows (force if update_every==1)
            if update_every == 1:
                yield (
                    f"▶️ Starte: `{src.name}` ({idx}/{total})",
                    logcap.text(),
                    (last_in if show_live_previews else None),
                    (last_out if show_live_previews else None),
                    gallery_items,
                )

            try:
                process_product_image(str(src), str(dst))
                gallery_items.append((str(dst), dst.name))
                last_out = str(dst)
                # Always show a "Fertig" update (throttled unless end)
                if (idx % update_every == 0) or (idx == total) or (update_every == 1):
                    yield (
                        f"✅ Fertig: `{src.name}` → `{dst.name}` ({idx}/{total})",
                        logcap.text(),
                        (last_in if show_live_previews else None),
                        (last_out if show_live_previews else None),
                        gallery_items,
                    )
                    _maybe_sleep()
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Fehler bei {src.name}: {e}\n{tb}")
                # Keep last_out so preview doesn't blink
                yield (
                    f"❌ Fehler bei `{src.name}` (siehe Log). Fahre fort… ({idx}/{total})",
                    logcap.text(),
                    (last_in if show_live_previews else None),
                    (last_out if show_live_previews else None),
                    gallery_items,
                )
                _maybe_sleep()

        elapsed = time.time() - start_time
        progress(1.0, desc="Fertig")
        yield (
            f"🎉 Batch fertig. {len(gallery_items)}/{total} Dateien verarbeitet. Dauer: {elapsed:.1f}s",
            logcap.text(),
            (last_in if show_live_previews else None),
            (last_out if show_live_previews else None),
            gallery_items,
        )
    finally:
        logcap.stop()


def preview_single(
    image_file,
    output_dir: str,
    overwrite: bool,
):
    """Process a single uploaded file, return output image + path + logs."""
    logcap = _LogCapture()
    logcap.start()
    try:
        if image_file is None:
            return None, "Bitte ein Bild hochladen.", logcap.text()

        out_dir = _ensure_dir(output_dir)
        src_path = Path(image_file)
        dst_path = out_dir / _safe_output_name(src_path)

        if dst_path.exists() and not overwrite:
            return str(dst_path), f"Ausgabe existiert bereits (overwrite aus): {dst_path.name}", logcap.text()

        process_product_image(str(src_path), str(dst_path))
        return str(dst_path), f"Fertig: {dst_path.name}", logcap.text()
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Single-Preview Fehler: {e}\n{tb}")
        return None, f"Fehler: {e}", logcap.text()
    finally:
        logcap.stop()


def build_ui():
    with gr.Blocks(title="Produktbild-Prozessor (rembg) – Gradio UI") as demo:
        gr.Markdown(
            """
# Produktbild-Prozessor – Gradio Oberfläche
- **Batch**: Eingabeordner → Ausgabeordner (WebP)
- **Vorschau**: Einzelbild hochladen, Ergebnis ansehen
            """.strip()
        )

        with gr.Tabs():
            with gr.Tab("Batch-Verarbeitung (Ordner)"):
                with gr.Row():
                    input_dir = gr.Textbox(
                        label="Eingabeordner (Bilder)",
                        placeholder="z.B. /home/user/input_images",
                        value=str(Path.cwd()),
                    )
                    output_dir = gr.Textbox(
                        label="Ausgabeordner (WebP)",
                        placeholder="z.B. /home/user/output_images",
                        value=str((Path.cwd() / "output").resolve()),
                    )

                with gr.Accordion("Hilfe: Ordner auswählen / finden", open=False):
                    gr.Markdown(
                        """
Je nach Browser/OS gibt es keinen echten Ordner-Dialog. Du kannst hier:
- **Pfad kopieren** (Explorer/Finder) und in die Textfelder einfügen, oder
- mit dem File Explorer unten navigieren und den Ordnerpfad übernehmen.
                        """.strip()
                    )
                    with gr.Row():
                        explorer_root = gr.Textbox(
                            label="Explorer root_dir",
                            value=str(Path.cwd()),
                        )
                    file_explorer = gr.FileExplorer(
                        label="Datei/Ordner Explorer",
                        root_dir=str(Path.cwd()),
                        glob="**/*",
                        file_count="single",
                        height=260,
                    )

                    def _set_root(root):
                        return gr.FileExplorer(root_dir=root)

                    explorer_root.change(_set_root, explorer_root, file_explorer)

                    def _use_selected_path(sel):
                        # sel can be a string path
                        if not sel:
                            return gr.update()
                        p = Path(sel)
                        if p.is_dir():
                            return str(p)
                        return str(p.parent)

                    with gr.Row():
                        btn_use_as_input = gr.Button("Auswahl → Eingabeordner")
                        btn_use_as_output = gr.Button("Auswahl → Ausgabeordner")

                    btn_use_as_input.click(_use_selected_path, file_explorer, input_dir)
                    btn_use_as_output.click(_use_selected_path, file_explorer, output_dir)

                with gr.Row():
                    overwrite = gr.Checkbox(label="Vorhandene Ausgaben überschreiben", value=False)
                    show_previews = gr.Checkbox(label="Live-Vorschau anzeigen (langsamer)", value=True)
                    limit = gr.Number(label="Max. Bilder (0 = alle)", value=0, precision=0)

                with gr.Row():
                    update_every = gr.Slider(
                        label="UI-Update alle N Bilder (höher = weniger Flackern, schneller)",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=1,
                    )
                    preview_delay_ms = gr.Slider(
                        label="Vorschau-Anzeigedauer (ms) pro Update",
                        minimum=0,
                        maximum=2000,
                        step=50,
                        value=200,
                    )

                run_btn = gr.Button("▶️ Batch starten", variant="primary")

                status = gr.Markdown("Status: bereit.")
                logs = gr.Textbox(label="Live-Log (Loguru)", lines=16, max_lines=30)
                with gr.Row():
                    current_in = gr.Image(label="Aktuelles Eingabebild", type="filepath", height=320)
                    current_out = gr.Image(label="Aktuelles Ausgabebild", type="filepath", height=320)

                gallery = gr.Gallery(
                    label="Fertige Ausgaben",
                    columns=4,
                    height=420,
                    show_label=True,
                    preview=True,
                )

                run_btn.click(
                    fn=process_batch,
                    inputs=[input_dir, output_dir, overwrite, show_previews, limit, update_every, preview_delay_ms],
                    outputs=[status, logs, current_in, current_out, gallery],
                )

            with gr.Tab("Einzelbild-Vorschau"):
                with gr.Row():
                    single_in = gr.File(label="Bild hochladen (JPG/PNG/WebP etc.)", file_types=list(IMAGE_EXTS))
                    single_outdir = gr.Textbox(
                        label="Ausgabeordner",
                        value=str((Path.cwd() / "output").resolve()),
                    )
                with gr.Row():
                    single_overwrite = gr.Checkbox(label="Überschreiben", value=True)
                    single_btn = gr.Button("✨ Verarbeiten", variant="primary")

                single_status = gr.Textbox(label="Status", lines=2)
                single_logs = gr.Textbox(label="Log", lines=16, max_lines=30)
                single_out_img = gr.Image(label="Ergebnis (WebP)", type="filepath", height=420)

                single_btn.click(
                    fn=preview_single,
                    inputs=[single_in, single_outdir, single_overwrite],
                    outputs=[single_out_img, single_status, single_logs],
                )

        gr.Markdown(
            """
**Tipps**
- Wenn du im Batch-Modus viele Bilder hast, kannst du **Live-Vorschau** deaktivieren (schneller).
- Outputs werden als **.webp** gespeichert (Qualität 95, 1024×1024, weißer Hintergrund).
            """.strip()
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # server_name="0.0.0.0" makes it reachable in a network/container;
    # change to "127.0.0.1" if you want only local access.
    demo.launch(server_name="127.0.0.1", server_port=7860)
