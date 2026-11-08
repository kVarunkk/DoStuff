"""Word, Excel, PowerPoint and PDF files as text the model can read (multimodal spec REQ-4.1, D7).

No model reads Office formats natively, so they are converted on our side and attached the way a text file is.
PDFs come here only for models that can't read them (or that are over the native limits). The libraries are the
optional `dostuff[docs]` extra, imported inside each converter: this module is cheap to import, and without the
extra a conversion raises `MissingExtra`, whose message names it.
"""
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterable

# A sheet is cut at this many rows (header included), with a note; a used range of a million rows would only fill
# the store and the context
MAX_SHEET_ROWS = 5000

_MODULES = {"docx": "docx", "xlsx": "openpyxl", "pptx": "pptx", "pdf": "pypdf"}


class MissingExtra(ImportError):
    """The `[docs]` extra isn't installed; the message says what to install."""


def _require(kind: str) -> Any:
    import importlib
    try:
        return importlib.import_module(_MODULES[kind])
    except ImportError:
        raise MissingExtra(f"install dostuff[docs] to read .{kind} files") from None


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def markdown_table(rows: Iterable[Iterable[Any]]) -> str:
    """Rows as a markdown table, the first row as its header; short rows are padded."""
    cells = [[_cell(v) for v in row] for row in rows]
    if not cells:
        return ""
    width = max(len(row) for row in cells)
    cells = [row + [""] * (width - len(row)) for row in cells]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(row) + " |" for row in cells[1:]]
    return "\n".join(lines)


def _heading_level(style_name: str) -> int:
    if style_name == "Title":
        return 1
    if style_name.startswith("Heading "):
        try:
            return max(1, min(int(style_name.split()[1]), 6))
        except (IndexError, ValueError):
            return 0
    return 0


def _docx(path: Path) -> str:
    docx = _require("docx")
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    document = docx.Document(str(path))
    out: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = paragraph.style.name if paragraph.style is not None else ""
            level = _heading_level(style or "")
            if level:
                out.append("#" * level + " " + text)
            elif (style or "").startswith("List"):
                out.append("- " + text)
            else:
                out.append(text)
        elif tag == "tbl":
            table = Table(child, document)
            out.append(markdown_table([cell.text for cell in row.cells] for row in table.rows))
    return "\n\n".join(out)


def _trimmed(rows: list[tuple[Any, ...]]) -> list[list[Any]]:
    """Rows without the empty ones at the end and without empty columns on the right."""
    while rows and all(v is None or v == "" for v in rows[-1]):
        rows.pop()
    width = 0
    for row in rows:
        for i in range(len(row) - 1, -1, -1):
            if row[i] is not None and row[i] != "":
                width = max(width, i + 1)
                break
    return [list(row[:width]) for row in rows]


def _xlsx(path: Path) -> str:
    openpyxl = _require("xlsx")
    # data_only: the value Excel last computed, never the formula (REQ-4.1)
    workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    out: list[str] = []
    try:
        for sheet in workbook.worksheets:
            rows: list[tuple[Any, ...]] = []
            more = 0
            for row in sheet.iter_rows(values_only=True):
                if len(rows) < MAX_SHEET_ROWS:  # the header row counts
                    rows.append(row)
                elif any(v is not None and v != "" for v in row):
                    more += 1
            table = _trimmed(rows)
            body = markdown_table(table) if table else "(empty sheet)"
            if more:
                body += f"\n[… {more} more rows]"
            out.append(f"## Sheet: {sheet.title}\n{body}")
    finally:
        workbook.close()
    return "\n\n".join(out)


def _shape_text(shape: Any) -> list[str]:
    if getattr(shape, "has_table", False) and shape.has_table:
        return [markdown_table([cell.text for cell in row.cells] for row in shape.table.rows)]
    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        text = shape.text_frame.text.strip()
        return [text] if text else []
    if hasattr(shape, "shapes"):  # a group
        return [t for child in shape.shapes for t in _shape_text(child)]
    return []


def _pptx(path: Path) -> str:
    pptx = _require("pptx")
    presentation = pptx.Presentation(str(path))
    out: list[str] = []
    for number, slide in enumerate(presentation.slides, start=1):
        lines = [f"--- Slide {number} ---"]
        title = slide.shapes.title
        if title is not None and title.text.strip():
            lines.append("# " + title.text.strip())
        for shape in slide.shapes:
            if title is not None and shape.shape_id == title.shape_id:
                continue
            lines += _shape_text(shape)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame
            if notes is not None and notes.text.strip():
                lines.append("Notes: " + notes.text.strip())
        out.append("\n".join(lines))
    return "\n\n".join(out)


def _pdf(path: Path) -> str:
    pypdf = _require("pdf")
    reader = pypdf.PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("the PDF is encrypted") from None
    out = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        out.append(f"--- Page {number} ---\n" + (text or "(no text on this page)"))
    return "\n\n".join(out)


_CONVERTERS = {"docx": _docx, "xlsx": _xlsx, "pptx": _pptx, "pdf": _pdf}


def convert_document(path: Path, kind: str) -> str:
    """The text of a docx / xlsx / pptx / pdf file. Raises `MissingExtra` without `dostuff[docs]`, and ValueError
    for a file the library can't read (broken, encrypted)."""
    converter = _CONVERTERS.get(kind)
    if converter is None:
        raise ValueError(f"can't convert {kind} files")
    try:
        return converter(path)
    except (MissingExtra, OSError):
        raise
    except Exception as e:  # each library has its own error types for a broken file
        raise ValueError(f"could not read the {kind} file: {e}") from e
