"""Images and PDFs in the conversation (plan/2026-11-08-multimodal-input/spec.md).

A file the user attaches is copied into `.dostuff/attachments/<sha256[:16]>.<ext>` in the project, and the user step
holds a *ref* to it (a `dostuff_attachment` part), never the bytes (CON-004). `hydrate` turns refs into provider
content parts when a request is built; the same history always hydrates to the same bytes (CON-005).

Kept free of litellm and Pillow: file mentions import this on the TUI's first frame.
"""
import base64
import hashlib
import math
import re
import struct
import time
import zipfile
from pathlib import Path
from typing import Any, Literal

from dostuff.helpers.tools.project_root import project_root

Kind = Literal["image", "pdf", "docx", "xlsx", "pptx"]
ATTACHMENT_DIR = Path(".dostuff") / "attachments"
REF_TYPE = "dostuff_attachment"

IMAGE_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp",
              "application/pdf": "pdf"}
OFFICE_MEMBERS = {"word/document.xml": "docx", "xl/workbook.xml": "xlsx", "ppt/presentation.xml": "pptx"}

# REQ-6.3 / REQ-6.4: Anthropic's caps (5 MB of base64 per image), the strictest of the providers LiteLLM reaches
MAX_IMAGE_BYTES = 5 * 1024 * 1024 * 3 // 4
MAX_PER_STEP = 20

# D9: Anthropic's documented image formula, capped; close enough for other providers
IMAGE_TOKENS_MAX = 1600
PDF_TOKENS_PER_PAGE = 2000


def _image_format(head: bytes) -> str | None:
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


def image_media_type(data: bytes) -> str | None:
    fmt = _image_format(data[:16])
    return IMAGE_TYPES[fmt] if fmt else None


def classify_attachment(path: Path) -> Kind | None:
    """What a file is by its first bytes, not its extension (D3); None for anything else (text, other binaries)."""
    try:
        with path.open("rb") as f:
            head = f.read(16)
    except OSError:
        return None
    if _image_format(head):
        return "image"
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
        except (OSError, zipfile.BadZipFile):
            return None
        for member, kind in OFFICE_MEMBERS.items():
            if member in names:
                return kind  # type: ignore[return-value]
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # no length
            i += 2
            continue
        (length,) = struct.unpack(">H", data[i + 2:i + 4])
        # SOF0..SOF15, except DHT (C4), JPG (C8) and DAC (CC)
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC) and i + 9 <= len(data):
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        i += 2 + length
    return None


def image_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) read from the header by hand, so chips and estimates work without Pillow (D5)."""
    fmt = _image_format(data[:16])
    try:
        if fmt == "png" and len(data) >= 24:
            return struct.unpack(">II", data[16:24])  # type: ignore[return-value]
        if fmt == "gif" and len(data) >= 10:
            return struct.unpack("<HH", data[6:10])  # type: ignore[return-value]
        if fmt == "jpeg":
            return _jpeg_size(data)
        if fmt == "webp" and len(data) >= 30:
            chunk = data[12:16]
            if chunk == b"VP8X":
                return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
            if chunk == b"VP8 ":
                w, h = struct.unpack("<HH", data[26:30])
                return w & 0x3FFF, h & 0x3FFF
            if chunk == b"VP8L":
                bits = int.from_bytes(data[21:25], "little")
                return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    except struct.error:
        return None
    return None


_PDF_PAGE_RE = re.compile(rb"/Type\s*/Page(?![a-zA-Z])")


def pdf_page_count(data: bytes) -> int:
    """Pages by counting page objects; rough (compressed object streams hide them), never below 1."""
    return max(len(_PDF_PAGE_RE.findall(data)), 1)


def _megabytes(n: int) -> str:
    return f"{n / (1024 * 1024):.2f} MB"


def _has_pillow() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


def image_problem(data: bytes) -> str | None:
    """Why this image can't be sent as it is (too large), or None (REQ-6.3)."""
    if len(data) <= MAX_IMAGE_BYTES:
        return None
    reason = f"{_megabytes(len(data))} is over the {_megabytes(MAX_IMAGE_BYTES)} image limit"
    return reason if _has_pillow() else reason + "; install dostuff[images] to resize"


class AttachmentStore:
    """Content-addressed copies of attached files under `<root>/.dostuff/attachments/` (D2)."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root or project_root()

    def add(self, source: Path | bytes, name: str = "") -> dict[str, Any]:
        """Copy an image or PDF into the store and return its ref. Raises ValueError for anything else."""
        data = source.read_bytes() if isinstance(source, Path) else source
        if not name:
            name = source.name if isinstance(source, Path) else "attachment"
        media_type = image_media_type(data)
        if media_type:
            kind = "image"
        elif data.startswith(b"%PDF-"):
            kind, media_type = "pdf", "application/pdf"
        else:
            raise ValueError(f"{name}: not an image or a PDF")
        digest = hashlib.sha256(data).hexdigest()[:16]
        folder = self.root / ATTACHMENT_DIR
        folder.mkdir(parents=True, exist_ok=True)
        ignore = folder / ".gitignore"
        if not ignore.exists():
            ignore.write_text("*\n", encoding="utf-8")  # keep attachments out of git without touching the repo's
        path = folder / f"{digest}.{EXTENSIONS[media_type]}"
        if not path.is_file():
            path.write_bytes(data)
        ref: dict[str, Any] = {"type": REF_TYPE, "id": digest, "kind": kind, "media_type": media_type, "name": name,
                               "path": (ATTACHMENT_DIR / path.name).as_posix()}
        if kind == "image":
            size = image_size(data)
            if size:
                ref["width"], ref["height"] = size
        else:
            ref["pages"] = pdf_page_count(data)
        return ref


def is_ref(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == REF_TYPE


def refs(content: Any) -> list[dict[str, Any]]:
    return [p for p in content if is_ref(p)] if isinstance(content, list) else []


def ref_label(ref: dict[str, Any]) -> str:
    return f"[{'PDF' if ref.get('kind') == 'pdf' else 'Image'}: {ref.get('name', 'attachment')}]"


def ref_chips(content: Any) -> str:
    """The attachments of a step as the transcript shows them (REQ-7.1):
    `[Image #1 · bug.png · 1280×720] [PDF · report.pdf · 12 pages]`; images numbered in order."""
    chips = []
    number = 0
    for ref in refs(content):
        name = ref.get("name", "attachment")
        if ref.get("kind") == "pdf":
            pages = int(ref.get("pages") or 0)
            chips.append(f"[PDF · {name}" + (f" · {pages} page{'s' if pages != 1 else ''}]" if pages else "]"))
            continue
        number += 1
        size = f" · {ref['width']}×{ref['height']}" if ref.get("width") and ref.get("height") else ""
        chips.append(f"[Image #{number} · {name}{size}]")
    return " ".join(chips)


def content_text(content: Any) -> str:
    """A step's content as text: text parts joined, refs as `[Image: name]` / `[PDF: name]` (REQ-1.4)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    pieces = []
    for part in content:
        if is_ref(part):
            pieces.append(ref_label(part))
        elif isinstance(part, dict) and part.get("type") == "text":
            pieces.append(str(part.get("text") or ""))
        elif isinstance(part, dict) and part.get("type") == "image_url":
            pieces.append("[Image]")
    return "\n".join(p for p in pieces if p)


def ref_tokens(ref: dict[str, Any]) -> int:
    """D9: what a ref costs in context, for estimates only (real usage comes from the provider)."""
    if ref.get("kind") == "pdf":
        return PDF_TOKENS_PER_PAGE * max(int(ref.get("pages") or 1), 1)
    width, height = ref.get("width"), ref.get("height")
    if not width or not height:
        return IMAGE_TOKENS_MAX
    return min(math.ceil(int(width) * int(height) / 750), IMAGE_TOKENS_MAX)


def _provider_part(ref: dict[str, Any], root: Path, missing: list[str] | None) -> dict[str, Any]:
    path = Path(ref.get("path") or "")
    if not path.is_absolute():
        path = root / path
    try:
        data = path.read_bytes()
    except OSError:
        if missing is not None:
            missing.append(str(ref.get("name", "attachment")))
        return {"type": "text", "text": f"[attachment unavailable: {ref.get('name', 'attachment')}]"}
    url = f"data:{ref.get('media_type')};base64," + base64.b64encode(data).decode("ascii")
    if ref.get("kind") == "pdf":
        return {"type": "file", "file": {"file_data": url}}
    return {"type": "image_url", "image_url": {"url": url}}


def hydrate(messages: list[dict[str, Any]], root: Path | None = None,
            missing: list[str] | None = None) -> list[dict[str, Any]]:
    """Messages with every ref swapped for a provider content part (D1). Messages without refs are returned as
    the same objects, so a turn without attachments sends exactly what it did before (CON-007). Names of refs
    whose file is gone are added to `missing`."""
    root = root or project_root()
    out = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list) or not refs(content):
            out.append(message)
            continue
        parts = [_provider_part(p, root, missing) if is_ref(p) else p for p in content]
        out.append({**message, "content": parts})
    return out


def sweep_old_attachments(retention_days: float, root: Path | None = None) -> int:
    """Delete stored attachments older than `retention_days` (0 or less: none); returns how many went (D2).

    A resumed session whose attachment was swept gets `[attachment unavailable: name]` (REQ-1.3)."""
    folder = (root or project_root()) / ATTACHMENT_DIR
    if retention_days <= 0 or not folder.is_dir():
        return 0
    cutoff = time.time() - retention_days * 86_400
    removed = 0
    for path in folder.iterdir():
        try:
            if path.name != ".gitignore" and path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def delete_attachments(steps: list[dict[str, Any]], root: Path | None = None) -> int:
    """Delete the stored files the refs in `steps` point at (`loop.attachment_retention_days: 0`, when the session
    ends); returns how many went."""
    folder = ((root or project_root()) / ATTACHMENT_DIR).resolve()
    removed = 0
    for path in {(folder / Path(str(r.get("path") or "")).name) for s in steps for r in refs(s.get("content"))}:
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed
