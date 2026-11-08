"""Images from the system clipboard (`alt+v`) and files dragged into the terminal (multimodal spec REQ-3, D6).

`alt+v` reads an image off the clipboard and puts `[Image #N]` in the input; on submit `expand_file_mentions` turns
each placeholder this process knows into a ref, like an `@image` mention. Pillow is tried first when it is
installed; otherwise one command per platform, whose stdout carries the image. Nothing here imports Pillow or
litellm at module level: the TUI imports this on its first frame.
"""
import base64
import binascii
import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

from dostuff.helpers.agent.attachments import image_media_type

CLIPBOARD_TIMEOUT_S = 5.0
NO_IMAGE = "No image on the clipboard"
PLACEHOLDER_RE = re.compile(r"\[Image #(\d+)\]")

# PowerShell has the clipboard in Windows Forms; -STA because the clipboard needs a single-threaded apartment
_WINDOWS_SCRIPT = (
    "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; "
    "$i = [System.Windows.Forms.Clipboard]::GetImage(); if ($i -eq $null) { exit 1 }; "
    "$m = New-Object System.IO.MemoryStream; $i.Save($m, [System.Drawing.Imaging.ImageFormat]::Png); "
    "[Console]::Out.Write([Convert]::ToBase64String($m.ToArray()))"
)
# osascript prints the data as «data PNGf89504E47…»; it fails when the clipboard holds no PNG
_MAC_SCRIPT = "the clipboard as «class PNGf»"
_MAC_DATA_RE = re.compile(r"«data PNGf([0-9A-Fa-f]+)»")


def _pillow_grab() -> bytes | None:
    """The clipboard image as PNG via Pillow (also the first image among copied files), or None — also when
    Pillow is missing or fails (on Linux it shells out to the same tools we try next)."""
    try:
        from PIL import ImageGrab
        grabbed = ImageGrab.grabclipboard()
    except Exception:
        return None
    if grabbed is None:
        return None
    if isinstance(grabbed, list):  # files copied in Explorer / Finder
        for name in grabbed:
            try:
                data = Path(name).read_bytes()
            except (OSError, TypeError):
                continue
            if image_media_type(data):
                return data
        return None
    buffer = io.BytesIO()
    try:
        grabbed.save(buffer, format="PNG")
    except Exception:
        return None
    return buffer.getvalue() or None


def _commands(platform: str) -> list[tuple[list[str], str]]:
    """(argv, how stdout is encoded) to try in order on `platform`."""
    if platform == "win32":
        return [(["powershell", "-NoProfile", "-NonInteractive", "-STA", "-Command", _WINDOWS_SCRIPT], "base64")]
    if platform == "darwin":
        return [(["osascript", "-e", _MAC_SCRIPT], "applescript")]
    return [(["wl-paste", "--no-newline", "--type", "image/png"], "raw"),
            (["xclip", "-selection", "clipboard", "-t", "image/png", "-o"], "raw")]


def _decode(out: bytes, encoding: str) -> bytes | None:
    try:
        if encoding == "base64":
            return base64.b64decode(out.strip(), validate=True)
        if encoding == "applescript":
            match = _MAC_DATA_RE.search(out.decode("utf-8", errors="replace"))
            return bytes.fromhex(match.group(1)) if match else None
    except (binascii.Error, ValueError):
        return None
    return out


def _find(name: str, platform: str) -> str | None:
    """`name` on PATH; on Windows, PowerShell where Windows always installs it (PATH can lack it, e.g. Git Bash),
    then PowerShell 7."""
    found = shutil.which(name)
    if found or platform != "win32" or name != "powershell":
        return found
    system = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0"
    if (system / "powershell.exe").is_file():
        return str(system / "powershell.exe")
    return shutil.which("pwsh")


def read_clipboard_image(timeout: float = CLIPBOARD_TIMEOUT_S, platform: str = sys.platform) -> tuple[bytes | None, str]:
    """(image bytes, "") or (None, why there is none). Blocks up to `timeout` per command: run it in a thread."""
    data = _pillow_grab()
    if data:
        return data, ""
    found_tool = False
    for argv, encoding in _commands(platform):
        executable = _find(argv[0], platform)
        if executable is None:
            continue
        found_tool = True
        try:
            done = subprocess.run([executable, *argv[1:]], capture_output=True, timeout=timeout,
                                  stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return None, f"Reading the clipboard timed out after {timeout:g} s"
        except OSError:
            continue
        if done.returncode != 0 or not done.stdout:
            continue
        data = _decode(done.stdout, encoding)
        if data and image_media_type(data):
            return data, ""
    if not found_tool and platform not in ("win32", "darwin"):
        return None, "Clipboard images need wl-paste or xclip"
    return None, NO_IMAGE


class PastedImages:
    """This process's clipboard images, by number (`[Image #N]`); numbers keep counting across /clear, like
    `PasteStore`, so a placeholder recalled from this run's history still means the same image."""

    def __init__(self) -> None:
        self._images: dict[int, bytes] = {}

    def add(self, data: bytes) -> str:
        """Keep `data` and return the placeholder to put in the input."""
        number = len(self._images) + 1
        self._images[number] = data
        return f"[Image #{number}]"

    def get(self, number: int) -> bytes | None:
        return self._images.get(number)


_WINDOWS_ABSOLUTE_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_POSIX_ESCAPE_RE = re.compile(r"\\(.)")


def dropped_path_text(text: str, platform: str = sys.platform) -> str | None:
    """The absolute path a drag-and-drop pasted, or None when the paste is anything else.

    Terminals paste a dropped file as its path, maybe quoted (`'…'`, `"…"`), with spaces escaped (`my\\ shot.png`,
    macOS) or as a `file://` URI. Anything that isn't absolute is None ("see /a.png" is a sentence); whether the
    file exists is the caller's check.
    """
    text = text.strip()
    if not text or "\n" in text:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    elif text.startswith("file://"):
        text = unquote(urlparse(text).path)
        if platform == "win32" and re.match(r"^/[A-Za-z]:", text):
            text = text[1:]
    elif platform != "win32":
        text = _POSIX_ESCAPE_RE.sub(r"\1", text)
    if platform == "win32":
        return text if _WINDOWS_ABSOLUTE_RE.match(text) else None
    return text if text.startswith(("/", "~/")) else None
