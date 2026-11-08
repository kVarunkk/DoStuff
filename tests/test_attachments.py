"""Multimodal input: plan/2026-11-08-multimodal-input/tasks.md."""
import sys
from pathlib import Path

import pytest

from conftest import requires_extra


# --- T1.0 extras -------------------------------------------------------------------------------

def test_requires_extra_fails_when_required(monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)  # hide Pillow whether or not it is installed
    monkeypatch.delenv("DOSTUFF_REQUIRE_EXTRAS", raising=False)
    with pytest.raises(pytest.skip.Exception, match=r"dostuff\[images\]"):
        requires_extra("images")
    monkeypatch.setenv("DOSTUFF_REQUIRE_EXTRAS", "1")
    with pytest.raises(pytest.fail.Exception, match=r"dostuff\[images\]"):
        requires_extra("images")


# --- fixtures: image headers built from a few bytes, never committed binaries -------------------

def png_bytes(width: int = 2, height: int = 2) -> bytes:
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\0" + b"\xff\x00\x00" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def jpeg_bytes(width: int = 3, height: int = 5) -> bytes:
    import struct
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0"
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3) + b"\x01\x11\0\x02\x11\0\x03\x11\0"
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


def gif_bytes(width: int = 4, height: int = 6) -> bytes:
    import struct
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\0\0\0;"


def webp_bytes(width: int = 7, height: int = 9) -> bytes:
    body = (b"WEBP" + b"VP8X" + (10).to_bytes(4, "little") + b"\0\0\0\0"
            + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little"))
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def pdf_bytes(pages: int = 2) -> bytes:
    objs = "".join(f"{i + 3} 0 obj << /Type /Page /Parent 2 0 R >> endobj\n" for i in range(pages))
    return (f"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            f"2 0 obj << /Type /Pages /Count {pages} >> endobj\n{objs}%%EOF\n").encode()


def office_bytes(member: str) -> bytes:
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr(member, "<x/>")
    return buf.getvalue()


# --- T1.1 classify ------------------------------------------------------------------------------

def test_classify_by_magic_bytes(tmp_path):
    from dostuff.helpers.agent.attachments import classify_attachment, image_size
    samples = {
        "a.png": (png_bytes(), "image"), "b.jpg": (jpeg_bytes(), "image"), "c.gif": (gif_bytes(), "image"),
        "d.webp": (webp_bytes(), "image"), "e.pdf": (pdf_bytes(), "pdf"),
        "f.docx": (office_bytes("word/document.xml"), "docx"), "g.xlsx": (office_bytes("xl/workbook.xml"), "xlsx"),
        "h.pptx": (office_bytes("ppt/presentation.xml"), "pptx"),
        "fake.png": (b"just some text\n", None),  # the extension doesn't decide
        "other.zip": (office_bytes("readme.txt"), None),
        "renamed.txt": (png_bytes(), "image"),
    }
    for name, (data, kind) in samples.items():
        (tmp_path / name).write_bytes(data)
        assert classify_attachment(tmp_path / name) == kind, name
    assert classify_attachment(tmp_path / "missing.png") is None
    assert image_size(png_bytes(1280, 720)) == (1280, 720)
    assert image_size(jpeg_bytes(3, 5)) == (3, 5)
    assert image_size(gif_bytes(4, 6)) == (4, 6)
    assert image_size(webp_bytes(7, 9)) == (7, 9)
    assert image_size(b"nope") is None


# --- T1.2 store ---------------------------------------------------------------------------------

def test_store_is_content_addressed(tmp_path):
    from dostuff.helpers.agent.attachments import ATTACHMENT_DIR, AttachmentStore
    src = tmp_path / "shot.png"
    src.write_bytes(png_bytes(4, 3))
    store = AttachmentStore(root=tmp_path)
    a = store.add(src)
    b = store.add(png_bytes(4, 3), name="again.png")
    assert a["type"] == "dostuff_attachment" and a["kind"] == "image" and a["media_type"] == "image/png"
    assert a["id"] == b["id"] and a["path"] == b["path"] and len(a["id"]) == 16
    assert a["name"] == "shot.png" and b["name"] == "again.png"
    assert (a["width"], a["height"]) == (4, 3)
    stored = tmp_path / a["path"]
    assert stored.parent == tmp_path / ATTACHMENT_DIR and stored.read_bytes() == png_bytes(4, 3)
    assert (tmp_path / ATTACHMENT_DIR / ".gitignore").read_text() == "*\n"
    mtime = stored.stat().st_mtime_ns
    store.add(src)
    assert stored.stat().st_mtime_ns == mtime  # written once
    c = store.add(png_bytes(5, 3), name="other.png")
    assert c["id"] != a["id"]
    assert "base64" not in str(a) and "data:" not in str(a)  # CON-004: refs only


# --- T1.3 / T1.4 hydrate ------------------------------------------------------------------------

def _image_step(tmp_path, text="what is this?"):
    from dostuff.helpers.agent.attachments import AttachmentStore
    ref = AttachmentStore(root=tmp_path).add(png_bytes(), name="bug.png")
    return ref, {"role": "user", "content": [{"type": "text", "text": text}, ref]}


def test_hydrate_refs_to_data_uri(tmp_path):
    import base64
    from dostuff.helpers.agent.attachments import hydrate
    ref, step = _image_step(tmp_path)
    out = hydrate([step], root=tmp_path)
    assert out[0]["content"][0] == {"type": "text", "text": "what is this?"}
    assert out[0]["content"][1] == {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(png_bytes()).decode()}}
    assert step["content"][1] is ref  # history untouched (CON-004)


def test_hydrate_is_deterministic(tmp_path):
    import json
    from dostuff.helpers.agent.attachments import hydrate
    _, step = _image_step(tmp_path)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}, step]
    assert json.dumps(hydrate(history, root=tmp_path)) == json.dumps(hydrate(history, root=tmp_path))


def test_missing_attachment_becomes_text(tmp_path):
    from dostuff.helpers.agent.attachments import hydrate
    ref, step = _image_step(tmp_path)
    (tmp_path / ref["path"]).unlink()
    missing: list[str] = []
    out = hydrate([step], root=tmp_path, missing=missing)
    assert out[0]["content"][1] == {"type": "text", "text": "[attachment unavailable: bug.png]"}
    assert missing == ["bug.png"]


def test_plain_turn_request_unchanged(tmp_path):
    from dostuff.helpers.agent.attachments import hydrate
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": None, "tool_calls": []},
               {"role": "tool", "tool_call_id": "c1", "content": "x"},
               {"role": "user", "content": [{"type": "text", "text": "plain list"}]}]
    out = hydrate(history, root=tmp_path)
    assert out == history and all(a is b for a, b in zip(out, history))


def test_call_agent_hydrates_and_keeps_history(fake_model, tmp_path, monkeypatch):
    import asyncio
    from dostuff.agent.call_agent import call_agent
    monkeypatch.chdir(tmp_path)
    ref, step = _image_step(tmp_path)
    history = [step]
    asyncio.run(call_agent(history, "SYS", tool_names=[]))
    sent = fake_model.calls[0]["messages"]
    assert sent[1]["content"][1]["type"] == "image_url"
    assert not any(p.get("type") == "dostuff_attachment"
                   for m in sent if isinstance(m["content"], list) for p in m["content"])
    assert history[0]["content"][1] == ref


# --- T1.7 token estimate ------------------------------------------------------------------------

def test_estimate_tokens_counts_ref_not_json(tmp_path):
    from dostuff.helpers.agent.attachments import AttachmentStore, ref_tokens
    from dostuff.helpers.agent.manage_context import estimate_tokens
    ref = AttachmentStore(root=tmp_path).add(png_bytes(1280, 720), name="wide.png")
    assert ref_tokens(ref) == 1229  # ceil(1280*720/750)
    assert ref_tokens({**ref, "width": 4000, "height": 3000}) == 1600  # capped
    assert ref_tokens({k: v for k, v in ref.items() if k not in ("width", "height")}) == 1600  # unknown size
    assert ref_tokens({"type": "dostuff_attachment", "kind": "pdf", "pages": 3}) == 6000
    step = {"role": "user", "content": [{"type": "text", "text": "x" * 40}, ref]}
    assert abs(estimate_tokens(step) - 1229) < 40


def test_context_shows_attachments_row(tmp_path):
    from dostuff.helpers.agent.attachments import AttachmentStore
    from dostuff.helpers.agent.context_usage import context_parts, render_context
    store = AttachmentStore(root=tmp_path)
    a = store.add(png_bytes(1280, 720), name="a.png")
    b = store.add(pdf_bytes(1), name="r.pdf")
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}, a, b]}]
    parts = dict(context_parts("S", "", {}, messages, "", len))
    assert parts["Messages"] == len("hello")  # the ref's JSON is not counted as text
    assert parts["Attachments (2)"] == 1229 + 2000
    assert "Attachments (2)" in render_context(list(parts.items()), 100_000, "m").plain
    assert "Attachments" not in "".join(dict(context_parts("S", "", {}, [{"role": "user", "content": "x"}], "", len)))


# --- T1.8 capability ----------------------------------------------------------------------------

def _config(tmp_path, monkeypatch, yaml_text: str):
    from dostuff import config as config_module
    from dostuff.lib import model as model_module
    home = tmp_path / "home"
    (home / ".dostuff").mkdir(parents=True, exist_ok=True)
    (home / ".dostuff" / "config.yaml").write_text(yaml_text, encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    cfg = config_module.Config()
    monkeypatch.setattr(model_module, "get_config", lambda: cfg)
    return cfg


def test_model_supports_overrides(tmp_path, monkeypatch):
    from dostuff.lib.model import model_supports
    _config(tmp_path, monkeypatch, "model:\n  name: openai/gpt-4o\n")
    assert model_supports("vision") is True and model_supports("pdf") is True  # auto: litellm knows gpt-4o
    _config(tmp_path, monkeypatch, "model:\n  name: openai/gpt-4o\n  vision: false\n  pdf_input: false\n")
    assert model_supports("vision") is False and model_supports("pdf") is False
    _config(tmp_path, monkeypatch, "model:\n  name: ollama/llama3\n")
    assert model_supports("vision") is False
    _config(tmp_path, monkeypatch, "model:\n  name: ollama/llama3\n  vision: true\n")
    assert model_supports("vision") is True
    _config(tmp_path, monkeypatch, "model:\n  name: custom/not-a-real-model-xyz\n  vision: auto\n")
    assert model_supports("vision") is False and model_supports("pdf") is False  # unknown: False, no exception
    # A `models:` entry carries its own override
    _config(tmp_path, monkeypatch, "models:\n  - name: ollama/llava\n    active: true\n    vision: true\n")
    assert model_supports("vision") is True


# --- T1.5 content_text and its readers ----------------------------------------------------------

def _list_step(tmp_path, text='look @bug.png\n\n<attached_file path="a.py">\n1\tx\n</attached_file>'):
    from dostuff.helpers.agent.attachments import AttachmentStore
    ref = AttachmentStore(root=tmp_path).add(png_bytes(), name="bug.png")
    return {"role": "user", "content": [{"type": "text", "text": text}, ref]}


def test_content_text(tmp_path):
    from dostuff.helpers.agent.attachments import content_text
    step = _list_step(tmp_path, "hi")
    assert content_text(step["content"]) == "hi\n[Image: bug.png]"
    assert content_text("plain") == "plain" and content_text(None) == ""
    assert content_text([{"type": "dostuff_attachment", "kind": "pdf", "name": "r.pdf"}]) == "[PDF: r.pdf]"


def test_strip_attachments_list_content(tmp_path):
    from dostuff.helpers.agent.file_mentions import attached_paths, strip_attachments
    step = _list_step(tmp_path)
    assert strip_attachments(step["content"]) == "look @bug.png"
    assert attached_paths(step["content"]) == ["a.py"]


def test_session_title_list_content(tmp_path):
    import json
    from dostuff.lib.memory.session_store import _session_title
    assert _session_title(json.dumps(_list_step(tmp_path))) == "look @bug.png"
    assert _session_title(json.dumps(_list_step(tmp_path, ""))) == "[Image: bug.png]"  # an image alone


def test_export_list_content(tmp_path):
    from dostuff.helpers.memory.export_session import session_to_markdown
    md = session_to_markdown([_list_step(tmp_path)], None, "s1")
    assert "## User\n\nlook @bug.png\n" in md
    assert "`a.py`" in md and "`bug.png`" in md
    assert "dostuff_attachment" not in md and "base64" not in md


def test_export_tool_attachment_step(tmp_path):
    from dostuff.helpers.agent.attachments import AttachmentStore
    from dostuff.helpers.agent.tool_attachments import tool_attachment_step
    from dostuff.helpers.memory.export_session import session_to_markdown
    step = tool_attachment_step([AttachmentStore(root=tmp_path).add(png_bytes(), name="diagram.png")])
    md = session_to_markdown([step], None, "s1")
    assert "_Attached by the tools: `diagram.png`_" in md and "## User" not in md


def test_recall_finds_list_prompt(tmp_path):
    from types import SimpleNamespace
    from dostuff.cli_tui import DostuffTUI
    history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"}, _list_step(tmp_path)]
    assert DostuffTUI._last_prompt_index(SimpleNamespace(steps_history=history)) == 2  # type: ignore[arg-type]


def test_trace_attr_is_text_not_parts(fake_model, tmp_path, monkeypatch):
    import asyncio
    from dostuff.agent import call_agent as call_agent_module
    seen: list = []
    monkeypatch.setattr(call_agent_module, "content_attr", lambda value, limit=2000: seen.append(value) or "")
    monkeypatch.chdir(tmp_path)
    asyncio.run(call_agent_module.call_agent([_list_step(tmp_path, "hi")], "SYS", tool_names=[]))
    assert seen == ["hi\n[Image: bug.png]"]


# --- T1.9 @image mentions -----------------------------------------------------------------------

@pytest.fixture
def vision(monkeypatch):
    """`vision(True/False)`: what the model can read, through the one wrapper (never litellm itself)."""
    from dostuff.lib import model as model_module
    state = {"vision": True, "pdf": False}

    def set_vision(value: bool = True, pdf: bool = False):
        state["vision"], state["pdf"] = value, pdf
    monkeypatch.setattr(model_module, "model_supports", lambda kind: state[kind])
    monkeypatch.setattr(model_module, "get_model", lambda: "test/model")
    return set_vision


def test_mention_image_attaches_ref(tmp_path, vision, monkeypatch):
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    monkeypatch.chdir(tmp_path)  # read_file's sandbox is the cwd
    (tmp_path / "shot.png").write_bytes(png_bytes(1280, 720))
    (tmp_path / "a.py").write_text("x = 1\n")
    vision(True)
    exp = expand_file_mentions("what's @shot.png, see @a.py", root=tmp_path)
    assert exp.refused == [] and exp.skipped == []
    assert [p["name"] for p in exp.parts] == ["shot.png"] and exp.parts[0]["kind"] == "image"
    assert (tmp_path / exp.parts[0]["path"]).read_bytes() == png_bytes(1280, 720)
    assert exp.attached == ["shot.png (1280×720)", "a.py"]
    assert "<attached_file" in exp.content and "shot.png\">" not in exp.content  # the image isn't pasted as text
    # The same image twice is attached once
    assert len(expand_file_mentions("@shot.png @shot.png", root=tmp_path).parts) == 1


def test_mention_image_refused_without_vision(tmp_path, vision):
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    (tmp_path / "shot.png").write_bytes(png_bytes())
    vision(False)
    exp = expand_file_mentions("what's @shot.png", root=tmp_path)
    assert exp.parts == [] and exp.refused == ["shot.png: test/model can't read images"]
    assert not (tmp_path / ".dostuff" / "attachments").exists()  # nothing stored for a refused turn


def test_mention_image_too_large(tmp_path, vision, monkeypatch):
    from dostuff.helpers.agent import attachments
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    monkeypatch.setitem(sys.modules, "PIL", None)  # REQ-6.3 without Pillow
    monkeypatch.setattr(attachments, "MAX_IMAGE_BYTES", 100)
    (tmp_path / "big.png").write_bytes(png_bytes(40, 40))
    vision(True)
    exp = expand_file_mentions("@big.png", root=tmp_path)
    assert exp.parts == [] and len(exp.refused) == 1
    assert exp.refused[0].startswith("big.png: ") and "install dostuff[images] to resize" in exp.refused[0]


def test_mention_too_many_images(tmp_path, vision, monkeypatch):
    from dostuff.helpers.agent import attachments
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    monkeypatch.setattr(attachments, "MAX_PER_STEP", 2)
    for i in range(3):
        (tmp_path / f"{i}.png").write_bytes(png_bytes(i + 1, 1))
    vision(True)
    exp = expand_file_mentions("@0.png @1.png @2.png", root=tmp_path)
    assert exp.parts == [] and exp.refused == ["3 images and PDFs; at most 2 per message"]


def _steer_app(tmp_path):
    from types import SimpleNamespace
    from dostuff.helpers.agent.clipboard import PastedImages
    from dostuff.helpers.agent.pastes import PasteStore
    from dostuff.helpers.agent.steering import SteeringQueue
    shown: list = []
    return SimpleNamespace(
        _append=lambda text, msg_type="": shown.append((msg_type, text)), _steering=SteeringQueue(),
        _pastes=PasteStore(), _images=PastedImages(), _skill_commands={}, _render_queue=lambda: None, session_id="s1",
        config=SimpleNamespace(max_tool_output_chars=0, paste_file_lines=0)), shown


def test_steer_image_prepared_with_parts(tmp_path, vision, monkeypatch):
    import asyncio
    from dostuff.cli_tui import DostuffTUI
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shot.png").write_bytes(png_bytes())
    app, shown = _steer_app(tmp_path)
    vision(True)
    msg = app._steering.put("and @shot.png")
    asyncio.run(DostuffTUI._prepare_steer(app, msg))  # type: ignore[arg-type]
    assert msg.content == "and @shot.png" and [p["name"] for p in msg.parts] == ["shot.png"]
    vision(False)
    msg = app._steering.put("and @shot.png")
    asyncio.run(DostuffTUI._prepare_steer(app, msg))  # type: ignore[arg-type]
    assert msg.content is None and msg.failed == "Not attached: shot.png: test/model can't read images"


# --- T1.13 compaction ---------------------------------------------------------------------------

def test_summary_request_has_no_images(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from litellm import ModelResponse
    from dostuff.helpers.agent import manage_context
    ref = _list_step(tmp_path, "what is this?")["content"][1]
    history = [{"role": "user", "content": [{"type": "text", "text": "what is this?"}, ref]},
               {"role": "assistant", "content": "a red square " * 200}]
    history += [{"role": "user", "content": f"next {i} " * 200} for i in range(4)]
    sent: list = []

    async def fake_acompletion(**kwargs):
        sent.append(kwargs["messages"])
        return ModelResponse(choices=[{"message": {"role": "assistant", "content": "NOTES"}}])
    monkeypatch.setattr(manage_context, "acompletion", fake_acompletion)
    monkeypatch.setattr(manage_context, "completion_kwargs", lambda: {"model": "m"})
    recent, notes = asyncio.run(manage_context.compact_context(history, 300, context_window=100_000))
    assert notes == "NOTES" and sent
    for messages in sent:
        assert all(isinstance(m["content"], str) for m in messages)  # the summary model may lack vision
        text = "".join(m["content"] for m in messages)
        assert "dostuff_attachment" not in text and "base64" not in text
    assert "[Image: bug.png]" in "".join(m["content"] for m in sent[0])


def test_split_keeps_refs(tmp_path):
    from dostuff.helpers.agent.manage_context import split_by_token_budget
    step = _list_step(tmp_path, "look")
    history = [{"role": "user", "content": "old " * 400}, {"role": "assistant", "content": "ok " * 400}, step]
    old, recent = split_by_token_budget(history, 300)
    assert recent == [step] and recent[0]["content"][1] is step["content"][1]  # still a ref, not text


# --- T1.14 sweep --------------------------------------------------------------------------------

def test_attachment_sweep_age(tmp_path, monkeypatch):
    import os
    import time
    from dostuff.helpers.agent.attachments import (ATTACHMENT_DIR, AttachmentStore, delete_attachments,
                                                   sweep_old_attachments)
    store = AttachmentStore(root=tmp_path)
    old = store.add(png_bytes(1, 1), name="old.png")
    new = store.add(png_bytes(2, 1), name="new.png")
    long_ago = time.time() - 40 * 86_400
    os.utime(tmp_path / old["path"], (long_ago, long_ago))
    assert sweep_old_attachments(0, root=tmp_path) == 0  # 0: no age sweep (deleted with the session instead)
    assert sweep_old_attachments(30, root=tmp_path) == 1
    assert not (tmp_path / old["path"]).exists() and (tmp_path / new["path"]).exists()
    assert (tmp_path / ATTACHMENT_DIR / ".gitignore").exists()
    # retention 0: the session's own attachments go when it ends
    history = [{"role": "user", "content": [{"type": "text", "text": "x"}, new]}, {"role": "assistant", "content": "y"}]
    assert delete_attachments(history, root=tmp_path) == 1 and not (tmp_path / new["path"]).exists()
    assert sweep_old_attachments(30, root=tmp_path / "nowhere") == 0


def test_attachment_retention_config(tmp_path, monkeypatch):
    cfg = _config(tmp_path, monkeypatch, "model:\n  name: openai/gpt-4o\n")
    assert cfg.attachment_retention_days == 30
    cfg = _config(tmp_path, monkeypatch, "loop:\n  attachment_retention_days: 0\n")
    assert cfg.attachment_retention_days == 0


# --- T2.1 clipboard readers ---------------------------------------------------------------------

class _Run:
    """A stubbed subprocess.run: pops one result per call, records the argv."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls: list[list[str]] = []
        self.kwargs: dict = {}

    def __call__(self, argv, **kwargs):
        import subprocess
        self.calls.append(argv)
        self.kwargs = kwargs
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        code, out = result
        return subprocess.CompletedProcess(argv, code, stdout=out, stderr=b"")


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_clipboard_reader_order(monkeypatch, platform):
    import base64
    from dostuff.helpers.agent import clipboard
    png = png_bytes(4, 3)
    # Pillow first: when it has the image, no command runs
    run = _Run()
    monkeypatch.setattr(clipboard.subprocess, "run", run)
    monkeypatch.setattr(clipboard, "_pillow_grab", lambda: png)
    assert clipboard.read_clipboard_image(platform=platform) == (png, "")
    assert run.calls == []
    # No Pillow (or nothing from it): the platform's command, its stdout decoded
    monkeypatch.setattr(clipboard, "_pillow_grab", lambda: None)
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: f"/bin/{name}")
    out = {"win32": base64.b64encode(png),
           "darwin": "«data PNGf".encode() + png.hex().upper().encode() + "»\n".encode(),
           "linux": png}[platform]
    run = _Run((0, out))
    monkeypatch.setattr(clipboard.subprocess, "run", run)
    assert clipboard.read_clipboard_image(platform=platform) == (png, "")
    assert run.calls[0][0] == {"win32": "/bin/powershell", "darwin": "/bin/osascript", "linux": "/bin/wl-paste"}[platform]
    assert run.kwargs["timeout"] == clipboard.CLIPBOARD_TIMEOUT_S
    # Nothing on the clipboard: a message, no bytes
    run = _Run((1, b""), (1, b""))
    monkeypatch.setattr(clipboard.subprocess, "run", run)
    assert clipboard.read_clipboard_image(platform=platform) == (None, "No image on the clipboard")


def test_clipboard_linux_falls_back_to_xclip(monkeypatch):
    from dostuff.helpers.agent import clipboard
    png = png_bytes()
    monkeypatch.setattr(clipboard, "_pillow_grab", lambda: None)
    run = _Run((1, b"No suitable type of content copied"), (0, png))
    monkeypatch.setattr(clipboard.subprocess, "run", run)
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert clipboard.read_clipboard_image(platform="linux") == (png, "")
    assert [c[0] for c in run.calls] == ["/usr/bin/wl-paste", "/usr/bin/xclip"]
    # Only xclip installed: wl-paste isn't tried
    run = _Run((0, png))
    monkeypatch.setattr(clipboard.subprocess, "run", run)
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: "/usr/bin/xclip" if name == "xclip" else None)
    assert clipboard.read_clipboard_image(platform="linux") == (png, "")
    assert [c[0] for c in run.calls] == ["/usr/bin/xclip"]
    # Neither: say what to install
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: None)
    assert clipboard.read_clipboard_image(platform="linux") == (None, "Clipboard images need wl-paste or xclip")


def test_clipboard_windows_powershell_off_path(monkeypatch, tmp_path):
    """Git Bash and some terminals have no `powershell` on PATH; Windows always installs it under SYSTEMROOT."""
    from dostuff.helpers.agent import clipboard
    exe = tmp_path / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path))
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: None)
    assert clipboard._find("powershell", "win32") == str(exe)
    exe.unlink()
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: "/opt/pwsh" if name == "pwsh" else None)
    assert clipboard._find("powershell", "win32") == "/opt/pwsh"
    assert clipboard._find("powershell", "linux") is None


def test_clipboard_text_is_not_an_image(monkeypatch):
    from dostuff.helpers.agent import clipboard
    monkeypatch.setattr(clipboard, "_pillow_grab", lambda: None)
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(clipboard.subprocess, "run", _Run((0, b"just some text"), (0, b"")))
    assert clipboard.read_clipboard_image(platform="linux") == (None, "No image on the clipboard")


def test_clipboard_timeout(monkeypatch):
    import subprocess
    from dostuff.helpers.agent import clipboard
    monkeypatch.setattr(clipboard, "_pillow_grab", lambda: None)
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(clipboard.subprocess, "run", _Run(subprocess.TimeoutExpired("powershell", 5)))
    data, message = clipboard.read_clipboard_image(platform="win32")
    assert data is None and message == "Reading the clipboard timed out after 5 s"


def test_clipboard_pillow_grab(monkeypatch, tmp_path):
    """Pillow's grabclipboard gives an Image, a list of copied files, or None; it is imported only when asked."""
    import types
    from dostuff.helpers.agent import clipboard
    png = png_bytes(3, 2)

    class FakeImage:
        def save(self, buffer, format):
            assert format == "PNG"
            buffer.write(png)

    grab = types.SimpleNamespace(grabclipboard=lambda: FakeImage())
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(ImageGrab=grab))
    monkeypatch.setitem(sys.modules, "PIL.ImageGrab", grab)
    assert clipboard._pillow_grab() == png
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "shot.png").write_bytes(png)
    grab.grabclipboard = lambda: [str(tmp_path / "notes.txt"), str(tmp_path / "shot.png")]  # copied in Explorer
    assert clipboard._pillow_grab() == png
    grab.grabclipboard = lambda: None
    assert clipboard._pillow_grab() is None

    def broken():
        raise OSError("xclip not found")
    grab.grabclipboard = broken  # Linux Pillow shells out itself: a failure falls through to our commands
    assert clipboard._pillow_grab() is None
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "PIL.ImageGrab", None)
    assert clipboard._pillow_grab() is None


# --- T2.2 / T2.3 pasted images: numbered placeholders, refs on submit ---------------------------

def test_pasted_images_numbering():
    from dostuff.helpers.agent.clipboard import PastedImages
    images = PastedImages()
    assert images.add(png_bytes(1, 1)) == "[Image #1]"
    assert images.add(png_bytes(2, 2)) == "[Image #2]"
    assert images.get(2) == png_bytes(2, 2) and images.get(3) is None


def test_placeholders_become_refs_in_order(tmp_path, vision):
    from dostuff.helpers.agent.attachments import ref_chips
    from dostuff.helpers.agent.clipboard import PastedImages
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    images = PastedImages()
    images.add(png_bytes(10, 20))
    images.add(png_bytes(30, 40))
    (tmp_path / "shot.png").write_bytes(png_bytes(5, 5))
    vision(True)
    text = "compare [Image #2] with @shot.png and [Image #1], again [Image #2]"
    exp = expand_file_mentions(text, root=tmp_path, images=images)
    assert exp.refused == [] and exp.content == text  # the placeholders stay in the text (D10)
    assert [p["name"] for p in exp.parts] == ["Image #2", "shot.png", "Image #1"]  # prompt order, once each
    assert (tmp_path / exp.parts[0]["path"]).read_bytes() == png_bytes(30, 40)
    assert exp.attached == ["Image #2 (30×40)", "shot.png (5×5)", "Image #1 (10×20)"]
    content = [{"type": "text", "text": text}, *exp.parts]
    # A pasted image keeps its own number; mentioned ones are numbered in order
    assert ref_chips(content) == "[Image #2 · 30×40] [Image #1 · shot.png · 5×5] [Image #1 · 10×20]"


def test_placeholder_refused_without_vision(tmp_path, vision):
    from dostuff.helpers.agent.clipboard import PastedImages
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    images = PastedImages()
    images.add(png_bytes())
    vision(False)
    exp = expand_file_mentions("what is [Image #1]", root=tmp_path, images=images)
    assert exp.parts == [] and exp.refused == ["Image #1: test/model can't read images"]
    assert not (tmp_path / ".dostuff" / "attachments").exists()


def test_unknown_placeholder_stays_text(tmp_path, vision):
    from dostuff.helpers.agent.clipboard import PastedImages
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    vision(True)
    for images in (PastedImages(), None):  # recalled after a restart / no store at all (-p)
        exp = expand_file_mentions("what was [Image #9]?", root=tmp_path, images=images)
        assert exp.content == "what was [Image #9]?" and exp.parts == [] and exp.refused == [] and exp.attached == []


# --- T2.4 drag-and-drop -------------------------------------------------------------------------

@pytest.mark.parametrize("pasted, platform, expected", [
    ("/home/me/shot.png", "linux", "/home/me/shot.png"),
    ("  /home/me/shot.png \n", "linux", "/home/me/shot.png"),
    ("'/home/me/my shot.png' ", "linux", "/home/me/my shot.png"),  # GNOME Terminal
    ('"/home/me/my shot.png"', "linux", "/home/me/my shot.png"),
    ("/Users/me/my\\ shot\\ (1).png ", "darwin", "/Users/me/my shot (1).png"),  # Terminal.app / iTerm escape
    ("file:///home/me/my%20shot.png", "linux", "/home/me/my shot.png"),
    ("C:\\Users\\me\\shot.png", "win32", "C:\\Users\\me\\shot.png"),
    ('"C:\\Users\\me\\my shot.png"', "win32", "C:\\Users\\me\\my shot.png"),  # Windows Terminal
    ("two\nlines.png", "linux", None),
    ("", "linux", None),
    ("see /home/me/shot.png", "linux", None),
    ("/home/me/my shot.png", "linux", "/home/me/my shot.png"),  # unquoted: the caller checks the file exists
    ("C:\\Users\\a b\\shot.png", "win32", "C:\\Users\\a b\\shot.png"),
    ("shot.png", "win32", None),
])
def test_dropped_path_text(pasted, platform, expected):
    from dostuff.helpers.agent.clipboard import dropped_path_text
    assert dropped_path_text(pasted, platform=platform) == expected


# --- Phase 3 fixtures: documents built with the extras' own libraries ----------------------------

def text_pdf(pages: list[str]) -> bytes:
    """A real PDF with one line of text per page (Helvetica), which pypdf can extract."""
    n = len(pages)
    objects = ["<< /Type /Catalog /Pages 2 0 R >>",
               "<< /Type /Pages /Kids [" + " ".join(f"{4 + 2 * i} 0 R" for i in range(n)) + f"] /Count {n} >>",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
                       "/Resources << /Font << /F1 3 0 R >> >> >>")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def make_docx(path: Path) -> None:
    import docx
    doc = docx.Document()
    doc.add_heading("Intro", level=1)
    doc.add_paragraph("Plain words.")
    doc.add_heading("Detail", level=2)
    table = doc.add_table(rows=2, cols=2)
    for r, row in enumerate([("name", "qty"), ("apple | red", "3")]):
        for c, value in enumerate(row):
            table.cell(r, c).text = value
    doc.add_paragraph("After the table.")
    doc.save(str(path))


def make_xlsx(path: Path, rows: int = 2) -> None:
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Budget"
    ws.append(["item", "cost"])
    for i in range(rows):
        ws.append([f"row{i}", i * 10])
    ws["C1"] = "=1+2"  # a formula: the converter shows values, never the formula text
    other = wb.create_sheet("Notes")
    other.append(["hello"])
    wb.create_sheet("Empty")
    wb.save(str(path))


def make_pptx(path: Path) -> None:
    import pptx
    prs = pptx.Presentation()
    for title, body, notes in [("Kickoff", "Goals for Q1", "Say hello"), ("Plan", "Ship it", "")]:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
        if notes:
            slide.notes_slide.notes_text_frame.text = notes
    prs.save(str(path))


# --- T3.2 converters ----------------------------------------------------------------------------

def test_convert_docx_headings_tables(tmp_path):
    requires_extra("docs")
    from dostuff.helpers.agent.convert_document import convert_document
    make_docx(tmp_path / "r.docx")
    text = convert_document(tmp_path / "r.docx", "docx")
    assert "# Intro\n" in text and "## Detail\n" in text and "Plain words." in text
    assert "| name | qty |\n| --- | --- |\n| apple \\| red | 3 |" in text
    assert text.index("Plain words.") < text.index("| name") < text.index("After the table.")  # body order


def test_convert_xlsx_sheets(tmp_path):
    requires_extra("docs")
    from dostuff.helpers.agent.convert_document import convert_document
    make_xlsx(tmp_path / "b.xlsx")
    text = convert_document(tmp_path / "b.xlsx", "xlsx")
    assert "## Sheet: Budget" in text and "## Sheet: Notes" in text and "## Sheet: Empty\n(empty sheet)" in text
    assert "| item | cost |" in text and "| row1 | 10 |" in text
    assert "=1+2" not in text  # values, not formulas


def test_convert_xlsx_caps_rows(tmp_path, monkeypatch):
    requires_extra("docs")
    from dostuff.helpers.agent import convert_document as conv
    monkeypatch.setattr(conv, "MAX_SHEET_ROWS", 5)
    make_xlsx(tmp_path / "b.xlsx", rows=20)
    text = conv.convert_document(tmp_path / "b.xlsx", "xlsx")
    assert "| row3 |" in text and "| row4 |" not in text and "[… 16 more rows]" in text


def test_convert_pptx_slides_notes(tmp_path):
    requires_extra("docs")
    from dostuff.helpers.agent.convert_document import convert_document
    make_pptx(tmp_path / "d.pptx")
    text = convert_document(tmp_path / "d.pptx", "pptx")
    assert "--- Slide 1 ---\n# Kickoff\nGoals for Q1\nNotes: Say hello" in text
    assert "--- Slide 2 ---\n# Plan\nShip it" in text and text.count("Notes:") == 1


def test_convert_pdf_pages(tmp_path):
    requires_extra("docs")
    from dostuff.helpers.agent.convert_document import convert_document
    (tmp_path / "p.pdf").write_bytes(text_pdf(["Hello first", "Second page"]))
    text = convert_document(tmp_path / "p.pdf", "pdf")
    assert text.index("--- Page 1 ---") < text.index("Hello first") < text.index("--- Page 2 ---")
    assert "Second page" in text


def test_convert_without_extra(tmp_path, monkeypatch):
    from dostuff.helpers.agent.convert_document import MissingExtra, convert_document
    monkeypatch.setitem(sys.modules, "docx", None)
    (tmp_path / "r.docx").write_bytes(office_bytes("word/document.xml"))
    with pytest.raises(MissingExtra, match=r"install dostuff\[docs\]"):
        convert_document(tmp_path / "r.docx", "docx")


def test_convert_broken_file_is_value_error(tmp_path):
    requires_extra("docs")
    from dostuff.helpers.agent.convert_document import convert_document
    (tmp_path / "r.docx").write_bytes(office_bytes("word/document.xml"))  # a zip, but no real document
    with pytest.raises(ValueError):
        convert_document(tmp_path / "r.docx", "docx")


# --- T3.3 converted mentions --------------------------------------------------------------------

def test_mention_xlsx_inline(tmp_path, vision, monkeypatch):
    requires_extra("docs")
    from dostuff.helpers.agent.file_mentions import expand_file_mentions, strip_attachments
    monkeypatch.chdir(tmp_path)
    make_xlsx(tmp_path / "b.xlsx")
    vision(False)  # converted attachments don't need a vision model
    exp = expand_file_mentions("sum @b.xlsx", root=tmp_path, file_lines=200)
    assert exp.refused == [] and exp.skipped == [] and exp.parts == [] and exp.attached == ["b.xlsx"]
    assert '<attached_file path="b.xlsx" converted_from="xlsx">' in exp.content and "| row1 | 10 |" in exp.content
    assert strip_attachments(exp.content) == "sum @b.xlsx"  # display and export still see the typed text
    assert not (tmp_path / ".dostuff" / "attachments").exists()


def test_mention_big_xlsx_spills_to_store(tmp_path, vision, monkeypatch):
    requires_extra("docs")
    import re
    from dostuff.helpers.agent.file_mentions import attached_paths, expand_file_mentions, strip_attachments
    from dostuff.tools.files.read_file import read_file
    monkeypatch.chdir(tmp_path)
    make_xlsx(tmp_path / "b.xlsx", rows=60)
    vision(False)
    exp = expand_file_mentions("@b.xlsx", root=tmp_path, file_lines=20)
    assert exp.refused == [] and exp.attached == ["b.xlsx"]
    spilled = re.search(r'file="(\.dostuff/attachments/[0-9a-f]{16}\.md)"', exp.content).group(1)
    assert "read_file" in exp.content and "| row0 | 0 |" in exp.content  # path, how to read it, a preview
    assert "| row59 |" not in exp.content
    assert "| row59 | 590 |" in read_file(spilled, offset=60)  # the rest, by offset
    assert strip_attachments(exp.content) == "@b.xlsx" and attached_paths(exp.content) == ["b.xlsx"]


def test_mention_docx_without_extra(tmp_path, vision, monkeypatch):
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    monkeypatch.setitem(sys.modules, "docx", None)
    (tmp_path / "r.docx").write_bytes(office_bytes("word/document.xml"))
    exp = expand_file_mentions("@r.docx", root=tmp_path)
    assert exp.refused == ["r.docx: install dostuff[docs] to read .docx files"] and exp.attached == []


def test_mention_legacy_doc_unsupported(tmp_path, vision):
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    (tmp_path / "old.doc").write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 64)
    (tmp_path / "old.xls").write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 64)
    exp = expand_file_mentions("@old.doc @old.xls", root=tmp_path)
    assert exp.refused == ["old.doc: .doc files aren't supported; save it as .docx",
                           "old.xls: .xls files aren't supported; save it as .xlsx"]


def test_mention_secret_document_skipped(tmp_path, vision):
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    (tmp_path / ".env.docx").write_bytes(office_bytes("word/document.xml"))
    exp = expand_file_mentions("@.env.docx", root=tmp_path)
    assert exp.skipped == [".env.docx: may contain secrets"] and exp.refused == []


# --- T3.4 native PDF ----------------------------------------------------------------------------

def test_pdf_native_when_supported(tmp_path, vision):
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    (tmp_path / "r.pdf").write_bytes(pdf_bytes(3))
    vision(True, pdf=True)
    exp = expand_file_mentions("summarise @r.pdf", root=tmp_path)
    assert exp.refused == [] and exp.attached == ["r.pdf (3 pages)"]
    assert [(p["kind"], p["pages"]) for p in exp.parts] == [("pdf", 3)]
    assert "<attached_file" not in exp.content


def test_pdf_falls_back_to_text(tmp_path, vision, monkeypatch):
    requires_extra("docs")
    from dostuff.helpers.agent import attachments
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    monkeypatch.chdir(tmp_path)
    (tmp_path / "r.pdf").write_bytes(text_pdf(["Hello first", "Second page"]))
    vision(True, pdf=False)
    exp = expand_file_mentions("@r.pdf", root=tmp_path)
    assert exp.parts == [] and exp.refused == [] and exp.attached == ["r.pdf"]
    assert 'converted_from="pdf"' in exp.content and "--- Page 1 ---" in exp.content
    # Over the native limits on a PDF model: text too, rather than a request the provider rejects
    monkeypatch.setattr(attachments, "MAX_PDF_PAGES", 1)
    vision(True, pdf=True)
    exp = expand_file_mentions("@r.pdf", root=tmp_path)
    assert exp.parts == [] and 'converted_from="pdf"' in exp.content


def test_pdf_fallback_without_extra(tmp_path, vision, monkeypatch):
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    monkeypatch.setitem(sys.modules, "pypdf", None)
    (tmp_path / "r.pdf").write_bytes(pdf_bytes(1))
    vision(True, pdf=False)
    exp = expand_file_mentions("@r.pdf", root=tmp_path)
    assert exp.refused == ["r.pdf: install dostuff[docs] to read .pdf files"]


# --- T3.5 resize --------------------------------------------------------------------------------

def _noise_png(width: int, height: int) -> bytes:
    import io
    import os
    from PIL import Image
    img = Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_resize_long_edge_1568(tmp_path, vision):
    requires_extra("images")
    from dostuff.helpers.agent.attachments import image_size, prepare_image
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    data, problem = prepare_image(_noise_png(3000, 1000))
    assert problem is None and image_size(data) == (1568, 523)
    small = _noise_png(100, 50)
    assert prepare_image(small) == (small, None)  # within the limits: the bytes are left alone
    (tmp_path / "wide.png").write_bytes(_noise_png(2000, 400))
    vision(True)
    exp = expand_file_mentions("@wide.png", root=tmp_path)
    assert exp.refused == [] and (exp.parts[0]["width"], exp.parts[0]["height"]) == (1568, 314)
    assert exp.attached == ["wide.png (1568×314)"]


def test_resize_still_too_large(monkeypatch):
    requires_extra("images")
    from dostuff.helpers.agent import attachments
    monkeypatch.setattr(attachments, "MAX_IMAGE_BYTES", 50)
    _, problem = attachments.prepare_image(_noise_png(40, 40))
    assert problem is not None and "over the" in problem and "install" not in problem


def test_oversized_without_pillow(monkeypatch):
    from dostuff.helpers.agent import attachments
    monkeypatch.setitem(sys.modules, "PIL", None)
    wide = png_bytes(3000, 2)
    assert attachments.prepare_image(wide) == (wide, None)  # no Pillow: sent as it is when it fits
    monkeypatch.setattr(attachments, "MAX_IMAGE_BYTES", 10)
    _, problem = attachments.prepare_image(wide)
    assert problem is not None and problem.endswith("install dostuff[images] to resize")


# --- T4.1 read_file -----------------------------------------------------------------------------

def test_read_file_image_returns_note(tmp_path, vision, monkeypatch):
    from dostuff.helpers.agent.tool_attachments import collecting
    from dostuff.tools.files.read_file import read_file
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "diagram.png").write_bytes(png_bytes(800, 600))
    vision(True)
    with collecting() as found:
        out = read_file("docs/diagram.png")
    assert out == "Image docs/diagram.png (800×600) is attached in the next message."
    assert [(r["kind"], r["name"], r["width"]) for r in found] == [("image", "diagram.png", 800)]
    assert (tmp_path / found[0]["path"]).read_bytes() == png_bytes(800, 600)


def test_read_file_image_without_vision(tmp_path, vision, monkeypatch):
    from dostuff.helpers.agent.tool_attachments import collecting
    from dostuff.tools.files.read_file import read_file
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "diagram.png").write_bytes(png_bytes())
    vision(False)
    with collecting() as found:
        assert read_file("docs/diagram.png") == "This model can't view images; docs/diagram.png was not read."
    assert found == [] and not (tmp_path / ".dostuff" / "attachments").exists()


def test_read_file_image_outside_a_turn(tmp_path, vision, monkeypatch):
    from dostuff.tools.files.read_file import read_file
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.png").write_bytes(png_bytes())
    vision(True)
    # no loop collecting: never claims it was attached
    assert read_file("a.png") == "a.png was not attached (it can only be attached during an agent turn); it was not read."


def test_read_file_docx_converted_with_offset(tmp_path, vision, monkeypatch):
    requires_extra("docs")
    from dostuff.tools.files.read_file import read_file
    monkeypatch.chdir(tmp_path)
    make_docx(tmp_path / "r.docx")
    full = read_file("r.docx")
    assert full.startswith(" 1\t# Intro\n 2\t\n 3\tPlain words.")  # numbers padded to the widest (11 lines)
    part = read_file("r.docx", offset=3, limit=1)
    assert part.splitlines()[0] == "3\tPlain words." and "continue with offset=4" in part


def test_read_file_docx_without_extra(tmp_path, monkeypatch):
    from dostuff.tools.files.read_file import read_file
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "docx", None)
    (tmp_path / "r.docx").write_bytes(office_bytes("word/document.xml"))
    with pytest.raises(ImportError, match=r"install dostuff\[docs\]"):
        read_file("r.docx")


def test_read_file_legacy_and_pdf(tmp_path, vision, monkeypatch):
    from dostuff.helpers.agent.tool_attachments import collecting
    from dostuff.tools.files.read_file import read_file
    monkeypatch.chdir(tmp_path)
    (tmp_path / "old.ppt").write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 64)
    with pytest.raises(ValueError, match=r"\.ppt files aren't supported"):
        read_file("old.ppt")
    (tmp_path / "r.pdf").write_bytes(pdf_bytes(2))
    vision(True, pdf=True)
    with collecting() as found:
        assert read_file("r.pdf") == "PDF r.pdf (2 pages) is attached in the next message."
    assert [r["kind"] for r in found] == ["pdf"]


def test_read_file_pdf_as_text(tmp_path, vision, monkeypatch):
    requires_extra("docs")
    from dostuff.helpers.agent.tool_attachments import collecting
    from dostuff.tools.files.read_file import read_file
    monkeypatch.chdir(tmp_path)
    (tmp_path / "r.pdf").write_bytes(text_pdf(["Hello first", "Second page"]))
    vision(True, pdf=False)
    with collecting() as found:
        out = read_file("r.pdf")
    assert found == [] and "1\t--- Page 1 ---" in out and "Second page" in out


def test_read_file_denied_image(tmp_path, vision, monkeypatch):
    from dostuff.lib.exceptions import PathAccessDenied
    from dostuff.tools.files.read_file import read_file
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.png").write_bytes(png_bytes())
    vision(True)
    with pytest.raises(PathAccessDenied):
        read_file(".env.png")  # CON-008: the sandbox decides before the file is classified


# --- T4.2 the follow-up step --------------------------------------------------------------------

def _tool_turn(tmp_path, monkeypatch, fake_model, calls, adapter=None):
    import asyncio
    from conftest import fake_response, fake_tool_call
    from dostuff.agent.loop import loop
    monkeypatch.chdir(tmp_path)
    fake_model.script(None, fake_response(tool_calls=[fake_tool_call(n, a, f"c{i}") for i, (n, a) in
                                                      enumerate(calls)]),
                      fake_response("it is red"))
    working: list = [{"role": "user", "content": "look at the diagram"}]
    result = asyncio.run(loop(session_id="s", turn_id="t", user_text="look", dynamic_system_instruction="sys",
                              mcp_client=None, working_history=working, turn_type="subagent_loop", stream=False,
                              max_iterations=5, tool_names=["read_file"], adapter=adapter))
    return result, working


def test_loop_appends_tool_attachment_step(tmp_path, vision, monkeypatch, fake_model):
    import json
    from conftest import FakeAdapter
    (tmp_path / "shot.png").write_bytes(png_bytes(800, 600))
    (tmp_path / "a.txt").write_text("hello\n")
    vision(True)
    adapter = FakeAdapter(auto="allow")
    result, working = _tool_turn(tmp_path, monkeypatch, fake_model,
                                 [("read_file", {"path": "shot.png"}), ("read_file", {"path": "a.txt"})], adapter)
    assert result == "it is red"
    assert [s["role"] for s in working] == ["user", "assistant", "tool", "tool", "user", "assistant"]
    step = working[4]
    assert step["meta"] == "tool_attachments"
    assert [p["type"] for p in step["content"]] == ["text", "dostuff_attachment"]
    assert "shot.png" in step["content"][0]["text"]
    assert "base64" not in json.dumps(working)  # CON-004: refs in history, bytes only in the request
    sent = fake_model.calls[1]["messages"]
    assert [m["role"] for m in sent[-4:]] == ["assistant", "tool", "tool", "user"]
    assert "meta" not in sent[-1] and [p["type"] for p in sent[-1]["content"]] == ["text", "image_url"]
    assert ("system", "[Image #1 · shot.png · 800×600]") in adapter.events  # the chip under the tool call


def test_loop_no_attachment_step_without_images(tmp_path, vision, monkeypatch, fake_model):
    (tmp_path / "a.txt").write_text("hello\n")
    vision(True)
    _, working = _tool_turn(tmp_path, monkeypatch, fake_model, [("read_file", {"path": "a.txt"})])
    assert [s["role"] for s in working] == ["user", "assistant", "tool", "assistant"]


def test_exchanges_keep_tool_attachment_step(tmp_path):
    from dostuff.helpers.agent.manage_context import _exchanges, split_by_token_budget
    ref = {"type": "dostuff_attachment", "id": "x", "kind": "image", "media_type": "image/png", "name": "a.png",
           "path": ".dostuff/attachments/x.png", "width": 10, "height": 10}
    steps = [{"role": "user", "content": "q"},
             {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
             {"role": "tool", "tool_call_id": "c1", "content": "Image a.png is attached in the next message."},
             {"role": "user", "content": [{"type": "text", "text": "a.png"}, ref], "meta": "tool_attachments"},
             {"role": "assistant", "content": "x" * 4000}]
    assert [len(u) for u in _exchanges(steps)] == [1, 3, 1]
    for budget in (1, 50, 100_000):
        _, recent = split_by_token_budget(steps, budget)
        assert not recent or recent[0].get("meta") != "tool_attachments"


# --- T4.3 MCP -----------------------------------------------------------------------------------

def test_mcp_image_content_becomes_ref(tmp_path, vision, monkeypatch):
    import base64
    from types import SimpleNamespace
    from dostuff.helpers.agent.tool_attachments import collecting
    from dostuff.tools.mcp.call_mcp_tool import _format_tool_result
    monkeypatch.chdir(tmp_path)
    data = base64.b64encode(png_bytes(4, 3)).decode()
    result = SimpleNamespace(content=[SimpleNamespace(type="text", text="a chart"),
                                      SimpleNamespace(type="image", data=data, mimeType="image/png")])
    vision(True)
    with collecting() as found:
        out = _format_tool_result(result)
    assert out == "a chart\nImage 1 from the tool (4×3) is attached in the next message."
    assert [r["kind"] for r in found] == ["image"]
    vision(False)
    with collecting() as found:
        assert _format_tool_result(result) == "a chart\n[image left out: this model can't view images]"
    assert found == []
    only_image = SimpleNamespace(content=[SimpleNamespace(type="image", data="!!not base64", mimeType="image/png")])
    with collecting():
        assert _format_tool_result(only_image) == "[image left out: not a valid image]"
