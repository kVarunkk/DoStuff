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
    from dostuff.helpers.agent.pastes import PasteStore
    from dostuff.helpers.agent.steering import SteeringQueue
    shown: list = []
    return SimpleNamespace(
        _append=lambda text, msg_type="": shown.append((msg_type, text)), _steering=SteeringQueue(),
        _pastes=PasteStore(), _skill_commands={}, _render_queue=lambda: None, session_id="s1",
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
