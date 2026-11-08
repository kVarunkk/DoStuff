"""Images and PDFs a tool brings into the conversation (multimodal spec REQ-5, D8).

A tool result is text only (OpenAI's `role: "tool"` content can't hold an image), so a tool that opens an image —
`read_file`, an MCP tool returning `ImageContent` — stores it and hands its ref to the collector the loop opened for
that call. After the iteration's tool steps, the loop appends one user step holding every ref (`TOOL_ATTACHMENTS`).
Outside a loop nothing is collecting, and the tool says the file can't be attached rather than claiming it was.
"""
import contextlib
from contextvars import ContextVar
from typing import Any, Iterator

from dostuff.helpers.agent import attachments
from dostuff.helpers.agent.attachments import AttachmentStore

TOOL_ATTACHMENTS = "tool_attachments"  # the follow-up step's `meta`

_collector: ContextVar[list[dict[str, Any]] | None] = ContextVar("tool_attachments", default=None)


@contextlib.contextmanager
def collecting() -> Iterator[list[dict[str, Any]]]:
    """Collect the refs tools attach inside this block (one tool call, in the loop). The list is the context's, so
    a tool running in a thread (`asyncio.to_thread` copies the context) adds to it too."""
    found: list[dict[str, Any]] = []
    token = _collector.set(found)
    try:
        yield found
    finally:
        _collector.reset(token)


def is_tool_attachment_step(step: Any) -> bool:
    return isinstance(step, dict) and step.get("meta") == TOOL_ATTACHMENTS


def tool_attachment_step(found: list[dict[str, Any]]) -> dict[str, Any]:
    """The user step that carries what the tools of one iteration attached (D8), at most MAX_PER_STEP refs."""
    kept = found[:attachments.MAX_PER_STEP]
    names = ", ".join(str(r.get("name", "attachment")) for r in kept)
    text = f"[Attached by the tool calls above: {names}]"
    if len(found) > len(kept):
        text += f"\n[{len(found) - len(kept)} more left out: at most {attachments.MAX_PER_STEP} per message]"
    return {"role": "user", "content": [{"type": "text", "text": text}, *kept], "meta": TOOL_ATTACHMENTS}


def _vision() -> bool:
    from dostuff.lib import model as model_lib  # lazy: litellm
    return model_lib.model_supports("vision")


NO_VISION = "this model can't view images"
NOT_IN_TURN = "it can only be attached during an agent turn"


def attach_ref(data: bytes, name: str) -> tuple[dict[str, Any] | None, str]:
    """Store an image or PDF a tool opened and hand its ref to the loop: (ref, "") or (None, why not).

    An image needs a model that can view images; the caller decides that a PDF goes natively."""
    found = _collector.get()
    if not data.startswith(b"%PDF-"):
        if not _vision():
            return None, NO_VISION
        data, problem = attachments.prepare_image(data)
        if problem:
            return None, problem
    if found is None:
        return None, NOT_IN_TURN
    ref = AttachmentStore().add(data, name=name)
    found.append(ref)
    return ref, ""


def attach(data: bytes, name: str, shown: str) -> str:
    """`attach_ref`, worded as read_file's result (REQ-5.2)."""
    ref, reason = attach_ref(data, name)
    if ref is None:
        if reason == NO_VISION:
            return f"This model can't view images; {shown} was not read."
        return f"{shown} was not attached ({reason}); it was not read."
    if ref.get("kind") == "pdf":
        pages = int(ref.get("pages") or 1)
        return f"PDF {shown} ({pages} page{'s' if pages != 1 else ''}) is attached in the next message."
    return f"Image {shown}{size_text(ref)} is attached in the next message."


def size_text(ref: dict[str, Any]) -> str:
    return f" ({ref['width']}×{ref['height']})" if ref.get("width") and ref.get("height") else ""
