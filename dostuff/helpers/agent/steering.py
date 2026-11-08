"""Mid-turn steering: messages the user sends while a turn runs, read at the next tool boundary.

One queue per TUI app (the main session runs one turn at a time), filled by the TUI at submit and
drained by `loop()` at the top of an iteration, after the notices and the subagent inbox (steering
spec D1/D4). A plain object with no locks: the TUI thread and the loop both run on Textual's asyncio
loop, so `take_ready` and `remove` cannot interleave.

Order is submit order (CON-006): `put` reserves the slot synchronously and preparation fills it in
later, so a message whose attachments are still being read holds back the ones sent after it, never
the reverse.
"""
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SteerMessage:
    """One steering message, from submit until the loop delivers it (or it is dropped)."""
    id: str                    # uuid hex, for the TUI to match the delivery event
    raw: str                   # what the user typed (shown, re-queued as a leftover, put back on Esc)
    content: str | None = None # prepared text for the model (D5); None until ready
    failed: str | None = None  # preparation error, shown instead of delivering
    parts: list[dict[str, Any]] = field(default_factory=list)  # image refs sent after the text (multimodal REQ-2.4)


class SteeringQueue:
    """Submit-ordered entries that become deliverable when their preparation finishes (D4)."""

    def __init__(self) -> None:
        self._entries: list[SteerMessage] = []

    def put(self, raw: str) -> SteerMessage:
        """Reserve a slot in submit order; the id is what `ready` / `fail` / `remove` key on."""
        msg = SteerMessage(id=uuid.uuid4().hex, raw=raw)
        self._entries.append(msg)
        return msg

    def ready(self, msg_id: str, content: str, parts: list[dict[str, Any]] | None = None) -> None:
        """Preparation finished: this entry can be delivered at the next tool boundary."""
        for msg in self._entries:
            if msg.id == msg_id:
                msg.content = content
                msg.parts = list(parts or [])
                return

    def fail(self, msg_id: str, error: str) -> None:
        """Preparation failed: taken with the ready prefix and reported, never delivered (REQ-2.4)."""
        for msg in self._entries:
            if msg.id == msg_id:
                msg.failed = error
                return

    @staticmethod
    def _taken(msg: SteerMessage) -> bool:
        return msg.content is not None or msg.failed is not None

    def take_ready(self) -> list[SteerMessage]:
        """Remove and return the longest ready (or failed) prefix; an unready entry stops it."""
        taken: list[SteerMessage] = []
        while self._entries and self._taken(self._entries[0]):
            taken.append(self._entries.pop(0))
        return taken

    def pending(self) -> list[SteerMessage]:
        """Everything still queued, left in place — the queue bar and the `/queue` picker."""
        return list(self._entries)

    def remove(self, msg_id: str) -> bool:
        """`/queue` Remove / Edit; False once the loop has taken it (already read) or unknown."""
        for i, msg in enumerate(self._entries):
            if msg.id == msg_id:
                del self._entries[i]
                return True
        return False

    def take_all(self) -> list[SteerMessage]:
        """Leftovers at turn end (D8): everything, ready or not, in order."""
        entries, self._entries = self._entries, []
        return entries

    def has_ready_or_preparing(self) -> bool:
        """D7: any pending entry keeps the turn going — a final answer must not end it first."""
        return bool(self._entries)


# The step a delivered steering message becomes (spec D6): the step is this text verbatim —
# message-adjacent prompting, which is how Claude Code's queued commands reach the model (the
# wrapper in anthropics/claude-code#84389, v2.1.223: "Address the message above as you continue
# this turn") — because without it a standalone question is answered with a final response and the
# turn ends with the original task unfinished: the loop has no notion of the task, only the
# history. Slim on purpose: #84389 is a complaint about Claude Code's ~5-line boilerplate repeating
# per message, so this is one fixed head and one fixed tail, no per-session explanation. Written
# to the history exactly once (CON-004). Lives here, next to the queue, so the TUI can import it
# without pulling in the loop's heavy dependencies.
STEER_FRAME_HEAD = "The user sent a new message while you were working:"
STEER_FRAME_TAIL = "Address the message above, then continue the task you were working on."


def frame_steered(content: str) -> str:
    """A delivered steering message as the model reads it: head, raw content, resume clause (D6)."""
    return f"{STEER_FRAME_HEAD}\n{content}\n{STEER_FRAME_TAIL}"


def unframe_steered(text: str) -> str:
    """The typed message back out of a stored step (D6/REQ-2.9): Esc-to-restore after a reload
    returns the raw text, never the frame. Anything not framed passes through unchanged."""
    if text.startswith(STEER_FRAME_HEAD + "\n") and text.endswith("\n" + STEER_FRAME_TAIL):
        return text[len(STEER_FRAME_HEAD) + 1:-len(STEER_FRAME_TAIL) - 1]
    return text
