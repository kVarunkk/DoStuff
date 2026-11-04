"""Mid-turn steering: `SteerMessage` / `SteeringQueue` (steering spec REQ-1, D4).

Pure in-memory queue — no widgets, no model, no litellm. The loop drains it at the tool boundary
(`take_ready`), the TUI fills it at submit (`put` then `ready`/`fail`) and edits it (`remove`).
"""
import subprocess
import sys
from pathlib import Path

from dostuff.helpers.agent.steering import SteerMessage, SteeringQueue

ROOT = Path(__file__).resolve().parents[1]


# --- T1.1 put / pending / ready / take_ready ------------------------------------------------

def test_put_keeps_order():
    """REQ-1.1: `put` reserves a slot in submit order; `pending` lists them that way."""
    q = SteeringQueue()
    a, b, c = q.put("first"), q.put("second"), q.put("third")
    assert [m.raw for m in q.pending()] == ["first", "second", "third"]
    assert len({a.id, b.id, c.id}) == 3  # fresh id each
    assert all(m.content is None and m.failed is None for m in (a, b, c))


def test_take_ready_prefix():
    """REQ-1.2: the longest ready prefix; an unready entry stops it even when later ones are ready."""
    q = SteeringQueue()
    a, b, c = q.put("a"), q.put("b"), q.put("c")
    q.ready(a.id, "A")
    q.ready(c.id, "C")  # b is still being prepared
    taken = q.take_ready()
    assert [m.raw for m in taken] == ["a"]
    assert [m.raw for m in q.pending()] == ["b", "c"]
    q.ready(b.id, "B")
    assert [m.raw for m in q.take_ready()] == ["b", "c"]
    assert q.pending() == []


def test_failed_counts_as_ready():
    """REQ-1.2 / REQ-2.4: a failed preparation is taken with the prefix, carrying its error."""
    q = SteeringQueue()
    a, b = q.put("a"), q.put("b")
    q.fail(a.id, "Could not read @missing.txt")
    q.ready(b.id, "B")
    taken = q.take_ready()
    assert [m.raw for m in taken] == ["a", "b"]
    assert taken[0].failed == "Could not read @missing.txt" and taken[0].content is None
    assert taken[1].failed is None and taken[1].content == "B"


def test_take_ready_empty():
    """REQ-1.2: nothing ready → nothing taken, and an empty queue is safe."""
    q = SteeringQueue()
    assert q.take_ready() == []
    q.put("waiting")
    assert q.take_ready() == []
    assert len(q.pending()) == 1


# --- T1.2 remove / take_all / has_ready_or_preparing ---------------------------------------

def test_remove_pending():
    """REQ-1.3: `/queue` Remove takes a pending entry out, in order."""
    q = SteeringQueue()
    a, b = q.put("a"), q.put("b")
    assert q.remove(a.id) is True
    assert [m.raw for m in q.pending()] == ["b"]
    assert q.remove(b.id) is True and q.pending() == []


def test_remove_taken_returns_false():
    """REQ-1.3: once the loop has taken it (or the id is unknown) there is nothing to remove."""
    q = SteeringQueue()
    a = q.put("a")
    q.ready(a.id, "A")
    assert [m.raw for m in q.take_ready()] == ["a"]
    assert q.remove(a.id) is False  # already read
    assert q.remove("no-such-id") is False


def test_take_all_includes_unready():
    """REQ-1.4: leftovers at turn end (D8) take everything, ready or not, in order."""
    q = SteeringQueue()
    a, b, c = q.put("a"), q.put("b"), q.put("c")
    q.ready(a.id, "A")  # b never finished preparing
    assert [m.raw for m in q.take_all()] == ["a", "b", "c"]
    assert q.pending() == [] and q.take_all() == []


def test_has_ready_or_preparing():
    """REQ-1.5: D7's check — any pending entry keeps the turn going, ready or not."""
    q = SteeringQueue()
    assert q.has_ready_or_preparing() is False
    m = q.put("a")
    assert q.has_ready_or_preparing() is True
    q.ready(m.id, "A")
    assert q.has_ready_or_preparing() is True  # delivered at the next boundary, not yet taken
    q.take_ready()
    assert q.has_ready_or_preparing() is False
    n = q.put("b")
    q.fail(n.id, "boom")
    assert q.has_ready_or_preparing() is True
    q.take_all()
    assert q.has_ready_or_preparing() is False


# --- T1.3 light import ---------------------------------------------------------------------

def test_steering_import_is_light():
    """REQ-1.6: importing the queue must not drag in litellm / chromadb / mcp (CON-010's neighbours)."""
    code = ("import sys; import dostuff.helpers.agent.steering; "
            "print([m for m in ('litellm', 'chromadb', 'mcp') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip() == "[]"


def test_steer_message_shape():
    """D4's dataclass: the fields the TUI and the loop read."""
    m = SteerMessage(id="abc", raw="@file.md look")
    assert (m.id, m.raw, m.content, m.failed) == ("abc", "@file.md look", None, None)
