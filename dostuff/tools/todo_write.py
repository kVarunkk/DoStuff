from typing import Literal

# pydantic on Python < 3.12 rejects typing.TypedDict when building tool schemas
from typing_extensions import TypedDict

TodoStatus = Literal["pending", "in_progress", "completed"]
_MARKS: dict[str, str] = {"pending": "☐", "in_progress": "◐", "completed": "☑"}

# session_id -> the latest list; the model also sees each list in its own tool-call history
_todos: dict[str, list[dict[str, str]]] = {}


class TodoItem(TypedDict):
    content: str
    status: TodoStatus


def get_todos(session_id: str) -> list[dict[str, str]]:
    return list(_todos.get(session_id, []))


def format_todos(todos: list[dict[str, str]]) -> str:
    if not todos:
        return "Todo list cleared."
    return "\n".join(f"{_MARKS.get(t['status'], '☐')} {t['content']}" for t in todos)


def todo_write(todos: list[TodoItem], session_id: str) -> str:
    """Replaces the task list for the current work and shows it to the user. Use it for tasks
    with 3 or more steps: write the plan first, keep exactly one item in_progress while you work
    on it, and mark items completed as soon as they are done. Send the whole list every time.

    Args:
        todos: The full, updated list. Each item has `content` (a short imperative step, e.g.
            "Add tests for the parser") and `status` (pending, in_progress or completed).
            An empty list clears it.

    Returns:
        The list as shown to the user, plus a note if it breaks the one-in-progress rule.
    """
    # Arguments arrive as raw JSON: tolerate a missing or unknown status
    cleaned = [
        {"content": str(t["content"]).strip(), "status": t.get("status") if t.get("status") in _MARKS else "pending"}
        for t in todos
        if isinstance(t, dict) and str(t.get("content", "")).strip()
    ]
    _todos[session_id] = cleaned
    text = format_todos(cleaned)
    in_progress = sum(t["status"] == "in_progress" for t in cleaned)
    if in_progress > 1:
        text += f"\n(note: {in_progress} items are in_progress; keep exactly one)"
    return text
