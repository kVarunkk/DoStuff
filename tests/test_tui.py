"""Headless TUI smoke test: async turns, streaming, approval prompt, /clear, exit (REQ-4.x)."""
import asyncio
from pathlib import Path

import pytest

pytest.importorskip("textual")


@pytest.fixture
def tui(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".dostuff").mkdir(parents=True)
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(proj)
    monkeypatch.setenv("MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    from dostuff import config as config_module
    config_module.reset_config()
    from dostuff import cli_tui
    from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore
    from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore

    async def no_memories(self, *args, **kwargs):
        return []
    monkeypatch.setattr(SemanticMemoryStore, "query", no_memories)
    monkeypatch.setattr(EpisodicMemoryStore, "query", no_memories)
    monkeypatch.setattr(cli_tui, "load_mcp_config", lambda: {})
    from dostuff.helpers.skills import discover_skills
    monkeypatch.setattr(discover_skills, "SKILLS_DIRS", [".dostuff/skills"])  # not the real ~/.agents/skills
    from dostuff.lib.memory import chroma
    monkeypatch.setattr(chroma, "embedding_model_missing", lambda: False)  # no 80 MB download in tests
    monkeypatch.setattr(cli_tui, "get_model_token_limit", lambda: 100_000)
    # A job ending while idle would start a turn of the fake loop below; the wake-up tests turn it on
    config_module.get_config().jobs_wake_on_end = False

    async def fake_loop(session_id, turn_id, user_text, instructions, mcp_client, working_history, turn_type,
                        current_session_history, steps_history, store, adapter=None, **kwargs):
        assert "<environment>" in instructions
        answer = await adapter.ask("Allow write_file?")
        text = f"answer to {user_text} ({answer})"
        adapter.emit("partial_thought", "thinking...")
        adapter.emit("partial", text[:5])
        adapter.emit("partial", text)
        adapter.emit("stream_end", None)
        adapter.emit("usage", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        step = {"role": "assistant", "content": text}
        steps_history.append(step)
        current_session_history.append(step)
        await store.append(session_id, step)
        return text
    monkeypatch.setattr(cli_tui, "agent_loop", fake_loop)
    yield cli_tui
    config_module.reset_config()


def _texts(app):
    return [str(getattr(w, "content", getattr(w, "renderable", ""))) for w in app.query("#msg-container Static")]


def test_turn_clear_and_exit(tui):
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="smoke")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("hello"))
            for _ in range(50):
                await pilot.pause(0.05)
                if app.adapter.has_pending_confirm():
                    break
            assert app.adapter.has_pending_confirm()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("y"))
            for _ in range(50):
                await pilot.pause(0.05)
                if not app._turn_in_progress:
                    break
            assert not app._turn_in_progress
            assert [s["role"] for s in app.steps_history] == ["user", "assistant"]
            assert await app.store.load("smoke") == app.steps_history
            # The streamed answer is not appended a second time
            texts = _texts(app)
            assert sum("answer to hello" in t for t in texts) <= 1

            # /clear starts a new session; the old one stays in the DB
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/clear"))
            for _ in range(20):
                await pilot.pause(0.05)
                if not app.steps_history:
                    break
            assert app.steps_history == [] and app.current_session_history == []
            assert app.session_id != "smoke"
            assert len(await app.store.load("smoke")) == 2
            assert app.adapter._session_usage["total_tokens"] == 0

            await app._cleanup_and_save_session()

    asyncio.run(scenario())


async def _until(pilot, cond, tries=60):
    for _ in range(tries):
        await pilot.pause(0.05)
        if cond():
            return
    raise AssertionError("timed out")


def test_resume_picker_and_session_commands(tui, tmp_path):
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="first")
        async with app.run_test() as pilot:
            await pilot.pause()
            submit = lambda text: app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))  # noqa: E731
            # No DB row until the first message
            assert await app.store.get_session_meta("first") is None

            submit("hello")
            await _until(pilot, app.adapter.has_pending_confirm)
            submit("y")
            await _until(pilot, lambda: not app._turn_in_progress)
            assert await app.store.get_session_meta("first") is not None

            submit("/rename Tiger work")
            await _until(pilot, lambda: any("renamed" in t for t in _texts(app)))
            assert (await app.store.get_session_meta("first"))["title"] == "Tiger work"

            submit("/export notes.md")
            await _until(pilot, lambda: (Path.cwd() / "notes.md").exists())
            exported = (Path.cwd() / "notes.md").read_text(encoding="utf-8")
            assert exported.startswith("# Tiger work") and "## User\n\nhello" in exported

            submit("/clear")
            await _until(pilot, lambda: app.session_id != "first")
            new_id = app.session_id

            # /resume shows this directory's other sessions; Enter switches back
            submit("/resume")
            await _until(pilot, app._picker_open)
            assert not app.query_one("#input", cli_tui.EnterSubmits).display
            await pilot.press("enter")
            await _until(pilot, lambda: app.session_id == "first")
            assert [s["role"] for s in app.steps_history] == ["user", "assistant"]
            assert app.query_one("#input", cli_tui.EnterSubmits).display
            # The empty session /clear made was never saved, so there's nothing to go back to
            assert await app.store.get_session_meta(new_id) is None
            submit("/resume")
            await _until(pilot, lambda: any("No other sessions" in t for t in _texts(app)))

            # ESC closes the picker without switching
            app._show_picker("Pick", [("x", "x")], app._switch_session)
            await _until(pilot, app._picker_open)
            await pilot.press("escape")
            await _until(pilot, lambda: not app._picker_open())
            assert app.session_id == "first"

            submit("/resume nope")
            await _until(pilot, lambda: any("No session matches 'nope'" in t for t in _texts(app)))

            submit("/model anthropic/claude-sonnet-4-5")
            await _until(pilot, lambda: cli_tui.get_model() == "anthropic/claude-sonnet-4-5")
            submit("/permissions auto")
            await _until(pilot, lambda: app.permissions.mode == "auto")
            assert "auto mode" in str(app.query_one("#status", cli_tui.Static).render())
            submit("/permissions nope")
            await _until(pilot, lambda: any("Unknown permission mode 'nope'" in t for t in _texts(app)))
            submit("/permissions")
            await _until(pilot, app._picker_open)
            await pilot.press("up", "enter")
            await _until(pilot, lambda: app.permissions.mode == "allowlist")
            submit("/status")
            await _until(pilot, lambda: any("Context" in t and "allowlist" in t and "anthropic/claude-sonnet-4-5" in t for t in _texts(app)))

            # /model saves a new model to ~/.dostuff/config.yaml (created by the first one, so it's the default) and switches
            assert [m["name"] for m in app.config.configured_models()] == ["anthropic/claude-sonnet-4-5"]
            submit("/model groq/llama-3.3-70b")
            await _until(pilot, lambda: cli_tui.get_model() == "groq/llama-3.3-70b")
            assert [m["active"] for m in app.config.configured_models()] == [True, False]
            submit("/model add openai/gpt-4o --active")  # the old spelling still works
            await _until(pilot, lambda: cli_tui.get_model() == "openai/gpt-4o")
            assert [m["active"] for m in app.config.configured_models()] == [False, False, True]
            submit("/model claude-sonnet-4-5")  # a saved model by its bare name just switches
            await _until(pilot, lambda: cli_tui.get_model() == "anthropic/claude-sonnet-4-5")
            assert len(app.config.configured_models()) == 3
            submit("/model nope")
            await _until(pilot, lambda: any("Unknown model 'nope'" in t for t in _texts(app)))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_model_picker_browses_providers_and_lists_saved_models(tui, monkeypatch):
    cli_tui = tui
    known = ["anthropic/claude-haiku-5-5", "anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5"]
    monkeypatch.setattr(cli_tui, "provider_models", lambda provider: known if provider == "anthropic" else [])

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="pick")
        async with app.run_test() as pilot:
            await pilot.pause()
            submit = lambda text: app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))  # noqa: E731
            picker = lambda: app.query_one("#picker", cli_tui.PermissionPrompt)  # noqa: E731

            async def browse(provider: str) -> None:
                submit("/model")
                await _until(pilot, app._picker_open)
                await pilot.press(*f"more {provider}")
                assert [v for v, _ in picker()._options] == [f"browse:{provider}"]
                await pilot.press("enter")
                await _until(pilot, lambda: app._picker_open() and picker()._question.startswith(f"Which {provider}"))

            await browse("anthropic")
            await pilot.press("o", "p", "u", "s")  # letters filter instead of acting as shortcuts
            assert [v for v, _ in picker()._options] == ["anthropic/claude-opus-5-5"]
            await pilot.press("enter")
            await _until(pilot, lambda: cli_tui.get_model() == "anthropic/claude-opus-5-5")
            assert [m["name"] for m in app.config.configured_models()] == ["anthropic/claude-opus-5-5"]

            # No match: Enter adds what was typed, with the provider prefix
            await browse("anthropic")
            await pilot.press("z", "z", "z", "backspace")
            assert picker()._options == [] and picker()._filter == "zz"
            await pilot.press("enter")
            await _until(pilot, lambda: cli_tui.get_model() == "anthropic/zz")

            # A provider with no list still opens the picker, for typing a name
            await browse("gemini")
            await pilot.press("x", "enter")
            await _until(pilot, lambda: cli_tui.get_model() == "gemini/x")

            # Saved models come first, current one marked; picking one switches without saving again
            submit("/model")
            await _until(pilot, app._picker_open)
            values = [v for v, _ in picker()._options]
            assert values[:3] == ["anthropic/claude-opus-5-5", "anthropic/zz", "gemini/x"]
            assert "(current)" in picker()._options[2][1] and "(default)" in picker()._options[0][1]
            await pilot.press("enter")  # cursor starts on the current model
            submit("/model")
            await _until(pilot, app._picker_open)
            await pilot.press("o", "p", "u", "s", "enter")
            await _until(pilot, lambda: cli_tui.get_model() == "anthropic/claude-opus-5-5")
            assert len(app.config.configured_models()) == 3

            # Typing a full name that isn't listed adds it
            submit("/model")
            await _until(pilot, app._picker_open)
            await pilot.press(*"groq/foo", "enter")
            await _until(pilot, lambda: cli_tui.get_model() == "groq/foo")
            assert len(app.config.configured_models()) == 4
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_esc_denies_pending_prompt(tui):
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="esc")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("do it"))
            for _ in range(50):
                await pilot.pause(0.05)
                if app.adapter.has_pending_confirm():
                    break
            await pilot.press("escape")
            for _ in range(50):
                await pilot.pause(0.05)
                if not app._turn_in_progress:
                    break
            assert app.steps_history[-1]["content"].endswith("(no)")
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_slash_menu_and_shell_command(tui):
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="menu")
        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#command-menu", cli_tui.CommandMenu)
            messages_region = app.query_one("#messages").region
            await pilot.press("/")
            await pilot.pause()
            assert menu.display and menu.option_count == len(cli_tui.COMMAND_DESCRIPTIONS)
            # The menu floats over the conversation instead of resizing it (a resize repaints the screen)
            assert app.query_one("#messages").region == messages_region
            assert menu.region.height > 0 and menu.region.bottom == messages_region.bottom
            await pilot.press("c", "o", "m")
            await pilot.pause()
            assert menu.option_count == 1 and menu.selected_command() == "/compact"
            await pilot.press("tab")
            await pilot.pause()
            assert app.query_one("#input", cli_tui.EnterSubmits).text == "/compact"
            await pilot.press("escape")
            app._clear_input()

            # Ghost text: the rest of the command, then its flag; Tab accepts each
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            inp.focus()
            await pilot.press("/", "m", "o")
            await pilot.pause()
            assert inp.suggestion == "del"
            # Dimmer than typed text (the ansi theme's $text-muted is the plain foreground)
            ghost = inp.get_component_rich_style("text-area--suggestion").color
            assert ghost is not None and ghost.triplet is not None and ghost.triplet.hex == "#6a6a6a"
            await pilot.press("tab")
            await pilot.pause()
            assert inp.text == "/model " and inp.suggestion == "--active"
            await pilot.press("tab")
            await pilot.pause()
            assert inp.text == "/model --active" and inp.suggestion == ""
            app._clear_input()

            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("!echo shell-ok"))
            for _ in range(100):
                await pilot.pause(0.05)
                if app.steps_history:
                    break
            step = app.steps_history[-1]
            assert step["role"] == "user" and "<bash-input>echo shell-ok</bash-input>" in step["content"]
            assert "shell-ok" in cli_tui._parse_shell_step(step["content"])[1]
            assert any("shell-ok" in t for t in _texts(app))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_permission_selector(tui):
    cli_tui = tui
    from dostuff.agent.permissions import ApprovalReply

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="selector")
        async with app.run_test() as pilot:
            await pilot.pause()
            replies = []
            inp = app.query_one("#input", cli_tui.EnterSubmits)

            async def until(cond):
                for _ in range(50):
                    await pilot.pause(0.05)
                    if cond():
                        return
                raise AssertionError("timed out")

            async def ask():
                replies.append(await app.adapter.ask("Write a.txt", always_label="file writes"))

            # Down + Enter picks "always"; the selector replaces the input while open
            app.run_worker(ask())
            await until(lambda: app.query("#permission-prompt"))
            assert not inp.display
            await pilot.press("down", "enter")
            await until(lambda: replies)
            assert replies[-1] == ApprovalReply("always")
            assert inp.display and not app.query("#permission-prompt")

            # "No" (number key) then typed feedback goes back with the denial
            inp.text = "my draft"
            app.run_worker(ask())
            await until(lambda: app.query("#permission-prompt"))
            await pilot.press("3")
            await until(lambda: inp.has_class("feedback-mode"))
            assert inp.display and inp.text == ""
            inp.text = "use edit_file instead"
            await pilot.press("enter")
            await until(lambda: len(replies) == 2)
            assert replies[-1] == ApprovalReply("no", feedback="use edit_file instead")
            assert not inp.has_class("feedback-mode") and inp.text == "my draft"

            # One prompt for a batch; "y" allows all
            async def ask_batch():
                replies.append(await app.adapter.ask_batch(["Write a", "Write b", "Write c"], always_label="file writes"))
            app.run_worker(ask_batch())
            await until(lambda: app.query("#permission-prompt"))
            await pilot.press("y")
            await until(lambda: len(replies) == 3)
            assert replies[-1] == ApprovalReply("yes")
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_thought_chunks_typed_out_and_formatted(tui):
    cli_tui = tui
    chunk1 = "**Solving the Problem**\n\nWorking it out.\n\n\n"
    chunk2 = chunk1 + "**Confirming**\n\nDone.\n\n\n"

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="thoughts")
        async with app.run_test() as pilot:
            await pilot.pause()
            app._update_stream_widget("thought", chunk1)
            await pilot.pause(0.1)
            widget = app._stream_widgets["thought"]
            # A whole chunk is revealed over several ticks, not at once
            assert 0 < app._thought_shown < len(chunk1)
            for _ in range(100):
                await pilot.pause(0.02)
                if app._thought_shown == len(chunk1):
                    break
            assert app._thought_timer is None
            # The next chunk extends the same widget; ending the call shows it all, formatted
            app._update_stream_widget("thought", chunk2)
            assert app._stream_widgets["thought"] is widget and app._thought_timer is not None
            app._finish_stream_widgets()
            assert app._thought_timer is None
            text = widget.render()
            assert str(text) == "Solving the Problem\n\nWorking it out.\n\nConfirming\n\nDone."
            assert "**" not in str(text)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_input_newline_keys():
    from textual.app import App
    from dostuff.cli_tui import EnterSubmits

    class InputApp(App):
        submitted: list[str] = []

        def compose(self):
            yield EnterSubmits(id="input")

        def on_enter_submits_submitted(self, message):
            self.submitted.append(message.text)

    async def run():
        app = InputApp()
        async with app.run_test() as pilot:
            box = app.query_one("#input", EnterSubmits)
            box.focus()
            await pilot.press("a", "shift+enter", "b", "ctrl+enter", "c", "ctrl+j", "d", "\\", "enter", "e")
            assert box.text == "a\nb\nc\nd\ne"
            assert app.submitted == []
            await pilot.press("enter")
            await pilot.pause()
            assert app.submitted == ["a\nb\nc\nd\ne"]

    asyncio.run(run())


def test_tool_results_expand_and_collapse(tui):
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="expand")
        async with app.run_test() as pilot:
            await pilot.pause()
            long_output = "\n".join(f"line {i}" for i in range(1, 21))
            app.adapter.emit("tool_result", f"read_file: {long_output}")
            app.adapter.emit("tool_result", "write_file: ok")
            await pilot.pause()
            long_result, short_result = list(app.query(cli_tui.ToolResult))
            text = str(long_result.render())
            assert "line 8" in text and "line 9" not in text
            assert "… +12 lines (ctrl+o to expand)" in text
            assert not short_result.collapsible

            await pilot.press("ctrl+o")
            assert long_result.expanded and "line 20" in str(long_result.render())
            assert not short_result.expanded
            # Results added while expanded arrive expanded
            app.adapter.emit("tool_result", f"grep: {long_output}")
            await pilot.pause()
            assert list(app.query(cli_tui.ToolResult))[-1].expanded

            await pilot.press("ctrl+o")
            assert not any(w.expanded for w in app.query(cli_tui.ToolResult))
            assert "line 9" not in str(long_result.render())

            # A click toggles just that one
            long_result.on_click()
            assert long_result.expanded
            long_result.on_click()
            assert not long_result.expanded
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_select_to_copy_and_input_keeps_focus(tui, monkeypatch):
    cli_tui = tui
    monkeypatch.setattr(cli_tui.sys, "platform", "linux")  # no clip.exe in tests
    from textual.selection import Selection

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="select")
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            app._append("**bold** answer\n\n- item one", msg_type="agent")
            await pilot.pause()
            msg = list(app.query(cli_tui.SelectableStatic))[-1]
            # Markdown is flattened to Text, so Textual can select it
            text, _ = msg.get_selection(Selection(None, None))
            assert "bold answer" in text and "item one" in text and "**" not in text

            # Releasing the mouse over a selection copies it
            app.screen.selections = {msg: Selection(None, None)}
            app.post_message(cli_tui.TextSelected())
            await pilot.pause()
            assert "bold answer" in app.clipboard

            # Clicking the conversation leaves focus in the input
            await pilot.click("#messages")
            await pilot.pause()
            assert app.focused is inp
            app.set_focus(None)
            await pilot.pause()
            assert app.focused is inp
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_prompt_history_up_down(tui, tmp_path):
    cli_tui = tui
    from dostuff.helpers.ui.prompt_history import append_history

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="history")
        append_history(app._history_path, str(tmp_path / "elsewhere"), "other project")
        append_history(app._history_path, str(Path.cwd()), "first")
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.query_one("#input", cli_tui.EnterSubmits)
            assert box.prompt_history == ["first"]  # only this directory's prompts
            app._remember_prompt("/help")
            app._remember_prompt("line one\nline two")
            menu = app.query_one("#command-menu", cli_tui.CommandMenu)

            await pilot.press("d", "r", "a", "f", "t")
            await pilot.press("up")
            assert box.text == "line one\nline two"
            await pilot.press("up")  # cursor is on the last line: move up within the prompt
            assert box.text == "line one\nline two" and box.cursor_location[0] == 0
            await pilot.press("up")
            await pilot.pause()
            assert box.text == "/help" and not menu.display  # a recalled command doesn't open the menu
            await pilot.press("up", "up")  # stops at the oldest
            assert box.text == "first"
            await pilot.press("down", "down")
            assert box.text == "line one\nline two"
            await pilot.press("down")
            assert box.text == "draft"
            await pilot.press("down")
            assert box.text == "draft"

            # Saved for the next run in this directory
            app2_history = cli_tui.load_history(app._history_path, str(Path.cwd()))
            assert app2_history == ["first", "/help", "line one\nline two"]
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_tui_import_skips_heavy_libraries():
    """litellm / chromadb / mcp load after the first frame (DostuffTUI._load_backend)."""
    import subprocess
    import sys
    code = "import sys, dostuff.cli, dostuff.cli_tui; print(sorted(m for m in ('litellm', 'chromadb', 'mcp') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.stdout.strip() == "[]"


def test_cost_in_status_bar_and_status(tui, tmp_path):
    cli_tui = tui
    from dostuff import config as config_module
    (tmp_path / "home" / ".dostuff" / "config.yaml").write_text("ui:\n  status_bar: [mode, cost]\n")
    config_module.reset_config()

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="cost")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            status = app.query_one("#status", cli_tui.Static)
            assert "$" not in str(status.render())  # nothing priced yet
            app.adapter.emit("usage", {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100, "cost": 0.0123})
            app.adapter.emit("usage", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": None})
            await pilot.pause()
            assert "$0.01" in str(status.render())
            assert app.adapter._turn_cost == 0.0123
            await app._show_status()
            assert any("Cost" in t and "$0.01" in t for t in _texts(app))
            assert any("Status bar" in t and "ui.status_bar" in t for t in _texts(app))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_episodes_and_forget_commands(tui):
    cli_tui = tui

    class FakeFacts:
        def __init__(self):
            self.keys = {"editor"}

        async def list_all(self, user_id):
            return [{"key": k, "value": "Uses VS Code", "source": "user_message"} for k in sorted(self.keys)]

        async def delete(self, user_id, key):
            if key not in self.keys:
                return False
            self.keys.discard(key)
            return True

    class FakeEpisodes:
        def __init__(self):
            self.episodes = [
                {"id": "ep_s2_bbbb2222", "event_type": "LEARNING", "anchor_event": "Retry 429s", "created_at_iso": "2026-10-20T00:00:00Z"},
                {"id": "ep_s1_aaaa1111", "event_type": "DECISION", "anchor_event": "Use WAL", "created_at_iso": "2026-10-01T00:00:00Z"},
            ]
            self.listed_dirs = []

        async def list_all(self, user_id, working_dir=None):
            self.listed_dirs.append(working_dir)
            return list(self.episodes)

        async def delete(self, user_id, episode_id):
            self.episodes = [ep for ep in self.episodes if ep["id"] != episode_id]
            return True

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="episodes")
        async with app.run_test() as pilot:
            await app._backend_ready.wait()
            app.memory_store, app.episodic_store = FakeFacts(), FakeEpisodes()
            await app._show_episodes()
            listing = next(t for t in _texts(app) if t.startswith("Saved episodes"))
            assert "for this directory (2)" in listing and "[LEARNING] Retry 429s  (2026-10-20, bbbb2222)" in listing
            assert listing.index("Retry 429s") < listing.index("Use WAL")
            assert app.episodic_store.listed_dirs == [str(Path.cwd().resolve())]  # episodic_scope: project

            await app._forget("aaaa1111")
            assert "Forgot episode [DECISION] Use WAL" in _texts(app)
            await app._forget("editor")
            assert "Forgot memory 'editor'." in _texts(app) and not app.memory_store.keys
            await app._forget("nope")
            assert any("No memory key or episode id 'nope'" in t for t in _texts(app))

            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/forget"))
            await pilot.pause()
            assert any(t.startswith("Usage: /forget") for t in _texts(app))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_embedding_model_download_reported(tui, monkeypatch):
    cli_tui = tui
    from dostuff.lib.memory import chroma
    monkeypatch.setattr(chroma, "embedding_model_missing", lambda: True)

    def offline():
        raise OSError("no network")
    monkeypatch.setattr(chroma, "download_embedding_model", offline)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="embed")
        async with app.run_test() as pilot:
            await app._backend_ready.wait()
            await pilot.pause()
            texts = _texts(app)
            assert any("Downloading the memory embedding model" in t for t in texts)
            assert any("Embedding model download failed: no network" in t for t in texts)
            assert not app._backend_error  # turns still run, just without recall
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_file_mention_completion_and_attachment(tui, monkeypatch):
    cli_tui = tui
    from dostuff.helpers.agent import file_mentions
    monkeypatch.setattr(file_mentions, "_index", {})
    Path("src").mkdir()
    Path("src", "main.py").write_text("print('hi')\n", encoding="utf-8")

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="mention")
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.query_one("#input", cli_tui.EnterSubmits)
            menu = app.query_one("#command-menu", cli_tui.CommandMenu)
            await pilot.press(*"fix @ma")
            await pilot.pause()
            assert menu.display and menu.mode == "file" and menu.selected_command() == "src/main.py"
            await pilot.press("enter")  # completes the path, doesn't submit
            assert box.text == "fix @src/main.py " and not menu.display
            await pilot.press(*"now")
            await pilot.pause()
            assert not menu.display

            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(box.text))
            for _ in range(50):
                await pilot.pause(0.05)
                if app.adapter.has_pending_confirm():
                    break
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("y"))
            for _ in range(50):
                await pilot.pause(0.05)
                if not app._turn_in_progress:
                    break
            sent = app.steps_history[0]["content"]
            assert sent.startswith("fix @src/main.py now\n\n<attached_file path=\"src/main.py\">")
            assert "print('hi')" in sent
            assert any(t == "Attached src/main.py" for t in _texts(app))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_status_bar_fits_width_and_follows_config(tui, tmp_path):
    cli_tui = tui
    from dostuff import config as config_module

    async def scenario(size):
        app = cli_tui.DostuffTUI(session_id="narrow-session")
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            app.permissions.mode = "auto"
            app._update_status()
            text = str(app.query_one("#status", cli_tui.Static).render())
            await app._cleanup_and_save_session()
            return text, _texts(app)

    # Lean by default: no help / session id / tokens / timer
    default, _ = asyncio.run(scenario((160, 24)))
    assert "openai/gpt-4o-mini" in default and "auto mode" in default
    assert "/help" not in default and "narrow-session" not in default and "↑" not in default

    config_path = tmp_path / "home" / ".dostuff" / "config.yaml"
    config_path.write_text("ui:\n  status_bar: [help, cwd, session, model, mode, tokens, cost, context]\n")
    config_module.reset_config()
    wide, _ = asyncio.run(scenario((160, 24)))
    assert all(s in wide for s in ("/help", "narrow-session", "openai/gpt-4o-mini", "auto mode"))

    # Narrow: the least important items go first; mode stays, nothing past the edge
    narrow, _ = asyncio.run(scenario((30, 24)))
    assert "auto mode" in narrow
    assert "/help" not in narrow and "narrow-session" not in narrow
    assert len(narrow) <= 28

    # "timer" (now in the loader) is ignored quietly in older configs
    config_path.write_text("ui:\n  status_bar: [timer, mode, bogus]\n")
    config_module.reset_config()
    picked, texts = asyncio.run(scenario((160, 24)))
    assert picked.strip() == "auto mode"
    assert any("unknown item(s) bogus" in t for t in texts)
    assert not any("timer" in t for t in texts)


def test_settings_changes_apply_without_restart(tui):
    cli_tui = tui
    from dostuff.tools import settings

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="live")
        async with app.run_test(size=(160, 24)) as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            status = app.query_one("#status", cli_tui.Static)
            connected = []

            async def fake_connect(server_name, **kwargs):
                connected.append((server_name, kwargs["command"]))
                app.mcp_client.servers[server_name] = object()
            app.mcp_client.connect_to_server = fake_connect

            await asyncio.to_thread(settings.edit_config, "", "ui:\n  status_bar: [mode]", "user", app.adapter)
            await _until(pilot, lambda: str(status.render()).strip() == "ask mode")

            await asyncio.to_thread(settings.set_mcp_server, "git", ["uvx", "mcp-server-git"], adapter=app.adapter)
            await _until(pilot, lambda: connected == [("git", "uvx")])
            # Changing a server reconnects it; removing one disconnects it
            await asyncio.to_thread(settings.set_mcp_server, "git", ["uvx", "other"], adapter=app.adapter)
            await _until(pilot, lambda: len(connected) == 2)
            await asyncio.to_thread(settings.remove_mcp_server, "git", adapter=app.adapter)
            await _until(pilot, lambda: "git" not in app.mcp_client.servers and app._mcp_configured == [])
            assert any("MCP: disconnected git" in t for t in _texts(app))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_settings_files_edited_by_hand_apply_while_idle(tui, tmp_path):
    """config.yaml and mcp_config.json written outside dostuff (an editor) apply without a prompt or restart."""
    cli_tui = tui
    config_path = tmp_path / "home" / ".dostuff" / "config.yaml"

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="hand-edit")
        async with app.run_test(size=(160, 24)) as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            status = app.query_one("#status", cli_tui.Static)
            connected = []

            async def fake_connect(server_name, **kwargs):
                connected.append(server_name)
                app.mcp_client.servers[server_name] = object()
            app.mcp_client.connect_to_server = fake_connect

            config_path.write_text("ui:\n  status_bar: [mode]\n", encoding="utf-8")
            await _until(pilot, lambda: str(status.render()).strip() == "ask mode")
            assert any("config.yaml changed on disk: reloaded." in t for t in _texts(app))

            # A broken edit keeps the loaded settings and is reported once
            config_path.write_text("ui: [unclosed\n", encoding="utf-8")
            await _until(pilot, lambda: any("config.yaml not reloaded" in t for t in _texts(app)))
            await pilot.pause(2.5)
            assert sum("config.yaml not reloaded" in t for t in _texts(app)) == 1
            assert str(status.render()).strip() == "ask mode"
            config_path.write_text("ui:\n  status_bar: [mode]\npermissions:\n  mode: plan\n", encoding="utf-8")
            await _until(pilot, lambda: str(status.render()).strip() == "plan mode")

            (tmp_path / "home" / ".dostuff" / "mcp_config.json").write_text(
                '{"mcpServers": {"git": {"command": "uvx", "args": ["mcp-server-git"]}}}', encoding="utf-8")
            await _until(pilot, lambda: connected == ["git"])
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_tool_result_preview_caps_one_long_line():
    """A one-line result (e.g. a JSON list of 800 tool names) is cut by chars, not shown whole."""
    from dostuff.cli_tui import PREVIEW_CHARS_PER_LINE, _preview, _preview_clipped

    long_line = "x" * 20_000
    preview = _preview(long_line, 8)
    budget = 8 * PREVIEW_CHARS_PER_LINE
    assert preview.splitlines() == ["x" * budget, f"… +{20_000 - budget} chars (ctrl+o to expand)"]
    assert _preview_clipped(long_line, 8)
    assert _preview(long_line, len(long_line)) == long_line  # expanded shows everything
    assert not _preview_clipped("short\nresult", 8)
    assert _preview("\n".join(map(str, range(20))), 8).endswith("… +12 lines (ctrl+o to expand)")


def test_worktree_enter_exit_and_quit(tui, tmp_path):
    import subprocess
    cli_tui = tui
    proj = Path.cwd()

    def git(*args, cwd=proj):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (proj / "a.txt").write_text("a\n")
    git("add", ".")
    git("commit", "-qm", "init")

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="wt")
        async with app.run_test() as pilot:
            await pilot.pause()
            submit = lambda text: app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))  # noqa: E731
            submit("/worktree exit")
            await _until(pilot, lambda: any("Not in a worktree" in t for t in _texts(app)))

            # Unchanged: /worktree exit removes it
            submit("/worktree one")
            await _until(pilot, lambda: app.worktree is not None)
            wt_path = proj.resolve() / ".dostuff" / "worktrees" / "one"
            assert Path.cwd().resolve() == wt_path
            submit("/worktree exit")
            await _until(pilot, lambda: app.worktree is None and any("removed worktree 'one'" in t for t in _texts(app)))
            assert Path.cwd().resolve() == proj.resolve() and not wt_path.exists()

            # Changed: asks keep/remove
            submit("/worktree two")
            await _until(pilot, lambda: app.worktree is not None)
            (Path.cwd() / "a.txt").write_text("changed\n")
            submit("/worktree exit")
            await _until(pilot, app._picker_open)
            await app._picker_callback("keep")
            assert app.worktree is None and (proj / ".dostuff" / "worktrees" / "two" / "a.txt").exists()

            # Quitting from an unchanged worktree removes it
            submit("/worktree three")
            await _until(pilot, lambda: app.worktree is not None)
            await app._cleanup_and_save_session()
            assert "Removed worktree 'three'" in app.exit_note
            assert not (proj / ".dostuff" / "worktrees" / "three").exists()
            assert Path.cwd().resolve() == proj.resolve()

    asyncio.run(scenario())


def test_turn_summary_timer_format_and_jump_to_bottom(tui):
    cli_tui = tui
    assert [cli_tui._fmt_elapsed(s) for s in (0.4, 45, 65, 3725)] == ["0s", "45s", "1m 5s", "1h 2m"]

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="summary")
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            app.adapter._turn_cost = 0.0123
            app._append_turn_summary(65)
            await pilot.pause()
            assert any(t.startswith("✻ Worked for 1m 5s · $0.01 · done ") for t in _texts(app))

            # Scrolling up shows the bar; clicking it goes back to the bottom and hides it
            jump = app.query_one("#jump-bottom", cli_tui.JumpToBottom)
            for i in range(40):
                app._append(f"line {i}", msg_type="system")
            await pilot.pause()
            messages = app.query_one("#messages", cli_tui.VerticalScroll)
            assert not jump.display
            messages.scroll_home(animate=False)
            await _until(pilot, lambda: jump.display)
            await pilot.click("#jump-bottom")
            await _until(pilot, lambda: messages.is_vertical_scroll_end and not jump.display)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_big_paste_collapses_and_spills_to_file(tui):
    cli_tui = tui
    from textual.events import Paste

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="pastesess")
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            body = "\r\n".join(f"log line {i}" for i in range(1, 501))
            inp.insert("see ")
            app.post_message(Paste(body))  # the driver sends pastes to the app, which forwards them
            await pilot.pause()
            assert inp.text == "see [Pasted text #1 +500 lines]"
            app.post_message(Paste("short"))
            await pilot.pause()
            assert inp.text.endswith("lines]short")  # small pastes go in as text
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(inp.text))
            await _until(pilot, app.adapter.has_pending_confirm)
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("y"))
            await _until(pilot, lambda: not app._turn_in_progress)
            user = app.steps_history[0]["content"]
            assert '<pasted_content file=".dostuff/pastes/pasteses-1.txt" lines="500"' in user
            assert "log line 20\n" in user and "log line 21" not in user
            saved = Path(".dostuff/pastes/pasteses-1.txt").read_text(encoding="utf-8")
            assert saved == body.replace("\r\n", "\n")
            assert any("[Pasted text #1 +500 lines]" in t for t in _texts(app))  # the screen keeps the placeholder
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_skill_slash_command_runs_skill(tui):
    cli_tui = tui
    skill_dir = Path(".dostuff", "skills", "release-notes")
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: release-notes\ndescription: Write release notes from git log\n---\n\nRead git log and summarise.\n",
        encoding="utf-8")
    Path("CHANGES.txt").write_text("v1 fixed things\n", encoding="utf-8")

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="skill")
        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#command-menu", cli_tui.CommandMenu)
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            await pilot.press(*"/rel")
            await pilot.pause()
            assert menu.display and menu.selected_command() == "/release-notes"
            assert inp.suggestion == "ease-notes"
            app._clear_input()

            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/help"))
            assert any("/release-notes" in t and "Skills" in t for t in _texts(app))

            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/release-notes"))
            await _until(pilot, app.adapter.has_pending_confirm)
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("y"))
            await _until(pilot, lambda: not app._turn_in_progress)
            sent = app.steps_history[0]["content"]
            assert "Read git log and summarise." in sent and "name: release-notes" not in sent
            assert sent.endswith("(no arguments)")
            assert not any("Unknown command" in t for t in _texts(app))

            # The arguments keep their @mentions
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/release-notes for @CHANGES.txt"))
            await _until(pilot, app.adapter.has_pending_confirm)
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("y"))
            await _until(pilot, lambda: not app._turn_in_progress)
            sent = app.steps_history[2]["content"]
            assert "Arguments: for @CHANGES.txt" in sent and "v1 fixed things" in sent
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_config_command_edits_settings_file(tui, tmp_path):
    cli_tui = tui
    config_path = tmp_path / "home" / ".dostuff" / "config.yaml"
    config_path.write_text("ui:\n  status_bar: [model]\n", encoding="utf-8")

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="config")

        def status_text():
            return str(app.query_one("#settings-status", cli_tui.Static).render())

        async with app.run_test(size=(160, 30)) as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            status = app.query_one("#status", cli_tui.Static)

            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/config"))
            await pilot.pause()
            picker = app.query_one("#picker", cli_tui.PermissionPrompt)
            assert [v for v, _ in picker._options] == ["config:user", "config:project", "mcp:user", "mcp:project"]
            await pilot.press("enter")
            await _until(pilot, app._settings_open)
            editor = app.query_one("#settings-editor", cli_tui.SettingsEditor)
            assert app.screen.focused is editor and editor.text == "ui:\n  status_bar: [model]\n"
            assert not app.query_one("#conversation").display and not app.query_one("#input").display

            # Invalid YAML: nothing written, the editor stays open with the error
            editor.text = "ui: [unclosed\n"
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert app._settings_open() and "Not saved" in status_text()
            assert config_path.read_text(encoding="utf-8") == "ui:\n  status_bar: [model]\n"

            # Valid: written, applied right away, editor closes
            editor.text = "ui:\n  status_bar: [mode]\n"
            await pilot.press("ctrl+s")
            await _until(pilot, lambda: not app._settings_open())
            assert config_path.read_text(encoding="utf-8") == "ui:\n  status_bar: [mode]\n"
            await _until(pilot, lambda: str(status.render()).strip() == "ask mode")
            assert any("Saved ~" in t and "config.yaml" in t for t in _texts(app))
            assert app.query_one("#input").display and app.query_one("#conversation").display

            # Unsaved changes: the first ESC warns, the second discards them
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/config mcp project"))
            await _until(pilot, app._settings_open)
            editor = app.query_one("#settings-editor", cli_tui.SettingsEditor)
            assert '"mcpServers"' in editor.text
            editor.text = "{}"
            await pilot.press("escape")
            await pilot.pause()
            assert app._settings_open() and "Unsaved" in status_text()
            await pilot.press("escape")
            await _until(pilot, lambda: not app._settings_open())
            assert not Path(".dostuff", "mcp_config.json").exists()

            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/config bogus"))
            assert any("Usage: /config" in t for t in _texts(app))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_roadmap_021_input_and_commands(tui, monkeypatch):
    """Shift+Tab mode cycle, Ctrl+Backspace / Ctrl+Delete, /copy, /compact auto, /notifications, title state."""
    cli_tui = tui
    copied = []
    monkeypatch.setattr(cli_tui, "_copy_to_clipboard", lambda app, text: copied.append(text))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="r021")
        async with app.run_test(size=(160, 30)) as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            inp = app.query_one("#input", cli_tui.EnterSubmits)

            # Shift+Tab: ask → auto → plan → allowlist → ask, shown in the input border
            modes = []
            for _ in range(4):
                await pilot.press("shift+tab")
                modes.append(app.permissions.mode)
            assert modes == ["auto", "plan", "allowlist", "ask"]
            await pilot.press("shift+tab", "shift+tab")
            assert app.permissions.mode == "plan" and "plan mode" in str(inp.border_title)

            # Ctrl+Backspace as Windows terminals send it (^H) deletes a word; elsewhere ^H stays Backspace.
            # (Pilot sends "backspace" with the character ^H.)
            inp.text = "hello big world"
            inp.move_cursor(inp.document.end)
            monkeypatch.setattr(cli_tui, "CTRL_H_DELETES_WORD", True)
            await pilot.press("backspace")
            assert inp.text == "hello big "
            monkeypatch.setattr(cli_tui, "CTRL_H_DELETES_WORD", False)
            await pilot.press("backspace")
            assert inp.text == "hello big"
            inp.move_cursor((0, 0))
            await pilot.press("ctrl+delete")
            assert inp.text == " big"
            inp.text = ""

            # /copy before and after an answer; the turn sets the title to done
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/copy"))
            await pilot.pause()
            assert copied == [] and any("No answer to copy" in t for t in _texts(app))
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("hi"))
            await _until(pilot, app.adapter.has_pending_confirm)
            assert app._title_state == "waiting"
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("y"))
            await _until(pilot, lambda: not app._turn_in_progress)
            assert app._title_state == "done"
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/copy"))
            await pilot.pause()
            assert copied == ["answer to hi (yes)"]

            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/compact auto 80"))
            assert app.config.compact_threshold == 80
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/compact auto off"))
            assert app.config.compact_threshold is None
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/compact auto 500"))
            assert app.config.compact_threshold is None

            assert app._notify_kind == "off"
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/notifications on"))
            assert app._notify_kind == "bell"
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/notifications osc9"))
            assert app._notify_kind == "osc9"
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/notifications loud"))
            assert app._notify_kind == "osc9" and any("Usage: /notifications" in t for t in _texts(app))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_roadmap_021_commands_memory_context_doctor_history(tui, tmp_path, monkeypatch):
    """Custom commands, `# note`, /memory, /context, /doctor, Esc Esc and Ctrl+R."""
    cli_tui = tui
    monkeypatch.setattr(cli_tui, "count_tokens", lambda text, model: len(text) // 4)
    commands = Path(".dostuff", "commands")
    commands.mkdir(parents=True)
    (commands / "review.md").write_text("---\ndescription: Review a file\n---\nReview $ARGUMENTS for bugs.\n",
                                        encoding="utf-8")
    user_commands = tmp_path / "home" / ".dostuff" / "commands"
    user_commands.mkdir(parents=True)
    (user_commands / "review.md").write_text("User review (hidden by the project one)\n", encoding="utf-8")
    (user_commands / "standup.md").write_text("Summarise yesterday's commits.\n", encoding="utf-8")

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="r021b")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            submit = lambda text: app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))  # noqa: E731
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            menu = app.query_one("#command-menu", cli_tui.CommandMenu)

            # Custom commands: in the menu, project wins over user, $ARGUMENTS filled in
            await pilot.press(*"/rev")
            await pilot.pause()
            assert menu.display and menu.selected_command() == "/review"
            assert menu.commands["/review"] == "(project command) Review a file"
            assert menu.commands["/standup"].startswith("(user command)")
            app._clear_input()
            submit("/review src/app.py")
            await _until(pilot, app.adapter.has_pending_confirm)
            submit("y")
            await _until(pilot, lambda: not app._turn_in_progress)
            assert app.steps_history[0]["content"] == "Review src/app.py for bugs."
            assert any("Running command review" in t for t in _texts(app))

            # Esc Esc with an empty input takes the prompt back as typed and drops the turn
            await pilot.press("escape")
            assert "Esc again" in str(inp.border_title)
            await pilot.press("escape")
            await _until(pilot, lambda: not app.steps_history)
            assert inp.text == "/review src/app.py"
            assert await app.store.load("r021b") == [] and await app.store.load_working_history("r021b") == []
            app._clear_input()

            # `# note` asks project or global, then appends to that AGENTS.md
            submit("# always use pytest")
            await pilot.pause()
            picker = app.query_one("#picker", cli_tui.PermissionPrompt)
            assert [v for v, _ in picker._options] == ["project", "global"]
            await pilot.press("enter")
            await _until(pilot, lambda: Path("AGENTS.md").exists())
            assert Path("AGENTS.md").read_text(encoding="utf-8") == "- always use pytest\n"
            await _until(pilot, lambda: "always use pytest" in app._system_instructions)
            submit("# second note")
            await pilot.pause()
            await pilot.press("down", "enter")
            global_md = tmp_path / "home" / ".dostuff" / "AGENTS.md"
            await _until(pilot, global_md.exists)
            assert global_md.read_text(encoding="utf-8") == "- second note\n"

            # /memory opens AGENTS.md in the editor; Ctrl+S saves it
            submit("/memory project")
            await _until(pilot, app._settings_open)
            editor = app.query_one("#settings-editor", cli_tui.SettingsEditor)
            assert editor.text == "- always use pytest\n"
            editor.text = "- always use pytest\n- run ruff\n"
            await pilot.press("ctrl+s")
            await _until(pilot, lambda: not app._settings_open())
            assert Path("AGENTS.md").read_text(encoding="utf-8").endswith("- run ruff\n")
            assert "run ruff" in app._system_instructions
            submit("/memory bogus")
            assert any("Usage: /memory" in t for t in _texts(app))

            # /context and /doctor
            submit("/context")
            await _until(pilot, lambda: any("Project instructions" in t for t in _texts(app)))
            context = next(t for t in _texts(app) if "Project instructions" in t)
            assert "System prompt" in context and "Tools" in context and "Free" in context
            submit("/doctor")
            await _until(pilot, lambda: any("Global config" in t for t in _texts(app)))

            # Ctrl+R: pick a past prompt (type to filter)
            inp.prompt_history = ["fix the tests", "write docs", "fix the build"]
            await pilot.press("ctrl+r")
            await pilot.pause()
            picker = app.query_one("#picker", cli_tui.PermissionPrompt)
            assert [label for _, label in picker._options] == ["fix the build", "write docs", "fix the tests"]
            await pilot.press(*"tests")
            await pilot.press("enter")
            await _until(pilot, lambda: inp.text == "fix the tests")
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- hooks (plan/2026-10-29-hooks, T3.1 / T3.4) ------------------------------------------------

def _write_hooks(hooks: dict) -> None:
    import yaml
    from dostuff import config as config_module
    (Path.home() / ".dostuff" / "config.yaml").write_text(yaml.safe_dump({"hooks": hooks}), encoding="utf-8")
    config_module.reset_config()


def _recording_loop(cli_tui, monkeypatch):
    calls: list[dict] = []

    async def fake_loop(session_id, turn_id, user_text, instructions, mcp_client, working_history, turn_type,
                        current_session_history, steps_history, store, adapter=None, **kwargs):
        calls.append({"instructions": instructions, "history": list(working_history),
                      "session_hook_context": dict(kwargs.get("session_hook_context") or {})})
        step = {"role": "assistant", "content": f"answer to {user_text}"}
        steps_history.append(step)
        current_session_history.append(step)
        await store.append(session_id, step)
        return step["content"]
    monkeypatch.setattr(cli_tui, "agent_loop", fake_loop)
    return calls


def test_tui_prompt_blocked_by_hook(tui, monkeypatch, hook_cmd):
    cli_tui = tui
    calls = _recording_loop(cli_tui, monkeypatch)
    blocker = hook_cmd("import sys; sys.stderr.write('not in this repo please'); sys.exit(2)")
    _write_hooks({"UserPromptSubmit": [{"hooks": [{"type": "command", "command": blocker}]}]})

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="blocked")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("push it"))
            await _until(pilot, lambda: any("not in this repo please" in t for t in _texts(app)), tries=200)
            await _until(pilot, lambda: not app._turn_in_progress)
            assert calls == [] and app.steps_history == []
            assert await app.store.load("blocked") == []
            assert any(t.startswith("Prompt blocked by a UserPromptSubmit hook") for t in _texts(app))

            # a context hook: the context reaches the turn's instructions, not the history
            ctx = hook_cmd("print('team rules: be brief')")
            _write_hooks({"UserPromptSubmit": [{"hooks": [{"type": "command", "command": ctx}]}]})
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("hello"))
            await _until(pilot, lambda: bool(calls) and not app._turn_in_progress, tries=200)
            assert "team rules: be brief" in calls[0]["instructions"]
            assert calls[0]["history"] == [{"role": "user", "content": "hello"}]
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_tui_session_start_sources(tui, monkeypatch, hook_cmd, tmp_path):
    cli_tui = tui
    calls = _recording_loop(cli_tui, monkeypatch)
    log = tmp_path / "sources.log"
    logger = hook_cmd(f"""
        import json, sys
        source = json.load(sys.stdin)["source"]
        open({str(log)!r}, "a").write(source + "\\n")
        print("ctx-" + source)
    """)
    _write_hooks({"SessionStart": [{"hooks": [{"type": "command", "command": logger}]}]})

    def sources():
        return log.read_text().split() if log.exists() else []

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="first")
        async with app.run_test() as pilot:
            await pilot.pause()

            async def turn(text):
                n = len(calls)
                app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))
                await _until(pilot, lambda: len(calls) > n and not app._turn_in_progress, tries=200)

            await turn("one")
            await turn("two")
            assert sources() == ["startup"]  # once per session, not per prompt
            assert calls[0]["session_hook_context"] == calls[1]["session_hook_context"] == {"startup": "ctx-startup"}

            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/clear"))
            await _until(pilot, lambda: app.session_id != "first")
            assert app.session_hook_context == {}
            await turn("three")
            assert sources() == ["startup", "clear"]
            assert calls[-1]["session_hook_context"] == {"clear": "ctx-clear"}

            await app._switch_session("first")
            await turn("four")
            assert sources() == ["startup", "clear", "resume"]
            assert calls[-1]["session_hook_context"] == {"resume": "ctx-resume"}
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_tui_reports_hook_warnings_once(tui):
    cli_tui = tui
    _write_hooks({"PreToolUse": 5, "Bogus": []})

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="hook-warnings")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            texts = _texts(app)
            await app._cleanup_and_save_session()
            return texts

    texts = asyncio.run(scenario())
    assert len([t for t in texts if "hooks.PreToolUse" in t]) == 1
    assert len([t for t in texts if "hooks.Bogus" in t]) == 1


# --- hooks (plan/2026-10-29-hooks, T5.1 / T5.2) ------------------------------------------------

def _write_project_hooks(hooks: dict) -> Path:
    import yaml
    from dostuff import config as config_module
    path = Path.cwd() / ".dostuff" / "config.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump({"hooks": hooks}), encoding="utf-8")
    config_module.reset_config()
    return path


@pytest.fixture
def fresh_trust(monkeypatch):
    from dostuff.helpers.hooks import dispatch, trust
    monkeypatch.setattr(trust, "_decisions", {})
    monkeypatch.setattr(dispatch, "_untrusted_warned", set())
    return trust


def test_hooks_command(tui, fresh_trust):
    cli_tui, trust = tui, fresh_trust

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="hooks-cmd")
        async with app.run_test() as pilot:
            await pilot.pause()
            submit = lambda text: app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))  # noqa: E731

            def listing(marker: str, count: int = 1) -> list[str]:
                return [t for t in _texts(app) if marker in t][count - 1].splitlines()

            # nothing configured: a hint and where to read about hooks
            submit("/hooks")
            await _until(pilot, lambda: any("No hooks configured" in t for t in _texts(app)))
            empty = next(t for t in _texts(app) if "No hooks configured" in t)
            assert "README" in empty and "config.example.yaml" in empty

            _write_hooks({"PreToolUse": [{
                "matcher": "bash_command", "if": "(?i)^\\s*git\\s+push", "fail_closed": True, "timeout": 10,
                "hooks": [{"type": "command", "name": "block-main-push",
                           "command": "python .dostuff/hooks/block_main_push.py"}]}]})
            project = _write_project_hooks({
                "PostToolUse": [{"matcher": "write_file|edit_file", "hooks": [
                    {"type": "command", "name": "format", "command": "python .dostuff/hooks/format_file.py"}]}],
                "PreToolUse": [{"fail_closed": True, "matcher": "bash_command", "if": "(", "hooks": [
                    {"type": "command", "name": "guard", "command": "echo ok"}]}],
            })
            submit("/hooks")
            await _until(pilot, lambda: any("block-main-push" in t for t in _texts(app)))
            lines = listing("block-main-push")
            text = "\n".join(lines)
            user_line = next(line for line in lines if "block-main-push" in line)
            for part in ("user", "matcher bash_command", "if (?i)^\\s*git\\s+push", "timeout 10s", "fail_closed",
                         "python .dostuff/hooks/block_main_push.py", str(Path.home() / ".dostuff" / "config.yaml")):
                assert part in user_line, (part, user_line)
            fmt_line = next(line for line in lines if " format " in line)
            assert "untrusted" in fmt_line and str(project) in fmt_line and "timeout 30s" in fmt_line
            assert "matcher write_file|edit_file" in fmt_line
            assert "broken" in next(line for line in lines if " guard " in line)
            # grouped per event, in the order the engine runs them
            assert lines.index("PreToolUse") < lines.index(user_line)
            guard_line = next(line for line in lines if " guard " in line)
            assert lines.index(user_line) < lines.index(guard_line) < lines.index("PostToolUse")  # user first
            assert "unsandboxed" in text and ".dostuff/hooks/" in text and "/hooks r" in text

            # trusted once the user said yes
            trust.remember(trust.trust_key(trust.project_entries(cli_tui.get_config().hooks)))
            submit("/hooks")
            await _until(pilot, lambda: sum("block-main-push" in t for t in _texts(app)) == 2)
            fmt_line = next(line for line in listing("block-main-push", 2) if " format " in line)
            assert "trusted" in fmt_line and "untrusted" not in fmt_line

            # r re-reads the config files after an on-disk edit
            (Path.home() / ".dostuff" / "config.yaml").write_text(
                "hooks:\n  Stop:\n    - hooks:\n        - {type: command, name: lint, command: echo lint}\n",
                encoding="utf-8")
            submit("/hooks r")
            await _until(pilot, lambda: any(" lint " in t for t in _texts(app)))
            assert any("Reloaded" in t for t in _texts(app))
            assert [e.name for e in cli_tui.get_config().hooks["Stop"]] == ["lint"]
            assert not any("block-main-push" in line for line in listing(" lint "))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_tui_hook_frame(tui, fresh_trust, monkeypatch, hook_cmd, tmp_path):
    """Trust prompt before the first project hook, then ⚓ lines for pre/post/stop, live and on resume."""
    import json
    from types import SimpleNamespace
    from dostuff.agent import loop as loop_module
    from dostuff.helpers.hooks import dispatch
    cli_tui, trust = tui, fresh_trust

    async def real_loop(*args, **kwargs):
        return await loop_module.loop(*args, **kwargs)
    monkeypatch.setattr(cli_tui, "agent_loop", real_loop)

    def msg(content=None, tool_calls=None):
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    raw = json.dumps({"path": "a.txt"})
    tc = SimpleNamespace(id="c1", type="function", function=SimpleNamespace(name="read_file", arguments=raw))
    tc.model_dump = lambda: {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": raw}}
    responses: list = []

    async def fake_call_agent(steps_history, system_instruction, tool_names=None, stream=False):
        return responses.pop(0), None

    async def fake_dispatch(name, args, *a, **kw):
        return "contents"
    monkeypatch.setattr(loop_module, "call_agent", fake_call_agent)
    monkeypatch.setattr(loop_module, "dispatch_tool", fake_dispatch)
    monkeypatch.setattr(loop_module, "get_model_token_limit", lambda: 100_000)

    log, counter = tmp_path / "ran.log", tmp_path / "stop.count"

    def logged(tag: str, body: str = "") -> str:
        return hook_cmd(f"open({str(log)!r}, 'a').write({tag!r} + chr(10))\n{body}")

    stop = logged("stop", f"""
import os, sys
n = int(open({str(counter)!r}).read()) if os.path.exists({str(counter)!r}) else 0
open({str(counter)!r}, "w").write(str(n + 1))
if n == 0:
    sys.stderr.write("tests are failing"); sys.exit(2)
""")
    post = logged("post", "print('lint ok')")
    _write_hooks({"PreToolUse": [{"hooks": [{"type": "command", "name": "crashy",
                                             "command": hook_cmd("import sys; sys.exit(3)")}]}]})
    project = _write_project_hooks({
        "PreToolUse": [{"matcher": "read_file", "hooks": [
            {"type": "command", "name": "pre", "command": logged("pre", "print('pre ctx')")}]}],
        "PostToolUse": [{"hooks": [{"type": "command", "name": "post", "command": post}]}],
        "Stop": [{"hooks": [{"type": "command", "name": "tests", "command": stop}]}],
    })

    def ran() -> list[str]:
        return log.read_text().split() if log.exists() else []

    async def first_turn(app, pilot):
        responses[:] = [msg(tool_calls=[tc]), msg("done"), msg("done again")]
        app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("go"))
        await _until(pilot, app.adapter.has_pending_confirm, tries=200)
        prompt = next(t for t in _texts(app) if "This project defines hooks" in t)
        assert str(project) in prompt and "PostToolUse 'post'" in prompt and post in prompt
        assert ran() == []  # asked before any project hook ran

    async def declined():
        app = cli_tui.DostuffTUI(session_id="frame-no")
        async with app.run_test() as pilot:
            await pilot.pause()
            await first_turn(app, pilot)
            app.adapter.choose("no")  # skips them; no "tell the model what to do differently" step
            await _until(pilot, lambda: not app._turn_in_progress, tries=300)
            assert any("project hooks not trusted; skipped" in t for t in _texts(app))
            await app._cleanup_and_save_session()

    async def accepted():
        app = cli_tui.DostuffTUI(session_id="frame")
        async with app.run_test() as pilot:
            await pilot.pause()
            await first_turn(app, pilot)
            app.adapter.choose("yes")
            await _until(pilot, lambda: not app._turn_in_progress, tries=300)
            assert ran() == ["pre", "post", "stop", "stop"]
            texts = _texts(app)
            anchors = [t for t in texts if t.startswith("⚓")]
            assert [a.split()[1] for a in anchors] == ["PreToolUse", "PostToolUse", "Stop"]
            assert "tests are failing" in anchors[2]
            errors = [str(getattr(w, "content", "")) for w in app.query("#msg-container Static")
                      if w.has_class("msg-error")]
            assert any("crashy" in e for e in errors)
            assert not any("[Stop hook" in t for t in texts)  # the Stop step is not shown as a user message
            assert any(str(s.get("content")).startswith("[Stop hook tests]") for s in app.steps_history)
            assert trust.is_trusted(trust.trust_key(trust.project_entries(cli_tui.get_config().hooks)))

            # after /resume the Stop step renders as a ⚓ line too
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/clear"))
            await _until(pilot, lambda: app.session_id != "frame")
            await app._switch_session("frame")
            texts = _texts(app)
            assert not any("[Stop hook" in t for t in texts)
            assert any(t.startswith("⚓ Stop") and "tests are failing" in t for t in texts)
            assert app._last_prompt_index() == 0  # Esc Esc edits the real prompt, not the Stop note
            await app._cleanup_and_save_session()

    asyncio.run(declined())
    assert ran() == [] and not (Path.home() / ".dostuff" / "trusted_hooks.json").exists()
    trust._decisions.clear()
    dispatch._untrusted_warned.clear()
    asyncio.run(accepted())


# --- background shell T3.2: `!command` runs as a `user` job (plan/2026-10-31-background-shell) ---

def test_bang_command_output_unchanged(tui, registry, monkeypatch, tmp_path, _py, _wait_dead):
    cli_tui = tui
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    pid_file = tmp_path / "pid"
    sleeper = _py(f"""
        import os, time
        open(r"{pid_file}", "w").write(str(os.getpid()))
        time.sleep(30)
    """)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="bang")
        async with app.run_test() as pilot:
            await pilot.pause()
            # Startup lines ("Running with local tools only.") must land before the snapshot
            await _until(pilot, lambda: any("local tools only" in t for t in _texts(app)), tries=200)
            before = len(_texts(app))
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("!echo hi"))
            # The loader's remove() lands a tick after the turn ends; wait for it too
            await _until(pilot, lambda: app.steps_history and not app._turn_in_progress
                         and not any(t.startswith("Running...") for t in _texts(app)), tries=200)
            # Snapshot taken with the pre-swap (pipe) run_shell
            assert _texts(app)[before:] == ["! echo hi", "  ⎿  hi"]
            assert app.steps_history[-1]["content"] == "<bash-input>echo hi</bash-input>\n<bash-output>\nhi\n</bash-output>"
            [job] = registry.jobs()
            assert job.owner == "user" and job.status == "exited"

            # Not Ctrl+B-able (OUT-002), and ESC kills it
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(f"!{sleeper}"))
            await _until(pilot, pid_file.exists, tries=200)
            assert registry.detach_foreground(app.session_id) == []
            app.adapter._cancel_event.set()
            await _until(pilot, lambda: not app._turn_in_progress, tries=200)
            assert any("Cancelled (process killed)" in t for t in _texts(app))
            assert _wait_dead(int(pid_file.read_text()), 5)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- background shell T5.5: quit, /clear and /resume stop the session's jobs ------------------

def _start_bg(registry, owner, job_id, cmd):
    return registry.start(cmd, "", owner, background=True, job_id=job_id)


def test_tui_quit_with_jobs(tui, registry, monkeypatch, _py, _alive, _wait_dead):
    cli_tui = tui
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    sleeper = _py("import time; time.sleep(30)")

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="quit")
        exited: list = []
        app.exit = lambda *a, **k: exited.append(True)  # type: ignore[method-assign]
        async with app.run_test() as pilot:
            await pilot.pause()
            job = await _start_bg(registry, "quit", "srv", sleeper)
            await pilot.press("ctrl+q")
            await _until(pilot, app.adapter.has_pending_confirm)
            assert any("Quit and stop 1 background job?" in t for t in _texts(app))
            app.adapter.choose("no")
            await _until(pilot, lambda: not app.adapter.has_pending_confirm())
            await pilot.pause(0.2)
            assert not exited and not app._quitting and _alive(job.pid)

            await pilot.press("ctrl+q")
            await _until(pilot, app.adapter.has_pending_confirm)
            app.adapter.choose("yes")
            await _until(pilot, lambda: bool(exited), tries=200)
            assert _wait_dead(job.pid, 5) and job.status == "killed"

    asyncio.run(scenario())

    async def no_jobs():
        app = cli_tui.DostuffTUI(session_id="quit2")
        exited: list = []
        app.exit = lambda *a, **k: exited.append(True)  # type: ignore[method-assign]
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+q")
            await _until(pilot, lambda: bool(exited), tries=200)  # no prompt: unchanged quit path
    asyncio.run(no_jobs())


def test_tui_clear_resume_with_jobs(tui, registry, monkeypatch, _py, _alive, _wait_dead):
    cli_tui = tui
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    sleeper = _py("import time; time.sleep(30)")

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="first")
        async with app.run_test() as pilot:
            await pilot.pause()
            submit = lambda text: app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))  # noqa: E731
            dev = await _start_bg(registry, "first", "dev", sleeper)
            watch = await _start_bg(registry, "first", "watch", sleeper)

            # /compact keeps jobs (nothing to compact here, and no prompt)
            submit("/compact")
            await pilot.pause(0.3)
            assert not app.adapter.has_pending_confirm() and _alive(dev.pid)

            # /clear: no prompt; both keep running and move to the new session
            submit("/clear")
            await _until(pilot, lambda: app.session_id != "first", tries=200)
            await pilot.pause(0.2)
            assert not app.adapter.has_pending_confirm()
            assert any("2 background jobs (dev, watch) kept running" in t for t in _texts(app))
            assert _alive(dev.pid) and _alive(watch.pid)
            cleared = app.session_id
            assert registry.get(cleared, "dev") is dev and dev.owner == cleared
            assert registry.jobs("first") == []

            # /resume <id>: same, the jobs follow into the resumed session
            await app.store.append("other", {"role": "user", "content": "x"})
            submit("/resume other")
            await _until(pilot, lambda: app.session_id == "other", tries=200)
            await pilot.pause(0.2)
            assert not app.adapter.has_pending_confirm()
            assert _alive(dev.pid) and _alive(watch.pid)
            assert {j.id for j in registry.jobs("other")} == {"dev", "watch"}

            # quit still stops them
            await app._cleanup_and_save_session()
            assert _wait_dead(dev.pid, 5) and _wait_dead(watch.pid, 5)

    asyncio.run(scenario())


# --- background shell T6.1: Ctrl+B moves foreground commands to the background ---------------

def _bash_loop(cli_tui, monkeypatch, commands, results, after=None):
    """A fake agent loop that runs each `(owner suffix, command)` as a foreground bash_command
    concurrently, emits the main call's result and stores one tool step per call."""
    from dostuff.tools.bash.bash_command import bash_command

    async def fake_loop(session_id, turn_id, user_text, instructions, mcp_client, working_history, turn_type,
                        current_session_history, steps_history, store, adapter=None, **kwargs):
        calls = [bash_command(cmd, timeout=100, owner=session_id + suffix) for suffix, cmd in commands]
        adapter.emit("tool_call", "bash_command(...)")
        for suffix, text in zip([s for s, _ in commands], await asyncio.gather(*calls)):
            results.append(text)
            if suffix:
                continue  # a subagent's call: its result goes to the subagent, not the transcript
            adapter.emit("tool_result", f"bash_command: {text}")
            step = {"role": "tool", "name": "bash_command", "tool_call_id": "c1", "content": text}
            steps_history.append(step)
            current_session_history.append(step)
            await store.append(session_id, step)
        if after is not None:
            await after(adapter)
        return "done"
    monkeypatch.setattr(cli_tui, "agent_loop", fake_loop)


def test_ctrl_b_backgrounds_call(tui, registry, monkeypatch, tmp_path, _py, _alive, _wait_dead):
    import re
    cli_tui = tui
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    go = tmp_path / "go"
    sleeper = _py(f"""
        import os, sys, time
        print("server up", flush=True)
        print("warming", file=sys.stderr, flush=True)
        while not os.path.exists(r"{go}"):
            time.sleep(0.05)
        print("bye", flush=True)
    """)
    results: list[str] = []
    _bash_loop(cli_tui, monkeypatch, [("", sleeper)], results)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ctrlb")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("start the server"))
            await _until(pilot, lambda: bool(registry.jobs("ctrlb")), tries=200)
            [job] = registry.jobs("ctrlb")
            await _until(pilot, lambda: job.out_path.read_text().strip() == "server up", tries=200)
            assert "Ctrl+B to background" in app._loader_label

            await pilot.press("ctrl+b")
            await _until(pilot, lambda: bool(results), tries=20)  # within 1 s
            [text] = results
            lines = text.splitlines()
            assert lines[0] == f"$ {sleeper}"
            assert re.fullmatch(rf'moved to background by the user after \d+s as job "bg-1" \(pid {job.pid}\)', lines[1])
            assert "stdout so far" in text and "server up" in text
            assert "stderr so far" in text and "warming" in text
            assert lines[-1] == 'Use bash_output("bg-1") to read more and kill_shell("bg-1") to stop it.'
            assert _alive(job.pid) and not job.foreground and job.status == "running"
            await _until(pilot, lambda: not app._turn_in_progress, tries=200)
            assert [s["content"] for s in app.steps_history if s.get("role") == "tool"] == [text]  # CON-007
            # Collapsed to the one line (Ctrl+O shows the whole result)
            [shown] = [t for t in _texts(app) if "moved to background" in t]
            assert shown.splitlines()[0].strip() == '⎿  moved to background as "bg-1"' and "server up" not in shown

            # The shown output is consumed: bash_output only returns what came after
            go.write_text("")
            await _until(pilot, lambda: job.ended is not None, tries=200)
            [notice] = registry.drain_notices("ctrlb")
            assert notice.id == "bg-1" and notice.status == "exited" and notice.code == 0
            assert registry.read_new(job).stdout == "bye\n"
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_ctrl_b_edge_cases(tui, registry, monkeypatch, tmp_path, _py, _alive, _wait_dead):
    cli_tui = tui
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    sleeper = _py("import time; time.sleep(30)")
    results: list[str] = []
    released = asyncio.Event()

    async def after(adapter):  # the turn keeps going until ESC
        await adapter._cancel_event.wait()
        released.set()
    _bash_loop(cli_tui, monkeypatch, [("", sleeper), ("/sub/r1", sleeper)], results, after=after)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="edge")
        async with app.run_test() as pilot:
            await pilot.pause()
            # Idle: no foreground job → no-op (the input still gets the key)
            await pilot.press("ctrl+b")
            await pilot.pause(0.1)
            assert registry.jobs() == [] and not app._turn_in_progress

            # A subagent's foreground call in the same turn is detached too
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("go"))
            await _until(pilot, lambda: len(registry.jobs()) == 2, tries=200)
            await pilot.pause(0.3)
            await pilot.press("ctrl+b")
            await _until(pilot, lambda: len(results) == 2, tries=20)
            assert all("moved to background by the user" in r for r in results)
            main, sub = registry.jobs("edge")[0], registry.jobs("edge/sub/r1")[0]
            assert "Ctrl+B" not in app._loader_label  # nothing left to background

            # ESC after Ctrl+B ends the turn but does not kill the now-background jobs
            await pilot.press("escape")
            await _until(pilot, lambda: not app._turn_in_progress, tries=200)
            assert released.is_set()
            await pilot.pause(0.3)
            assert _alive(main.pid) and _alive(sub.pid)
            assert main.status == "running" and sub.status == "running"

            # During a `!command`: Ctrl+B is a no-op (OUT-002)
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(f"!{sleeper}"))
            await _until(pilot, lambda: bool(registry.jobs("user")), tries=200)
            await pilot.press("ctrl+b")
            await pilot.pause(0.3)
            [bang] = registry.jobs("user")
            assert bang.foreground and bang.status == "running"
            app.adapter._cancel_event.set()
            await _until(pilot, lambda: bang.ended is not None, tries=200)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- background shell T6.2: /jobs, /jobs <id>, /jobs kill <id> --------------------------------

def test_jobs_command(tui, registry, monkeypatch, _py, _alive, _wait_dead):
    cli_tui = tui
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    sleeper = _py("import sys, time; print('dev ready'); print('warn', file=sys.stderr); sys.stdout.flush(); "
                  "sys.stderr.flush(); time.sleep(30)")

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="jobs")
        async with app.run_test() as pilot:
            await pilot.pause()
            submit = lambda text: app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))  # noqa: E731

            def last(marker: str) -> str:
                return [t for t in _texts(app) if marker in t][-1]

            # no jobs
            submit("/jobs")
            await _until(pilot, lambda: any("No background jobs" in t for t in _texts(app)))

            # table: id, owner, status, runtime, command — the main agent's and a subagent's
            dev = await _start_bg(registry, "jobs", "dev", sleeper)
            registry.set_role("jobs/sub/r1", "reviewer")
            lint = await _start_bg(registry, "jobs/sub/r1", "lint", "exit 3")
            await _until(pilot, lambda: lint.ended is not None, tries=100)
            await _until(pilot, lambda: dev.out_path.read_text() and dev.err_path.read_text(), tries=100)
            await _start_bg(registry, "other-session", "nope", "exit 0")  # another session's: not listed
            submit("/jobs")
            await _until(pilot, lambda: any("ID" in t and "OWNER" in t for t in _texts(app)))
            table = last("OWNER").splitlines()
            header = table[0].split()
            assert header == ["ID", "OWNER", "STATUS", "RUNTIME", "COMMAND"]
            [dev_row] = [r for r in table if r.lstrip().startswith("dev ")]
            [lint_row] = [r for r in table if r.lstrip().startswith("lint ")]
            assert "main" in dev_row and "running" in dev_row and sleeper in dev_row
            assert "subagent: reviewer" in lint_row and "exited (3)" in lint_row and "exit 3" in lint_row
            assert not any("nope" in r for r in table)

            # /jobs <id>: tails and log paths
            submit("/jobs dev")
            await _until(pilot, lambda: any(str(dev.out_path) in t for t in _texts(app)))
            detail = last(str(dev.out_path))
            assert "dev ready" in detail and "warn" in detail and str(dev.err_path) in detail
            assert registry.read_new(dev).stdout == "dev ready\n"  # looking doesn't consume the model's output

            # unknown id → hint
            submit("/jobs ghost")
            await _until(pilot, lambda: any('No job "ghost"' in t for t in _texts(app)))
            assert "/jobs" in last('No job "ghost"')
            submit("/jobs kill")
            await _until(pilot, lambda: any("Usage: /jobs" in t for t in _texts(app)))

            # /jobs kill <id>: dead, and the owner hears it was the user
            submit("/jobs kill dev")
            await _until(pilot, lambda: dev.ended is not None, tries=200)
            assert _wait_dead(dev.pid, 5) and dev.status == "killed" and dev.killed_by == "user"
            await _until(pilot, lambda: any('Killed job "dev"' in t for t in _texts(app)))
            [notice] = registry.drain_notices("jobs")
            assert notice.id == "dev" and notice.killed_by == "user"
            assert 'killed_by="user"' in jobs_module.format_notice([notice])

            # listed in the menu and /help
            assert "/jobs" in cli_tui.COMMAND_DESCRIPTIONS
            submit("/help")
            await _until(pilot, lambda: any("Commands:" in t for t in _texts(app)))
            assert "/jobs" in last("Commands:")
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- background shell T6.3: the jobs line under the input -----------------------------------------

def test_jobs_bar(tui, registry, monkeypatch, _py):
    cli_tui = tui
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    sleeper = _py("import time; time.sleep(30)")

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="bar")
        async with app.run_test(size=(160, 24)) as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            jobs_bar = app.query_one("#jobs-bar", cli_tui.JobsBar)

            def bar() -> str:
                return str(jobs_bar.render()) if jobs_bar.display else ""

            def question() -> str:
                return app.query_one("#picker", cli_tui.PermissionPrompt)._question if app._picker_open() else ""

            assert not jobs_bar.display  # nothing at zero
            app.config.status_bar_items = ["mode"]  # shown whatever ui.status_bar lists
            dev = await _start_bg(registry, "bar", "dev", sleeper)
            await _until(pilot, lambda: bar() == "⚙ 1 shell")  # no keypress needed
            status = app.query_one("#status", cli_tui.Static)
            assert "⚙" not in str(status.render())
            # on the status line, right after its items
            assert jobs_bar.region.y == status.region.y and jobs_bar.region.x == status.region.right
            lint = await _start_bg(registry, "bar/sub/r1", "lint", sleeper)  # a subagent's counts too
            await _until(pilot, lambda: bar() == "⚙ 2 shells")
            await _start_bg(registry, "elsewhere", "x", "exit 0")  # another session's doesn't

            # a click lists them: status, start time, runtime, owner, command
            await pilot.click("#jobs-bar")
            await _until(pilot, lambda: question() == "Background jobs")
            labels = [label for _, label in app.query_one("#picker", cli_tui.PermissionPrompt)._all_options]
            assert len(labels) == 2 and labels[0].startswith("dev  running  since ") and "main" in labels[0]
            assert labels[1].startswith("lint  running  since ") and sleeper[:40] in labels[1]
            # clicking a row picks it, on any of its lines (line 0 is the question; dev's label wraps)
            await pilot.click("#picker", offset=(5, 2))
            await _until(pilot, lambda: question().startswith('Job "dev"'))
            # clicking the item again closes the list, and once more opens it; Esc closes it too
            await pilot.click("#jobs-bar")
            await _until(pilot, lambda: not app._picker_open())
            await pilot.click("#jobs-bar")
            await _until(pilot, lambda: question() == "Background jobs")
            await pilot.press("escape")
            await _until(pilot, lambda: not app._picker_open())
            await pilot.click("#jobs-bar")
            await _until(pilot, lambda: question() == "Background jobs")
            # pick dev → its actions; "Stop it" kills it and the agent is told
            await pilot.press("enter")
            await _until(pilot, lambda: question().startswith('Job "dev"'))
            await pilot.press("2")
            await _until(pilot, lambda: dev.ended is not None, tries=200)
            assert dev.killed_by == "user"
            await _until(pilot, lambda: bar() == "⚙ 1 shell")

            # ↓ in an empty input opens the list too; lint's picker shows its recent output and log
            # path in a few rows, not in the conversation
            await pilot.press("down")
            await _until(pilot, lambda: question() == "Background jobs")
            await pilot.press("1")
            await _until(pilot, lambda: question().startswith('Job "lint"'))
            picker = app.query_one("#picker", cli_tui.PermissionPrompt)
            assert picker._detail[-1] == (f"Full log: {lint.out_path}", "dim")
            assert len(picker._detail) <= cli_tui.JOB_DETAIL_LINES + 1
            assert [label for _, label in picker._options] == ["Refresh output", "Stop it (the agent is told)",
                                                               "Back to the list"]
            assert not any(str(lint.out_path) in t for t in _texts(app))
            # one #picker, reused: a second one mounted while the first was still being removed
            # raised DuplicateIds (Textual removes asynchronously) and the list never opened again
            await pilot.press("1")  # Refresh output
            await _until(pilot, lambda: question().startswith('Job "lint"'))
            await pilot.press("escape")
            await _until(pilot, lambda: not app._picker_open())
            await pilot.press("down")
            await _until(pilot, lambda: question() == "Background jobs")
            assert len(app.query("#picker")) == 1 and app.query_one("#picker") is picker
            await pilot.press("escape")
            await registry.kill_all()
            await _until(pilot, lambda: not jobs_bar.display)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- background shell T6.4: notice rendering, orphan warning, notification ---------------------

def test_job_notice_rendering(tui, registry, monkeypatch, _py):
    cli_tui = tui
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    sleeper = _py("import time; time.sleep(30)")

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="notice")
        async with app.run_test() as pilot:
            await pilot.pause()
            submit = lambda text: app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))  # noqa: E731

            # live: a system line when the job ends, also while idle
            test = await _start_bg(registry, "notice", "test", "exit 1")
            await _until(pilot, lambda: any('⚙ job "test" exited (1) after' in t for t in _texts(app)), tries=100)
            dev = await _start_bg(registry, "notice", "dev", sleeper)
            await registry.kill(dev, "user")
            await _until(pilot, lambda: any('⚙ job "dev" killed by user after' in t for t in _texts(app)))
            await _start_bg(registry, "notice", "quiet", sleeper)
            await registry.kill(registry.get("notice", "quiet"), "model")  # the model knows: no line
            await registry.kill_owner("other-session", "owner_ended")
            await pilot.pause(0.3)
            assert not any('"quiet"' in t for t in _texts(app))
            assert test.ended is not None

            # resume: the stored notice step and a Ctrl+B'd call re-render as system lines
            notice_step = jobs_module.format_notice(registry.drain_notices("notice"))
            detached = ('$ npm run dev\nmoved to background by the user after 3s as job "bg-1" (pid 42)\n'
                        'stdout so far:\n  up\nUse bash_output("bg-1") to read more and kill_shell("bg-1") to stop it.')
            for step in ({"role": "user", "content": "start it"},
                         {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                          "function": {"name": "bash_command", "arguments": '{"command": "npm run dev"}'}}]},
                         {"role": "tool", "tool_call_id": "c1", "name": "bash_command", "content": detached},
                         {"role": "user", "content": notice_step}):
                await app.store.append("other", step)
            submit("/resume other")
            await _until(pilot, lambda: app.session_id == "other", tries=200)
            await pilot.pause(0.3)
            texts = _texts(app)
            idx = max(i for i, t in enumerate(texts) if "start it" in t)
            resumed = texts[idx:]
            assert any('⚙ job "test" exited (1) after' in t for t in resumed)
            assert any('⚙ job "dev" killed by user after' in t for t in resumed)
            assert not any("<background-job" in t for t in resumed)  # not shown as a user message
            assert any('moved to background as "bg-1"' in t for t in resumed)
            # and Esc Esc's "previous prompt" skips the notice
            assert app.steps_history[app._last_prompt_index()]["content"] == "start it"
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_orphan_warning(tui, jobs_home, monkeypatch, tmp_path):
    import json
    import os
    import time
    cli_tui = tui
    stale = jobs_home / "old-session"
    stale.mkdir(parents=True)
    (stale / "jobs.json").write_text(json.dumps([
        {"id": "dev", "pid": 4812, "command": "npm run dev", "started": 1.0, "ended": None},
        {"id": "t", "pid": 4813, "command": "pytest", "started": 1.0, "ended": 2.0},
    ]), encoding="utf-8")
    ancient = jobs_home / "ancient"
    ancient.mkdir()
    (ancient / "x.out").write_text("old")
    week = time.time() - 8 * 86400
    os.utime(ancient, (week, week))

    async def scenario(session_id):
        app = cli_tui.DostuffTUI(session_id=session_id)
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            await _until(pilot, lambda: not ancient.exists(), tries=60)
            await pilot.pause(0.2)
            texts = _texts(app)
            await app._cleanup_and_save_session()
            return texts

    texts = asyncio.run(scenario("fresh"))
    warnings = [t for t in texts if "earlier session may still be running" in t]
    assert len(warnings) == 1
    assert "1 background job" in warnings[0] and "npm run dev, pid 4812" in warnings[0]
    assert "pytest" not in warnings[0] and "OS tools" in warnings[0]
    assert not ancient.exists()  # old log dirs pruned at startup

    (stale / "jobs.json").write_text("[]", encoding="utf-8")
    ancient.mkdir()
    os.utime(ancient, (week, week))
    texts = asyncio.run(scenario("fresh2"))
    assert not any("earlier session" in t for t in texts)


def test_job_notice_notification(tui, registry, monkeypatch):
    cli_tui = tui
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="notify")
        written: list[str] = []
        async with app.run_test() as pilot:
            await pilot.pause()
            app._write_terminal = written.append  # type: ignore[method-assign]
            app._notify_kind = "osc9"
            app.app_focus = False
            await _start_bg(registry, "notify", "build", "exit 2")
            await _until(pilot, lambda: any('job "build" exited (2)' in w for w in written), tries=100)

            written.clear()  # focused: no notification for a job that just ended
            app.app_focus = True
            await _start_bg(registry, "notify", "build2", "exit 0")
            await _until(pilot, lambda: any('"build2"' in t for t in _texts(app)), tries=100)
            assert not any("build2" in w for w in written)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- background shell D15: a job ending while idle starts a turn ---------------------------------

def _wake_loop(cli_tui, monkeypatch, registry, calls, restart=None):
    """A fake agent loop that drains its notices like the real one and records
    `(user_text, drained ids)`; `restart()` true starts another quick background job."""
    async def fake_loop(session_id, turn_id, user_text, instructions, mcp_client, working_history, turn_type,
                        current_session_history, steps_history, store, adapter=None, **kwargs):
        calls.append((user_text, [n.id for n in registry.drain_notices(session_id)]))
        if restart is not None and restart():
            await registry.start("exit 0", "", session_id, background=True)
        return "handled"
    monkeypatch.setattr(cli_tui, "agent_loop", fake_loop)


def test_job_wake_up(tui, registry, monkeypatch, _py):
    cli_tui = tui
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    sleeper = _py("import time; time.sleep(30)")
    calls: list[tuple[str, list[str]]] = []
    _wake_loop(cli_tui, monkeypatch, registry, calls)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="wake")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            app.config.jobs_wake_on_end = True

            # idle: the job's end starts a turn whose input is the notice
            await _start_bg(registry, "wake", "test", "exit 1")
            await _until(pilot, lambda: len(calls) == 1, tries=100)
            text, ids = calls[0]
            assert text.startswith('<background-job id="test" status="exited" code="1"') and ids == ["test"]
            assert any('⚙ Resuming the agent: background job "test" ended' in t for t in _texts(app))
            await _until(pilot, lambda: not app._turn_in_progress)
            assert not any(s.get("content") == text for s in app.steps_history)  # the loop stores it, not the TUI

            # jobs that end together: one turn
            await _start_bg(registry, "wake", "a", "exit 0")
            await _start_bg(registry, "wake", "b", "exit 0")
            await _until(pilot, lambda: len(calls) == 2, tries=100)
            assert sorted(calls[1][1]) == ["a", "b"]

            # a /jobs kill alone doesn't wake it, a model kill leaves no notice; both ride along with the next one
            dev = await _start_bg(registry, "wake", "dev", sleeper)
            await registry.kill(dev, "user")
            quiet = await _start_bg(registry, "wake", "quiet", sleeper)
            await registry.kill(quiet, "model")
            await _start_bg(registry, "wake/sub/r1", "sub", "exit 0")  # a subagent's job: not ours to wake for
            await pilot.pause(1.5)
            assert len(calls) == 2
            await _start_bg(registry, "wake", "c", "exit 0")
            await _until(pilot, lambda: len(calls) == 3, tries=100)
            assert calls[2][1] == ["dev", "c"]
            registry.drain_notices("wake/sub/r1")
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_job_wake_waits_for_user(tui, registry, monkeypatch):
    cli_tui = tui
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    calls: list[tuple[str, list[str]]] = []
    _wake_loop(cli_tui, monkeypatch, registry, calls)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="later")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            inp = app.query_one("#input", cli_tui.EnterSubmits)

            # off: the notice waits for the next message
            await _start_bg(registry, "later", "off", "exit 0")
            await _until(pilot, lambda: any('"off" exited' in t for t in _texts(app)), tries=100)
            await pilot.pause(1.5)
            assert calls == []
            registry.drain_notices("later")
            app.config.jobs_wake_on_end = True

            # typing: wait; an emptied input lets it through
            inp.text = "half a thought"
            await _start_bg(registry, "later", "typed", "exit 0")
            await pilot.pause(1.5)
            assert calls == []
            inp.text = ""
            await _until(pilot, lambda: len(calls) == 1, tries=100)
            assert calls[0][1] == ["typed"]
            await _until(pilot, lambda: not app._turn_in_progress)

            # the user sends a prompt first: that turn gets the notice, no wake-up after it
            inp.text = "still typing"
            await _start_bg(registry, "later", "sent", "exit 0")
            await _until(pilot, lambda: any('"sent" exited' in t for t in _texts(app)), tries=100)
            inp.text = ""
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("what happened?"))
            await _until(pilot, lambda: len(calls) == 2, tries=100)
            assert calls[1] == ("what happened?", ["sent"])
            await pilot.pause(1.5)
            assert len(calls) == 2
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_job_wake_limit(tui, registry, monkeypatch):
    cli_tui = tui
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    calls: list[tuple[str, list[str]]] = []
    chain = {"on": True}
    _wake_loop(cli_tui, monkeypatch, registry, calls, restart=lambda: chain["on"])

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="chain")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            app.config.jobs_wake_on_end = True

            # every wake-up turn starts a job that ends at once: it stops after JOB_WAKE_LIMIT turns
            await _start_bg(registry, "chain", "first", "exit 0")
            await _until(pilot, lambda: any("Not resuming the agent" in t for t in _texts(app)), tries=200)
            assert len(calls) == cli_tui.JOB_WAKE_LIMIT
            await pilot.pause(1.5)
            assert len(calls) == cli_tui.JOB_WAKE_LIMIT
            assert sum("Not resuming the agent" in t for t in _texts(app)) == 1

            # a message from the user resets the count
            chain["on"] = False
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("carry on"))
            await _until(pilot, lambda: len(calls) == cli_tui.JOB_WAKE_LIMIT + 1, tries=100)
            assert calls[-1][0] == "carry on" and len(calls[-1][1]) == 1
            await _start_bg(registry, "chain", "again", "exit 0")
            await _until(pilot, lambda: len(calls) == cli_tui.JOB_WAKE_LIMIT + 2, tries=100)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- prompt queue: listed above the input, /queue edits / reorders / removes, Esc puts it back ---

def test_prompt_queue(tui, monkeypatch):
    cli_tui = tui
    ran: list[str] = []

    async def gated_loop(session_id, turn_id, user_text, instructions, mcp_client, working_history, turn_type,
                         current_session_history, steps_history, store, adapter=None, **kwargs):
        ran.append(user_text)
        await adapter._cancel_event.wait()  # every turn runs until Esc
        return ""
    monkeypatch.setattr(cli_tui, "agent_loop", gated_loop)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="queue")
        async with app.run_test() as pilot:
            await pilot.pause()
            submit = lambda text: app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))  # noqa: E731
            bar = app.query_one("#queue-bar", cli_tui.QueueBar)
            picker = app.query_one("#picker", cli_tui.PermissionPrompt)
            inp = app.query_one("#input", cli_tui.EnterSubmits)

            async def choose(value: str) -> None:
                app.on_permission_prompt_chosen(cli_tui.PermissionPrompt.Chosen(value, "picker"))
                await pilot.pause()

            submit("first")
            await _until(pilot, lambda: ran == ["first"])
            assert not bar.display
            for text in ("second", "third", "fourth"):
                submit(text)
            await pilot.pause()
            # each queued prompt is listed, not just counted
            assert bar.display
            shown = str(bar.render())
            assert "3 queued" in shown and "1. second" in shown and "3. fourth" in shown

            # /queue: run "fourth" next
            submit("/queue")
            await pilot.pause()
            assert [label for _, label in picker._options] == ["1. second", "2. third", "3. fourth", "Clear the queue"]
            await choose("q2")
            assert [v for v, _ in picker._options] == ["edit", "next", "remove", "back"]
            await choose("next")
            assert app._prompt_queue == ["fourth", "second", "third"]

            # remove "second", edit "third" (back to the input)
            submit("/queue")
            await pilot.pause()
            await choose("q1")
            await choose("remove")
            assert app._prompt_queue == ["fourth", "third"]
            submit("/queue")
            await pilot.pause()
            await choose("q1")
            await choose("edit")
            assert app._prompt_queue == ["fourth"] and inp.text == "third"

            # the turn ends on its own: the next queued prompt starts and leaves the queue
            app.adapter._cancel_event.set()
            await _until(pilot, lambda: ran == ["first", "fourth"])
            assert app._prompt_queue == [] and not bar.display

            # Esc with prompts queued: they go back in the input, ahead of the draft, instead of running
            inp.text = ""
            submit("fifth")
            submit("sixth")
            inp.text = "draft"
            await pilot.pause()
            await app.action_cancel()
            await _until(pilot, lambda: not app._turn_in_progress)
            await pilot.pause()
            assert ran == ["first", "fourth"] and app._prompt_queue == [] and not bar.display
            assert inp.text == "fifth\n\nsixth\n\ndraft"
            assert any("Moved 2 queued prompts back to the input" in t for t in _texts(app))

            # /queue clear, and listed in the menu
            submit("/queue clear")
            await _until(pilot, lambda: any("No queued prompts." in t for t in _texts(app)))
            assert "/queue" in cli_tui.COMMAND_DESCRIPTIONS
            await app._cleanup_and_save_session()

    asyncio.run(scenario())
