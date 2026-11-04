"""Headless TUI smoke test: async turns, streaming, approval prompt, /clear, exit (REQ-4.x)."""
import asyncio
import time
import types
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
    # A job or agent ending while idle would start a turn of the fake loop below; the wake-up tests turn it on
    config_module.get_config().jobs_wake_on_end = False
    config_module.get_config().agents_wake_on_end = False

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


def test_emit_after_exit_is_dropped(tui):
    """The emit shim keeps pointing at the adapter of an app that has exited; a worker thread's emit then
    must be dropped before call_from_thread, which builds its coroutine and then fails on the closed loop
    ("coroutine ... run_callback was never awaited")."""
    import threading
    from dostuff.helpers.ui import emit as emit_module
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="gone")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._cleanup_and_save_session()
        return app

    app = asyncio.run(scenario())  # closes the app's loop
    assert emit_module._active_adapter is app.adapter
    calls = []
    app.call_from_thread = lambda *a, **k: calls.append(a)  # type: ignore[method-assign]
    worker = threading.Thread(target=lambda: emit_module.emit("[worktree] created"))
    worker.start()
    worker.join()
    assert calls == []


async def _until(pilot, cond, tries=60):
    """Poll until `cond` holds, pausing between tries.

    For a prompt taking the input (approval, ask-user panel) wait for `widget.display and app.focused is
    widget`, never `widget.display` alone: `_show_ask_panel` sets `display` synchronously but focuses it in
    `call_after_refresh`, because a widget cannot take focus before it is laid out. Between the two there is
    a real frame where the prompt is on screen and the (hidden) input still has focus — on a slow runner
    that frame is long enough to observe, and keys pressed in it go nowhere. `test_ask_panel_render` failed
    exactly there on CI's windows/3.11 while passing locally.
    """
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
            # A space after a command taking arguments, so the argument hint shows
            assert app.query_one("#input", cli_tui.EnterSubmits).text == "/compact "
            assert app.query_one("#input", cli_tui.EnterSubmits).suggestion == "[auto <1-100 | off>]"
            await pilot.press("escape")
            app._clear_input()

            # Ghost text: the rest of the command, then its flag; Tab accepts each
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            inp.focus()
            await pilot.press("/", "m", "o")
            await pilot.pause()
            # The rest of the name, then a hint for its arguments (shown, never inserted)
            assert inp.suggestion == "del [provider/model] [--active]" and inp.completion == "del"
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
            # Tab and Right never insert a bare argument hint
            await pilot.press(*"/rename ")
            await pilot.pause()
            assert inp.suggestion == "<title>" and inp.completion == ""
            await pilot.press("right")
            await pilot.pause()
            assert inp.text == "/rename " and inp.suggestion == "<title>"
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

            # Scrolling up shows the bar; clicking the centered button goes back to the bottom and hides it
            jump = app.query_one("#jump-bottom", cli_tui.JumpToBottom)
            for i in range(40):
                app._append(f"line {i}", msg_type="system")
            await pilot.pause()
            messages = app.query_one("#messages", cli_tui.VerticalScroll)
            assert not jump.display
            messages.scroll_home(animate=False)
            await _until(pilot, lambda: jump.display)
            # The bar spans the conversation so it can center a small button (not a full-width band)
            label = app.query_one("#jump-bottom-label", cli_tui.Static)
            assert label.region.width < jump.region.width
            assert abs(label.region.x + label.region.width / 2 - (jump.region.x + jump.region.width / 2)) <= 1
            await pilot.click("#jump-bottom-label")
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
            assert inp.suggestion == "ease-notes [arguments]" and inp.completion == "ease-notes"
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

            # Esc with text in the input has nothing to do: it says how to quit instead
            inp.text = "draft"
            await pilot.press("escape")
            assert str(inp.border_title) == cli_tui.QUIT_HINT
            app._clear_input()
            await pilot.press("ctrl+o")
            assert "expanded" in str(inp.border_title)
            await pilot.press("ctrl+o")
            assert "collapsed" in str(inp.border_title)

            # Esc Esc with an empty input takes the prompt back as typed and drops the turn
            await pilot.press("escape")
            assert "Esc again" in str(inp.border_title) and cli_tui.QUIT_HINT in str(inp.border_title)
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


def test_tools_command(tui):
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="tools-cmd")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/tools"))
            await _until(pilot, lambda: any(t.startswith("Tools (") for t in _texts(app)))
            text = next(t for t in _texts(app) if t.startswith("Tools ("))
            assert "read_file" in text and "bash_command" in text and "MCP tools" in text
            assert "/tools" in cli_tui.COMMAND_DESCRIPTIONS

    asyncio.run(scenario())


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


def test_tui_quit_with_agents(tui, agents, fake_model, registry, _py, _alive, _wait_dead):
    """Ctrl+Q with a background agent asks first; "No" keeps it, "Yes" stops every run (no notices), kills
    its jobs and exits (subagents T4.10, CON-007)."""
    from conftest import fake_response, fake_tool_call

    from dostuff.agent.permissions import SessionPermissions
    from dostuff.helpers.agent.notices import get_notice_queue
    cli_tui = tui
    sleeper = _py("import time; time.sleep(30)")
    fake_model.script("[A]", fake_response(tool_calls=[fake_tool_call("bash_command", {
        "command": sleeper, "run_in_background": True})]))
    fake_model.gate("[A]", fake_response("never"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="aquit")
        exited: list = []
        app.exit = lambda *a, **k: exited.append(True)  # type: ignore[method-assign]
        async with app.run_test() as pilot:
            await pilot.pause()
            run = await agents.start(owner="aquit", agent_type="general-purpose", description="long one",
                                     prompt="[A] x", background=True, adapter=app.adapter,
                                     permissions=SessionPermissions(mode="auto"))
            await _until(pilot, lambda: len(fake_model.calls_for("[A]")) == 2, tries=100)
            [job] = [j for j in registry.jobs() if j.owner == f"aquit/sub/{run.id}"]
            await pilot.press("ctrl+q")
            await _until(pilot, app.adapter.has_pending_confirm)
            assert any('Quit and stop 1 agent? ("long one")' in t for t in _texts(app))
            app.adapter.choose("no")
            await _until(pilot, lambda: not app.adapter.has_pending_confirm())
            await pilot.pause(0.2)
            assert not exited and run.status == "running" and _alive(job.pid)

            srv = await _start_bg(registry, "aquit", "srv", sleeper)
            await pilot.press("ctrl+q")
            await _until(pilot, app.adapter.has_pending_confirm)
            assert any('Quit and stop 1 background job and 1 agent? (srv, "long one")' in t for t in _texts(app))
            app.adapter.choose("yes")
            await _until(pilot, lambda: bool(exited), tries=300)
            assert run.status == "stopped" and run.stopped_by == "session_end"
            assert _wait_dead(job.pid, 5) and _wait_dead(srv.pid, 5)
            assert get_notice_queue().pending("aquit", "agent") == []

    asyncio.run(scenario())


def test_tui_clear_resume_with_agents(tui, agents, fake_model):
    """Running background agents follow /clear and /resume into the new session: their end line, notice and
    wake-up go there, and stop_agent from the new conversation works (subagents T4.11)."""
    from conftest import fake_response

    from dostuff.helpers.agent.notices import get_notice_queue
    from dostuff.tools.stop_agent import stop_agent
    cli_tui = tui
    gate = fake_model.gate("[A]", fake_response("report A"))
    fake_model.gate("[B]", fake_response("never"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="afirst")
        async with app.run_test() as pilot:
            await pilot.pause()
            submit = lambda text: app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))  # noqa: E731

            async def bg(marker, description):
                return await agents.start(owner=app.session_id, agent_type="explore", description=description,
                                          prompt=f"{marker} x", background=True, adapter=app.adapter)
            a = await bg("[A]", "find a")
            await _until(pilot, lambda: bool(fake_model.calls_for("[A]")))
            submit("/clear")
            await _until(pilot, lambda: app.session_id != "afirst", tries=200)
            await pilot.pause(0.2)
            cleared = app.session_id
            assert a.owner == cleared and agents.runs("afirst") == []
            assert any('1 agent ("find a") kept running' in t for t in _texts(app))
            gate.set()
            await _until(pilot, lambda: a.status == "completed")
            assert [n.id for n in get_notice_queue().pending(cleared)] == [a.id]

            b = await bg("[B]", "hold b")
            await app.store.append("aother", {"role": "user", "content": "x"})
            submit("/resume aother")
            await _until(pilot, lambda: app.session_id == "aother", tries=200)
            await pilot.pause(0.2)
            assert b.owner == "aother" and a.owner == cleared  # an ended run stays where it was
            assert [n.id for n in get_notice_queue().pending("aother")] == [a.id]  # notices follow too
            assert await stop_agent(agent_id=b.id, session_id="aother") == f'Stopped agent "{b.id}" (explore).'
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


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


def test_agent_notice_rendering(tui, agents, fake_model, monkeypatch):
    """A background agent that ends shows one system line (also while idle) and a notification; its stored
    notice step re-renders as that line on /resume, and Esc Esc's "previous prompt" skips it (subagents T4.3)."""
    from conftest import fake_response

    from dostuff.helpers.agent.notices import format_notices, get_notice_queue
    cli_tui = tui
    fake_model.script("[A]", fake_response("found it"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="agentn")
        async with app.run_test() as pilot:
            await pilot.pause()
            submit = lambda text: app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))  # noqa: E731
            notified: list = []
            monkeypatch.setattr(app, "_notify", lambda message, elapsed: notified.append(message))
            line = '⚙ agent "find auth code" (explore) finished after'
            await agents.start(owner="agentn", agent_type="explore", description="find auth code", prompt="[A] x",
                               background=True, adapter=app.adapter)
            await _until(pilot, lambda: any(line in t for t in _texts(app)), tries=100)
            await pilot.pause(0.2)
            assert sum(line in t for t in _texts(app)) == 1
            assert any("find auth code" in m and "finished" in m for m in notified)

            notice_step = format_notices(get_notice_queue().drain("agentn"))
            for step in ({"role": "user", "content": "start it"}, {"role": "user", "content": notice_step}):
                await app.store.append("other", step)
            submit("/resume other")
            await _until(pilot, lambda: app.session_id == "other", tries=200)
            await pilot.pause(0.3)
            texts = _texts(app)
            resumed = texts[max(i for i, t in enumerate(texts) if "start it" in t):]
            assert any(line in t for t in resumed)
            assert not any("<background-agent" in t for t in resumed)
            assert app.steps_history[app._last_prompt_index()]["content"] == "start it"
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_background_approval_waits_for_input(tui, agents, fake_model):
    """A background agent's approval prompt waits while the input has text or a picker is open; a
    foreground agent's shows at once, as the main agent's does (subagents D8, T4.6)."""
    from conftest import fake_response, fake_tool_call

    from dostuff.agent.permissions import SessionPermissions
    cli_tui = tui
    for m in "ABC":
        fake_model.script(f"[{m}]", fake_response(tool_calls=[fake_tool_call("write_file", {"path": f"{m}.txt",
                                                                                       "content": "x"})]),
                          fake_response(f"done {m}"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="bgask")
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            perms = SessionPermissions(mode="ask")

            async def start(marker, background):
                return await agents.start(owner="bgask", agent_type="general-purpose", description=f"write {marker}",
                                          prompt=f"[{marker}] x", background=background, adapter=app.adapter,
                                          permissions=perms)

            # typing: the background prompt waits, and shows once the input is cleared
            inp.text = "ab"
            run = await start("A", True)
            await _until(pilot, lambda: run.status == "waiting_approval")
            await pilot.pause(0.5)
            assert not app.query("#permission-prompt") and inp.text == "ab"
            inp.text = ""
            await _until(pilot, lambda: bool(app.query("#permission-prompt")))
            assert any("[agent: write A · general-purpose]" in t for t in _texts(app))
            app.adapter.choose("yes")
            await _until(pilot, lambda: run.status == "completed")

            # a picker open: same
            async def chosen(value):
                pass
            app._show_picker("Pick one", [("a", "A"), ("b", "B")], chosen)
            await pilot.pause()
            run = await start("B", True)
            await _until(pilot, lambda: run.status == "waiting_approval")
            await pilot.pause(0.5)
            assert not app.query("#permission-prompt")
            app._close_picker()
            await _until(pilot, lambda: bool(app.query("#permission-prompt")))
            app.adapter.choose("yes")
            await _until(pilot, lambda: run.status == "completed")

            # a foreground agent's prompt shows at once, text or not
            inp.text = "typing"
            run = await start("C", False)
            await _until(pilot, lambda: bool(app.query("#permission-prompt")))
            app.adapter.choose("yes")
            await _until(pilot, lambda: run.status == "completed")
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def cli_tui_moved(agent_id: str) -> str:
    return f"Agent \"{agent_id}\" moved to the background; you'll get its report as a notice."


def test_ctrl_b_backgrounds_agent(tui, agents, fake_model, monkeypatch):
    """While a foreground agent runs the loader offers Ctrl+B; pressing it returns the waiting call at once
    with the "moved" text, the run keeps going and its end queues a notice. Idle, Ctrl+B is skipped (T4.8)."""
    import time

    from conftest import fake_response
    from textual.actions import SkipAction

    from dostuff.helpers.agent.notices import get_notice_queue
    from dostuff.tools.agent import agent
    cli_tui = tui
    gate = fake_model.gate("[A]", fake_response("report A"))
    results: list = []

    async def fake_loop(session_id, turn_id, user_text, instructions, mcp_client, working_history, turn_type,
                        current_session_history, steps_history, store, adapter=None, **kwargs):
        results.append((await agent(description="slow one", prompt="[A] x", mcp_client=object(),  # type: ignore[arg-type]
                                    session_id=session_id, turn_id=turn_id, adapter=adapter), time.monotonic()))
        return "handled"
    monkeypatch.setattr(cli_tui, "agent_loop", fake_loop)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ctrlbagent")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            try:
                app.action_background_jobs()
                raise AssertionError("expected SkipAction")
            except SkipAction:
                pass
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("go"))
            await _until(pilot, lambda: bool(agents.runs()) and agents.runs()[0].status == "running", tries=100)
            [run] = agents.runs()
            await _until(pilot, lambda: app._loader_label == cli_tui.WORKING_LABEL_JOBS)
            pressed = time.monotonic()
            await pilot.press("ctrl+b")
            await _until(pilot, lambda: bool(results))
            text, returned = results[0]
            assert text == cli_tui_moved(run.id)
            assert returned - pressed < 0.5
            assert run.background and run.status == "running"
            await _until(pilot, lambda: not app._turn_in_progress)
            gate.set()
            await _until(pilot, lambda: run.status == "completed")
            assert [n.id for n in get_notice_queue().pending("ctrlbagent")] == [run.id]
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_agents_status_item(tui, agents, fake_model):
    """`⚙ 2 agents` on the status line while agents run, `⚠ 1 agent waiting` while one waits for approval,
    hidden at zero; it updates without a keypress and a click opens /agents (T4.9)."""
    from conftest import fake_response, fake_tool_call

    from dostuff.agent.permissions import SessionPermissions
    cli_tui = tui
    gates = [fake_model.gate(m, fake_response("done")) for m in ("[A]", "[B]")]
    fake_model.script("[W]", fake_response(tool_calls=[fake_tool_call("write_file", {"path": "w.txt", "content": "x"})]),
                      fake_response("wrote"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="abar")
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one("#agents-bar", cli_tui.AgentsBar)
            assert not bar.display

            async def bg(marker, **kwargs):
                return await agents.start(owner="abar", agent_type="general-purpose", description=marker,
                                          prompt=f"{marker} x", background=True, adapter=app.adapter, **kwargs)
            a, b = await bg("[A]"), await bg("[B]")
            await _until(pilot, lambda: bar.display and str(bar.render()) == "⚙ 2 agents")
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            inp.text = "typing"  # holds the background prompt back
            w = await bg("[W]", permissions=SessionPermissions(mode="ask"))
            await _until(pilot, lambda: str(bar.render()) == "⚠ 1 agent waiting")
            for gate in gates:
                gate.set()
            await _until(pilot, lambda: a.status == b.status == "completed")
            await _until(pilot, lambda: str(bar.render()) == "⚠ 1 agent waiting")
            inp.text = ""
            await _until(pilot, lambda: bool(app.query("#permission-prompt")))
            app.adapter.choose("yes")
            await _until(pilot, lambda: w.status == "completed")
            await _until(pilot, lambda: not bar.display)

            gate = fake_model.gate("[C]", fake_response("done"))
            c = await bg("[C]")
            await _until(pilot, lambda: bar.display)
            await pilot.click("#agents-bar")  # opens the /agents list (T5.7)
            await _until(pilot, lambda: app._picker_open())
            assert c.id in dict(app.query_one("#picker", cli_tui.PermissionPrompt)._options)
            app._close_picker()
            gate.set()
            await _until(pilot, lambda: c.status == "completed")
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


def test_agent_wake_up(tui, agents, fake_model, monkeypatch):
    """A background agent that ends on its own while idle starts one wake turn; agents ending together share
    one; a stopped one rides along; agents.wake_on_end off doesn't wake; the limit is shared with jobs (T4.7)."""
    from conftest import fake_response

    from dostuff.agent.subagents import registry as registry_module
    from dostuff.helpers.agent.notices import get_notice_queue
    cli_tui = tui
    monkeypatch.setattr(cli_tui, "JOB_WAKE_DELAY_S", 0.3)
    monkeypatch.setattr(registry_module, "STOP_GRACE_S", 0.1)
    calls: list[tuple[str, list[str]]] = []

    async def fake_loop(session_id, turn_id, user_text, instructions, mcp_client, working_history, turn_type,
                        current_session_history, steps_history, store, adapter=None, **kwargs):
        calls.append((user_text, [n.id for n in get_notice_queue().drain(session_id)]))
        return "handled"
    monkeypatch.setattr(cli_tui, "agent_loop", fake_loop)
    from dostuff.tools.bash.jobs import get_registry  # the agents fixture installs a fresh one

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="awake")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._backend_ready.wait()

            async def bg(description, marker="[Z]"):
                return await agents.start(owner="awake", agent_type="explore", description=description,
                                          prompt=f"{marker} x", background=True, adapter=app.adapter)

            # agents.wake_on_end off: no wake-up (the tui fixture's default)
            off = await bg("quiet one")
            await _until(pilot, lambda: off.status == "completed")
            await pilot.pause(0.8)
            assert calls == []
            app.config.agents_wake_on_end = True

            # idle: the end starts a turn whose input is the notice (the earlier one rides along)
            first = await bg("find a")
            await _until(pilot, lambda: len(calls) == 1, tries=100)
            text, ids = calls[0]
            assert text.startswith(f'<background-agent id="{off.id}"') and ids == [off.id, first.id]
            assert any('⚙ Resuming the agent: background agent "quiet one", "find a" ended' in t for t in _texts(app))
            await _until(pilot, lambda: not app._turn_in_progress)

            # two ending together: one turn
            a, b = await bg("two a"), await bg("two b")
            await _until(pilot, lambda: len(calls) == 2, tries=100)
            assert sorted(calls[1][1]) == sorted([a.id, b.id])
            await _until(pilot, lambda: not app._turn_in_progress)

            # stopped by the user: no wake; it rides along with the next one
            fake_model.gate("[G]", fake_response("never"))
            held = await bg("held", "[G]")
            await _until(pilot, lambda: bool(fake_model.calls_for("[G]")))
            await agents.stop(held, "user")
            await pilot.pause(0.8)
            assert len(calls) == 2
            last = await bg("last")
            await _until(pilot, lambda: len(calls) == 3, tries=100)
            assert calls[2][1] == [held.id, last.id]
            await _until(pilot, lambda: not app._turn_in_progress)

            # three wake-ups in a row: a job's end no longer wakes it (the limit is shared)
            app.config.jobs_wake_on_end = True
            await _start_bg(get_registry(), "awake", "j1", "exit 0")
            await _until(pilot, lambda: any(f"{cli_tui.WAKE_LIMIT} automatic turns in a row" in t
                                            for t in _texts(app)), tries=100)
            assert len(calls) == 3
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

            # every wake-up turn starts a job that ends at once: it stops after WAKE_LIMIT turns
            await _start_bg(registry, "chain", "first", "exit 0")
            await _until(pilot, lambda: any("Not resuming the agent" in t for t in _texts(app)), tries=200)
            assert len(calls) == cli_tui.WAKE_LIMIT
            await pilot.pause(1.5)
            assert len(calls) == cli_tui.WAKE_LIMIT
            assert sum("Not resuming the agent" in t for t in _texts(app)) == 1

            # a message from the user resets the count
            chain["on"] = False
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("carry on"))
            await _until(pilot, lambda: len(calls) == cli_tui.WAKE_LIMIT + 1, tries=100)
            assert calls[-1][0] == "carry on" and len(calls[-1][1]) == 1
            await _start_bg(registry, "chain", "again", "exit 0")
            await _until(pilot, lambda: len(calls) == cli_tui.WAKE_LIMIT + 2, tries=100)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- prompt queue: listed above the input, /queue edits / reorders / removes, Esc puts it back ---

def test_prompt_queue(queue_mode, monkeypatch):
    """The prompt queue itself, pinned in `queue` mode (the opt-in batching mode): plain messages
    wait for the turn (steering spec D3 — the default mode routes them to the steering queue instead)."""
    cli_tui, _ = queue_mode
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
            assert [v for v, _ in picker._options] == ["edit", "next", "send", "remove", "back"]
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


# --- subagents rebuild T1.3: a session that used delegate_to_subagent still renders on /resume ----

def test_pin_old_delegate_history_renders(tui):
    import json
    cli_tui = tui
    snapshot = json.loads((Path(__file__).parent / "snapshots" / "old_delegate_session.json").read_text("utf-8"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="pin")
        async with app.run_test() as pilot:
            await pilot.pause()
            for step in snapshot:
                await app.store.append("old", step)
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/resume old"))
            await _until(pilot, lambda: app.session_id == "old", tries=200)
            await pilot.pause(0.3)
            texts = _texts(app)
            idx = max(i for i, t in enumerate(texts) if "Where is auth handled?" in t)
            resumed = texts[idx:]
            assert any(t.startswith("● delegate_to_subagent(") for t in resumed)
            assert any("auth/session.py:42 (verify_token)" in t for t in resumed)
            agent_md = [w for w in app.query("#msg-container .msg-agent") if hasattr(w.content, "markup")]
            assert any("Auth is handled in auth/session.py:42." in w.content.markup for w in agent_md)
            assert not app.query("#msg-container .msg-error")
            assert app.steps_history == snapshot
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- subagents rebuild T2.7: the input stays responsive while agents run (CON-004) ----------------

def test_tui_responsive_during_agents(tui, monkeypatch, fake_model):
    import time
    cli_tui = tui
    from conftest import fake_response
    from dostuff.agent.subagents import registry as agents_module
    reg = agents_module.AgentRegistry()
    monkeypatch.setattr(agents_module, "_agent_registry", reg)
    gates: list = []

    async def agents_loop(session_id, turn_id, user_text, instructions, mcp_client, working_history, turn_type,
                          current_session_history, steps_history, store, adapter=None, **kwargs):
        runs = [await reg.start(owner=session_id, agent_type="general-purpose", description=f"a{m}",
                                prompt=f"[{m}] x", adapter=adapter, turn_id=turn_id) for m in "AB"]
        await asyncio.gather(*(r.done for r in runs))
        return "done"
    monkeypatch.setattr(cli_tui, "agent_loop", agents_loop)

    async def scenario():
        gates.extend(fake_model.gate(f"[{m}]", fake_response(f"r{m}")) for m in "AB")
        app = cli_tui.DostuffTUI(session_id="responsive")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("go"))
            await _until(pilot, lambda: len([r for r in reg.runs() if r.status == "running"]) == 2)
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            inp.focus()
            start = time.monotonic()
            await pilot.press("a", "b", "c")
            await _until(pilot, lambda: inp.text == "abc")
            assert time.monotonic() - start < 0.2 * 3 + 0.5  # three keypresses, each well under 0.2 s
            assert all(r.status == "running" for r in reg.runs())
            for gate in gates:
                gate.set()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_agents_types_command(tui, monkeypatch):
    """/agents types: every type with its source, model and tools, then the loader warnings (subagents T3.9)."""
    cli_tui = tui
    from dostuff.agent.subagents import definitions as defs_module
    from dostuff.config import get_config
    monkeypatch.setattr(defs_module, "_cache", None)
    folder = Path.cwd() / ".dostuff" / "agents"
    folder.mkdir(parents=True)
    rev = folder / "rev.md"
    rev.write_text("---\nname: code-reviewer\ndescription: Reviews diffs\ntools: Read, Grep, Frobnicate\n"
                   "model: fast\n---\nReview.", encoding="utf-8")
    (folder / "broken.md").write_text("no frontmatter", encoding="utf-8")
    get_config().agents_models = {"fast": "anthropic/claude-haiku-5-5"}
    assert "/agents" in cli_tui.COMMAND_DESCRIPTIONS

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="agents")
        async with app.run_test() as pilot:
            await pilot.pause()
            submit = lambda text: app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))  # noqa: E731
            submit("/agents types")
            await _until(pilot, lambda: any("code-reviewer" in t for t in _texts(app)))
            text = [t for t in _texts(app) if "code-reviewer" in t][-1]
            assert str(rev) in text and "anthropic/claude-haiku-5-5" in text and "read_file, grep" in text
            assert "general-purpose" in text and "built-in" in text and "openai/gpt-4o-mini" in text
            assert "explore" in text and "read-only shell" in text
            assert "fork  ·  built-in" in text and "always the main model" in text  # (T6.2)
            assert "Frobnicate" in text and "broken.md" in text  # the warnings

            (folder / "second.md").write_text("---\nname: second\ndescription: d\n---\nb", encoding="utf-8")
            submit("/agents reload")
            await _until(pilot, lambda: any("Reloaded" in t and "agent types" in t for t in _texts(app)))
            reloaded = [t for t in _texts(app) if "Reloaded" in t][-1]
            assert "5 agent types" in reloaded  # general-purpose, explore, plan, code-reviewer, second

            assert "/agents types <name>" in text  # the hint

            submit("/agents types code-reviewer")  # one type's config and system prompt
            await _until(pilot, lambda: any("System prompt:" in t and "code-reviewer" in t for t in _texts(app)))
            detail = [t for t in _texts(app) if "System prompt:" in t][-1]
            assert str(rev) in detail and "anthropic/claude-haiku-5-5 (from `fast`)" in detail
            assert "max iterations:" in detail and detail.rstrip().endswith("Review.")
            submit("/agents types explore")
            await _until(pilot, lambda: any("You locate code quickly" in t for t in _texts(app)))
            submit("/agents types fork")
            await _until(pilot, lambda: any("You are a fork of the main agent" in t for t in _texts(app)))
            submit("/agents types nosuch")
            await _until(pilot, lambda: any("No agent type 'nosuch'" in t for t in _texts(app)))

            submit("/agents nope")
            await _until(pilot, lambda: any("Usage: /agents" in t for t in _texts(app)))
            submit("/agents reload extra")
            await _until(pilot, lambda: sum("Usage: /agents" in t for t in _texts(app)) == 2)
    asyncio.run(scenario())


def _agent_texts(app):
    """The Markdown source of the agent's messages."""
    return [w.content.markup for w in app.query("#msg-container .msg-agent") if hasattr(w.content, "markup")]


def _view_texts(app):
    return [str(getattr(w, "content", getattr(w, "renderable", ""))) for w in app.query("#agent-view-msgs Static")]


def _view_agent_texts(app):
    return [w.content.markup for w in app.query("#agent-view-msgs .msg-agent") if hasattr(w.content, "markup")]


def test_agents_command(tui, agents, fake_model):
    """/agents lists the session's agents (running first); picking one shows its transcript in the agent view (in
    place of the conversation, never in it) and offers Stop it (running) or Answer approval (waiting); loaded
    interrupted agents are listed too (subagents T5.7)."""
    from conftest import fake_response, fake_tool_call

    from dostuff.agent.permissions import SessionPermissions
    from dostuff.helpers.agent.notices import get_notice_queue
    cli_tui = tui
    fake_model.script("[D]", fake_response(tool_calls=[fake_tool_call("get_current_datetime", {}, "c1")]),
                      fake_response("report D"))
    fake_model.gate("[R]", fake_response("never"))
    fake_model.script("[W]", fake_response(tool_calls=[fake_tool_call("write_file", {"path": "w.txt", "content": "x"})]),
                      fake_response("wrote"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="alist")
        async with app.run_test() as pilot:
            await pilot.pause()
            submit = lambda text: app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))  # noqa: E731
            picker = app.query_one("#picker", cli_tui.PermissionPrompt)
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            submit("/agents")
            await _until(pilot, lambda: any("No agents in this session." in t for t in _texts(app)))

            async def start(marker, **kwargs):
                return await agents.start(owner="alist", agent_type="general-purpose", description=f"do {marker}",
                                          prompt=f"[{marker}] x", adapter=app.adapter, **kwargs)
            done = await start("D")
            await done.done
            running = await start("R", background=True)
            inp.text = "typing"  # holds W's approval prompt back
            waiting = await start("W", background=True, permissions=SessionPermissions(mode="ask"))
            await _until(pilot, lambda: waiting.status == "waiting_approval" and running.status == "running")

            submit("/agents")
            await _until(pilot, app._picker_open)
            inp.text = "typing"  # submitting cleared it
            assert [v for v, _ in picker._options] == [running.id, waiting.id, done.id]
            labels = dict(picker._options)
            for part in (done.id, "do D", "general-purpose", "completed", "tokens", "$", "foreground"):
                assert part in labels[done.id], (part, labels[done.id])
            assert "background" in labels[running.id] and "waiting for approval" in labels[waiting.id]

            # a finished agent: its transcript in the agent view, rendered like the conversation
            main_before = _texts(app)
            await app._picker_callback(done.id)
            await _until(pilot, lambda: any("report D" in t for t in _view_agent_texts(app)))
            assert any(t.startswith("● ") and "get_current_datetime" in t for t in _view_texts(app))
            assert any("[D] x" in t for t in _view_texts(app))
            assert not app.query_one("#conversation").display
            assert _texts(app) == main_before  # nothing of the agent's lands in the conversation
            assert [v for v, _ in picker._options] == ["type", "back"]
            await app._picker_callback("back")
            await _until(pilot, lambda: [v for v, _ in picker._options] == [running.id, waiting.id, done.id])
            assert not app._agent_view_open() and app.query_one("#conversation").display

            # a running agent's view fills in live, and Esc closes it
            await app._picker_callback(running.id)
            await _until(pilot, lambda: any("[R] x" in t for t in _view_texts(app)))
            app._close_picker()
            await app.action_cancel()
            await pilot.pause()
            assert not app._agent_view_open() and app.query_one("#conversation").display
            assert _texts(app) == main_before
            submit("/agents")
            await _until(pilot, lambda: app._picker_open() and running.id in dict(picker._options))
            inp.text = "typing"

            # a running one: Stop it → stopped, a notice without a wake-up
            await app._picker_callback(running.id)
            assert [v for v, _ in picker._options] == ["stop", "type", "back"]
            await app._picker_callback("stop")
            await _until(pilot, lambda: running.status == "stopped")
            [notice] = [n for n in get_notice_queue().pending("alist") if n.id == running.id]
            assert notice.status == "stopped" and not notice.wake

            # a waiting one: Answer approval shows its prompt now, although the input has text
            assert waiting.status == "waiting_approval"  # still held back: the input has text
            submit("/agents")
            await _until(pilot, lambda: app._picker_open() and waiting.id in dict(picker._options))
            inp.text = "typing"
            await app._picker_callback(waiting.id)
            assert [v for v, _ in picker._options] == ["answer", "stop", "type", "back"]
            await app._picker_callback("answer")
            await _until(pilot, lambda: bool(app.query("#permission-prompt")))
            assert inp.text == "typing"
            app.adapter.choose("yes")
            await _until(pilot, lambda: waiting.status == "completed")

            # an agent loaded from the store as interrupted is listed, and its transcript shows
            row = {"id": "old123", "session_id": "alist", "agent_type": "explore", "description": "old one",
                   "model": "m", "status": "running", "background": True, "worktree": None, "branch": None,
                   "started_at": "2026-11-01T10:00:00", "ended_at": None, "tokens": 3, "cost_usd": 0.0,
                   "result": None, "compaction_notes": ""}
            await app.store.upsert_agent_run(row)
            await app.store.append_agent_step("old123", {"role": "user", "content": "task of old one"})
            await agents.load("alist", app.store)
            submit("/agents")
            await _until(pilot, lambda: app._picker_open() and "old123" in dict(picker._options))
            assert "interrupted" in dict(picker._options)["old123"]
            await app._picker_callback("old123")
            await _until(pilot, lambda: any("task of old one" in t for t in _view_texts(app)))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_trust_prompt_does_not_bounce(tui, fresh_trust, hook_cmd):
    """Answering the session-start hook trust prompt doesn't move the transcript down and back up: the rows
    the prompt frees are held under the loader for the turn, and one loader stays up from submit to the end."""
    cli_tui = tui

    async def quick_loop(session_id, turn_id, user_text, instructions, *a, adapter=None, **kw):
        await asyncio.sleep(0.2)
        adapter.emit("partial", "ok")
        adapter.emit("stream_end", None)
        return "ok"
    cli_tui.agent_loop = quick_loop
    _write_project_hooks({"SessionStart": [{"hooks": [{"type": "command", "name": "hello",
                                                       "command": hook_cmd("print('BANANA')")}]}]})

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="bounce")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            loaders: set[int] = set()

            def look() -> None:
                loaders.update(id(w) for w in app.query(".msg-loader"))
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("hi"))
            # Fine-grained polling (to see every loader) with a time limit, not an iteration count:
            # 200 x 5 ms was too short for a slow Windows CI runner to reach the trust prompt
            deadline = time.monotonic() + 20
            while not app.adapter.has_pending_confirm() and time.monotonic() < deadline:
                await pilot.pause(0.005)
                look()
            assert app.adapter.has_pending_confirm()
            await pilot.pause(0.05)
            details = next(w for w in app.query("#msg-container Static") if "This project defines hooks" in str(w.content))
            assert app.query_one("#messages", cli_tui.VerticalScroll).max_scroll_y > 0  # the prompt made it overflow
            y = details.region.y
            app.adapter.choose("yes")
            ys = []
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                await asyncio.sleep(0)
                ys.append(details.region.y)
                await pilot.pause(0.005)
                ys.append(details.region.y)
                look()
                if any(t.startswith("⚓ SessionStart") for t in _texts(app)):
                    break
            await pilot.pause(0.05)
            ys.append(details.region.y)
            assert all(v <= y for v in ys), ys  # never dropped when the input came back
            await _until(pilot, lambda: not app._turn_in_progress)
            assert len(loaders) == 1  # "Loading model libraries..." relabelled, not swapped for another loader
            assert app.query_one("#msg-container").styles.inline.min_height is None  # released at the end
            await app._cleanup_and_save_session()
    asyncio.run(scenario())


def test_picker_does_not_bounce_the_transcript(tui):
    """Opening or closing a selector must not move the transcript, not even for the frame that lays out
    the new height: Textual pins an anchored scrollbox with `container_size`, which it only writes back
    after arranging, so the first frame after the input/picker swap used the previous viewport height
    and drew the conversation a few rows off (the bounce). Every painted frame stays flush."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="bounce")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            for i in range(25):  # enough to overflow the viewport
                app._append(f"row{i:02d} " + "-" * 30, msg_type="user")
            await pilot.pause()
            await pilot.pause()
            messages = app.query_one("#messages")
            assert isinstance(messages, cli_tui.Transcript)
            off_bottom: list[int] = []
            original_display = app._display

            def display(screen, renderable):
                """Check what is actually painted, before it goes to the terminal."""
                rows = [strip.text.rstrip() for strip in
                        app.screen._compositor.render_strips()[:messages.region.height]]
                occupied = [y for y, row in enumerate(rows) if row]
                if occupied:  # empty viewport: nothing to be flush with
                    off_bottom.append(messages.region.height - 1 - occupied[-1])
                return original_display(screen, renderable)
            app._display = display  # type: ignore[method-assign]

            for cursor in (0, 3, 5):  # as if arrowing through the options: the picker can change height
                app._show_picker("Pick", [(f"v{i}", f"o{i}") for i in range(6)], app._switch_session)
                await pilot.pause()
                await pilot.pause()
                for _ in range(cursor):
                    await pilot.press("down")
                await pilot.pause()
                app._close_picker()
                await pilot.pause()
                await pilot.pause()

            # An approval prompt (mounted, not just shown) opens and closes the same way
            task = asyncio.get_running_loop().create_task(app.adapter.ask("Allow write_file?"))
            deadline = time.monotonic() + 20
            while not app.query("#permission-prompt") and time.monotonic() < deadline:
                await pilot.pause(0.01)
            await pilot.pause(0.1)
            app.adapter.choose("yes")
            await task
            await pilot.pause(0.1)
            app._display = original_display  # type: ignore[method-assign]

            assert off_bottom, "no frame was measured"
            assert max(off_bottom) <= 0, f"the transcript drifted off the bottom: {off_bottom}"
            await app._cleanup_and_save_session()
    asyncio.run(scenario())


def test_picker_keeps_a_scrolled_up_transcript_in_place(tui):
    """The anchoring fix must not drag a transcript the user scrolled up back to the bottom."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="scrolled")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            for i in range(25):
                app._append(f"row{i:02d} " + "-" * 30, msg_type="user")
            await pilot.pause()
            messages = app.query_one("#messages", cli_tui.Transcript)
            messages.scroll_to(y=10, animate=False)
            await pilot.pause()
            await pilot.pause()
            assert messages.scroll_y == 10

            app._show_picker("Pick", [("a", "a")], app._switch_session)
            await pilot.pause()
            await pilot.pause()
            assert messages.scroll_y == 10
            app._close_picker()
            await pilot.pause()
            await pilot.pause()
            assert messages.scroll_y == 10
            await app._cleanup_and_save_session()
    asyncio.run(scenario())


# --- ask_user Phase 0: the approval contract is pinned before anything changes (CON-005) ---------

def _params(fn):
    """(name, default) per parameter (without `self`); None means required."""
    import inspect
    return [(p.name, None if p.default is inspect.Parameter.empty else p.default)
            for p in inspect.signature(fn).parameters.values() if p.name != "self"]


def test_pin_approval_surface(tui):
    """ask_user is a new, separate channel: the approval surface keeps its signatures and behaviour."""
    import dataclasses
    cli_tui = tui
    from dostuff.agent.permissions import ApprovalReply
    from dostuff.headless import HeadlessAdapter

    assert _params(cli_tui.TuiAdapter.ask) == [("message", None), ("always_label", ""), ("background", False)]
    assert _params(cli_tui.TuiAdapter.ask_batch) == [
        ("messages", None), ("always_label", ""), ("background", False)]
    assert _params(cli_tui.TuiAdapter.ask_trust) == [("message", None)]
    assert _params(cli_tui.TuiAdapter.confirm) == [
        ("details", None), ("question", None), ("yes", None), ("no", None)]
    assert _params(HeadlessAdapter.ask) == [("message", None), ("always_label", ""), ("background", False)]

    assert [f.name for f in dataclasses.fields(ApprovalReply)] == ["answer", "feedback"]
    assert dataclasses.fields(ApprovalReply)[1].default == ""
    assert cli_tui.PermissionPrompt.SHORTCUTS == {"y": "yes", "a": "always", "n": "no"}
    assert cli_tui.PermissionPrompt.ALLOW_SELECT is False

    # has_pending_confirm() stays approval-only once a second prompt type exists (asserted again in T3.1)
    async def scenario():
        app = cli_tui.DostuffTUI(session_id="pin-approval")
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.adapter.has_pending_confirm() is False
            replies = []

            async def ask():
                replies.append(await app.adapter.ask("Write a.txt", always_label="file writes"))
            app.run_worker(ask())
            await _until(pilot, lambda: app.adapter.has_pending_confirm())
            app.adapter.choose("yes")
            await _until(pilot, lambda: replies)
            assert replies == [ApprovalReply("yes")]
            assert app.adapter.has_pending_confirm() is False
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_pin_approval_prompt_still_picks(tui):
    """test_permission_selector's behaviour, once through the pin name: a Phase 4 panel change that
    touched the approval widget fails here first."""
    cli_tui = tui
    from dostuff.agent.permissions import ApprovalReply

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="pin-picks")
        async with app.run_test() as pilot:
            await pilot.pause()
            replies = []
            inp = app.query_one("#input", cli_tui.EnterSubmits)

            async def ask():
                replies.append(await app.adapter.ask("Write a.txt", always_label="file writes"))

            # A number key picks that option at once
            app.run_worker(ask())
            await _until(pilot, lambda: app.query("#permission-prompt"))
            assert not inp.display
            await pilot.press("1")
            await _until(pilot, lambda: replies)
            assert replies[-1] == ApprovalReply("yes")
            assert inp.display

            # y allows, n asks for feedback that goes back with the denial
            app.run_worker(ask())
            await _until(pilot, lambda: app.query("#permission-prompt"))
            await pilot.press("y")
            await _until(pilot, lambda: len(replies) == 2)
            assert replies[-1] == ApprovalReply("yes")

            app.run_worker(ask())
            await _until(pilot, lambda: app.query("#permission-prompt"))
            await pilot.press("n")
            await _until(pilot, lambda: inp.has_class("feedback-mode"))
            inp.text = "use edit_file instead"
            await pilot.press("enter")
            await _until(pilot, lambda: len(replies) == 3)
            assert replies[-1] == ApprovalReply("no", feedback="use edit_file instead")
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def _message_lines(app) -> list[str]:
    """Every message widget's rendered plain text, one per message (wrapped lines joined with a space)."""
    lines = []
    for w in app.query("#msg-container Static"):
        rendered = w.render() if hasattr(w, "render") else getattr(w, "content", "")
        text = rendered.plain if hasattr(rendered, "plain") else str(rendered)
        lines.append(text.replace("\n", " "))
    return lines


def test_pin_transcript_without_ask_user(tui):
    """CON-007: a stored session with no ask_user step renders byte-identically after Phase 5.

    The message lines are pinned to `snapshots/transcript_no_ask_user.txt` (one per message, written from
    today's output)."""
    import json
    cli_tui = tui
    snapshot = json.loads((Path(__file__).parent / "snapshots" / "old_delegate_session.json").read_text("utf-8"))
    expected_file = Path(__file__).parent / "snapshots" / "transcript_no_ask_user.txt"

    async def scenario() -> list[str]:
        app = cli_tui.DostuffTUI(session_id="pin-transcript")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for step in snapshot:
                await app.store.append("old", step)
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/resume old"))
            await _until(pilot, lambda: app.session_id == "old", tries=200)
            await pilot.pause(0.3)
            lines = _message_lines(app)
            idx = max(i for i, t in enumerate(lines) if "Where is auth handled?" in t)
            await app._cleanup_and_save_session()
            return lines[idx:]

    lines = asyncio.run(scenario())
    assert lines == expected_file.read_text(encoding="utf-8").splitlines()


# --- ask-user T3.2: the input answers a question, and the three gates (REQ-3.1, D7) ----------------

def _ask_question(app, questions):
    """Start an `ask_user` call the way the tool does, and return its task."""
    return asyncio.ensure_future(app.adapter.ask_user(questions))


def test_typed_text_answers_the_question(tui):
    """A free-text question takes the input: typed text is the answer, never a new prompt."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask")
        async with app.run_test() as pilot:
            await pilot.pause()
            assert not app._turn_in_progress
            task = _ask_question(app, [{"question": "Name the app?", "options": [], "header": "Q1"}])
            await _until(pilot, app.adapter.has_pending_ask)
            assert app.query_one("#input", cli_tui.EnterSubmits).has_class("feedback-mode")
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("Tiger"))
            await _until(pilot, lambda: task.done())
            assert task.result() == [("Name the app?", "Tiger")]
            assert not app._turn_in_progress  # no turn started under the question
            assert not app.adapter.has_pending_ask()
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            assert not inp.has_class("feedback-mode") and inp.placeholder == ""
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_empty_enter_keeps_the_question(tui):
    """Open point 4: an empty Enter records nothing; Esc is the way out."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-empty")
        async with app.run_test() as pilot:
            await pilot.pause()
            task = _ask_question(app, [{"question": "Name the app?"}])
            await _until(pilot, app.adapter.has_pending_ask)
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(""))
            await pilot.pause()
            assert app.adapter.has_pending_ask() and not task.done()
            assert "type an answer" in (app.query_one("#input", cli_tui.EnterSubmits).border_title or "")
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            assert task.result() is None
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_esc_cancels_question(tui):
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-esc")
        async with app.run_test() as pilot:
            await pilot.pause()
            task = _ask_question(app, [{"question": "Name the app?"}])
            await _until(pilot, app.adapter.has_pending_ask)
            await app.action_cancel()
            await _until(pilot, lambda: task.done())
            assert task.result() is None
            assert not app.adapter.has_pending_ask()
            assert app.adapter._cancelled  # Esc still ends the turn, as on an approval
            assert app._title_state != "waiting"
            app.adapter._cancelled = False
            app.adapter._cancel_event.clear()
            # idle again: Esc goes back to the "nothing to cancel" path
            await app.action_cancel()
            await _until(pilot, lambda: str(app.query_one("#input", cli_tui.EnterSubmits).border_title) == cli_tui.QUIT_HINT)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_ctrl_r_blocked_during_question(tui):
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-ctrlr")
        async with app.run_test() as pilot:
            await pilot.pause()
            task = _ask_question(app, [{"question": "Name the app?"}])
            await _until(pilot, app.adapter.has_pending_ask)
            app.action_history_search()
            await pilot.pause()
            assert not app._picker_open()  # no history picker on top of the question
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_gate_is_approval_only(tui, monkeypatch):
    """`has_pending_confirm()` keeps its approval-only meaning; the app asks `has_pending_prompt()`."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-gate")
        async with app.run_test() as pilot:
            await pilot.pause()
            task = _ask_question(app, [{"question": "Name the app?"}])
            await _until(pilot, app.adapter.has_pending_ask)
            assert app.adapter.has_pending_prompt() and not app.adapter.has_pending_confirm()
            seen = []
            monkeypatch.setattr(app.adapter, "has_pending_prompt", lambda: seen.append(1) or True)
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("Tiger"))
            assert seen, "the input gate must ask has_pending_prompt()"
            await _until(pilot, lambda: task.done())
            assert task.result() == [("Name the app?", "Tiger")]
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- ask-user Phase 4: the panel (REQ-4.1 - REQ-4.4, D6) -----------------------------------------

def _ask_questions(app, questions):
    """An `ask_user` call as the tool makes it; the panel it drives is `#ask-panel`."""
    return asyncio.ensure_future(app.adapter.ask_user(questions))


DB_QUESTION = {
    "question": "Which database should I use?",
    "header": "DB",
    "options": [{"label": "Postgres", "description": "battle-tested, but another service to run"},
                {"label": "SQLite", "description": "zero setup, single file, no concurrent writers"}],
}
AUTH_QUESTION = {
    "question": "Which auth methods?",
    "header": "Auth",
    "multi_select": True,
    "options": [{"label": "JWT"}, {"label": "OAuth"}],
}


def test_ask_panel_render(tui):
    """`#ask-panel` is composed once and hidden; shown, it is the tab strip, the question, the numbered
    options with one-row descriptions, `Other` last, the separator, the submit row and the hint."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-render")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            assert panel.is_mounted and not panel.display  # composed hidden, never mounted per question
            assert len(app.query("#ask-panel")) == 1
            task = _ask_questions(app, [DB_QUESTION, AUTH_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            assert app.focused is panel  # the panel owns the input while a question is open
            await pilot.pause()
            lines = str(panel.content).split("\n")
            assert lines[0].strip() == "▌ DB ·   Auth"
            assert lines[1].strip() == "Which database should I use?"
            assert "1. Postgres" in lines[2] and "battle-tested" in lines[2]
            assert "2. SQLite" in lines[3] and "zero setup" in lines[3]
            assert lines[4].strip() == "3. Other (type your own)"
            assert set(lines[5].strip()) == {"─"}
            assert "Submit answers" in lines[6] and "answer every question first" in lines[6]
            assert "Esc cancel" in lines[7] and "Enter confirm" in lines[7]
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_ask_panel_keys(tui):
    """`↓`/`j`/`↑`/`k` move and wrap; `1`-`9` pick; `Tab`/`→` next question or the submit row;
    `Shift+Tab`/`←` back; a click picks the row; long lists scroll with the picker's markers."""
    cli_tui = tui
    many = {"question": "Pick one", "header": "Many",
            "options": [{"label": f"option {i}"} for i in range(1, 13)]}

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-keys")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            task = _ask_questions(app, [many, AUTH_QUESTION, DB_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.pause()

            # the list is longer than the panel: the markers and the cursor window are the picker's
            assert "↓ 3 more" in str(panel.content)  # 13 rows, the picker's 10 visible
            assert "↑ " not in str(panel.content)  # nothing has scrolled off the top yet
            for _ in range(7):
                await pilot.press("down")
            assert panel.cursor_row == 7 and "↑ 2 more" in str(panel.content)
            await pilot.press("up", "up")
            assert panel.cursor_row == 5
            await pilot.press("k")
            assert panel.cursor_row == 4
            await pilot.press("j")
            assert panel.cursor_row == 5

            # wrap at both ends, counting the `Other` row
            panel.cursor_row = 0
            await pilot.press("up")
            assert panel.cursor_row == len(many["options"])  # the Other row, wrapped from the top
            await pilot.press("down")
            assert panel.cursor_row == 0

            # a number picks that row (single-select records and auto-advances)
            await pilot.press("2")
            await pilot.pause()
            assert panel.answers()[0] == "option 2"
            assert panel.cursor_question == 1  # the next unanswered question

            # Tab / right move on, shift+tab / left come back, Tab on the last one goes to the submit row
            await pilot.press("tab")
            assert panel.cursor_question == 2
            await pilot.press("shift+tab")
            assert panel.cursor_question == 1
            await pilot.press("right")
            assert panel.cursor_question == 2
            await pilot.press("left")
            assert panel.cursor_question == 1
            await pilot.press("2", "right", "right")
            assert panel.submit_focused  # past the last question
            await pilot.press("left")
            assert panel.cursor_question == 2 and not panel.submit_focused

            # a click picks the row under the pointer (one border row above the tab strip, then the question)
            app._show_ask_panel([DB_QUESTION])
            await pilot.pause()
            await _until(pilot, lambda: panel.cursor_question == 0 and not panel.answers()[0])
            await pilot.click("#ask-panel", offset=(6, 4))  # the second option's row
            await pilot.pause()
            assert panel.answers() == ["SQLite"]

            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_ask_panel_reused_hides_and_shows(tui):
    """Two consecutive calls use the same widget instance, and it is hidden in between."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-reuse")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            first = _ask_questions(app, [DB_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("1", "tab", "enter")
            await _until(pilot, lambda: first.done())
            assert not panel.display and app.query_one("#input", cli_tui.EnterSubmits).display
            second = _ask_questions(app, [AUTH_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            assert app.query_one("#ask-panel", cli_tui.AskUserPanel) is panel  # never mounted again
            assert panel.answers() == [None]  # the previous call's answers are gone
            assert "DB" not in str(panel.content)  # ...and its tab strip too
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: second.done())
            assert first.result() == [("Which database should I use?", "Postgres")]
            assert second.result() is None
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_pick_records_and_advances(tui):
    """A pick records the answer and stays open: the cursor moves to the next unanswered question (D6)."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-pick")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            task = _ask_questions(app, [DB_QUESTION, AUTH_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("1")
            await pilot.pause()
            assert panel.answers()[0] == "Postgres"
            assert panel.display and app.adapter.has_pending_ask()  # nothing submitted
            assert not task.done()
            assert panel.cursor_question == 1  # the next unanswered question
            assert "✓ DB" in str(panel.content)  # the tab strip is the review
            assert "▌ Auth" in str(panel.content)  # ...and marks where the cursor went
            await pilot.press("1")  # multi-select: a toggle, and the cursor stays for the next pick
            await pilot.pause()
            assert panel.answers()[1] == "JWT"
            assert panel.cursor_question == 1 and not task.done()
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_submit_row_inert_until_answered(tui):
    """The submit row is dim with the hint until every question is answered; Enter there only flashes it."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-inert")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            task = _ask_questions(app, [DB_QUESTION, AUTH_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("1")  # only question 1 answered: auto-advance put the cursor on question 2
            await pilot.pause()
            await pilot.press("tab")  # past the last question: the submit row
            await pilot.pause()
            assert panel.submit_focused
            assert "answer every question first" in str(panel.content)
            await pilot.press("enter")
            await pilot.pause()
            assert not task.done() and app.adapter.has_pending_ask()  # Enter resolved nothing
            hint = app.query_one("#input", cli_tui.EnterSubmits).border_title or ""
            assert "answer every question first" in hint or "answer every" in str(hint)
            await pilot.press("1")  # answer question 2 from the submit row
            await pilot.pause()
            assert panel.all_answered()
            assert "answer every question first" not in str(panel.content)
            await pilot.press("tab", "enter")
            await _until(pilot, lambda: task.done())
            assert task.result() == [("Which database should I use?", "Postgres"), ("Which auth methods?", "JWT")]
            assert not panel.display
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_change_an_answer_before_submitting(tui):
    """`←` back to a question and pick again: single-select replaces, and the result carries one pair."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-change")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            task = _ask_questions(app, [DB_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("1")
            await pilot.pause()
            assert panel.answers() == ["Postgres"]
            await pilot.press("left")  # back to the only question
            await pilot.pause()
            assert panel.cursor_question == 0 and not panel.submit_focused
            await pilot.press("2")
            await pilot.pause()
            assert panel.answers() == ["SQLite"]  # replaced, not appended
            await pilot.press("tab", "enter")
            await _until(pilot, lambda: task.done())
            assert task.result() == [("Which database should I use?", "SQLite")]
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_single_question_still_needs_submit(tui):
    """One question, one pick: the call is not resolved until the submit row (open question: no auto-submit)."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-single")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            captures: list = []

            async def ask():
                result = await app.adapter.ask_user([DB_QUESTION])
                captures.append(result)
                return result
            task = asyncio.ensure_future(ask())
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("1")
            await pilot.pause()
            assert not captures and not task.done()  # 0 resolutions after the pick
            assert panel.submit_focused  # the only question is answered, so the cursor is on the submit row
            await pilot.press("enter")
            await _until(pilot, lambda: bool(captures))
            assert len(captures) == 1  # 1 after the submit
            assert captures[0] == [("Which database should I use?", "Postgres")]
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_submit_row_click(tui):
    """Clicking the submit row (bright, every question answered) submits; clicking it while inert does not."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-click-submit")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            task = _ask_questions(app, [DB_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.pause()
            # the submit row is 4 rows down from the panel's top: border, tabs, question, 3 rows, separator
            await pilot.click("#ask-panel", offset=(6, 7))
            await pilot.pause()
            assert not task.done()  # inert: no question answered yet
            await pilot.press("1")  # auto-advance puts the cursor on the submit row
            await pilot.pause()
            await pilot.click("#ask-panel", offset=(6, 7))
            await _until(pilot, lambda: task.done())
            assert task.result() == [("Which database should I use?", "Postgres")]
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_multi_select_toggles(tui):
    """`space` and the number key toggle (not replace); submit joins in option order, never pick order."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-multi")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            task = _ask_questions(app, [AUTH_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("2")  # OAuth first...
            await pilot.pause()
            assert panel.answers() == ["OAuth"]
            await pilot.press("1")  # ...then JWT: still option order
            await pilot.pause()
            assert panel.answers() == ["JWT, OAuth"]
            assert panel.cursor_question == 0 and not panel.submit_focused  # a toggle stays put
            # the rows show the toggles in place, and the order never changes
            rows = [line for line in str(panel.content).split("\n") if "1. JWT" in line or "2. OAuth" in line]
            assert "✓ 1. JWT" in rows[0] and "✓ 2. OAuth" in rows[1]
            # a second press toggles it off, with space too
            await pilot.press("1")
            await pilot.pause()
            assert panel.answers() == ["OAuth"]
            panel.cursor_row = 1
            await pilot.press("space")
            await pilot.pause()
            assert panel.answers() == [None]
            await pilot.press("2", "1", "tab", "enter")
            await _until(pilot, lambda: task.done())
            assert task.result() == [("Which auth methods?", "JWT, OAuth")]
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_multi_select_with_other(tui):
    """`JWT` + a typed `my own` → `JWT, my own` (typed last, D5); toggling Other off drops it; Other alone is
    just the typed text."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-multi-other")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            task = _ask_questions(app, [AUTH_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("1")  # JWT
            await pilot.pause()
            await pilot.press("3")  # the Other row
            await pilot.pause()
            assert not panel.display and inp.display and inp.has_class("feedback-mode")
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("my own"))
            await pilot.pause()
            assert panel.display and not inp.display and app.focused is panel
            assert panel.answers() == ["JWT, my own"]
            assert "✓ 3. Other (type your own)" in str(panel.content)
            await pilot.press("3", "tab", "enter")  # Other off again
            await _until(pilot, lambda: task.done())
            assert task.result() == [("Which auth methods?", "JWT")]

            # Other alone is just the typed text (open point 3)
            panel_task = _ask_questions(app, [AUTH_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("3")
            await pilot.pause()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("saml"))
            await pilot.pause()
            assert panel.answers() == ["saml"]
            await pilot.press("enter")  # the cursor is on the submit row
            await _until(pilot, lambda: panel_task.done())
            assert panel_task.result() == [("Which auth methods?", "saml")]
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_other_takes_the_input(tui):
    """Picking `Other` hides the panel and shows the input; Enter records the text and the panel comes back."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-other")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            task = _ask_questions(app, [DB_QUESTION, AUTH_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("3")
            await pilot.pause()
            assert not panel.display and inp.display and app.focused is inp
            assert "Type your answer" in inp.placeholder or "Which database" in inp.placeholder
            assert inp.has_class("feedback-mode")  # the yellow prompt border
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("DuckDB"))
            await pilot.pause()
            assert panel.display and not inp.display
            assert panel.answers() == ["DuckDB", None]
            await pilot.press("left")  # back to question 1: the typed answer is shown in place
            await pilot.pause()
            assert "✓ 3. Other (type your own): DuckDB" in str(panel.content)
            await pilot.press("right")  # back to the unanswered question
            await pilot.pause()
            assert panel.cursor_question == 1
            # a paste is inlined, as the input already does for a prompt
            await pilot.press("3")
            await pilot.pause()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(app._pastes.inline("JWT,\n  OAuth")))
            await pilot.pause()
            assert panel.answers()[1] == "JWT, OAuth"
            await pilot.press("enter")  # the cursor is on the submit row
            await _until(pilot, lambda: task.done())
            assert task.result() == [("Which database should I use?", "DuckDB"), ("Which auth methods?", "JWT, OAuth")]
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_free_text_only_skips_the_panel(tui):
    """No question with options never shows the panel: each Enter is one answer, the last one is the submission."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-free")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            task = _ask_questions(app, [{"question": "Name the app?", "header": "Name"},
                                        {"question": "Who is it for?", "header": "Who"}])
            await _until(pilot, app.adapter.has_pending_ask)
            assert not panel.display  # nothing to submit separately
            assert inp.display and inp.has_class("feedback-mode")
            assert inp.placeholder == "Name the app?"
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("Tiger"))
            await pilot.pause()
            assert not task.done() and app.adapter.has_pending_ask()
            assert inp.placeholder == "Who is it for?"  # the next question, same input
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("me"))
            await _until(pilot, lambda: task.done())
            assert task.result() == [("Name the app?", "Tiger"), ("Who is it for?", "me")]
            assert not inp.has_class("feedback-mode") and inp.placeholder == ""
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_mixed_call_hands_the_input_over(tui):
    """Some option questions, some free text: the panel is used, and the free-text question's Enter
    takes the input (it never grabs it — see `test_free_text_question_does_not_take_the_input_by_surprise`)."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-mixed")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            task = _ask_questions(app, [DB_QUESTION, {"question": "Name the app?", "header": "Name"}])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("1")  # Postgres → advance to the free-text question, panel still up
            await pilot.pause()
            assert panel.display and not inp.display
            assert panel.cursor_question == 1
            await pilot.press("enter")  # asking for the typed answer is one deliberate key
            await _until(pilot, lambda: inp.display)
            assert not panel.display and inp.has_class("feedback-mode")
            assert "Name the app?" in inp.placeholder
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("Tiger"))
            await pilot.pause()
            assert panel.display and not inp.display
            assert panel.answers() == ["Postgres", "Tiger"]
            await pilot.press("enter")  # every question answered: the cursor is on the submit row
            await _until(pilot, lambda: task.done())
            assert task.result() == [("Which database should I use?", "Postgres"), ("Name the app?", "Tiger")]
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_empty_input_keeps_the_question(tui):
    """Open point 4: an empty Enter records nothing and flashes the hint; the question stays open."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-empty-panel")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            task = _ask_questions(app, [DB_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("3")  # Other
            await pilot.pause()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("   "))
            await pilot.pause()
            assert panel.answers() == [None] and app.adapter.has_pending_ask() and not task.done()
            hint = app.query_one("#input", cli_tui.EnterSubmits).border_title or ""
            assert "type an answer" in hint
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_free_text_answer_is_one_line(tui):
    """A pasted two-line answer is collapsed before it is recorded, so the result still parses."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-oneline")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            task = _ask_questions(app, [{"question": "Describe it?"}])
            await _until(pilot, app.adapter.has_pending_ask)
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("line one\n   line two"))
            await _until(pilot, lambda: task.done())
            assert task.result() == [("Describe it?", "line one line two")]
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_esc_mid_question(tui):
    """Esc on a call, the first question already answered: the whole call is cancelled, and the input is
    usable again with its draft."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-esc-panel")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            task = _ask_questions(app, [DB_QUESTION, AUTH_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("1")
            await pilot.pause()
            assert panel.answers()[0] == "Postgres"
            await app.action_cancel()
            await _until(pilot, lambda: task.done())
            assert task.result() is None  # cancelled, not partially answered (open point 7)
            assert not panel.display and inp.display
            assert not inp.has_class("feedback-mode") and inp.placeholder == ""
            assert app.adapter._cancelled  # Esc still ends the turn, as on an approval
            app.adapter._cancelled = False
            app.adapter._cancel_event.clear()
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_esc_while_typing_other(tui):
    """Esc during the `Other` input cancels the call: the typed draft is not sent, and the input is normal."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-esc-other")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            task = _ask_questions(app, [DB_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("3")
            await pilot.pause()
            assert inp.display and inp.has_class("feedback-mode")
            inp.text = "half-typed"
            await app.action_cancel()
            await _until(pilot, lambda: task.done())
            assert task.result() is None
            assert not panel.display and inp.display and not inp.has_class("feedback-mode")
            assert "half-typed" not in inp.text  # the prompt input's draft, not the cancelled answer
            app.adapter._cancelled = False
            app.adapter._cancel_event.clear()
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_ctrl_q_mid_question(tui):
    """Ctrl+Q with a question open resolves it as cancelled first, then follows the normal quit path."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-quit")
        exited: list = []
        app.exit = lambda *a, **k: exited.append(True)  # type: ignore[method-assign]
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            task = _ask_questions(app, [DB_QUESTION])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("ctrl+q")
            await _until(pilot, lambda: bool(exited), tries=200)
            assert task.done() and task.result() is None  # resolved, not cancelled by the event loop
            assert not app.adapter.has_pending_ask()

    asyncio.run(scenario())


def test_exit_command_mid_question(tui):
    """/exit behaves as Ctrl+Q: the question is resolved as cancelled before the app leaves."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-exit")
        exited: list = []
        app.exit = lambda *a, **k: exited.append(True)  # type: ignore[method-assign]
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            task = _ask_questions(app, [DB_QUESTION])
            await _until(pilot, app.adapter.has_pending_ask)
            # /exit can only be typed in the input, which the panel has taken: Esc is the panel's way out,
            # and the command still works while a question is open through the adapter's cancel
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            assert task.result() is None
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/exit"))
            await _until(pilot, lambda: bool(exited), tries=200)

    asyncio.run(scenario())


def test_cancelled_call_costs_one_step(tui, monkeypatch, hook_cmd, tmp_path):
    """After Esc the turn summary says `Cancelled after …`, and a `PostToolUse` hook sees the cancel text."""
    import json
    from types import SimpleNamespace
    from dostuff.agent import loop as loop_module
    cli_tui = tui

    async def real_loop(*args, **kwargs):
        return await loop_module.loop(*args, **kwargs)
    monkeypatch.setattr(cli_tui, "agent_loop", real_loop)

    def msg(content=None, tool_calls=None):
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    raw = json.dumps({"questions": [{"question": "Which database should I use?",
                                     "options": [{"label": "Postgres"}, {"label": "SQLite"}]}]})
    tc = SimpleNamespace(id="c1", type="function", function=SimpleNamespace(name="ask_user", arguments=raw))
    tc.model_dump = lambda: {"id": "c1", "type": "function", "function": {"name": "ask_user", "arguments": raw}}
    responses: list = [msg(tool_calls=[tc])]

    async def fake_call_agent(steps_history, system_instruction, tool_names=None, stream=False):
        if not responses:
            return msg("done"), None
        return responses.pop(0), None
    monkeypatch.setattr(loop_module, "call_agent", fake_call_agent)
    monkeypatch.setattr(loop_module, "get_model_token_limit", lambda: 100_000)

    log = tmp_path / "post.log"
    post = hook_cmd(f"import json, sys\nopen({str(log)!r}, 'w').write(json.load(sys.stdin)['tool_response'])\n"
                    f"print('hook-ran')")
    _write_hooks({"PostToolUse": [{"hooks": [{"type": "command", "name": "see", "command": post}]}]})

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-step")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("build it"))
            await _until(pilot, app.adapter.has_pending_ask, tries=200)
            await pilot.press("escape")
            await _until(pilot, lambda: not app._turn_in_progress, tries=300)
            texts = _texts(app)
            assert any(t.startswith("✻ Cancelled after") for t in texts)
            tool_steps = [s for s in app.steps_history if s.get("role") == "tool"]
            assert tool_steps and "cancelled the questions" in tool_steps[0]["content"]
            # The hook ran on the cancelled call and saw the D5 cancel text (spec D7: accepted, pinned)
            assert log.exists() and "cancelled the questions" in log.read_text()
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- ask-user T4.6: the panel does not bounce the transcript (REQ-4.1) ----------------------------

def test_ask_panel_does_not_bounce_the_transcript(tui):
    """Opening the questions panel, arrowing through questions of different height and closing it must leave
    every painted frame flush at the bottom, like the /resume picker and the approval prompt before it
    (`test_picker_does_not_bounce_the_transcript`). The panel is taller than the input it replaces."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-bounce")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            for i in range(25):  # enough to overflow the viewport
                app._append(f"row{i:02d} " + "-" * 30, msg_type="user")
            await pilot.pause()
            await pilot.pause()
            messages = app.query_one("#messages")
            assert isinstance(messages, cli_tui.Transcript)
            off_bottom: list[int] = []
            original_display = app._display

            def display(screen, renderable):
                """Measure what is actually painted, before it goes to the terminal."""
                rows = [strip.text.rstrip() for strip in
                        app.screen._compositor.render_strips()[:messages.region.height]]
                occupied = [y for y, row in enumerate(rows) if row]
                if occupied:  # empty viewport: nothing to be flush with
                    off_bottom.append(messages.region.height - 1 - occupied[-1])
                return original_display(screen, renderable)
            app._display = display  # type: ignore[method-assign]

            # As if the user arrows through the questions: the panel changes height with the option count
            task = _ask_questions(app, [DB_QUESTION, AUTH_QUESTION])
            await _until(pilot, app.adapter.has_pending_ask)
            await pilot.pause()
            for key in ("tab", "right", "shift+tab", "left", "tab"):
                await pilot.press(key)
                await pilot.pause()
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            await pilot.pause()
            await pilot.pause()
            app._display = original_display  # type: ignore[method-assign]

            assert off_bottom, "no frame was measured"
            assert max(off_bottom) <= 0, f"the transcript drifted off the bottom: {off_bottom}"
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_ask_panel_keeps_a_scrolled_up_transcript_in_place(tui):
    """`_hold_transcript` must not drag a transcript the user scrolled up back to the bottom either."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-scrolled")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            for i in range(25):
                app._append(f"row{i:02d} " + "-" * 30, msg_type="user")
            await pilot.pause()
            messages = app.query_one("#messages", cli_tui.Transcript)
            messages.scroll_to(y=10, animate=False)
            await pilot.pause()
            await pilot.pause()
            assert messages.scroll_y == 10

            task = _ask_questions(app, [DB_QUESTION])
            await _until(pilot, app.adapter.has_pending_ask)
            await pilot.pause()
            await pilot.pause()
            assert messages.scroll_y == 10  # the panel opening did not scroll the conversation
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            await pilot.pause()
            await pilot.pause()
            assert messages.scroll_y == 10  # nor did it closing
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- ask-user T4.7: title and notification (REQ-4.6, D7) ------------------------------------------

def test_question_notifies(tui):
    """An open question puts the title on `waiting` and notifies a terminal that is in the background;
    answering it puts the title back, exactly as an approval prompt does."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-notify")
        async with app.run_test() as pilot:
            await pilot.pause()
            written: list[str] = []
            app._write_terminal = written.append  # type: ignore[method-assign]
            app._notify_kind = "osc9"
            app.app_focus = False
            task = _ask_questions(app, [DB_QUESTION])
            await _until(pilot, app.adapter.has_pending_ask)
            await pilot.pause()
            assert app._title_state == "waiting"
            assert any("Which database should I use?" in w for w in written), written
            assert any("approval needed" in w for w in written)  # open point 10: one voice for both prompts

            written.clear()
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            await pilot.pause()
            assert app._title_state == "done"  # idle again, not `running`: no turn is going
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_question_notifies_after_the_wait(tui):
    """A question left open past `ui.notify_after_seconds` notifies even while the terminal is focused
    (`should_notify`, the rule approvals already use)."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-notify-late")
        async with app.run_test() as pilot:
            await pilot.pause()
            written: list[str] = []
            app._write_terminal = written.append  # type: ignore[method-assign]
            app._notify_kind = "osc9"
            app.app_focus = True
            app.config.notify_after_seconds = 30
            app._turn_start_time = time.time() - 60  # the turn has been waiting over the threshold
            task = _ask_questions(app, [DB_QUESTION])
            await _until(pilot, app.adapter.has_pending_ask)
            await pilot.pause()
            assert any("Which database should I use?" in w for w in written), written
            app._turn_start_time = None
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- ask-user T5.1: the result renders as answers (REQ-4.5, D9) -------------------------------------

ASK_RESULT_TEXT = (
    "Asked the user 2 questions; they answered:\n"
    "1. Which database should I use?\n"
    "   → Postgres\n"
    "2. Which auth methods?\n"
    "   → JWT, OAuth"
)


def test_ask_result_block(tui):
    """A live result renders as one ⎿ block with one row per answer, in full — not as a raw
    `ask_user: Asked the user 2 questions…` line, and not cut to the preview height."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-result")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.adapter.emit("tool_result", f"ask_user: {ASK_RESULT_TEXT}")
            await pilot.pause()
            text = "\n".join(_message_lines(app))
            assert "Which database should I use? → Postgres" in text
            assert "Which auth methods? → JWT, OAuth" in text
            assert "ask_user:" not in text  # the raw prefix is gone
            # not a truncated ToolResult: the block is a plain message, so Ctrl+O has nothing to expand
            assert not app.query(cli_tui.ToolResult) or all(
                "asked the user" not in str(w.content).lower() for w in app.query(cli_tui.ToolResult))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_ask_result_label_is_short(tui):
    """The call line is `● ask_user("Which database should I use?" +2 more)`, not a JSON blob."""
    cli_tui = tui
    label = cli_tui.format_call_label("ask_user", '{"questions": [{"question": "Which database should I use?"},'
                                                 ' {"question": "Which auth?"}, {"question": "Which cache?"}]}')
    assert label == 'ask_user("Which database should I use?" +2 more)'


def test_ask_result_replays(tui):
    """A stored session renders the same call label and answer block on /resume as it did live."""
    cli_tui = tui
    call = {"id": "c1", "type": "function", "function": {
        "name": "ask_user",
        "arguments": '{"questions": [{"question": "Which database should I use? methods"},'
                     ' {"question": "Which cache?"}]}',
    }}
    steps = [
        {"role": "user", "content": "build it"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "name": "ask_user", "tool_call_id": "c1", "content": ASK_RESULT_TEXT},
        {"role": "assistant", "content": "Done."},
    ]

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-replay")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for step in steps:
                await app.store.append("ask-old", step)
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/resume ask-old"))
            await _until(pilot, lambda: app.session_id == "ask-old", tries=200)
            await pilot.pause(0.3)
            lines = _message_lines(app)
            await app._cleanup_and_save_session()
            return lines

    lines = asyncio.run(scenario())
    call_lines = [t for t in lines if t.startswith("●")]
    assert any('ask_user("Which database should I use? methods" +1 more)' in t for t in call_lines), call_lines
    # one ⎿ block, a row per pair (the message contains the newlines; _message_lines joins them)
    block = next(t for t in lines if t.lstrip().startswith("⎿"))
    assert "Which database should I use? → Postgres" in block
    assert "Which auth methods? → JWT, OAuth" in block
    assert not any(t.startswith("ask_user: ") for t in lines)


def test_cancelled_result_renders_as_tool_result(tui):
    """The cancel, unavailable and refusal texts render as ordinary ⎿ results: no answer block, no lost text —
    live and on /resume (T5.3, CON-007)."""
    cli_tui = tui
    from dostuff.helpers.ui.ask_answers import CANCELLED_ASK_RESULT, UNAVAILABLE_ASK_RESULT, parse_answers
    assert parse_answers(CANCELLED_ASK_RESULT) is None
    assert parse_answers(UNAVAILABLE_ASK_RESULT) is None
    assert parse_answers("error: too many questions") is None
    stored = [
        {"role": "user", "content": "build it"},
        {"role": "tool", "name": "ask_user", "tool_call_id": "c1", "content": UNAVAILABLE_ASK_RESULT},
    ]

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-plain")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.adapter.emit("tool_result", f"ask_user: {CANCELLED_ASK_RESULT}")
            await pilot.pause()
            live = "\n".join(_message_lines(app))
            for step in stored:
                await app.store.append("ask-plain-old", step)
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/resume ask-plain-old"))
            await _until(pilot, lambda: app.session_id == "ask-plain-old", tries=200)
            await pilot.pause(0.3)
            replayed = "\n".join(_message_lines(app))
            await app._cleanup_and_save_session()
            return live, replayed

    live, replayed = asyncio.run(scenario())
    assert "cancelled the questions" in live and "→" not in live
    assert "No user is available" in replayed


def test_placeholder_is_dimmed(tui):
    """The "Other"/free-text questions put their text in the input's placeholder, which Textual paints at
    `$text 40%` — plain white under the ansi-dark theme, so it reads as a real prompt rather than a hint.
    The input overrides it to a dim grey, and that is the single place both callers go through."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-placeholder-dim")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            style = inp.get_visual_style("text-area--placeholder")
            color = style.foreground
            assert color is not None, "the placeholder has no colour"
            assert getattr(color, "a", 1.0) == 1.0, f"still blended: {color!r}"
            # #6a6a6a — the same grey the ghost suggestion text uses
            assert (color.r, color.g, color.b) == (0x6A, 0x6A, 0x6A), color
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_free_text_question_does_not_take_the_input_by_surprise(tui):
    """A mixed call's free-text question must not swallow the panel: arrowing onto it (or Tab, or the
    automatic advance after a pick) leaves the question on screen, and Enter — or `o` — takes the input.
    Only Esc gives up the whole call from there, so the input must be reachable but not a trap."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-free-in-panel")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            question = {"question": "Name the app?", "header": "Name"}
            task = _ask_questions(app, [DB_QUESTION, question])
            await _until(pilot, lambda: panel.display and app.focused is panel)

            # Tab onto it without answering: the panel stays, the input stays hidden
            await pilot.press("tab")
            await pilot.pause()
            assert panel.display and not inp.display
            assert panel.cursor_question == 1
            # and back again
            await pilot.press("shift+tab")
            await pilot.pause()
            assert panel.display and panel.cursor_question == 0

            # Tab to it again, then Enter takes the input deliberately
            await pilot.press("tab")
            await pilot.pause()
            await pilot.press("enter")
            await _until(pilot, lambda: inp.display)
            assert not panel.display and inp.has_class("feedback-mode")
            assert inp.placeholder == "Name the app?"
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("Tiger"))
            await pilot.pause()
            # recorded, and the panel is back
            assert panel.display and not inp.display
            assert panel.answers() == [None, "Tiger"]
            # the free-text answer did not answer the option question, so submit stays inert
            await pilot.press("enter")
            await pilot.pause()
            assert app.adapter.has_pending_ask() and not task.done()
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_auto_advance_onto_a_free_text_question_keeps_the_panel(tui):
    """The advance after a pick lands on a free-text question instead of hiding the panel."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-advance-free")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            task = _ask_questions(app, [DB_QUESTION, {"question": "Name the app?", "header": "Name"}])
            await _until(pilot, lambda: panel.display and app.focused is panel)
            await pilot.press("1")  # Postgres, then the advance must not take the input
            await pilot.pause()
            assert panel.display and not inp.display
            assert panel.cursor_question == 1 and panel.answers() == ["Postgres", None]
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_o_key_takes_the_input_from_the_other_row(tui):
    """`o` is the one-key way in — and it edits rather than toggling a typed Other off, which is what
    Enter on that row does for a multi-select question."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-o-key")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            multi = {"question": "Which auth?", "header": "Auth", "multi_select": True,
                     "options": [{"label": "JWT"}, {"label": "OAuth"}]}
            task = _ask_questions(app, [{"question": "Name the app?", "header": "Name"}, multi])
            await _until(pilot, lambda: panel.display and app.focused is panel)

            # a free-text question: o takes the input, Enter records it, the panel comes back
            await pilot.press("o")
            await _until(pilot, lambda: inp.display)
            assert not panel.display and inp.placeholder == "Name the app?"
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("Tiger"))
            await pilot.pause()
            assert panel.display and panel.answers() == ["Tiger", None]

            # a multi-select question with a typed Other: o re-opens the input, it does not clear it
            await pilot.pause()
            assert panel.cursor_question == 1  # the advance already landed on the multi-select
            await pilot.press("down")
            await pilot.press("down")  # JWT → OAuth → the Other row
            await pilot.press("o")
            await _until(pilot, lambda: inp.display)
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("SAML"))
            await pilot.pause()
            assert panel.answers() == ["Tiger", "SAML"]
            # recording it advanced to the submit row; go back to the Other row
            assert panel.submit_focused
            await pilot.press("shift+tab")
            await pilot.pause()
            await pilot.press("down")
            await pilot.press("down")  # JWT → OAuth → the Other row
            assert panel.cursor_row == panel._other_row(panel.cursor_question)
            await pilot.press("o")  # the cursor is on the Other row: edit, do not clear
            await _until(pilot, lambda: inp.display)
            assert inp.display and not panel.display
            assert panel.answers() == ["Tiger", "SAML"], "o must not toggle the typed answer off"
            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_tab_strip_pads_and_marks_the_current_question(tui):
    """The tab strip is the only thing that says which question you are on, so the current one is marked
    twice: a few spaces of padding and the answer mark's slot are reserved on every tab so the tabs stay
    aligned, and the mark itself is bold + orange. `✓` (a plain glyph, not an emoji) still means answered."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="ask-tabs")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#ask-panel", cli_tui.AskUserPanel)
            three = [
                {"question": "Which database?", "header": "DB",
                 "options": [{"label": "SQLite"}, {"label": "Postgres"}]},
                {"question": "Which auth?", "header": "Auth", "multi_select": True,
                 "options": [{"label": "JWT"}, {"label": "OAuth"}]},
                {"question": "Name the app?", "header": "Name"},
            ]
            task = _ask_questions(app, three)
            await _until(pilot, lambda: panel.display and app.focused is panel)

            first = str(panel.content).split("\n")[0]
            assert first.strip() == "▌ DB ·   Auth ·   Name"
            # the mark sits in a fixed slot before the label, so every tab is the same width
            assert first.index("▌") < first.index("DB")
            assert "▌ Auth" not in first  # the mark belongs to DB, not Auth

            await pilot.press("tab")  # onto Auth without answering
            await pilot.pause()
            second = str(panel.content).split("\n")[0]
            assert second.strip() == "DB · ▌ Auth ·   Name"
            assert second.index("▌") > second.index("DB")
            assert len(second) == len(first), (repr(second), repr(first))

            # answering DB puts a ✓ in DB's slot, and the widths still do not change
            await pilot.press("shift+tab")
            await pilot.pause()
            await pilot.press("1")
            await pilot.pause()
            answered = str(panel.content).split("\n")[0]
            assert answered.strip() == "✓ DB · ▌ Auth ·   Name", answered
            assert len(answered) == len(first), (repr(answered), repr(first))

            # No emoji: `⚠` (U+26A0) on the question line and `✔` (U+2714) on the submit row both have
            # Emoji_Presentation, so terminals painted them in colour. `✓`/`▌`/`❯` are text glyphs and stay.
            body = str(panel.content)
            for emoji in ("\u26a0", "\u2714"):
                assert emoji not in body, f"{emoji!r} is still in the panel"
            assert not [ch for ch in body if ord(ch) >= 0x1F000], "a pictographic emoji slipped in"

            app.adapter.cancel_pending_ask()
            await _until(pilot, lambda: task.done())
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- /btw Phase 3: the command and the overlay (plan/2026-11-03-btw/tasks.md T3.x) --------------------------------

from conftest import fake_response  # noqa: E402

class HeldTurn:
    """A fake `agent_loop` that publishes a real snapshot (as `loop()` does) and holds until `release()`.
    `systems` records the system string each turn would have sent, composed as the loop composes it."""

    def __init__(self):
        self.gate: asyncio.Event | None = None
        self.systems: list[str] = []
        self.snapshots: list = []
        self.kwargs: list = []  # every call's kwargs, for the steering wiring pin
        self.hold = True

    def release(self):
        if self.gate is not None:
            self.gate.set()

    async def __call__(self, session_id, turn_id, user_text, instructions, mcp_client, working_history, turn_type,
                       current_session_history, steps_history, store, adapter=None, **kwargs):
        from dostuff.agent.subagents.fork import ForkContext
        from dostuff.helpers.agent import side_question
        from dostuff.helpers.agent.append_step import append_step
        self.kwargs.append(kwargs)
        notes = kwargs.get("compaction_notes") or ""
        context = kwargs.get("session_hook_context")
        with side_question.published_snapshot(session_id):
            snap = ForkContext(system_instruction=instructions, compaction_notes=notes,
                               history=[dict(s) for s in working_history], tool_names=["read_file", "bash"],
                               session_hook_context=dict(context) if context is not None else None)
            side_question.publish_snapshot(session_id, snap)
            self.snapshots.append(side_question.snapshot(session_id))
            self.systems.append(side_question.compose_system_instruction(
                instructions, side_question.plan_mode_on(kwargs.get("permissions")), notes, context))
            if self.hold:
                self.gate = asyncio.Event()
                await self.gate.wait()
            if adapter is not None and adapter._cancelled:
                return "[Cancelled]"
            step = {"role": "assistant", "content": f"answer to {user_text}"}
            await append_step(step, steps_history, working_history, current_session_history, session_id, store,
                              turn_type)
            return step["content"]


@pytest.fixture
def btw_tui(tui, monkeypatch, side_call):
    """(cli_tui, HeldTurn, side_call): turns are held until released; the side call is a `SideCall`."""
    import dostuff.agent.call_agent  # noqa: F401  the real module, so side_call patches it rather than stubbing it
    from dostuff.helpers.agent import side_question
    side_question._reset_registry()
    held = HeldTurn()
    monkeypatch.setattr(tui, "agent_loop", held)
    yield tui, held, side_call
    held.release()
    side_question._reset_registry()


def _submit(app, cli_tui, text):
    app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(text))


async def _btw_answered(pilot, app):
    panel = app._btw_panel()
    await _until(pilot, lambda: panel.display and panel.answer is not None and app.focused is panel)
    return panel


async def _state(app):
    """Everything a side question must leave alone (CON-004, CON-005), by value."""
    import copy
    import sqlite3
    meta = await app.store.get_session_meta(app.session_id) or {}
    counts = {}
    with sqlite3.connect(app.store.db_path) as db:
        for table in ("session_steps", "working_steps", "agent_runs"):
            try:
                counts[table] = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = None
    return {
        "steps": copy.deepcopy(app.steps_history), "current": copy.deepcopy(app.current_session_history),
        "working": await app.store.load_working_history(app.session_id), "queue": list(app._prompt_queue),
        "meta": {k: meta.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd",
                                          "compaction_notes", "last_input_tokens")},
        "counts": counts, "usage": dict(app.adapter._session_usage), "cost": app.adapter._session_cost,
    }


def test_btw_registered(tui):
    from dostuff.helpers.agent.constants import COMMAND_ARGS, COMMAND_DESCRIPTIONS, COMMAND_FLAGS
    cli_tui = tui
    assert "/btw" in COMMAND_DESCRIPTIONS and COMMAND_ARGS["/btw"] == "<question>" and "/btw" not in COMMAND_FLAGS

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-help")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "/help")
            await _until(pilot, lambda: any("/btw" in t for t in _texts(app)))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_usage_when_empty(btw_tui):
    cli_tui, held, side_call = btw_tui
    call = side_call()

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-usage")
        async with app.run_test() as pilot:
            await pilot.pause()
            for text in ("/btw", "/btw   "):
                _submit(app, cli_tui, text)
                await pilot.pause()
            assert sum(t == cli_tui.BTW_USAGE for t in _texts(app)) == 2
            assert call.calls == [] and app._prompt_queue == [] and not app._btw_open()
            _submit(app, cli_tui, "/btwx hello")  # not the command: an ordinary prompt
            await _until(pilot, lambda: held.gate is not None)
            assert call.calls == []
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_mid_turn(btw_tui):
    """Mid-turn: not queued, the turn keeps running, the live snapshot is what is sent, and nothing about the
    session changes (REQ-3.1, REQ-3.2, CON-004/005/006)."""
    from dostuff.helpers.agent.side_question import SIDE_QUESTION_PROMPT
    cli_tui, held, side_call = btw_tui
    usage = types.SimpleNamespace(prompt_tokens=1200, completion_tokens=340, total_tokens=1540,
                                  prompt_tokens_details=types.SimpleNamespace(cached_tokens=1100))
    call = side_call(fake_response("It edited a.py.", usage=usage))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-mid")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "fix the bug")
            await _until(pilot, lambda: held.gate is not None)
            before = await _state(app)
            lines_before = _message_lines(app)
            status_before = str(app.query_one("#status", cli_tui.Static).render())
            _submit(app, cli_tui, "/btw which file did it edit?")
            panel = await _btw_answered(pilot, app)
            assert app._turn_in_progress and app._prompt_queue == []
            assert len(call.calls) == 1
            assert call.calls[0]["steps_history"] == held.snapshots[0].history + [
                {"role": "user", "content": SIDE_QUESTION_PROMPT.format(question="which file did it edit?")}]
            assert call.calls[0]["system_instruction"] == held.systems[0]
            assert call.calls[0]["tool_names"] == ["read_file", "bash"]
            assert panel.body_text == "It edited a.py."
            footer = str(panel.query_one("#btw-footer", cli_tui.Static).render())
            assert footer.startswith("Esc/Enter/Space close · f fork out · c copy · ↑1.2k (1.1k cached) ↓340")
            assert await _state(app) == before
            assert _message_lines(app) == lines_before  # nothing in the transcript
            assert str(app.query_one("#status", cli_tui.Static).render()) == status_before
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            assert [s["role"] for s in app.steps_history] == ["user", "assistant"]
            assert not any("which file" in str(s.get("content")) for s in await app.store.load(app.session_id))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_expands_pastes(btw_tui):
    cli_tui, held, side_call = btw_tui
    call = side_call()

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-paste")
        async with app.run_test() as pilot:
            await pilot.pause()
            marker = app._pastes.add("line one\nline two\nline three")
            _submit(app, cli_tui, f"/btw what does this mean? {marker}")
            await _btw_answered(pilot, app)
            assert "line one\nline two\nline three" in call.calls[0]["steps_history"][-1]["content"]
            assert marker not in call.calls[0]["steps_history"][-1]["content"]
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_not_reached_under_prompt(tui, side_call):
    """D6: with an approval waiting, the input belongs to it, so `/btw` never reaches the dispatch."""
    import dostuff.agent.call_agent  # noqa: F401
    cli_tui = tui
    call = side_call()

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-prompt")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "hello")  # the tui fixture's loop asks for approval
            await _until(pilot, app.adapter.has_pending_confirm)
            _submit(app, cli_tui, "/btw x")
            await _until(pilot, lambda: not app._turn_in_progress)
            assert call.calls == [] and not app._btw_open()
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_idle_uses_kept_parts_and_store(btw_tui):
    cli_tui, held, side_call = btw_tui
    held.hold = False
    call = side_call()

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-idle")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "fix the bug")
            await _until(pilot, lambda: not app._turn_in_progress and len(app.steps_history) == 2)
            _submit(app, cli_tui, "/btw what did I ask?")
            await _btw_answered(pilot, app)
            stored = await app.store.load_working_history(app.session_id)
            assert call.calls[0]["steps_history"][:-1] == stored  # the finished turn, from the store
            assert call.calls[0]["system_instruction"] == held.systems[0]  # byte-equal: the cache stays warm
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_idle_after_compact(btw_tui, monkeypatch):
    """tasks.md open point 2: an idle /compact rewrites the notes; the side call reads them from the store."""
    cli_tui, held, side_call = btw_tui
    held.hold = False
    call = side_call()

    async def fake_compact(history, budget, notes, *args, **kwargs):
        return history[-1:], "NEW NOTES"
    monkeypatch.setattr(cli_tui, "compact_context", fake_compact)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-compact")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "fix the bug")
            await _until(pilot, lambda: not app._turn_in_progress and len(app.steps_history) == 2)
            _submit(app, cli_tui, "/compact")
            await _until(pilot, lambda: any("Context compacted" in t for t in _texts(app)))
            _submit(app, cli_tui, "/btw what did I ask?")
            await _btw_answered(pilot, app)
            system = call.calls[0]["system_instruction"]
            assert system.endswith("[Summary of earlier conversation]: NEW NOTES")
            assert system.startswith(held.snapshots[0].system_instruction)
            assert len(call.calls[0]["steps_history"]) == 2  # the compacted history + the question
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_cold_session(btw_tui, monkeypatch):
    """No turn in this process: the system prompt is built now, with the default tools."""
    from dostuff.tools.definitions import default_tool_names
    cli_tui, held, side_call = btw_tui
    call = side_call()
    built = []

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-cold")
        original = app._build_instructions

        async def build(text):
            built.append(text)
            return await original(text)
        app._build_instructions = build  # type: ignore[method-assign]
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "/btw anything yet?")
            await _btw_answered(pilot, app)
            assert built == ["anything yet?"]
            assert "<environment>" in call.calls[0]["system_instruction"]
            assert call.calls[0]["tool_names"] == default_tool_names()
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_panel_composed_hidden(tui):
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-compose")
        async with app.run_test() as pilot:
            await pilot.pause()
            panels = app.query("#btw-panel")
            assert len(panels) == 1 and panels.first().display is False
            ids = [w.id for w in app.screen.children]
            assert ids.index("queue-bar") < ids.index("btw-panel") < ids.index("picker")
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_panel_asking_then_answer(btw_tui, monkeypatch):
    from dostuff.helpers.agent import model_cost
    cli_tui, held, side_call = btw_tui
    monkeypatch.setattr(model_cost, "call_cost", lambda model, usage: None)  # an unpriced model (Ollama)
    gate = asyncio.Event()
    usage = types.SimpleNamespace(prompt_tokens=50, completion_tokens=7, total_tokens=57)
    side_call(fake_response("**Short** answer.", usage=usage), gate=gate)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-asking")
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "/btw  a\n  long   question")
            panel = app._btw_panel()
            await _until(pilot, lambda: panel.display and app.focused is panel)
            title = str(panel.query_one("#btw-title", cli_tui.Static).render())
            assert title == "/btw — a long question"
            assert panel.body_text == "Asking…"
            assert str(panel.query_one("#btw-footer", cli_tui.Static).render()) == "Esc close"
            gate.set()
            await _btw_answered(pilot, app)
            assert panel.body_text == "**Short** answer."
            # no cost (unpriced) and no cached tokens: both parts are left out
            assert str(panel.query_one("#btw-footer", cli_tui.Static).render()) == (
                "Esc/Enter/Space close · f fork out · c copy · ↑50 ↓7")
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_usage_footer_with_cost(tui):
    cli_tui = tui
    app = cli_tui.DostuffTUI(session_id="btw-cost")
    assert app._format_side_usage({"prompt_tokens": 1200, "completion_tokens": 340, "cached_tokens": 1100,
                                   "cost": 0.002}) == "↑1.2k (1.1k cached) ↓340 · $0.0020"
    assert app._format_side_usage(None) == ""


def test_btw_panel_not_in_transcript_or_export(btw_tui, tmp_path):
    cli_tui, held, side_call = btw_tui
    held.hold = False
    side_call(fake_response("SIDE-ANSWER-TEXT"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-export")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "fix the bug")
            await _until(pilot, lambda: not app._turn_in_progress and len(app.steps_history) == 2)
            lines = _message_lines(app)
            _submit(app, cli_tui, "/btw SIDE-QUESTION-TEXT")
            await _btw_answered(pilot, app)
            assert _message_lines(app) == lines
            await pilot.press("escape")
            out = tmp_path / "export.md"
            _submit(app, cli_tui, f"/export {out}")
            await _until(pilot, out.exists)
            text = out.read_text(encoding="utf-8")
            assert "fix the bug" in text
            assert "SIDE-QUESTION-TEXT" not in text and "SIDE-ANSWER-TEXT" not in text
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_panel_leaves_conversation_visible(btw_tui):
    cli_tui, held, side_call = btw_tui
    side_call(fake_response("\n\n".join(f"paragraph {i}" for i in range(60))))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-visible")
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "/btw long?")
            panel = await _btw_answered(pilot, app)
            await pilot.pause()
            assert app.query_one("#conversation").display and app.query_one("#conversation").size.height > 0
            assert panel.size.height + 2 <= 40 // 2 + 2  # max-height 50vh (border included)
            for _ in range(5):  # the body scrolls with the keys
                await pilot.press("down")
            assert panel.query_one("#btw-scroll").scroll_y > 0
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_rapid_reopen(btw_tui):
    cli_tui, held, side_call = btw_tui
    call = side_call()

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-rapid")
        async with app.run_test() as pilot:
            await pilot.pause()
            for question in ("a", "b", "c"):
                _submit(app, cli_tui, f"/btw {question}")
                await _btw_answered(pilot, app)
                await pilot.press("escape")
                await pilot.pause()
            assert len(app.query("#btw-panel")) == 1 and not app._btw_open()
            assert len(call.calls) == 3
            assert len(app._side_exchanges) == 3  # the thread replays: the third saw the first two
            assert len(call.calls[2]["steps_history"]) == len(call.calls[0]["steps_history"]) + 4
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_panel_does_not_bounce_the_transcript(btw_tui):
    """Opening and closing the side answer leaves every painted frame flush at the bottom, like the picker and
    the ask panel (`test_ask_panel_does_not_bounce_the_transcript`)."""
    cli_tui, held, side_call = btw_tui
    side_call(fake_response("one\n\ntwo\n\nthree"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-bounce")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            for i in range(25):
                app._append(f"row{i:02d} " + "-" * 30, msg_type="user")
            await pilot.pause()
            await pilot.pause()
            messages = app.query_one("#messages")
            off_bottom: list[int] = []
            original_display = app._display

            def display(screen, renderable):
                rows = [strip.text.rstrip() for strip in
                        app.screen._compositor.render_strips()[:messages.region.height]]
                occupied = [y for y, row in enumerate(rows) if row]
                if occupied:
                    off_bottom.append(messages.region.height - 1 - occupied[-1])
                return original_display(screen, renderable)
            app._display = display  # type: ignore[method-assign]
            _submit(app, cli_tui, "/btw q")
            await _btw_answered(pilot, app)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.pause()
            app._display = original_display  # type: ignore[method-assign]
            assert off_bottom, "no frame was measured"
            assert max(off_bottom) <= 0, f"the transcript drifted off the bottom: {off_bottom}"
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_takes_focus_and_gives_it_back(btw_tui):
    cli_tui, held, side_call = btw_tui
    side_call()

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-focus")
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            _submit(app, cli_tui, "/btw q")
            inp.text = "half-typed draft"
            panel = await _btw_answered(pilot, app)
            assert app.focused is panel and inp.text == "half-typed draft"
            await pilot.press("escape")
            await _until(pilot, lambda: app.focused is inp)
            assert not app._btw_open() and inp.text == "half-typed draft"
            for key in ("enter", "space"):
                _submit(app, cli_tui, "/btw again")  # clears the input, as sending it would
                inp.text = f"draft before {key}"
                await _btw_answered(pilot, app)
                await pilot.press(key)
                await _until(pilot, lambda: not app._btw_open() and app.focused is inp)
                # the key closed the panel and nothing else: the input was neither sent nor edited
                assert inp.text == f"draft before {key}" and not app._turn_in_progress
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_esc_does_not_cancel_turn(btw_tui):
    cli_tui, held, side_call = btw_tui
    side_call()

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-esc")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "fix the bug")
            await _until(pilot, lambda: held.gate is not None)
            _submit(app, cli_tui, "/btw q")
            await _btw_answered(pilot, app)
            await pilot.press("escape")
            await _until(pilot, lambda: not app._btw_open())
            assert app.adapter._cancelled is False and app._turn_in_progress
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            assert app.steps_history[-1] == {"role": "assistant", "content": "answer to fix the bug"}
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_esc_idle_no_recall(btw_tui):
    cli_tui, held, side_call = btw_tui
    held.hold = False
    side_call()

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-recall")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "first prompt")
            await _until(pilot, lambda: not app._turn_in_progress and len(app.steps_history) == 2)
            _submit(app, cli_tui, "/btw q")
            await _btw_answered(pilot, app)
            app._last_escape = 0.0
            await pilot.press("escape")
            await _until(pilot, lambda: not app._btw_open())
            await pilot.press("escape")  # one more: had the first been an idle Esc, this would recall the prompt
            await pilot.pause(0.2)
            assert app.query_one("#input", cli_tui.EnterSubmits).text == ""
            assert len(app.steps_history) == 2
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_copy(btw_tui, monkeypatch):
    from dostuff.helpers.agent.side_question import TOOL_CALL_NOTE
    cli_tui, held, side_call = btw_tui
    side_call(fake_response('Run {"name": "read_file", "arguments": {}}'))
    copied = []
    monkeypatch.setattr(cli_tui, "_copy_to_clipboard", lambda app, text: copied.append(text))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-copy")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "/btw the question")
            await _btw_answered(pilot, app)
            await pilot.press("c")
            await _until(pilot, lambda: copied)
            assert copied == [f'Run {{"name": "read_file", "arguments": {{}}}}\n\n{TOOL_CALL_NOTE}']
            assert app._btw_open()
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_keys_inert_while_asking(btw_tui, monkeypatch):
    cli_tui, held, side_call = btw_tui
    gate = asyncio.Event()
    side_call(gate=gate)
    copied = []
    monkeypatch.setattr(cli_tui, "_copy_to_clipboard", lambda app, text: copied.append(text))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-inert")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "/btw q")
            panel = app._btw_panel()
            await _until(pilot, lambda: panel.display and app.focused is panel)
            forked = []
            app.on_btw_panel_fork_out = lambda event: forked.append(event)  # type: ignore[attr-defined]
            await pilot.press("c")
            await pilot.press("f")
            await pilot.pause()
            assert copied == [] and forked == [] and app._btw_open()
            gate.set()
            await _btw_answered(pilot, app)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


@pytest.mark.parametrize("prompt", ["approval", "ask_user"])
def test_btw_prompt_wins_focus(btw_tui, prompt):
    """An approval or a question opened while the side answer is up takes the focus and the first Esc; the
    next Esc closes the side answer (D6)."""
    cli_tui, held, side_call = btw_tui
    side_call()

    async def scenario():
        app = cli_tui.DostuffTUI(session_id=f"btw-{prompt}")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "fix the bug")
            await _until(pilot, lambda: held.gate is not None)
            _submit(app, cli_tui, "/btw q")
            panel = await _btw_answered(pilot, app)
            if prompt == "approval":
                task = asyncio.ensure_future(app.adapter.ask("Allow write_file?"))
                await _until(pilot, lambda: bool(app.query("#permission-prompt"))
                             and app.focused is app.query_one("#permission-prompt"))
            else:
                task = _ask_questions(app, [DB_QUESTION])
                ask_panel = app._ask_panel()
                await _until(pilot, lambda: ask_panel.display and app.focused is ask_panel)
            assert panel.display
            await pilot.press("escape")
            await _until(pilot, task.done)
            if prompt == "approval":
                assert task.result().answer == "no"
            else:
                assert task.result() is None
            assert app._btw_open()
            app.adapter._cancelled = False  # that Esc also cancelled the turn; this test is about the panel
            app.adapter._cancel_event.clear()
            await pilot.press("escape")
            await _until(pilot, lambda: not app._btw_open())
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_error_in_panel(btw_tui):
    cli_tui, held, side_call = btw_tui
    side_call(raise_exc=RuntimeError("rate limited"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-error")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "fix the bug")
            await _until(pilot, lambda: held.gate is not None)
            lines = _message_lines(app)
            _submit(app, cli_tui, "/btw q")
            panel = app._btw_panel()
            await _until(pilot, lambda: panel.display and panel.error)
            assert "rate limited" in panel.body_text
            assert _message_lines(app) == lines and not any("Turn error" in t for t in _texts(app))
            assert app._side_exchanges == [] and app._turn_in_progress
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            assert app.steps_history[-1]["content"] == "answer to fix the bug"
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_second_refused(btw_tui):
    cli_tui, held, side_call = btw_tui
    gate = asyncio.Event()
    call = side_call(gate=gate)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-second")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "/btw first")
            await _until(pilot, lambda: len(call.calls) == 1)
            _submit(app, cli_tui, "/btw second")
            await pilot.pause()
            assert app.query_one("#input", cli_tui.EnterSubmits).border_title == cli_tui.BTW_BUSY
            assert len(call.calls) == 1 and app._btw_panel().question == "first"
            gate.set()
            await _btw_answered(pilot, app)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_answer_survives_turn_cancel(btw_tui):
    cli_tui, held, side_call = btw_tui
    side_call(fake_response("kept"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-cancel")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "fix the bug")
            await _until(pilot, lambda: held.gate is not None)
            _submit(app, cli_tui, "/btw q")
            panel = await _btw_answered(pilot, app)
            app.adapter._cancelled = True  # cancel the turn directly, not with Esc (that closes the panel)
            app.adapter._cancel_event.set()
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            assert panel.display and panel.body_text == "kept"
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- /btw Phase 4: fork-out (plan/2026-11-03-btw/tasks.md T4.x) ---------------------------------------------------

def test_btw_fork_out_starts_fork(btw_tui, monkeypatch):
    from conftest import FakeAgentRegistry
    from dostuff.helpers.agent.side_question import BTW_FORK_DESCRIPTION, TOOL_CALL_NOTE
    cli_tui, held, side_call = btw_tui
    side_call(fake_response('Probably a.py. {"name": "read_file", "arguments": {}}'))
    fake = FakeAgentRegistry()
    monkeypatch.setattr(cli_tui, "_agent_registry", lambda: fake)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-fork")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "fix the bug")
            await _until(pilot, lambda: held.gate is not None)
            _submit(app, cli_tui, "/btw which file?")
            panel = await _btw_answered(pilot, app)
            used = panel.snapshot
            assert used is held.snapshots[0]  # the live snapshot the side call read
            await pilot.press("f")
            await _until(pilot, lambda: not app._btw_open())
            [kwargs] = fake.started
            assert kwargs["owner"] == "btw-fork" and kwargs["agent_type"] == "fork"
            assert kwargs["background"] is True and kwargs["description"] == BTW_FORK_DESCRIPTION
            answer = f'Probably a.py. {{"name": "read_file", "arguments": {{}}}}\n\n{TOOL_CALL_NOTE}'
            assert kwargs["prompt"] == f"which file?\n\nMy answer without tools was:\n{answer}\n\nGo and check it."
            assert kwargs["fork"] is used  # not re-resolved at `f` time
            assert cli_tui.BTW_HANDED_OFF in _texts(app)
            assert app._turn_in_progress  # the main turn is untouched
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_fork_out_twice(btw_tui, monkeypatch):
    from conftest import FakeAgentRegistry
    cli_tui, held, side_call = btw_tui
    side_call()
    gate = asyncio.Event()

    class SlowRegistry(FakeAgentRegistry):
        async def start(self, **kwargs):
            self.started.append(kwargs)
            await gate.wait()
            return types.SimpleNamespace(id="r1")
    fake = SlowRegistry()
    monkeypatch.setattr(cli_tui, "_agent_registry", lambda: fake)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-twice")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "/btw q")
            await _btw_answered(pilot, app)
            await pilot.press("f")
            await _until(pilot, lambda: len(fake.started) == 1)
            await pilot.press("f")
            await pilot.pause(0.1)
            assert len(fake.started) == 1
            gate.set()
            await _until(pilot, lambda: not app._btw_open())
            assert len(fake.started) == 1 and not app._btw_forking
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_fork_out_error_stays_open(btw_tui, monkeypatch):
    from conftest import FakeAgentRegistry
    cli_tui, held, side_call = btw_tui
    side_call(fake_response("the answer"))
    fake = FakeAgentRegistry(raise_exc=ValueError("a fork needs the main agent's conversation"))
    monkeypatch.setattr(cli_tui, "_agent_registry", lambda: fake)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-fork-error")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "/btw q")
            panel = await _btw_answered(pilot, app)
            for exc in (ValueError("a fork needs the main agent's conversation"), RuntimeError("queue full")):
                fake.raise_exc = exc
                await pilot.press("f")
                await _until(pilot, lambda: str(exc) in panel.body_text)
                assert app._btw_open() and not app._btw_forking
                assert panel.body_text.startswith("the answer")  # the answer stays above the reason
            assert len(fake.started) == 2  # the guard was reset after each failure
            assert cli_tui.BTW_HANDED_OFF not in _texts(app)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_fork_out_report_wakes(tui, agents, fake_model, side_call, monkeypatch):
    """REQ-4.2: the real registry and the untouched notice path — the fork's report wakes the idle session and
    arrives as a `<background-agent>` user step."""
    from dostuff.helpers.agent.notices import get_notice_queue
    import dostuff.agent.call_agent  # noqa: F401
    cli_tui = tui
    monkeypatch.setattr(cli_tui, "JOB_WAKE_DELAY_S", 0.3)
    side_call(fake_response("my guess [F]"))
    fake_model.script("[F]", fake_response("checked: it is a.py"))
    woke: list[tuple[str, list[str]]] = []

    async def wake_loop(session_id, turn_id, user_text, instructions, mcp_client, working_history, turn_type,
                        current_session_history, steps_history, store, adapter=None, **kwargs):
        woke.append((user_text, [n.id for n in get_notice_queue().drain(session_id)]))
        return "handled"
    monkeypatch.setattr(cli_tui, "agent_loop", wake_loop)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-wake")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            app.config.agents_wake_on_end = True
            _submit(app, cli_tui, "/btw which file? [F]")
            await _btw_answered(pilot, app)
            await pilot.press("f")
            await _until(pilot, lambda: bool(agents.runs("btw-wake")))
            [run] = agents.runs("btw-wake")
            await _until(pilot, lambda: run.status == "completed", tries=100)
            await _until(pilot, lambda: bool(woke), tries=100)
            text, ids = woke[0]
            assert text.startswith(f'<background-agent id="{run.id}"') and ids == [run.id]
            assert "checked: it is a.py" in text
            fork_call = fake_model.calls_for("[F]")[0]["messages"]
            assert "My answer without tools was:\nmy guess [F]" in fork_call[-1]["content"]
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_fork_out_survives_esc(btw_tui, agents, fake_model):
    """Esc on the main turn cancels foreground runs only: the fork-out keeps going (D8)."""
    cli_tui, held, side_call = btw_tui
    side_call(fake_response("guess [S]"))
    gate = fake_model.gate("[S]", fake_response("report"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-esc-fork")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            _submit(app, cli_tui, "fix the bug")
            await _until(pilot, lambda: held.gate is not None)
            _submit(app, cli_tui, "/btw q [S]")
            await _btw_answered(pilot, app)
            await pilot.press("f")
            await _until(pilot, lambda: bool(fake_model.calls_for("[S]")), tries=100)
            [run] = agents.runs("btw-esc-fork")
            await pilot.press("escape")  # the panel is closed: this cancels the main turn
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            assert run.status == "running"
            gate.set()
            await _until(pilot, lambda: run.status == "completed", tries=100)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- /btw Phase 5: session boundaries (plan/2026-11-03-btw/tasks.md T5.x) -----------------------------------------

@pytest.mark.parametrize("switch", ["clear", "resume", "fork"])
def test_btw_thread_cleared_on_switch(btw_tui, switch):
    from dostuff.helpers.agent.side_question import kept_prompt
    cli_tui, held, side_call = btw_tui
    held.hold = False
    call = side_call(fake_response("answer A"), fake_response("answer B"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id=f"btw-{switch}")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "fix the bug")
            await _until(pilot, lambda: not app._turn_in_progress and len(app.steps_history) == 2)
            old = app.session_id
            assert kept_prompt(old) is not None
            _submit(app, cli_tui, "/btw QUESTION-A")
            await _btw_answered(pilot, app)  # left open: the switch must close it
            if switch == "clear":
                await app._clear_session()
            elif switch == "resume":
                await app._switch_session("btw-elsewhere")
            else:
                await app._fork_session("")
            await pilot.pause()
            assert app.session_id != old
            assert not app._btw_open() and app._side_exchanges == []
            assert kept_prompt(old) is None
            _submit(app, cli_tui, "/btw QUESTION-B")
            await _btw_answered(pilot, app)
            sent = call.calls[1]["steps_history"]
            assert not any("QUESTION-A" in str(s.get("content")) or s.get("content") == "answer A" for s in sent)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_btw_fork_out_carried_on_clear(tui, agents, fake_model, side_call, monkeypatch):
    """A running fork-out moves to the new session with /clear (`_carry_background`) and wakes it there."""
    from dostuff.helpers.agent.notices import get_notice_queue
    import dostuff.agent.call_agent  # noqa: F401
    cli_tui = tui
    monkeypatch.setattr(cli_tui, "JOB_WAKE_DELAY_S", 0.3)
    side_call(fake_response("guess [C]"))
    gate = fake_model.gate("[C]", fake_response("carried report"))
    woke: list[tuple[str, str]] = []

    async def wake_loop(session_id, turn_id, user_text, instructions, mcp_client, working_history, turn_type,
                        current_session_history, steps_history, store, adapter=None, **kwargs):
        get_notice_queue().drain(session_id)
        woke.append((session_id, user_text))
        return "handled"
    monkeypatch.setattr(cli_tui, "agent_loop", wake_loop)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-carry")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            app.config.agents_wake_on_end = True
            _submit(app, cli_tui, "/btw q [C]")
            await _btw_answered(pilot, app)
            await pilot.press("f")
            await _until(pilot, lambda: bool(fake_model.calls_for("[C]")), tries=100)
            [run] = agents.runs("btw-carry")
            await app._clear_session()
            new = app.session_id
            assert run.owner == new
            gate.set()
            await _until(pilot, lambda: bool(woke), tries=100)
            assert woke[0][0] == new and "carried report" in woke[0][1]
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


@pytest.mark.parametrize("which", ["fork-out", "model fork", "none"])
def test_fork_warns_about_running_fork_out(tui, agents, fake_model, side_call, monkeypatch, which):
    from dostuff.agent.subagents.fork import ForkContext
    import dostuff.agent.call_agent  # noqa: F401
    cli_tui = tui
    side_call(fake_response("guess [W]"))
    gate = fake_model.gate("[W]", fake_response("never mind"))

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="btw-strand")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._backend_ready.wait()
            _submit(app, cli_tui, "hello")  # the tui fixture's loop: approve, then it answers
            await _until(pilot, app.adapter.has_pending_confirm)
            _submit(app, cli_tui, "y")
            await _until(pilot, lambda: not app._turn_in_progress)
            if which == "fork-out":
                _submit(app, cli_tui, "/btw q [W]")
                await _btw_answered(pilot, app)
                await pilot.press("f")
            elif which == "model fork":
                snap = ForkContext(system_instruction="s", compaction_notes="", history=[], tool_names=[])
                await agents.start(owner="btw-strand", agent_type="fork", description="something else",
                                   prompt="[W] x", background=True, fork=snap, adapter=app.adapter)
            if which != "none":
                await _until(pilot, lambda: bool(fake_model.calls_for("[W]")), tries=100)
            await app._fork_session("")
            await pilot.pause()
            warned = cli_tui.BTW_FORK_STRANDED.format(old="btw-strand") in _texts(app)
            assert warned is (which == "fork-out")
            gate.set()
            await _until(pilot, lambda: all(r.ended is not None for r in agents.runs()), tries=100)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- Steering Phase 0: pins (plan/2026-11-04-steering/tasks.md T0.x) ------------------------


@pytest.fixture
def steer_tui(tui, monkeypatch):
    """(cli_tui, HeldTurn): a turn in flight, held until released — the steering pins' setup."""
    held = HeldTurn()
    monkeypatch.setattr(tui, "agent_loop", held)
    yield tui, held
    held.hold = False
    held.release()


@pytest.mark.parametrize("text,expect", [
    ("fix the bug", "queued"),
    ("/etc/hosts look", "queued"),
    ("/release-notes now", "queued"),
    ("!ls", "refused"),
    ("/clear", "refused"),
    ("/jobs", "ran"),
], ids=["plain", "path", "skill", "shell", "clear", "jobs"])
def test_pin_mid_turn_routing(queue_mode, text, expect):
    """T0.1 (steering spec D3): the `queue`-mode mid-turn routing, pinned with the mode set explicitly.

    With a turn in flight a plain message, a path message and a skill command all join `_prompt_queue`;
    `!command` and `/clear` are refused with their own message; `/jobs` runs at once. This table is the
    opt-in `queue` mode; `test_steering_mode_mid_turn_routing` pins `steer` mode and
    `test_default_mid_turn_is_steer` the default (no config key at all).
    """
    cli_tui, held = queue_mode
    skill_dir = Path(".dostuff", "skills", "release-notes")
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: release-notes\ndescription: Write release notes from git log\n---\nSummarise git log.\n",
        encoding="utf-8")

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-pin")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "fix the bug")
            #200 tries: the first turn of a cold run also waits for the backend to load
            await _until(pilot, lambda: held.gate is not None, tries=200)
            _submit(app, cli_tui, text)
            await pilot.pause(0.3)
            if expect == "queued":
                assert app._prompt_queue == [text], app._prompt_queue
                assert not any("while a turn is running" in t for t in _texts(app))
            elif expect == "refused":
                assert app._prompt_queue == []
                assert any("while a turn is running" in t for t in _texts(app))
            else:  # ran now, not queued
                await _until(pilot, lambda: any("No background jobs." in t for t in _texts(app)))
                assert app._prompt_queue == []
            held.hold = False
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- Steering Phase 3: /steer and routing by mode (plan/2026-11-04-steering/tasks.md T3.x) ---


def test_steer_command_mid_turn(steer_tui):
    """T3.2 (steering spec D2, REQ-3.1): `/steer <message>` while a turn runs puts the message on the
    steering queue — never `_prompt_queue` — and shows no user message yet (REQ-3.4: only the loop writes it)."""
    cli_tui, held = steer_tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-cmd")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "fix the bug")
            await _until(pilot, lambda: held.gate is not None, tries=200)
            _submit(app, cli_tui, "/steer use pytest, not unittest")
            await pilot.pause(0.3)
            assert [m.raw for m in app._steering.pending()] == ["use pytest, not unittest"]
            assert app._prompt_queue == []
            assert not any("use pytest" in t for t in _texts(app))
            held.hold = False
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_steer_command_idle_starts_turn(steer_tui):
    """T3.2 (D2): no turn running — `/steer <message>` is an ordinary prompt with `<message>`, so the
    habit is never an error and never a dead end."""
    cli_tui, held = steer_tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-idle")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "/steer fix the bug")
            await _until(pilot, lambda: held.gate is not None, tries=200)
            assert app._turn_in_progress
            assert app._steering.pending() == []
            assert any("fix the bug" in t and "/steer" not in t for t in _texts(app))
            held.hold = False
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_steer_command_usage(tui):
    """T3.2 (REQ-3.1): `/steer` with no message says how to use it and starts nothing."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-usage")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "/steer")
            await pilot.pause(0.3)
            assert any("Usage: /steer <message>" in t for t in _texts(app))
            assert not app._turn_in_progress
            assert app._steering.pending() == []
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


@pytest.fixture
def steer_mode(steer_tui, tmp_path):
    """(cli_tui, HeldTurn) with `ui.mid_turn_messages: steer` written explicitly in the user config —
    REQ-6.1's real path (the yaml, not a set attribute). `steer` is also the default (D2); this pins
    the yaml path, `test_default_mid_turn_is_steer` the default."""
    from dostuff import config as config_module

    def apply():
        config_module.reset_config()
        # The tui fixture turns these off after its own reset; a second reset brings the defaults back
        config_module.get_config().jobs_wake_on_end = False
        config_module.get_config().agents_wake_on_end = False

    (tmp_path / "home" / ".dostuff" / "config.yaml").write_text("ui:\n  mid_turn_messages: steer\n",
                                                                encoding="utf-8")
    apply()
    yield steer_tui
    (tmp_path / "home" / ".dostuff" / "config.yaml").unlink()
    apply()


@pytest.fixture
def queue_mode(steer_tui, tmp_path):
    """(cli_tui, HeldTurn) with `ui.mid_turn_messages: queue` in the user config — the opt-in batching
    mode, pinned explicitly now that the default is `steer` (steering spec D2)."""
    from dostuff import config as config_module

    def apply():
        config_module.reset_config()
        # The tui fixture turns these off after its own reset; a second reset brings the defaults back
        config_module.get_config().jobs_wake_on_end = False
        config_module.get_config().agents_wake_on_end = False

    (tmp_path / "home" / ".dostuff" / "config.yaml").write_text("ui:\n  mid_turn_messages: queue\n",
                                                                encoding="utf-8")
    apply()
    yield steer_tui
    (tmp_path / "home" / ".dostuff" / "config.yaml").unlink()
    apply()


@pytest.mark.parametrize("text,expect", [
    ("fix the bug", "steered"),
    ("/etc/hosts look", "steered"),
    ("/release-notes now", "queued"),
    ("!ls", "refused"),
    ("/clear", "refused"),
    ("/jobs", "ran"),
], ids=["plain", "path", "skill", "shell", "clear", "jobs"])
def test_steering_mode_mid_turn_routing(steer_mode, text, expect):
    """T3.3 (steering spec D3, REQ-3.2/3.3): in `steer` mode a plain message and a path message go to
    the steering queue, a skill command still waits for the end of the turn, and the refused / built-in
    rows are unchanged. `test_pin_mid_turn_routing` pins this same table for `queue` mode (REQ-3.2)."""
    cli_tui, held = steer_mode
    skill_dir = Path(".dostuff", "skills", "release-notes")
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: release-notes\ndescription: Write release notes from git log\n---\nSummarise git log.\n",
        encoding="utf-8")

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-mode")
        assert app.config.mid_turn_messages == "steer"
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "start the turn")  # distinct from every parametrised text
            await _until(pilot, lambda: held.gate is not None, tries=200)
            _submit(app, cli_tui, text)
            await pilot.pause(0.3)
            if expect == "steered":
                assert [m.raw for m in app._steering.pending()] == [text]
                assert app._prompt_queue == []
                assert not any(text in t for t in _texts(app))  # never shown or stored at submit (REQ-3.4)
            elif expect == "queued":
                assert app._prompt_queue == [text], app._prompt_queue
                assert app._steering.pending() == []
                assert not any("while a turn is running" in t for t in _texts(app))
            elif expect == "refused":
                assert app._prompt_queue == [] and app._steering.pending() == []
                assert any("while a turn is running" in t for t in _texts(app))
            else:  # ran now, not queued
                await _until(pilot, lambda: any("No background jobs." in t for t in _texts(app)))
                assert app._prompt_queue == [] and app._steering.pending() == []
            held.hold = False
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- Steering Phase 3 continued: preparation and the wiring (plan/2026-11-04-steering/tasks.md T3.4) ---


def test_run_turn_passes_the_steering_queue(steer_tui):
    """T3.4 (D1): `_run_turn` hands the app's queue to `loop()` — without it nothing is ever drained and
    no steering message is ever delivered."""
    cli_tui, held = steer_tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-wiring")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "start the turn")
            await _until(pilot, lambda: held.gate is not None, tries=200)
            assert held.kwargs and held.kwargs[-1].get("steering") is app._steering, held.kwargs
            held.hold = False
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_steer_prepared_in_order(steer_tui, monkeypatch):
    """T3.4 (D5, CON-006): submit order is delivery order. A's @mention expansion is held while B, sent
    after it with no mention, finishes first — the loop's drain (`take_ready`) returns nothing until A is
    ready, then A, B. Preparation runs off the UI thread (REQ-3.5)."""
    import threading
    cli_tui, held = steer_tui
    from dostuff.helpers.agent import file_mentions
    monkeypatch.setattr(file_mentions, "_index", {})
    Path("docs").mkdir()
    Path("docs", "big.md").write_text("big content\n", encoding="utf-8")

    started, release = threading.Event(), threading.Event()
    real_expand = cli_tui.expand_file_mentions

    def held_expand(text, *args, **kwargs):
        result = real_expand(text, *args, **kwargs)
        if "big.md" in text:
            started.set()
            release.wait(10)  # the 2 MB @file is still being read
        return result

    monkeypatch.setattr(cli_tui, "expand_file_mentions", held_expand)

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-order")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "start the turn")
            await _until(pilot, lambda: held.gate is not None, tries=200)
            _submit(app, cli_tui, "/steer read @docs/big.md")
            await _until(pilot, lambda: started.is_set())  # A is in its expansion, holding the prefix
            _submit(app, cli_tui, "/steer a small note, no mention")
            await _until(pilot, lambda: [m.content for m in app._steering.pending()] == [None, "a small note, no mention"])
            assert app._steering.take_ready() == []  # B ready, A not: the prefix stops (REQ-1.2)
            release.set()
            await _until(pilot, lambda: all(m.content is not None for m in app._steering.pending()))
            taken = app._steering.take_ready()
            assert [m.raw for m in taken] == ["read @docs/big.md", "a small note, no mention"]
            held.hold = False
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_steer_mentions_and_pastes_expanded(steer_tui, monkeypatch):
    """T3.4 (D5): the loop is handed the expanded text, not what was typed, and the "Attached …" line
    shows when the preparation finishes (as it does for a prompt)."""
    cli_tui, held = steer_tui
    from dostuff.helpers.agent import file_mentions
    monkeypatch.setattr(file_mentions, "_index", {})
    Path("notes.md").write_text("the plan is here\n", encoding="utf-8")

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-mention")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "start the turn")
            await _until(pilot, lambda: held.gate is not None, tries=200)
            _submit(app, cli_tui, "/steer look at @notes.md")
            await _until(pilot, lambda: bool(app._steering.pending())
                         and app._steering.pending()[0].content is not None)
            msg = app._steering.pending()[0]
            content = msg.content
            assert msg.raw == "look at @notes.md"  # what the user typed: shown, re-queued, put back on Esc
            assert content is not None
            assert '<attached_file path="notes.md">' in content
            assert "the plan is here" in content
            assert any(t == "Attached notes.md" for t in _texts(app))
            assert app._prompt_queue == []
            held.hold = False
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_default_mid_turn_is_steer(steer_tui):
    """D2 (flipped default): with no `ui.mid_turn_messages` in any config, a plain message sent while a
    turn runs goes to the steering queue — `steer` is the default, `queue` is the opt-in."""
    cli_tui, held = steer_tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-default")
        assert app.config.mid_turn_messages == "steer"  # no config.yaml anywhere: the default
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "start the turn")
            await _until(pilot, lambda: held.gate is not None, tries=200)
            _submit(app, cli_tui, "fix the bug")
            await pilot.pause(0.3)
            assert [m.raw for m in app._steering.pending()] == ["fix the bug"]
            assert app._prompt_queue == []
            assert not any("fix the bug" in t for t in _texts(app))  # never shown at submit (REQ-3.4)
            held.hold = False
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- Steering Phase 4: display and leftovers (plan/2026-11-04-steering/tasks.md T4.x) ----------


class _SteerRun:
    """What `steer_loop`'s scripted `call_agent` / `dispatch_tool` recorded."""

    def __init__(self):
        self.responses: list = []
        self.calls: list = []       # the history each model call saw
        self.dispatched: list = []  # (tool name, args)
        self.results: list = []     # (tool name, result) once a live dispatch returned
        self.on_dispatch = None     # async: submit a steering message while the tool runs


@pytest.fixture
def _steer_loop(tui, monkeypatch):
    """(cli_tui, _SteerRun): the **real** `loop()` driving the TUI — Phase 2's delivery path end to end.
    Only `call_agent` is scripted; hooks, `append_note`, the events and `_emit_on_ui` are the real thing,
    which is what Phase 4 is about. `steer_loop` scripts `dispatch_tool` too; `steer_loop_live` doesn't."""
    from conftest import fake_response
    from dostuff.agent import loop as loop_module
    cli_tui = tui
    state = _SteerRun()

    async def real_loop(*args, **kwargs):
        return await loop_module.loop(*args, **kwargs)

    async def fake_call_agent(steps_history, system_instruction, tool_names=None, stream=False):
        state.calls.append([dict(s) for s in steps_history])
        item = state.responses.pop(0) if state.responses else fake_response("done")
        if isinstance(item, BaseException):
            raise item
        return item, None

    monkeypatch.setattr(cli_tui, "agent_loop", real_loop)
    monkeypatch.setattr(loop_module, "call_agent", fake_call_agent)
    monkeypatch.setattr(loop_module, "get_model_token_limit", lambda: 100_000)
    return cli_tui, state


@pytest.fixture
def steer_loop(_steer_loop, monkeypatch):
    """`_steer_loop` with a scripted `dispatch_tool`: tools never really run, and a test can submit a
    steering message from inside one (`state.on_dispatch`)."""
    from dostuff.agent import loop as loop_module
    cli_tui, state = _steer_loop

    async def fake_dispatch(name, args, *a, **kw):
        state.dispatched.append((name, dict(args)))
        if state.on_dispatch is not None:
            await state.on_dispatch(name, dict(args))
        return "ok"

    monkeypatch.setattr(loop_module, "dispatch_tool", fake_dispatch)
    return cli_tui, state


@pytest.fixture
def steer_loop_live(_steer_loop, registry, monkeypatch):
    """`_steer_loop` with the **real** `dispatch_tool`: the model's tool calls really run (T4.4 needs a
    foreground `bash_command` blocking the loop). `registry` is the session's jobs registry, wired into
    the tools the way every jobs test does it (conftest's fixture only yields it)."""
    from dostuff.agent import loop as loop_module
    from dostuff.tools.bash import jobs as jobs_module
    cli_tui, state = _steer_loop
    monkeypatch.setattr(jobs_module, "_registry", registry)
    real_dispatch = loop_module.dispatch_tool

    async def recording_dispatch(name, args, *a, **kw):
        state.dispatched.append((name, dict(args) if isinstance(args, dict) else {}))
        result = await real_dispatch(name, args, *a, **kw)
        state.results.append((name, str(result)[:300]))
        return result

    monkeypatch.setattr(loop_module, "dispatch_tool", recording_dispatch)
    return cli_tui, state


def test_steered_message_shown_after_tool_output(steer_loop):
    """T4.1 (D9, REQ-4.1): the delivered message shows where it was read — after the tool output it was
    reacting to, as an ordinary user message with the raw text — and `_sent_prompts` records that raw text
    at the delivered index (so Esc Esc can put it back)."""
    cli_tui, state = steer_loop
    from conftest import fake_response, fake_tool_call
    state.responses = [
        fake_response(tool_calls=[fake_tool_call("read_file", {"path": "plan.md"}, "c1")]),
        fake_response("the finished answer"),
    ]

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-show")

        async def submit_while_tool_runs(name, args):
            # a message arriving during a tool call: put from inside the tool (spec test conventions)
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/steer use pytest, not unittest"))

        state.on_dispatch = submit_while_tool_runs
        async with app.run_test() as pilot:
            await pilot.pause()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("start the turn"))
            await _until(pilot, lambda: not app._turn_in_progress, tries=300)
            lines = _message_lines(app)
            tool_at = next(i for i, line in enumerate(lines) if line.startswith("●"))
            shown_at = next(i for i, line in enumerate(lines) if "use pytest, not unittest" in line)
            assert tool_at < shown_at, lines  # after the output it was reacting to, not before it
            assert lines[shown_at].startswith("> ")  # a user message, as typed
            hits = [s for s in app.steps_history if "use pytest, not unittest" in str(s.get("content"))]
            assert len(hits) == 1 and hits[0]["role"] == "user"  # exactly once (CON-004)
            index = app.steps_history.index(hits[0])
            assert app._sent_prompts.get((app.session_id, index)) == "use pytest, not unittest"
            assert not any("STEERED" in line for line in lines)  # the event is consumed, not printed raw
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_esc_esc_after_steer(steer_loop, monkeypatch):
    """T4.1 (D9): Esc Esc puts back what the user typed, not the expanded attachments, and the rewind
    starts at the steering message — the last prompt the user actually sent."""
    cli_tui, state = steer_loop
    from conftest import fake_response, fake_tool_call
    from dostuff.helpers.agent import file_mentions
    monkeypatch.setattr(file_mentions, "_index", {})
    Path("plan.md").write_text("use pytest\n", encoding="utf-8")
    state.responses = [
        fake_response(tool_calls=[fake_tool_call("read_file", {"path": "plan.md"}, "c1")]),
        fake_response("finished"),
    ]

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-esc")

        async def submit_while_tool_runs(name, args):
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/steer check @plan.md"))

        state.on_dispatch = submit_while_tool_runs
        async with app.run_test() as pilot:
            await pilot.pause()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("start the turn"))
            await _until(pilot, lambda: not app._turn_in_progress, tries=300)
            delivered = [s for s in app.steps_history if "<attached_file" in str(s.get("content"))]
            assert len(delivered) == 1  # the loop stored the expanded text (D5)
            # D9: the raw text is recorded at the delivered index, which is what Esc Esc reads back
            assert app._sent_prompts.get((app.session_id, app.steps_history.index(delivered[0]))) == "check @plan.md"
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            assert not inp.text.strip()
            await pilot.press("escape")
            await pilot.press("escape")
            await _until(pilot, lambda: not any("<attached_file" in str(s.get("content"))
                                                for s in app.steps_history))
            assert inp.text == "check @plan.md", inp.text  # raw, not the expansion
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_esc_esc_after_steer_across_a_reload(steer_loop, monkeypatch):
    """Review round 2 (D6 frame): after a reload the stored step is framed and `_sent_prompts` is
    empty — Esc-to-restore still puts back what the user typed, never the frame (the frame reaches
    only the model and the stored transcript)."""
    cli_tui, state = steer_loop
    from conftest import fake_response, fake_tool_call
    from dostuff.helpers.agent import file_mentions
    monkeypatch.setattr(file_mentions, "_index", {})
    Path("plan.md").write_text("use pytest\n", encoding="utf-8")
    state.responses = [
        fake_response(tool_calls=[fake_tool_call("read_file", {"path": "plan.md"}, "c1")]),
        fake_response("finished"),
    ]

    async def scenario():
        app1 = cli_tui.DostuffTUI(session_id="steer-reload")

        async def submit_while_tool_runs(name, args):
            app1.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("/steer check @plan.md"))

        state.on_dispatch = submit_while_tool_runs
        async with app1.run_test() as pilot:
            await pilot.pause()
            app1.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("start the turn"))
            await _until(pilot, lambda: not app1._turn_in_progress, tries=300)
            framed = [s for s in app1.steps_history
                      if "The user sent a new message" in str(s.get("content"))]
            assert len(framed) == 1 and "<attached_file" in framed[0]["content"]  # framed + expanded
            await app1._cleanup_and_save_session()

        # a fresh app on the same session: the stored step is framed, _sent_prompts is empty
        app2 = cli_tui.DostuffTUI(session_id="steer-reload")
        async with app2.run_test() as pilot:
            await pilot.pause()
            assert not app2._sent_prompts  # the bug's precondition: only the fallback can restore
            inp = app2.query_one("#input", cli_tui.EnterSubmits)
            await pilot.press("escape")
            await pilot.press("escape")
            await _until(pilot, lambda: "check @plan.md" in inp.text, tries=200)
            assert "The user sent a new message" not in inp.text  # unframed, not the frame
            assert "Address the message above" not in inp.text
            await app2._cleanup_and_save_session()

    asyncio.run(scenario())


def test_steer_blocked_and_failed_reported_not_delivered(tui):
    """T4.1 (D6, REQ-2.4): the TUI shows what the loop reports for a blocked or failed message — the
    hook's reason / the preparation error — and neither becomes a user message in the transcript."""
    cli_tui = tui

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-blocked")
        async with app.run_test() as pilot:
            await pilot.pause()
            before = len(_message_lines(app))
            app.adapter.emit("steer_blocked", {"id": "b1", "reason": "no passwords in prompts"})
            app.adapter.emit("steer_failed", {"id": "f1", "error": "Could not read @missing.txt"})
            await _until(pilot, lambda: any("no passwords in prompts" in t for t in _texts(app)))
            await _until(pilot, lambda: any("Could not read @missing.txt" in t for t in _texts(app)))
            lines = _message_lines(app)[before:]
            assert not any(line.startswith("> ") for line in lines), lines  # nothing delivered
            assert not any("STEER BLOCKED" in line or "STEER FAILED" in line for line in lines)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- Steering Phase 4 continued: bar, loader and the /queue picker (tasks.md T4.2) -------------


def test_queue_bar_and_loader_count_steering(queue_mode):
    """T4.2 (D9, REQ-4.2): the bar leads with the steering messages and their rule, then the queued
    group as before; the loader counts them separately — `· 1 steering · 2 queued`. Pinned in `queue`
    mode (the two groups only exist together there: plain messages steer in the default mode)."""
    cli_tui, held = queue_mode

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-bar")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "start the turn")
            await _until(pilot, lambda: held.gate is not None, tries=200)
            _submit(app, cli_tui, "/steer first correction")
            _submit(app, cli_tui, "queued one")
            _submit(app, cli_tui, "queued two")
            await pilot.pause(0.3)
            bar = app.query_one("#queue-bar", cli_tui.QueueBar)
            assert bar.display
            rendered = bar.render()
            bar_text = rendered.plain if hasattr(rendered, "plain") else str(rendered)
            assert "→ 1 for the agent · read at the next tool boundary" in bar_text, bar_text
            assert "first correction" in bar_text  # steering rows are listed, marked →
            assert "2 queued" in bar_text and "click or /queue" in bar_text, bar_text
            loader = app._loader_text().plain
            assert "· 1 steering · 2 queued" in loader, loader
            app._prompt_queue.clear()  # so ending the turn doesn't start the queued prompts
            app._render_queue()
            held.hold = False
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_queue_picker_steering_and_send_now(queue_mode):
    """T4.2 (REQ-4.3, D9): `/queue` lists steering entries marked `→` with Edit / Remove, queued prompts
    gain *Send now*, and Remove on an entry the loop has already taken says it was already read
    (REQ-1.3). Pinned in `queue` mode: a plain message is the queued prompt in that mode only."""
    cli_tui, held = queue_mode

    def labels(picker):
        return [label for _, label in picker._options]

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-picker")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "start the turn")
            await _until(pilot, lambda: held.gate is not None, tries=200)
            _submit(app, cli_tui, "/steer a correction")
            _submit(app, cli_tui, "run this afterwards")
            await pilot.pause(0.3)
            picker = app.query_one("#picker", cli_tui.PermissionPrompt)

            # the list: the steering entry first, marked →, then the queued prompt
            app.action_show_queue()
            rows = labels(picker)
            assert rows[0].startswith("→") and "a correction" in rows[0], rows
            assert any("run this afterwards" in row for row in rows), rows

            # a queued prompt now offers "Send now", which moves it onto the steering queue (D2)
            await app._picker_callback("q0")
            rows = labels(picker)
            assert any("Send now" in row for row in rows), rows
            await app._picker_callback("send")
            assert app._prompt_queue == []
            assert [m.raw for m in app._steering.pending()] == ["a correction", "run this afterwards"]

            # Remove takes a pending entry out
            app.action_show_queue()
            await app._picker_callback("s0")
            rows = labels(picker)
            assert any("Remove" in row for row in rows) and any("Edit" in row for row in rows), rows
            await app._picker_callback("remove")
            assert [m.raw for m in app._steering.pending()] == ["run this afterwards"]
            assert "a correction" not in app._steer_raws.values()

            # already taken by the loop between the listing and the choice: say so, remove nothing
            _submit(app, cli_tui, "/steer second correction")
            await pilot.pause(0.3)
            app.action_show_queue()
            app._steering.take_ready()  # the loop reads it at the next tool boundary
            await app._picker_callback("s0")
            await _until(pilot, lambda: any("That message was already read." in t for t in _texts(app)))
            assert app._steering.pending() == []
            held.hold = False
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_queue_clear_closes_the_picker_with_steering_pending(queue_mode):
    """Review fix: "Clear the queue" closes the picker even when steering messages keep the bar up —
    the prompt rows it listed are gone, and choosing one would hit "That prompt already started."."""
    cli_tui, held = queue_mode

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-clear")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "start the turn")
            await _until(pilot, lambda: held.gate is not None, tries=200)
            _submit(app, cli_tui, "/steer a correction")
            _submit(app, cli_tui, "run this afterwards")
            await pilot.pause(0.3)
            picker = app.query_one("#picker", cli_tui.PermissionPrompt)
            app.action_show_queue()
            assert any("run this afterwards" in label for _, label in picker._options)
            await app._picker_callback("clear")
            await pilot.pause()
            assert app._prompt_queue == []
            assert not picker.display  # closed, not left open over the cleared rows
            assert [m.raw for m in app._steering.pending()] == ["a correction"]  # untouched
            assert any("Cleared 1 queued prompt." in t for t in _texts(app))
            assert app.query_one("#queue-bar", cli_tui.QueueBar).display  # steering keeps the bar up
            held.hold = False
            held.release()
            await _until(pilot, lambda: not app._turn_in_progress)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- Steering Phase 4 continued: leftovers (plan/2026-11-04-steering/tasks.md T4.3) -------------


def test_leftover_runs_next(steer_loop):
    """T4.3 (D8, REQ-5.2): messages that outlive their turn — the iteration cap closed the loop before
    they reached a boundary — go to the front of `_prompt_queue` in order, announce themselves once, and
    each runs as a normal turn whose content is stored exactly once (CON-004)."""
    cli_tui, state = steer_loop
    from conftest import fake_response, fake_tool_call
    state.responses = [fake_response(tool_calls=[fake_tool_call("read_file", {"path": "plan.md"}, "c1")])]
    raws = ["fix it before you finish", "and this too"]

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-leftover")
        app.max_iterations = 1  # the cap: nothing after the tool call gets a boundary

        async def submit_while_tool_runs(name, args):
            for raw in raws:
                app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted(f"/steer {raw}"))

        state.on_dispatch = submit_while_tool_runs
        async with app.run_test() as pilot:
            await pilot.pause()
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("start the turn"))
            await _until(pilot, lambda: any("2 messages arrived after the turn ended; running them next" in t
                                            for t in _texts(app)), tries=300)
            assert app._steering.pending() == []
            await _until(pilot, lambda: not app._turn_in_progress
                         and all(any(s.get("content") == raw for s in app.steps_history) for raw in raws),
                         tries=300)
            # in order, once each, in all four places (CON-004)
            delivered = [s.get("content") for s in app.steps_history if s.get("content") in raws]
            assert delivered == raws, delivered
            working = await app.store.load_working_history(app.session_id)
            stored = await app.store.load(app.session_id)
            for raw in raws:
                assert sum(1 for s in working if s.get("content") == raw) == 1
                assert sum(1 for s in app.current_session_history if s.get("content") == raw) == 1
                assert sum(1 for s in stored if s.get("content") == raw) == 1
            assert app._steer_raws == {}  # the ids are forgotten with the leftovers
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


def test_leftover_on_cancel_back_to_input(queue_mode):
    """T4.3 (D8, REQ-5.1): Esc means stop the work, corrections included — the pending steering message
    goes back to the input **ahead of** the queued prompts, through the usual cancel path. Pinned in
    `queue` mode so one message is queued and one steers (the default mode would steer both)."""
    cli_tui, held = queue_mode

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-cancel")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "start the turn")
            await _until(pilot, lambda: held.gate is not None, tries=200)
            _submit(app, cli_tui, "queued prompt one")
            _submit(app, cli_tui, "/steer a correction")
            await pilot.pause(0.3)
            assert [m.raw for m in app._steering.pending()] == ["a correction"]
            await pilot.press("escape")  # cancel the running turn
            held.release()               # the fake loop wakes, sees the cancel and returns
            await _until(pilot, lambda: not app._turn_in_progress)
            inp = app.query_one("#input", cli_tui.EnterSubmits)
            assert inp.text.startswith("a correction"), inp.text  # ahead of the queued prompt
            assert "queued prompt one" in inp.text
            assert inp.text.index("a correction") < inp.text.index("queued prompt one")
            assert app._steering.pending() == [] and app._steer_raws == {}
            assert any("Moved 2 queued prompts back to the input" in t for t in _texts(app))
            assert not any("arrived after the turn ended" in t for t in _texts(app))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- Steering Phase 4 continued: Ctrl+B and the boundary (tasks.md T4.4, spec open point 3) ------


def test_ctrl_b_frees_the_boundary_for_steering(steer_loop_live, registry, _py):
    """T4.4 (D10, open point 3): a steering message waits behind a foreground `bash_command`, and the
    queue bar says so and points at Ctrl+B. Pressing it detaches the command and the message is delivered
    at once, while the command is still running — the assumption D10's hint is written on."""
    cli_tui, state = steer_loop_live
    from conftest import fake_response, fake_tool_call
    sleeper = _py("import time; time.sleep(30)")
    state.responses = [
        fake_response(tool_calls=[fake_tool_call("bash_command", {"command": sleeper}, "c1")]),
        fake_response("done"),
    ]

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-ctrlb")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.permissions.mode = "auto"  # the shell call must not stop for an approval
            app.on_enter_submits_submitted(cli_tui.EnterSubmits.Submitted("start the turn"))
            await _until(pilot, lambda: any(j.foreground and j.ended is None for j in registry.jobs()), tries=200)
            app.on_enter_submits_submitted(
                cli_tui.EnterSubmits.Submitted("/steer use the fixture, not the mock"))
            await _until(pilot, lambda: bool(app._steering.pending())
                         and app._steering.pending()[0].content is not None)
            # it waits for the command (CON-005: nothing is interrupted), and the bar says why
            assert not any("use the fixture" in line for line in _message_lines(app))
            rendered = app.query_one("#queue-bar", cli_tui.QueueBar).render()
            bar_text = rendered.plain if hasattr(rendered, "plain") else str(rendered)
            assert "waiting for a running command · Ctrl+B to background it" in bar_text, bar_text

            await pilot.press("ctrl+b")  # D10's escape hatch: detach the command, keep the turn going
            await _until(pilot, lambda: any("use the fixture" in line and line.startswith("> ")
                                            for line in _message_lines(app)), tries=200)
            [job] = registry.jobs()
            assert job.ended is None and not job.foreground  # still sleeping: the message did not wait for it
            await _until(pilot, lambda: not app._turn_in_progress, tries=200)
            await app._cleanup_and_save_session()

    asyncio.run(scenario())


# --- Steering Phase 5: docs (plan/2026-11-04-steering/tasks.md T5.1–T5.2) ------------------------


def test_steer_registered(tui):
    """T5.1: `/steer` is a first-class command — in the "/" menu, in `/help`, with its args hint."""
    from dostuff.helpers.agent.constants import COMMAND_ARGS, COMMAND_DESCRIPTIONS, COMMAND_FLAGS
    cli_tui = tui
    assert "/steer" in COMMAND_DESCRIPTIONS and COMMAND_ARGS["/steer"] == "<message>"
    assert "/steer" not in COMMAND_FLAGS

    async def scenario():
        app = cli_tui.DostuffTUI(session_id="steer-help")
        async with app.run_test() as pilot:
            await pilot.pause()
            _submit(app, cli_tui, "/help")
            await _until(pilot, lambda: any("/steer" in t for t in _texts(app)))
            await app._cleanup_and_save_session()

    asyncio.run(scenario())
