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
    monkeypatch.setattr(cli_tui, "get_model_token_limit", lambda: 100_000)

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
            await pilot.press("/")
            await pilot.pause()
            assert menu.display and menu.option_count == len(cli_tui.COMMAND_DESCRIPTIONS)
            await pilot.press("c", "o")
            await pilot.pause()
            assert menu.option_count == 1 and menu.selected_command() == "/compact"
            await pilot.press("tab")
            await pilot.pause()
            assert app.query_one("#input", cli_tui.EnterSubmits).text == "/compact"
            await pilot.press("escape")
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
