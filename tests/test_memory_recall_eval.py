"""Offline recall eval: what a labelled prompt delivers to the model from seeded long-term memory.

Embeddings only (Chroma's MiniLM), no LLM, no network once the model is cached. Seeds a scratch store from
`tests/data/memory_recall_cases.yaml`, runs every case through the real delivery code (`build_turn_context`, and the
facts step once one exists), and asserts recall / false-hit floors. The floors are today's numbers, so a change can
only raise them; the memory-recall work (plan/2026-11-11-memory-recall) raises the bar as it lands.

Floor sweep (not a test): `python -m tests.test_memory_recall_eval --sweep` prints recall and episode false hits
for a range of similarity floors, which is how a shipped floor gets changed.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import yaml

CASES_FILE = Path(__file__).parent / "data" / "memory_recall_cases.yaml"
USER_ID = "recall-eval-user"
FACT_TAGS = ("relevant_user_facts", "user_facts", "project_facts", "other_relevant_facts")
SECTION = re.compile(r"<(" + "|".join(FACT_TAGS + ("past_episodes",)) + r")(?:\s[^>]*)?>\n?(.*?)\n?</\1>", re.S)


def load_cases() -> dict[str, Any]:
    return yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))


class CachedEmbedding:
    """Chroma's embedding function with a per-text cache, filled by one batched call.

    Every query embeds its prompt one text at a time (~0.3 s each on a laptop CPU); one batch over all prompts is
    several times faster. Same model, same vectors, so recall is unchanged.
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self._cache: dict[str, Any] = {}

    def warm(self, texts: list[str]) -> None:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            self._cache.update(zip(todo, self._inner(todo)))

    def __call__(self, input: list[str]) -> list[Any]:  # noqa: A002 - chroma passes it by this name
        self.warm(list(input))
        return [self._cache[t] for t in input]

    def embed_query(self, input: list[str]) -> list[Any]:  # noqa: A002 - chroma calls this for query_texts
        return self(input)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class Seeded:
    """A scratch semantic + episodic store holding the labelled set's memory, one directory per project."""

    def __init__(self, root: Path, data: dict[str, Any], backend: str = "chroma"):
        self.data = data
        self.backend = backend
        self.projects = {}
        for name in "ABC":
            (root / f"proj{name}").mkdir(parents=True, exist_ok=True)
            self.projects[name] = (root / f"proj{name}").resolve()
        self.sem: Any
        self.epi: Any
        if backend == "sqlite":
            from dostuff.lib.memory.sqlite_memory_store import SQLiteEpisodeStore, SQLiteFactStore
            self.sem, self.epi = SQLiteFactStore(str(root / "memory.db")), SQLiteEpisodeStore(str(root / "memory.db"))
        else:
            from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore
            from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore
            persist = str(root / "chroma")
            self.sem = SemanticMemoryStore(persist_path=persist)
            self.epi = EpisodicMemoryStore(persist_path=persist)
            self.embedding = CachedEmbedding(self.sem._collection._embedding_function)
            for store in (self.sem, self.epi):
                store._collection._embedding_function = self.embedding
            self.embedding.warm([c["prompt"] for c in data["cases"]])
        self.fact_owner = {f["key"]: p for p, rows in data["project_facts"].items() for f in rows}
        self.episode_owner = {e["anchor"]: p for p, rows in data["episodes"].items() for e in rows}

    async def seed(self) -> None:
        for fact in self.data["user_facts"]:
            await self.sem.upsert_fact(USER_ID, {**fact, "source": "user_message"})
        # Project facts go in the way the commit under test stores them: with a scope when it has one, otherwise as
        # the user-wide rows a model writes today (the cross-project leak this set measures)
        scoped = "scope" in inspect.signature(self.sem.upsert_fact).parameters
        for proj, rows in self.data["project_facts"].items():
            for fact in rows:
                row = {**fact, "source": "user_message"}
                scope = {"scope": "project", "working_dir": str(self.projects[proj])} if scoped else {}
                await self.sem.upsert_fact(USER_ID, row, **scope)
        now = int(time.time())
        for proj, rows in self.data["episodes"].items():
            batch = []
            for i, ep in enumerate(rows):
                batch.append({
                    "id": f"ep_{proj}_{i:02d}",
                    "vector_text": f"- EVENT TYPE: {ep['type']}\n  - ANCHOR EVENT: {ep['anchor']}\n"
                                   f"  - SUMMARY: {ep['summary']}",
                    "metadata": {"user_id": USER_ID, "session_id": "recall-eval",
                                 "working_dir": str(self.projects[proj]),
                                 "created_at_epoch": now - ep["days_old"] * 86400, "created_at_iso": "",
                                 "event_type": ep["type"], "anchor_event": ep["anchor"], "summary": ep["summary"],
                                 "tags": json.dumps([])},
                })
            await self.epi.add(batch)

    async def raw_similarity(self, prompt: str, proj: str, anchor_part: str) -> float:
        """Undecayed similarity of the project's episode whose anchor contains `anchor_part`: tells a miss in the
        ranking (decay, top-k) from one the store never matched (below the floor whatever the ranking)."""
        got = (await self.epi.query(USER_ID, prompt, top_k=100, working_dir=str(self.projects[proj]),
                                         recency_decay_rate=0.0, min_similarity=-1.0))
        sims = [float(e["score"]) for e in got if anchor_part.lower() in str(e.get("anchor_event") or "").lower()]
        return round(max(sims, default=0.0), 3)

    async def delivered(self, prompt: str, proj: str) -> tuple[set[str], list[str]]:
        """Fact keys and episode lines the model would see for `prompt` in project `proj`."""
        from dostuff.agent.instructions import build_turn_context

        old = os.getcwd()
        os.chdir(self.projects[proj])
        try:
            text = await build_turn_context(prompt, memory_store=self.sem, episodic_store=self.epi, user_id=USER_ID)
            text = getattr(text, "text", text)  # a TurnContext since memory recall T2.1, a str before (--worktree)
        finally:
            os.chdir(old)
        try:  # the facts step the loop sends before the call, when the commit under test has one
            from dostuff.helpers.agent.turn_context import facts_step_content
        except ImportError:
            pass
        else:
            here = str(self.projects[proj])
            text += "\n" + facts_step_content(await self.sem.list_all(USER_ID, working_dir=here), here)[0]
        facts: set[str] = set()
        episodes: list[str] = []
        for tag, body in SECTION.findall(text):
            if tag == "past_episodes":
                episodes += [line for line in body.splitlines() if "[" in line]
            else:
                facts |= set(re.findall(r"^- ([\w.-]+):", body, re.M))
        return facts, episodes


def hit(label: str, facts: set[str], episodes: list[str]) -> bool:
    kind, _, what = label.partition(":")
    if kind == "f":
        return what in facts
    return any(what.lower() in e.lower() for e in episodes) if what else bool(episodes)


async def evaluate(seeded: Seeded) -> list[dict[str, Any]]:
    rows = []
    for case in seeded.data["cases"]:
        proj = case["project"]
        facts, episodes = await seeded.delivered(case["prompt"], proj)
        leaks = sorted({f"f:{f}" for f in facts if seeded.fact_owner.get(f, proj) != proj}
                       | {f"e:{a[:30]}" for a, owner in seeded.episode_owner.items()
                          if owner != proj and any(a in e for e in episodes)})
        miss = [m for m in case["must_recall"] if not hit(m, facts, episodes)]
        rows.append({**case, "miss": miss,
                     "false": [m for m in case["must_not_recall"] if hit(m, facts, episodes)],
                     "leaks": leaks,
                     "miss_similarity": {m: await seeded.raw_similarity(case["prompt"], proj, m.partition(":")[2])
                                         for m in miss if m.startswith("e:")}})
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    def recall(prefix: str) -> tuple[int, int]:
        labels = [(m, r) for r in rows for m in r["must_recall"] if m.startswith(prefix)]
        return sum(m not in r["miss"] for m, r in labels), len(labels)

    return {
        "fact recall": recall("f:"),
        "episode recall": recall("e:"),
        "false hits": (sum(len(r["false"]) for r in rows), sum(len(r["must_not_recall"]) for r in rows)),
        "leaky cases": (sum(bool(r["leaks"]) for r in rows), len(rows)),
    }


def table(rows: list[dict[str, Any]]) -> str:
    lines = ["| measure | value |", "|---|---|"]
    lines += [f"| {k} | {a}/{b} |" for k, (a, b) in summary(rows).items()]
    by_group: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(not (r["miss"] or r["false"] or r["leaks"]))
    lines += ["", "| group | cases fully right |", "|---|---|"]
    lines += [f"| {g} | {sum(oks)}/{len(oks)} |" for g, oks in by_group.items()]
    lines += ["", "| proj | prompt | missed | false hits | leaks |", "|---|---|---|---|---|"]
    lines += [f"| {r['project']} | {r['prompt'][:60]} | {', '.join(r['miss'])} | {', '.join(r['false'])} | "
              f"{', '.join(r['leaks'])} |" for r in rows if r["miss"] or r["false"] or r["leaks"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Tests


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> list[dict[str, Any]]:
    pytest.importorskip("chromadb")
    seeded = Seeded(tmp_path_factory.mktemp("recall_eval"), load_cases())

    async def run() -> list[dict[str, Any]]:
        await seeded.seed()
        return await evaluate(seeded)

    return asyncio.run(run())


def test_case_file_is_well_formed():
    data = load_cases()
    fact_keys = {f["key"] for f in data["user_facts"]} | {
        f["key"] for rows in data["project_facts"].values() for f in rows}
    anchors = [e["anchor"] for rows in data["episodes"].values() for e in rows]
    for case in data["cases"]:
        assert case["project"] in "ABC", case
        for label in case["must_recall"] + case["must_not_recall"]:
            kind, _, what = label.partition(":")
            assert kind in ("f", "e"), label
            if kind == "f":
                assert what in fact_keys, label
            elif what:
                assert any(what.lower() in a.lower() for a in anchors), label
    # Open point 5: at least 20 negatives with no episode allowed, before any episode floor moves
    negatives = [c for c in data["cases"] if "e:" in c["must_not_recall"]]
    assert len(negatives) >= 20


# Raised by memory recall phase 1 (facts step): every fact below the cap is delivered and project facts stay in their
# project (was 18 / 15 / 10 / 8 on package-structure before). Episode recall 15 -> 17 with T2.3: two decay cases
# (an old WAL decision among newer SQLite events) that the flat decay ranked out. Never lower one without a sweep
# table in the PR that says why.
FLOORS = {"fact recall": 34, "episode recall": 17}
CEILINGS = {"false hits": 2, "leaky cases": 0}


@pytest.mark.parametrize("measure", list(FLOORS))
def test_recall_at_least_baseline(results, measure):
    got, total = summary(results)[measure]
    assert got >= FLOORS[measure], f"{measure} {got}/{total} < {FLOORS[measure]}\n\n{table(results)}"


@pytest.mark.parametrize("measure", list(CEILINGS))
def test_misses_at_most_baseline(results, measure):
    got, total = summary(results)[measure]
    assert got <= CEILINGS[measure], f"{measure} {got}/{total} > {CEILINGS[measure]}\n\n{table(results)}"


def test_decay_group_nothing_lost_in_ranking(results):
    """Memory recall T2.3 (D9): an old DECISION or LEARNING that matches the prompt is recalled despite its age. A
    miss is allowed only when the episode never cleared the similarity floor, which no decay rate can fix (the
    embedding doesn't link the prompt to it, e.g. "line endings LF" vs ".gitattributes eol=lf", 0.18)."""
    from dostuff.lib.memory.episodic_memory_store import MIN_EPISODE_SIMILARITY
    lost = [(r["prompt"], m, s) for r in results if r["group"] == "decay"
            for m, s in r["miss_similarity"].items() if s >= MIN_EPISODE_SIMILARITY]
    assert not lost, f"recalled above the floor but ranked out: {lost}"
    assert all(not r["false"] and not r["leaks"] for r in results if r["group"] == "decay")


@pytest.fixture(scope="module")
def sqlite_results(tmp_path_factory) -> list[dict[str, Any]]:
    from dostuff.lib.memory.sqlite_memory_store import fts5_available
    if not fts5_available():
        pytest.skip("sqlite3 built without FTS5")
    seeded = Seeded(tmp_path_factory.mktemp("recall_eval_sqlite"), load_cases(), backend="sqlite")

    async def run() -> list[dict[str, Any]]:
        await seeded.seed()
        return await evaluate(seeded)

    return asyncio.run(run())


# memory.backend: sqlite (FTS5 keywords, no embeddings) on the same set, floors from its sweep (2026-11-14): the
# embedding store's recall with no episode false hits
SQLITE_FLOORS = {"fact recall": 34, "episode recall": 17}
SQLITE_CEILINGS = {"false hits": 0, "leaky cases": 0}


@pytest.mark.parametrize("measure", list(SQLITE_FLOORS))
def test_sqlite_recall_at_least_baseline(sqlite_results, measure):
    got, total = summary(sqlite_results)[measure]
    assert got >= SQLITE_FLOORS[measure], f"{measure} {got}/{total}\n\n{table(sqlite_results)}"


@pytest.mark.parametrize("measure", list(SQLITE_CEILINGS))
def test_sqlite_misses_at_most_baseline(sqlite_results, measure):
    got, total = summary(sqlite_results)[measure]
    assert got <= SQLITE_CEILINGS[measure], f"{measure} {got}/{total}\n\n{table(sqlite_results)}"


# ---------------------------------------------------------------------------------------------------------------
# Floor sweep: python -m tests.test_memory_recall_eval --sweep [--backend sqlite]

SWEEP_FLOORS = (0.15, 0.18, 0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.40)


async def sweep(seeded: Seeded) -> str:
    """Recall and episode false hits by raw similarity floor, with top-k facts 5, episodes 3 (the per-turn
    overflow and episode recall). Through the store API, so it sweeps either backend's own score."""
    sims = []
    n_facts = len(await seeded.sem.list_all(USER_ID))
    for case in seeded.data["cases"]:
        proj = case["project"]
        fr = await seeded.sem.query(USER_ID, case["prompt"], top_k=n_facts, min_similarity=-1.0)
        f_s = [(f["key"], float(f.get("similarity", 0.0))) for f in fr]
        e_s: list[tuple[str, float]] = []
        if proj in seeded.data["episodes"]:
            er = await seeded.epi.query(USER_ID, case["prompt"], top_k=50, working_dir=str(seeded.projects[proj]),
                                        recency_decay_rate=0.0, min_similarity=-1.0)
            e_s = [(str(e["anchor_event"]), float(e["score"])) for e in er]
        sims.append((case, f_s, e_s))
    lines = ["| floor | fact recall | episode recall | episode false hits (off-topic + unrelated + leak) |",
             "|---|---|---|---|"]
    for floor in SWEEP_FLOORS:
        fo = ft = eo = et = fh = fn = 0
        for case, f_s, e_s in sims:
            f_top = [k for k, s in f_s if s >= floor][:5]
            e_top = [a for a, s in e_s if s >= floor][:3]
            for m in case["must_recall"]:
                kind, _, what = m.partition(":")
                if kind == "f":
                    ft += 1
                    fo += what in f_top
                else:
                    et += 1
                    eo += any(what.lower() in a.lower() for a in e_top)
            if case["group"] in ("offtopic", "unrelated", "leak"):
                fn += 1
                fh += bool(e_top) if case["group"] != "leak" else any(
                    m.startswith("e:") and any(m[2:].lower() in a.lower() for a in e_top)
                    for m in case["must_not_recall"])
        lines.append(f"| {floor:g} | {fo}/{ft} | {eo}/{et} | {fh}/{fn} |")
    return "\n".join(lines)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", action="store_true", help="also print the similarity-floor sweep")
    ap.add_argument("--backend", choices=("chroma", "sqlite"), default="chroma", help="memory.backend to evaluate")
    args = ap.parse_args()
    seeded = Seeded(Path(tempfile.mkdtemp(prefix="recall_eval_")), load_cases(), backend=args.backend)

    async def run() -> None:
        await seeded.seed()
        print(table(await evaluate(seeded)))
        if args.sweep:
            print("\n" + await sweep(seeded))

    asyncio.run(run())


if __name__ == "__main__":
    _main()
