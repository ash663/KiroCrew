"""Conductor work ledger routes — Phase 2 exit criteria, one test per criterion.

Pins what the conductor-work-ledger RFC (``docs/request-for-change/rfc-conductor-work-ledger.md``,
revision v2) §Migration plan Phase 2 lists for the four tools and their routes: a
worker's report reaches its own item and no other, asserted against a
two-conductor two-worker fixture; every error code carries its tabulated HTTP
status; ``accept_batch`` parses in the real ``accept_eval.py`` and ignores a
worker's claimed ``pr``; a round trip through ``work_report`` writes no
conductor-owned field, asserted field by field; and each of the four caller states
in the dispatch table resolves as tabulated against both tool halves.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from kiro_crew import validation
from kiro_crew import work_ledger as wl
from kiro_crew.dashboard.handlers import work_ledger as routes
from kiro_crew.platform import redact_via_context
from kiro_crew.validation import sanitize_string

#: The real implementation, captured before the autouse fixture stubs the name.
#: A test that drives the true branch has to reach past its own default.
_REAL_REACHES_A_CHANNEL = routes._reaches_a_channel

CONDUCTOR_A = "chat-a-conductor"
CONDUCTOR_B = "chat-b-conductor"
WORKER_A = "chat-a-worker"
WORKER_B = "chat-b-worker"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test writes into its own data home, never the live one."""
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
    yield


@pytest.fixture(autouse=True)
def _open_route(monkeypatch):
    """Bypass session recognition/restriction (their own suites cover them).

    Several tests below re-patch these to assert the refusals still fire, so the
    bypass is a default rather than an assumption.

    The channel-mirror probe is stubbed to "not mirrored" for the same reason and
    with the same caveat: the real probe reads the session store, which a
    ``make_mocked_request`` slot cannot answer, and it FAILS CLOSED — so leaving it
    live would make every test's caller read as mirrored and refused.
    ``test_an_outbound_mirrored_session_is_refused`` drives the true branch.
    """

    async def _recognized(*a: Any, **k: Any) -> None:
        return None

    monkeypatch.setattr(routes, "_recognize_session", _recognized)
    monkeypatch.setattr(routes, "_is_restricted_session", lambda *a: False)
    monkeypatch.setattr(routes, "_reaches_a_channel", lambda request, sk: False)
    # The routes record every write into the caller's crew log and refuse when
    # they cannot: a ``MagicMock`` state resolves no unit, so the gate is opened
    # here and the append is swallowed. ``test_work_ledger_projection.py`` drives
    # the gate and the recorded entries themselves.
    monkeypatch.setattr(routes.crew_log_emit, "enabled", lambda: True)
    monkeypatch.setattr(routes, "unit_for_session_key", lambda sessions, key: f"unit:{key}")
    monkeypatch.setattr(routes.crew_log_emit, "on_work_recorded", lambda unit, data: True)


class _Slot:
    """The two slot attributes these routes read, and nothing else.

    ``_created_by`` is what ``session_control.create_session`` stamps with the
    calling session's key — the attribution a bind's ownership check rests on — and
    ``workspace`` is the memory boundary it must not cross.
    """

    def __init__(
        self, created_by: str = "", workspace: str = "default", running: bool = False
    ) -> None:
        self._created_by = created_by
        self.workspace = workspace
        # ``_ChatSlot.running`` is a property over its asyncio task; a bool is the
        # whole of what these routes read from it.
        self.running = running


#: Slot table for the request under test, keyed exactly as the routes look keys
#: up. Reset per test by the autouse fixture, because a leaked worker slot would
#: make a later test's ``stale`` assertion pass for the wrong reason.
_SLOTS: dict[str, _Slot] = {}


@pytest.fixture(autouse=True)
def _clean_slots():
    _SLOTS.clear()
    yield
    _SLOTS.clear()


def _dispatched(
    worker: str, conductor: str, *, workspace: str = "default", running: bool = False
) -> None:
    """Register *worker* as a session *conductor* created, as session_create would."""
    _SLOTS[worker] = _Slot(created_by=conductor, workspace=workspace, running=running)


def _req(method: str, path: str, *, body: Any = ..., sk: str) -> web.Request:
    app = web.Application()
    state = MagicMock()
    # A real-shaped slot table rather than a MagicMock: a mock would answer every
    # ``get_slot`` with a truthy object, making every item look alive and every
    # bind look owned — the two things these routes decide from.
    state.get_slot = MagicMock(side_effect=lambda key: _SLOTS.get(key))
    app["state"] = state
    req = make_mocked_request(method, path, app=app, headers={"X-Session-Key": sk})
    # The routes require the internal-secret principal, positively confirmed — on
    # loopback a strict internal path with no secret header falls through to cookie
    # auth and is granted, so the handler cannot infer it from the path. This is
    # what ``token_auth`` sets for a verified secret caller.
    req["internal_auth"] = True
    if body is not ...:
        req.json = AsyncMock(return_value=body)  # type: ignore[method-assign]
    return req


async def _record(sk: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    resp = await routes.api_work_ledger_record(
        _req("POST", "/api/work-ledger/record", body=body, sk=sk)
    )
    return resp.status, json.loads(resp.text)


async def _report(sk: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    resp = await routes.api_work_report(_req("POST", "/api/work-ledger/report", body=body, sk=sk))
    return resp.status, json.loads(resp.text)


async def _read(sk: str) -> tuple[int, dict[str, Any]]:
    resp = await routes.api_work_ledger_get(_req("GET", "/api/work-ledger", sk=sk))
    return resp.status, json.loads(resp.text)


async def _brief(sk: str) -> tuple[int, dict[str, Any]]:
    resp = await routes.api_work_brief(_req("GET", "/api/work-ledger/brief", sk=sk))
    return resp.status, json.loads(resp.text)


async def _dispatch(conductor: str, worker: str, title: str, acceptance: dict) -> str:
    """The conductor's own create-then-bind sequence, through the routes only."""
    status, body = await _record(
        conductor, {"action": "goal", "goal": f"goal for {title}", "round": 1}
    )
    assert status == 200, body
    status, body = await _record(
        conductor, {"action": "create", "title": title, "acceptance": acceptance}
    )
    assert status == 200, body
    item_id = body["item"]["item_id"]
    # The conductor dispatches the session BEFORE binding it, which is the order
    # the RFC's binding lifecycle specifies and the order the ownership check on
    # ``bind`` requires: the slot has to exist and be attributed to this conductor.
    _dispatched(worker, conductor)
    status, body = await _record(
        conductor, {"action": "bind", "item_id": item_id, "worker_session_key": worker}
    )
    assert status == 200, body
    return item_id


async def two_by_two() -> dict[str, str]:
    """Two conductors, one bound worker each — the fixture the RFC names.

    An awaited helper rather than a pytest fixture: an async fixture needs a
    plugin-provided decorator, and every consumer here is already a coroutine.
    """
    item_a = await _dispatch(CONDUCTOR_A, WORKER_A, "item A", {"kind": "human_approval"})
    item_b = await _dispatch(CONDUCTOR_B, WORKER_B, "item B", {"kind": "human_approval"})
    return {"item_a": item_a, "item_b": item_b}


# ── the isolation the whole design exists for ─────────────────────────────


@pytest.mark.asyncio
async def test_a_workers_report_reaches_its_own_item_and_no_other():
    """Worker A's report lands on A's item; B's item is byte-identical after it."""
    ids = await two_by_two()
    before = wl.item_path(CONDUCTOR_B, ids["item_b"]).read_bytes()

    status, _ = await _report(WORKER_A, {"status": "progress", "summary": "A moving"})
    assert status == 200

    item_a = wl.read_work_item(CONDUCTOR_A, ids["item_a"])
    item_b = wl.read_work_item(CONDUCTOR_B, ids["item_b"])
    assert item_a is not None and item_b is not None
    assert item_a.summary == "A moving"
    assert item_b.summary == ""
    assert item_b.status is None
    assert wl.item_path(CONDUCTOR_B, ids["item_b"]).read_bytes() == before


@pytest.mark.asyncio
async def test_a_worker_cannot_name_another_item_because_there_is_no_parameter():
    """The bound item is resolved, not supplied — naming one is an unknown field."""
    ids = await two_by_two()
    status, body = await _report(
        WORKER_A,
        {"status": "done", "summary": "mine now", "item_id": ids["item_b"]},
    )
    assert status == 400
    assert body["code"] == wl.CODE_INVALID_VALUE
    item_b = wl.read_work_item(CONDUCTOR_B, ids["item_b"])
    assert item_b is not None and item_b.status is None


@pytest.mark.asyncio
async def test_a_report_writes_no_conductor_owned_field():
    """Field by field: every conductor-owned value survives a worker round trip.

    Asserted as a set difference rather than as a list of five names, so a field
    added to the item record joins this check automatically instead of silently
    escaping it.
    """
    ids = await two_by_two()
    worker_owned = {"status", "summary", "artifacts", "pr", "last_report_at"}
    before = (wl.read_work_item(CONDUCTOR_A, ids["item_a"]) or wl.WorkItem()).to_dict()

    status, _ = await _report(
        WORKER_A,
        {
            "status": "done",
            "summary": "built it",
            "artifacts": {"pr": "42", "branch": "feat/x"},
            "pr": 42,
        },
    )
    assert status == 200

    after = (wl.read_work_item(CONDUCTOR_A, ids["item_a"]) or wl.WorkItem()).to_dict()
    changed = {k for k in after if before.get(k) != after.get(k)}
    assert changed <= worker_owned, f"a worker changed conductor-owned field(s): {changed}"
    # And the conductor's own values are still exactly what it wrote.
    assert after["state"] == "open"
    assert after["verdict"] is None
    assert after["acceptance"] == {"kind": "human_approval"}
    assert after["title"] == "item A"


@pytest.mark.asyncio
async def test_a_brief_shows_one_item_and_never_a_sibling_or_the_goal():
    ids = await two_by_two()
    status, body = await _brief(WORKER_A)
    assert status == 200
    brief = body["brief"]
    assert brief["item_id"] == ids["item_a"]
    assert brief["title"] == "item A"
    assert "goal" not in brief
    assert "items" not in brief
    assert ids["item_b"] not in json.dumps(brief)


# ── the dispatch table, cell by cell ──────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_table_binding_only_is_a_worker():
    """A binding file and no ledger directory: worker half answers, conductor 404s."""
    await two_by_two()
    assert (await _brief(WORKER_A))[0] == 200
    assert (await _report(WORKER_A, {"status": "progress", "summary": "x"}))[0] == 200
    status, body = await _read(WORKER_A)
    assert status == 404
    assert body["code"] == wl.CODE_NO_LEDGER
    status, body = await _record(
        WORKER_A, {"action": "verdict", "item_id": "it_00000000", "verdict": "pass"}
    )
    assert status == 404
    assert body["code"] == wl.CODE_NO_LEDGER


@pytest.mark.asyncio
async def test_dispatch_table_ledger_only_is_a_conductor():
    """A ledger directory and no binding: conductor half answers, worker 403s."""
    await two_by_two()
    assert (await _read(CONDUCTOR_A))[0] == 200
    status, body = await _brief(CONDUCTOR_A)
    assert status == 403
    assert body["code"] == routes.CODE_NOT_BOUND
    status, body = await _report(CONDUCTOR_A, {"status": "done", "summary": "x"})
    assert status == 403
    assert body["code"] == routes.CODE_NOT_BOUND


@pytest.mark.asyncio
async def test_dispatch_table_both_is_a_second_level_conductor():
    """Worker A opens its own ledger: all four answer, and depth is one past A's."""
    ids = await two_by_two()
    status, body = await _record(WORKER_A, {"action": "goal", "goal": "sub-goal", "round": 1})
    assert status == 200, body
    assert body["conductor"]["depth"] == 1
    assert body["conductor"]["parent_item"] == ids["item_a"]

    assert (await _brief(WORKER_A))[0] == 200
    assert (await _report(WORKER_A, {"status": "progress", "summary": "sub"}))[0] == 200
    assert (await _read(WORKER_A))[0] == 200
    status, _ = await _record(
        WORKER_A,
        {"action": "create", "title": "sub item", "acceptance": {"kind": "human_approval"}},
    )
    assert status == 200


@pytest.mark.asyncio
async def test_dispatch_table_neither_gets_both_refusals():
    status, body = await _brief("chat-nobody")
    assert status == 403
    assert body["code"] == routes.CODE_NOT_BOUND
    status, body = await _report("chat-nobody", {"status": "done", "summary": "x"})
    assert status == 403
    assert body["code"] == routes.CODE_NOT_BOUND
    status, body = await _read("chat-nobody")
    assert status == 404
    assert body["code"] == wl.CODE_NO_LEDGER
    status, body = await _record(
        "chat-nobody", {"action": "decide", "item_id": "it_00000000", "decision": "x"}
    )
    assert status == 404
    assert body["code"] == wl.CODE_NO_LEDGER


@pytest.mark.asyncio
async def test_a_third_level_conductor_is_refused_at_the_depth_cap():
    """depth <= 2: a grandchild may work, and may not conduct."""
    await two_by_two()
    assert (await _record(WORKER_A, {"action": "goal", "goal": "level 1", "round": 1}))[0] == 200
    grandchild = "chat-a-grandchild"
    _dispatched(grandchild, WORKER_A)
    status, body = await _record(
        WORKER_A, {"action": "create", "title": "leaf", "acceptance": {"kind": "human_approval"}}
    )
    assert status == 200
    leaf = body["item"]["item_id"]
    assert (
        await _record(
            WORKER_A, {"action": "bind", "item_id": leaf, "worker_session_key": grandchild}
        )
    )[0] == 200
    # The grandchild is a working worker...
    assert (await _report(grandchild, {"status": "progress", "summary": "leaf work"}))[0] == 200
    # ...and its ledger opens at the cap (depth 2 == MAX_DEPTH) ...
    status, body = await _record(grandchild, {"action": "goal", "goal": "level 2", "round": 1})
    assert status == 200, body
    assert body["conductor"]["depth"] == wl.MAX_DEPTH
    # ...where it can dispatch nothing, which is what the cap buys: a summary of
    # summaries of summaries is not evidence any more.
    status, body = await _record(
        grandchild,
        {"action": "create", "title": "great-grandchild", "acceptance": {"kind": "human_approval"}},
    )
    assert status == 409
    assert body["code"] == wl.CODE_DEPTH_EXCEEDED


# ── every error code carries its tabulated status ─────────────────────────


@pytest.mark.asyncio
async def test_every_store_code_maps_to_the_status_the_rfc_tabulates():
    """The map is exhaustive over the store's ``CODE_*`` constants.

    Without this, a code the store gains later degrades to 400 unnoticed, and a
    409-shaped conflict would reach the model as a bad-argument error.
    """
    store_codes = {
        value
        for name, value in vars(wl).items()
        if name.startswith("CODE_") and isinstance(value, str)
    }
    assert store_codes == set(routes._CODE_STATUS), (
        "the code->status map and the store's CODE_* constants disagree: "
        f"{store_codes ^ set(routes._CODE_STATUS)}"
    )
    assert routes._CODE_STATUS == {
        "no_ledger": 404,
        "unknown_item": 404,
        "already_bound": 409,
        "item_closed": 409,
        "item_cap_exceeded": 409,
        "item_store_full": 409,
        "depth_exceeded": 409,
        "crew_log_incomplete": 409,
        "cache_dirty": 409,
        "field_too_long": 400,
        "invalid_action": 400,
        "invalid_status": 400,
        "invalid_value": 400,
        # Maintenance-only: raised by ``purge_conductor``, reachable from no
        # route. Mapped so the exhaustiveness property above keeps its meaning.
        "ledger_not_finished": 409,
    }


@pytest.mark.asyncio
async def test_unknown_item_is_404():
    await two_by_two()
    status, body = await _record(
        CONDUCTOR_A, {"action": "decide", "item_id": "it_deadbeef", "decision": "nope"}
    )
    assert status == 404
    assert body["code"] == wl.CODE_UNKNOWN_ITEM


@pytest.mark.asyncio
async def test_already_bound_is_409():
    ids = await two_by_two()
    _dispatched("chat-other", CONDUCTOR_A)
    status, body = await _record(
        CONDUCTOR_A,
        {"action": "bind", "item_id": ids["item_a"], "worker_session_key": "chat-other"},
    )
    assert status == 409
    assert body["code"] == wl.CODE_ALREADY_BOUND


@pytest.mark.asyncio
async def test_item_closed_is_409_for_both_halves():
    ids = await two_by_two()
    status, _ = await _record(
        CONDUCTOR_A, {"action": "close", "item_id": ids["item_a"], "state": "accepted"}
    )
    assert status == 200
    status, body = await _report(WORKER_A, {"status": "done", "summary": "too late"})
    assert status == 409
    assert body["code"] == wl.CODE_ITEM_CLOSED
    status, body = await _record(
        CONDUCTOR_A, {"action": "decide", "item_id": ids["item_a"], "decision": "late"}
    )
    assert status == 409
    assert body["code"] == wl.CODE_ITEM_CLOSED


@pytest.mark.asyncio
async def test_item_cap_exceeded_is_409():
    await two_by_two()
    for n in range(wl.MAX_ITEMS_PER_CONDUCTOR - 1):
        status, _ = await _record(
            CONDUCTOR_A,
            {"action": "create", "title": f"item {n}", "acceptance": {"kind": "human_approval"}},
        )
        assert status == 200
    status, body = await _record(
        CONDUCTOR_A,
        {"action": "create", "title": "one too many", "acceptance": {"kind": "human_approval"}},
    )
    assert status == 409
    assert body["code"] == wl.CODE_ITEM_CAP_EXCEEDED


@pytest.mark.asyncio
async def test_item_store_full_is_409(monkeypatch):
    """A board full of closed records refuses the next create as a 409 conflict.

    The code is the store's own, ``item_store_full``: the open cap is not what bit,
    since nothing on the board is open.
    """
    monkeypatch.setattr(wl, "MAX_STORED_ITEMS_PER_CONDUCTOR", 2)
    items = await two_by_two()
    status, _ = await _record(
        CONDUCTOR_A, {"action": "close", "item_id": items["item_a"], "state": "accepted"}
    )
    assert status == 200
    status, body = await _record(
        CONDUCTOR_A,
        {"action": "create", "title": "second", "acceptance": {"kind": "human_approval"}},
    )
    assert status == 200
    status, _ = await _record(
        CONDUCTOR_A, {"action": "close", "item_id": body["item"]["item_id"], "state": "accepted"}
    )
    assert status == 200
    status, body = await _record(
        CONDUCTOR_A,
        {"action": "create", "title": "one too many", "acceptance": {"kind": "human_approval"}},
    )
    assert status == 409
    assert body["code"] == wl.CODE_ITEM_STORE_FULL


@pytest.mark.asyncio
async def test_field_too_long_is_400_and_names_the_field():
    await two_by_two()
    status, body = await _report(
        WORKER_A, {"status": "progress", "summary": "x" * (wl.MAX_SUMMARY_CHARS + 1)}
    )
    assert status == 400
    assert body["code"] == wl.CODE_FIELD_TOO_LONG
    assert "summary" in body["error"]


@pytest.mark.asyncio
async def test_invalid_status_is_400():
    await two_by_two()
    status, body = await _report(WORKER_A, {"status": "finished", "summary": "x"})
    assert status == 400
    assert body["code"] == wl.CODE_INVALID_STATUS


@pytest.mark.asyncio
async def test_invalid_action_is_400():
    await two_by_two()
    status, body = await _record(CONDUCTOR_A, {"action": "delete"})
    assert status == 400
    assert body["code"] == wl.CODE_INVALID_ACTION


@pytest.mark.asyncio
async def test_an_unrecognized_session_is_refused_before_the_store(monkeypatch):
    async def _refused(*a: Any, **k: Any) -> web.Response:
        return web.json_response(
            {"error": "unknown session", "code": "unknown_session"}, status=403
        )

    monkeypatch.setattr(routes, "_recognize_session", _refused)
    for call in (
        routes.api_work_brief(_req("GET", "/api/work-ledger/brief", sk="chat-x")),
        routes.api_work_ledger_get(_req("GET", "/api/work-ledger", sk="chat-x")),
    ):
        assert (await call).status == 403


@pytest.mark.asyncio
async def test_a_restricted_session_is_refused(monkeypatch):
    await two_by_two()
    monkeypatch.setattr(routes, "_is_restricted_session", lambda *a: True)
    resp = await routes.api_work_report(
        _req(
            "POST", "/api/work-ledger/report", body={"status": "done", "summary": "x"}, sk=WORKER_A
        )
    )
    assert resp.status == 403
    assert json.loads(resp.text)["code"] == "restricted_session"


# ── acceptance stays the conductor's ──────────────────────────────────────


@pytest.mark.asyncio
async def test_accept_batch_parses_in_the_real_accept_eval():
    """Piped into the bundled script, unmodified, and read back as its verdicts."""
    ids = await two_by_two()
    script = (
        Path(__file__).resolve().parents[1]
        / "src/kiro_crew/builtin_skills/goal-conductor/scripts/accept_eval.py"
    )
    assert script.is_file(), script
    _, body = await _read(CONDUCTOR_A)
    batch = body["accept_batch"]
    assert batch["items"], batch
    proc = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(batch).encode(),
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    parsed = json.loads(proc.stdout.decode())
    # ``accept_eval.py`` answers ``{"results": [{"id", "verdict", "evidence"}]}`` —
    # one verdict per item it was handed, which is what "the batch parses" means.
    evaluated = {row.get("id") for row in parsed["results"]}
    assert ids["item_a"] in evaluated, parsed
    verdicts = {row["verdict"] for row in parsed["results"]}
    assert verdicts <= wl.VERDICTS, verdicts


@pytest.mark.asyncio
async def test_accept_batch_ignores_a_worker_supplied_pr():
    """The claim is surfaced beside the item and never enters the bar."""
    ids = await two_by_two()
    status, _ = await _report(WORKER_A, {"status": "done", "summary": "green", "pr": 999_111})
    assert status == 200
    _, body = await _read(CONDUCTOR_A)
    row = next(r for r in body["items"] if r["item_id"] == ids["item_a"])
    assert row["pr"] == 999_111, "the claim must be visible to the conductor"
    batch_text = json.dumps(body["accept_batch"])
    assert "999111" not in batch_text
    for entry in body["accept_batch"]["items"]:
        assert "pr" not in entry["accept"], entry


@pytest.mark.asyncio
async def test_the_conductor_promotes_a_claim_with_the_accept_action():
    """``accept`` is the explicit write that moves the bar — and only it does."""
    await two_by_two()
    item_id = await _dispatch(
        CONDUCTOR_A, "chat-a-worker-2", "pending pr", {"kind": "pr_checks", "pr": "TBD"}
    )
    status, _ = await _report(
        "chat-a-worker-2", {"status": "done", "summary": "opened it", "pr": 4321}
    )
    assert status == 200
    _, body = await _read(CONDUCTOR_A)
    # Absent from the batch while the bar still says "TBD", and the item row says why.
    assert not [e for e in body["accept_batch"]["items"] if e["id"] == item_id]
    row = next(r for r in body["items"] if r["item_id"] == item_id)
    assert row["acceptance_concrete"] is False
    assert row["acceptance"]["pr"] == "TBD", "the stored bar is untouched, just not batched"

    status, body = await _record(
        CONDUCTOR_A,
        {
            "action": "accept",
            "item_id": item_id,
            "acceptance": {"kind": "pr_checks", "pr": 4321, "repo": "kirodotdev/KiroCrew"},
        },
    )
    assert status == 200, body
    _, body = await _read(CONDUCTOR_A)
    entry = next(e for e in body["accept_batch"]["items"] if e["id"] == item_id)
    assert entry["accept"]["pr"] == 4321
    # One event per write, and the promotion is one of them.
    row = next(r for r in body["items"] if r["item_id"] == item_id)
    assert row["acceptance_concrete"] is True
    assert any(e["kind"] == "decision" for e in row["events"])


@pytest.mark.asyncio
async def test_a_read_says_per_item_why_an_item_is_not_in_the_batch():
    """``acceptance_concrete`` is derived on the row, not stored, so a conductor can
    tell "the bar is not filled in yet" from "the read dropped my item"."""
    ids = await two_by_two()
    pending = await _dispatch(
        CONDUCTOR_A, "chat-a-worker-3", "no number yet", {"kind": "pr_checks", "pr": "TBD"}
    )
    _, body = await _read(CONDUCTOR_A)
    rows = {r["item_id"]: r for r in body["items"]}
    assert rows[pending]["acceptance_concrete"] is False
    assert rows[ids["item_a"]]["acceptance_concrete"] is True
    batched = {e["id"] for e in body["accept_batch"]["items"]}
    assert pending not in batched
    assert ids["item_a"] in batched
    stored = json.loads(wl.item_path(CONDUCTOR_A, pending).read_text())
    assert "acceptance_concrete" not in stored


@pytest.mark.asyncio
async def test_each_batch_entry_carries_the_items_status_unfiltered():
    """The conductor filters to ``done``; the read hands it the field to filter on and
    filters nothing itself."""
    ids = await two_by_two()
    status, _ = await _report(WORKER_A, {"status": "progress", "summary": "moving"})
    assert status == 200
    _, body = await _read(CONDUCTOR_A)
    entry = next(e for e in body["accept_batch"]["items"] if e["id"] == ids["item_a"])
    assert entry["status"] == "progress"
    assert entry["accept"] == {"kind": "human_approval"}

    status, _ = await _report(WORKER_A, {"status": "done", "summary": "met"})
    assert status == 200
    _, body = await _read(CONDUCTOR_A)
    entry = next(e for e in body["accept_batch"]["items"] if e["id"] == ids["item_a"])
    assert entry["status"] == "done"


@pytest.mark.asyncio
async def test_a_done_item_waiting_on_the_conductor_is_not_stale():
    """``stale`` means the WORKER went quiet. Once it has claimed its bar is met the
    next move is the conductor's or a human's, so the flag must not point back at the
    reader — a quiet ``done`` item is silent because it is finished."""
    ids = await two_by_two()
    status, _ = await _report(WORKER_A, {"status": "done", "summary": "opened the pr"})
    assert status == 200
    item = wl.read_work_item(CONDUCTOR_A, ids["item_a"])
    assert item is not None
    long_after = datetime.now().astimezone() + timedelta(seconds=wl.DEFAULT_STALE_WINDOW_SECS * 6)
    assert (
        routes._slot_running(_req("GET", "/api/work-ledger", sk=CONDUCTOR_A).app["state"], WORKER_A)
        is False
    )
    assert wl.is_stale(item, worker_running=False, now=long_after) is False
    _, body = await _read(CONDUCTOR_A)
    row = next(r for r in body["items"] if r["item_id"] == ids["item_a"])
    assert row["stale"] is False


@pytest.mark.asyncio
async def test_accept_is_a_route_action_not_a_store_action():
    """The store's six are pinned by Phase 1, so the seventh lives one layer up."""
    assert "accept" not in wl.CONDUCTOR_ACTIONS
    assert "accept" in routes.RECORD_ACTIONS
    assert routes.RECORD_ACTIONS == wl.CONDUCTOR_ACTIONS | {"accept"}
    assert routes.RECORD_ACTIONS == validation._WORK_RECORD_ACTIONS


# ── derived flags, and the schema caps that restate the store's ───────────


@pytest.mark.asyncio
async def test_orphaned_and_stale_are_derived_not_stored():
    """Neither flag is a field, and both come back on a read."""
    ids = await two_by_two()
    _, body = await _read(CONDUCTOR_A)
    row = next(r for r in body["items"] if r["item_id"] == ids["item_a"])
    assert row["orphaned"] is True  # the mocked slot table reports nothing open
    # NOT stale: just created, so it is inside the window even with no worker
    # running. The window exists to cover the gap between bind and the first
    # report, not to flag an item the moment it is dispatched.
    assert row["stale"] is False
    stored = json.loads(wl.item_path(CONDUCTOR_A, ids["item_a"]).read_text())
    assert "orphaned" not in stored
    assert "stale" not in stored


@pytest.mark.asyncio
async def test_an_item_past_the_window_with_no_running_worker_is_stale():
    """Both conditions, and only both: the conjunction is what the flag means."""
    ids = await two_by_two()
    item = wl.read_work_item(CONDUCTOR_A, ids["item_a"])
    assert item is not None
    aged = datetime.now().astimezone() - timedelta(seconds=wl.DEFAULT_STALE_WINDOW_SECS + 60)
    assert wl.is_stale(item, worker_running=False, now=aged + timedelta(days=1)) is True
    assert wl.is_stale(item, worker_running=True, now=aged + timedelta(days=1)) is False


@pytest.mark.asyncio
async def test_an_idle_open_worker_past_the_window_is_stale():
    """ "Running" means a turn in flight, not an open tab. A worker whose session
    is still open but which stopped without reporting is the case the flag exists
    for, so slot EXISTENCE must not satisfy it."""
    ids = await two_by_two()
    item = wl.read_work_item(CONDUCTOR_A, ids["item_a"])
    assert item is not None
    aged = datetime.now().astimezone() + timedelta(seconds=wl.DEFAULT_STALE_WINDOW_SECS + 60)
    request = _req("GET", "/api/work-ledger", sk=CONDUCTOR_A)
    # The worker slot exists (dispatched by the fixture) but has no turn running.
    assert routes._slot_open(request.app["state"], WORKER_A) is True
    assert routes._slot_running(request.app["state"], WORKER_A) is False
    assert wl.is_stale(item, worker_running=False, now=aged) is True
    # And with a turn in flight the same item is not stale.
    _dispatched(WORKER_A, CONDUCTOR_A, running=True)
    assert routes._slot_running(request.app["state"], WORKER_A) is True
    assert wl.is_stale(item, worker_running=True, now=aged) is False


@pytest.mark.asyncio
async def test_a_running_worker_is_never_stale(monkeypatch):
    """The conjunction is the point: silence alone does not flag an item."""
    ids = await two_by_two()
    monkeypatch.setattr(routes, "_slot_open", lambda state, key: bool(key))
    monkeypatch.setattr(routes, "_slot_running", lambda state, key: bool(key))
    _, body = await _read(CONDUCTOR_A)
    row = next(r for r in body["items"] if r["item_id"] == ids["item_a"])
    assert row["stale"] is False
    assert row["orphaned"] is False


def test_the_schema_caps_restate_the_stores_own():
    """``validation`` cannot import the store on the gateway's request path, so the
    two spell the same numbers — and drifting apart would let a value through the
    schema that the store then refuses with a different code."""
    report = validation.WORK_REPORT_SCHEMA
    summary = next(f for f in report.fields if f.name == "summary")
    assert summary.max_len == wl.MAX_SUMMARY_CHARS
    pr = next(f for f in report.fields if f.name == "pr")
    assert (pr.min_val, pr.max_val) == (wl.MIN_PR, wl.MAX_PR)
    status = next(f for f in report.fields if f.name == "status")
    assert status.allowed == wl.WORKER_STATUSES

    record = validation.WORK_LEDGER_RECORD_SCHEMA
    assert next(f for f in record.fields if f.name == "title").max_len == wl.MAX_TITLE_CHARS
    assert next(f for f in record.fields if f.name == "goal").max_len == wl.MAX_GOAL_CHARS
    assert next(f for f in record.fields if f.name == "decision").max_len == wl.MAX_DECISION_CHARS
    assert next(f for f in record.fields if f.name == "verdict").allowed == wl.VERDICTS
    assert next(f for f in record.fields if f.name == "state").allowed <= wl.ITEM_STATES


def test_the_worker_schema_has_no_conductor_field():
    """The absence IS the guarantee — stronger than an allowlist kept correct by hand."""
    names = {f.name for f in validation.WORK_REPORT_SCHEMA.fields}
    assert names == {"status", "summary", "artifacts", "pr"}
    for forbidden in (
        "item_id",
        "session",
        "session_key",
        "acceptance",
        "verdict",
        "state",
        "decision",
        "title",
        "goal",
        "round",
        "fails",
        "worker_session_key",
    ):
        assert forbidden not in names, forbidden


@pytest.mark.asyncio
async def test_an_oversized_artifacts_map_is_refused_not_truncated():
    ids = await two_by_two()
    status, body = await _report(
        WORKER_A,
        {
            "status": "progress",
            "summary": "x",
            "artifacts": {f"k{n}": "v" for n in range(wl.MAX_ARTIFACT_KEYS + 1)},
        },
    )
    assert status == 400
    assert body["code"] == wl.CODE_FIELD_TOO_LONG
    item = wl.read_work_item(CONDUCTOR_A, ids["item_a"])
    assert item is not None and item.artifacts == {}


# ── the three refusals this layer owns, above the store ──────────────────


@pytest.mark.asyncio
async def test_bind_refuses_a_worker_this_conductor_did_not_create():
    """Otherwise conductor A binds B's idle worker, and B's worker then reads A's
    brief and A's ``decision`` — the one field a worker treats as an instruction.
    The store cannot see this: it checks only that the item is unbound."""
    ids = await two_by_two()
    victim = "chat-b-worker-2"
    _dispatched(victim, CONDUCTOR_B)  # created by the OTHER conductor
    status, body = await _record(
        CONDUCTOR_A,
        {"action": "create", "title": "hijack", "acceptance": {"kind": "human_approval"}},
    )
    assert status == 200
    item_id = body["item"]["item_id"]
    status, body = await _record(
        CONDUCTOR_A,
        {"action": "bind", "item_id": item_id, "worker_session_key": victim},
    )
    assert status == 403
    assert body["code"] == "worker_not_owned"
    # And the victim is still bound to nothing of A's.
    assert wl.read_binding(victim) is None
    item = wl.read_work_item(CONDUCTOR_A, item_id)
    assert item is not None and item.worker_session_key is None
    assert ids  # the two-by-two fixture stands untouched


@pytest.mark.asyncio
async def test_bind_refuses_a_session_that_is_not_open():
    ids = await two_by_two()
    status, body = await _record(
        CONDUCTOR_A,
        {"action": "create", "title": "no session", "acceptance": {"kind": "human_approval"}},
    )
    assert status == 200
    status, body = await _record(
        CONDUCTOR_A,
        {
            "action": "bind",
            "item_id": body["item"]["item_id"],
            "worker_session_key": "chat-never-created",
        },
    )
    assert status == 404
    assert body["code"] == "unknown_worker_session"
    assert ids


@pytest.mark.asyncio
async def test_bind_refuses_across_a_workspace_boundary():
    """Workspace is the memory boundary ``authorize_target`` already refuses across."""
    ids = await two_by_two()
    _SLOTS[CONDUCTOR_A] = _Slot(workspace="alpha")
    _dispatched("chat-a-worker-elsewhere", CONDUCTOR_A, workspace="beta")
    status, body = await _record(
        CONDUCTOR_A,
        {"action": "create", "title": "elsewhere", "acceptance": {"kind": "human_approval"}},
    )
    assert status == 200
    status, body = await _record(
        CONDUCTOR_A,
        {
            "action": "bind",
            "item_id": body["item"]["item_id"],
            "worker_session_key": "chat-a-worker-elsewhere",
        },
    )
    assert status == 403
    assert body["code"] == "worker_cross_workspace"
    assert ids


@pytest.mark.asyncio
async def test_a_channel_session_is_refused_on_every_tool():
    """The channel-agent block in ``channel.py`` matches a rendered PERMISSION
    REQUEST, and the four tools are auto-approved on three specs — so an
    ``allowedTools`` grant emits no permission event and that block never runs.
    Containment therefore has to hold here, where no spec can route around it."""
    for sk in ("slack:C123:456.789", "discord:guild:chan:1", "telegram:99:1"):
        status, body = await _brief(sk)
        assert status == 403, sk
        assert body["code"] == "channel_session", sk
        assert (await _report(sk, {"status": "done", "summary": "x"}))[0] == 403
        assert (await _read(sk))[0] == 403
        assert (await _record(sk, {"action": "goal", "goal": "g", "round": 1}))[0] == 403


@pytest.mark.asyncio
async def test_an_outbound_mirrored_session_is_refused_like_a_channel_one(monkeypatch):
    """A dashboard-BORN session given an outbound mirror republishes every turn to a
    channel, so its key looks local while the disclosure is identical. The key alone
    is therefore not the test."""
    ids = await two_by_two()
    monkeypatch.setattr(routes, "_reaches_a_channel", lambda request, sk: True)
    for status, body in (
        await _brief(WORKER_A),
        await _report(WORKER_A, {"status": "done", "summary": "x"}),
        await _read(CONDUCTOR_A),
        await _record(CONDUCTOR_A, {"action": "goal", "goal": "g", "round": 1}),
    ):
        assert status == 403
        assert body["code"] == "channel_session"
    assert ids


def test_the_mirror_probe_fails_closed_on_an_unreadable_store(monkeypatch):
    """An unreadable link counts as mirrored rather than opening the boundary —
    the same direction ``session_control._has_channel_mirror`` defaults to."""
    from kiro_crew.dashboard import session_control

    _SLOTS["chat-probe"] = _Slot()

    def _boom(*a: Any, **k: Any) -> bool:
        raise RuntimeError("session store unreadable")

    monkeypatch.setattr(session_control, "_has_channel_mirror", _boom)
    request = _req("GET", "/api/work-ledger", sk="chat-probe")
    assert _REAL_REACHES_A_CHANNEL(request, "chat-probe") is True


@pytest.mark.asyncio
async def test_accept_refuses_to_clear_the_bar_when_acceptance_is_omitted():
    """Coercing a missing acceptance to ``{}`` replaced the real condition, dropped
    the item out of ``accept_batch`` (which filters on a non-empty acceptance), and
    answered 200 — conductor-owned state lost with no signal. A promotion never
    legitimately clears a bar."""
    ids = await two_by_two()
    before = wl.read_work_item(CONDUCTOR_A, ids["item_a"])
    assert before is not None and before.acceptance == {"kind": "human_approval"}

    for body in (
        {"action": "accept", "item_id": ids["item_a"]},
        {"action": "accept", "item_id": ids["item_a"], "acceptance": {}},
    ):
        status, payload = await _record(CONDUCTOR_A, body)
        assert status == 400, payload
        assert payload["code"] == wl.CODE_INVALID_VALUE
        assert payload["field"] == "acceptance"

    after = wl.read_work_item(CONDUCTOR_A, ids["item_a"])
    assert after is not None and after.acceptance == {"kind": "human_approval"}
    # And it is still in the batch the evaluator reads.
    _, read = await _read(CONDUCTOR_A)
    assert any(e["id"] == ids["item_a"] for e in read["accept_batch"]["items"])


@pytest.mark.asyncio
async def test_an_unreadable_parent_ledger_refuses_the_bootstrap_rather_than_resetting_depth():
    """``read_conductor`` answers None for an absent record AND a torn one. Treating
    that as 'no parent' would compute depth 1 for a child of a depth-1 parent and
    PERSIST it, granting a generation no later read corrects — a cap that fails open
    on unreadable input is not a cap."""
    ids = await two_by_two()
    assert (await _record(WORKER_A, {"action": "goal", "goal": "level 1", "round": 1}))[0] == 200
    grandchild = "chat-a-grandchild"
    _dispatched(grandchild, WORKER_A)
    status, body = await _record(
        WORKER_A, {"action": "create", "title": "leaf", "acceptance": {"kind": "human_approval"}}
    )
    assert status == 200
    assert (
        await _record(
            WORKER_A,
            {
                "action": "bind",
                "item_id": body["item"]["item_id"],
                "worker_session_key": grandchild,
            },
        )
    )[0] == 200

    # Truncate the parent's own record so it reads as absent.
    wl.conductor_dir(WORKER_A).joinpath("conductor.json").write_text("{ tor", encoding="utf-8")
    assert wl.read_conductor(WORKER_A) is None

    status, body = await _record(grandchild, {"action": "goal", "goal": "level 2", "round": 1})
    assert status == 409
    assert body["code"] == "parent_unreadable"
    # Nothing was persisted at the wrong depth.
    assert wl.read_conductor(grandchild) is None
    assert ids


@pytest.mark.asyncio
async def test_a_worker_session_is_bound_once_and_never_rebound():
    """The store permits replacing a binding whose item is terminal, so a session
    COULD be reused — but a report already in flight from the old work resolves its
    binding when it LANDS, and there is no unbind path to undo the overwrite. An idle
    check cannot close that: a turn can start during the awaits before the commit.
    So the route refuses reuse outright, which makes the race unrepresentable."""
    ids = await two_by_two()
    # Close A's item, so the STORE would now allow the binding to be replaced.
    status, _ = await _record(
        CONDUCTOR_A, {"action": "close", "item_id": ids["item_a"], "state": "accepted"}
    )
    assert status == 200
    status, body = await _record(
        CONDUCTOR_A,
        {"action": "create", "title": "second", "acceptance": {"kind": "human_approval"}},
    )
    assert status == 200
    second = body["item"]["item_id"]

    status, body = await _record(
        CONDUCTOR_A,
        {"action": "bind", "item_id": second, "worker_session_key": WORKER_A},
    )
    assert status == 409
    assert body["code"] == "worker_already_dispatched"
    item = wl.read_work_item(CONDUCTOR_A, second)
    assert item is not None and item.worker_session_key is None
    # The original binding is untouched, so no report can be misrouted.
    assert wl.read_binding(WORKER_A) == (CONDUCTOR_A, ids["item_a"])

    # A FRESH session for the new item is the supported path.
    _dispatched("chat-a-worker-2", CONDUCTOR_A)
    status, body = await _record(
        CONDUCTOR_A,
        {"action": "bind", "item_id": second, "worker_session_key": "chat-a-worker-2"},
    )
    assert status == 200, body


def test_the_binding_presence_check_fails_closed(monkeypatch):
    """An unreadable store must not admit a rebind: the alternative is deciding the
    one question the refusal exists for on a store that could not answer."""

    def _boom(_key: str):
        raise OSError("store unreadable")

    monkeypatch.setattr(routes.work_ledger, "binding_path", _boom)
    assert routes._has_binding("chat-whoever") is True


@pytest.mark.asyncio
async def test_a_read_refuses_when_a_mirror_appears_during_the_read(monkeypatch):
    """Containment decided on ENTRY says nothing about containment after an await.
    The same reason ``session_control`` applies ``_refuse_ineligible_creator``
    twice — once on entry, once immediately before it allocates."""
    ids = await two_by_two()
    calls = {"n": 0}

    def _mirror_appears_after_entry(request: Any, sk: str) -> bool:
        # False for the entry check, True for the post-read re-check.
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(routes, "_reaches_a_channel", _mirror_appears_after_entry)
    status, body = await _brief(WORKER_A)
    assert status == 403
    assert body["code"] == "channel_session"
    assert "while the brief was being read" in body["error"]

    calls["n"] = 0
    status, body = await _read(CONDUCTOR_A)
    assert status == 403
    assert body["code"] == "channel_session"
    assert "while the ledger was being read" in body["error"]
    assert ids

    # The fit is the LAST await on the read path, and the re-check sits after it: a
    # mirror that attaches while the payload is being fitted is still refused.
    fitted = {"done": False}
    real_fit = routes._fit_to_budget

    def _fit_then_mirror(payload: Any, budget: int, drop_order: Any) -> str:
        text = real_fit(payload, budget, drop_order)
        fitted["done"] = True
        return text

    monkeypatch.setattr(routes, "_fit_to_budget", _fit_then_mirror)
    monkeypatch.setattr(routes, "_reaches_a_channel", lambda request, sk: fitted["done"])
    status, body = await _read(CONDUCTOR_A)
    assert status == 403
    assert body["code"] == "channel_session"


def test_the_route_owned_codes_are_disjoint_from_the_stores():
    """``_CODE_STATUS`` is asserted exhaustive over the store's ``CODE_*``, so a
    route-only code must not be folded into it or that assertion goes hollow."""
    assert not (routes.ROUTE_CODES & set(routes._CODE_STATUS))
    assert routes.CODE_NOT_BOUND not in routes.ROUTE_CODES  # its own named constant
    for code in (
        "channel_session",
        "parent_unreadable",
        "unknown_worker_session",
        "worker_not_owned",
        "worker_cross_workspace",
        "worker_already_dispatched",
    ):
        assert code in routes.ROUTE_CODES, code


# ── the routes are reachable by the tools and by nothing else ─────────────


def test_the_store_directory_is_fenced_from_file_tools():
    """The routes are the only sanctioned path in. The worker agent carries the
    full default file toolset, so without both fences those auto-approved tools
    reach every conductor's records directly — and a corrupted record reads as
    ABSENT to the store, so the loss is silent. Same treatment as the sibling
    ``ledger/`` directory, for the same reason."""
    from kiro_crew import sandbox
    from kiro_crew.security import is_sensitive_path

    for prefix in (".kiro/crew", ".kirocrew"):
        assert is_sensitive_path(f"~/{prefix}/work-ledger") is True
        assert is_sensitive_path(f"~/{prefix}/work-ledger/bindings/x.json") is True
        assert is_sensitive_path(f"~/{prefix}/work-ledger/c-abc12345/items/it_0.json") is True
    assert "work-ledger" in sandbox._CREW_HIDDEN_LEAVES


@pytest.mark.asyncio
async def test_a_worker_edit_of_its_own_item_file_is_refused_and_the_store_still_writes():
    """The fence, driven end to end through the tool gate on the REAL paths.

    The worker agent is a full-capability agent with ``fs_write``, so the tool
    layer's writer-ownership rule (a worker owns ``status``/``summary``/
    ``artifacts``/``pr``, a conductor owns ``acceptance``/``verdict``/``state``/
    ``decision``) is only as strong as the file fence under it: a prompt-injected
    worker that could open ``items/it_*.json`` directly would forge
    ``state: accepted``, gut ``acceptance`` before the conductor's evaluator runs,
    or read a sibling item past ``read_work_brief``'s scoping, and the conductor
    would read the forgery as its own writing.

    Three things are pinned here that the spelling test above cannot: the
    ``KIROCREW_HOME`` override this suite runs under is re-anchored (the store
    lives OUTSIDE ``~/.kiro/crew``, and it is still fenced); the refusal is what
    ``hooks.on_tool_call`` answers for an edit AND for a read of the item, the
    binding file and the conductor record; and the store's own ``atomic_write``
    path — the gateway process, not the agent tool — is untouched by the fence, so
    the sanctioned writers keep working on the very file the tool was refused.

    Shell writes (``echo >``, ``python -c``) are deliberately NOT text-matched by
    the gate (see ``is_sensitive_bash_command``); the OS sandbox mask asserted
    above is the shell-side control, so this test covers the file-tool plane.
    """
    from kiro_crew.config.paths import data_home
    from kiro_crew.hooks import TOOL_DENY, HookManager, HooksConfig

    ids = await two_by_two()
    item_file = wl.item_path(CONDUCTOR_A, ids["item_a"])
    sibling_file = wl.item_path(CONDUCTOR_B, ids["item_b"])
    binding_file = wl.binding_path(WORKER_A)
    conductor_file = wl.conductor_dir(CONDUCTOR_A) / wl._CONDUCTOR_FILE
    for path in (item_file, sibling_file, binding_file, conductor_file):
        assert path.is_file(), path
    # The suite's override puts the store outside the default home, which is the
    # anchoring case a hand-written ``~/.kiro/crew/...`` spelling never exercises.
    assert Path.home() / ".kiro" / "crew" not in data_home().parents
    assert data_home() not in (Path.home() / ".kiro" / "crew").parents

    gate = HookManager(HooksConfig.from_dict({}))
    for kind, path in (
        ("edit", item_file),
        ("edit", binding_file),
        ("edit", conductor_file),
        ("read", item_file),
        ("read", sibling_file),
    ):
        decision = gate.on_tool_call(
            f"Editing {path.name}" if kind == "edit" else f"Reading {path.name}",
            session_key="cli_chat",
            tool_kind=kind,
            raw_params={"path": str(path)},
        )
        assert decision.action == TOOL_DENY, (kind, path, decision)
        assert "sensitive path" in (decision.reason or ""), (kind, path, decision)

    # The sanctioned writers reach the same file the tool was refused: the worker's
    # report through its route, and the conductor's decision through the store.
    before = item_file.read_bytes()
    status, _ = await _report(WORKER_A, {"status": "progress", "summary": "still moving"})
    assert status == 200
    wl.apply_conductor_action(CONDUCTOR_A, "decide", item_id=ids["item_a"], decision="carry on")
    after = item_file.read_bytes()
    assert after != before
    item = wl.read_work_item(CONDUCTOR_A, ids["item_a"])
    assert item is not None
    assert item.summary == "still moving"
    assert item.decision == "carry on"


def test_the_routes_are_on_the_strict_internal_allowlist():
    """The four tools authenticate with the internal secret; without this entry the
    call falls through to cookie auth and every one fails with 403 before the
    handler's own session recognition can run."""
    from kiro_crew.dashboard import server

    assert "/api/work-ledger" in server._STRICT_INTERNAL_API_PATHS


def test_every_route_is_registered_lazily_on_the_app():
    """Registered by path AND by method, since a GET-only registration of the two
    write routes would fail only at call time — and through the DEFERRED binder, so
    an opt-in subsystem's module is not imported on the gateway boot path.
    """
    import inspect

    from kiro_crew.dashboard import server

    src = inspect.getsource(server)
    for method, path, handler in (
        ("add_get", "/api/work-ledger", "api_work_ledger_get"),
        ("add_post", "/api/work-ledger/record", "api_work_ledger_record"),
        ("add_get", "/api/work-ledger/brief", "api_work_brief"),
        ("add_post", "/api/work-ledger/report", "api_work_report"),
    ):
        assert f'{method}("{path}", _deferred_work_ledger("{handler}"))' in src, (method, path)
        # And never eagerly, which is what the boot-path rule forbids.
        assert f'{method}("{path}", handlers.{handler})' not in src, (method, path)


def test_the_handler_package_does_not_import_the_subsystem_at_boot():
    """``handlers/__init__.py`` is reached from ``start_dashboard``, so an import
    there IS the boot path. The four handlers belong to an opt-in MCP server, and
    clause 5 of ``no-new-work-on-gateway-boot-path`` requires gating the IMPORT, not
    just the handler."""
    package_init = (
        Path(__file__).resolve().parents[1] / "src/kiro_crew/dashboard/handlers/__init__.py"
    )
    text = package_init.read_text(encoding="utf-8")
    assert "from kiro_crew.dashboard.handlers.work_ledger import" not in text
    for name in (
        "api_work_brief",
        "api_work_report",
        "api_work_ledger_get",
        "api_work_ledger_record",
    ):
        assert f"    {name},\n" not in text, name


def test_the_deferred_binder_resolves_each_handler():
    """The binder resolves by NAME, so a renamed handler would fail only at request
    time — this pins all four names against the module."""
    from kiro_crew.dashboard import server

    for name in (
        "api_work_brief",
        "api_work_report",
        "api_work_ledger_get",
        "api_work_ledger_record",
    ):
        bound = server._deferred_work_ledger(name)
        assert bound.__name__ == name
        assert getattr(routes, name, None) is not None, name


# ── the read's shape: store order kept, narrowing arguments, compact, budget ──
#
# The reader is a model whose runtime cuts a tool result at a fixed length. An
# argument-less read is what it always was — the whole board in the store's order,
# twenty events per item — and the tests below pin what was ADDED around it: the
# narrowing arguments, the compact row, and the budget trim that sacrifices the
# oldest-created items first so the reply is valid JSON that still holds the
# newest one instead of a torn document.


async def _read_with(sk: str, query: str) -> tuple[int, dict[str, Any]]:
    resp = await routes.api_work_ledger_get(_req("GET", f"/api/work-ledger?{query}", sk=sk))
    return resp.status, json.loads(resp.text)


async def _create_stamped(
    monkeypatch, conductor: str, title: str, stamp: str, *, first: bool = False
) -> str:
    """One item created at *stamp*, through the routes, with no worker bound.

    ``_now_iso`` is the store's one clock, so pinning it here dates the item's
    ``created_at`` exactly — the field the response order and ``since`` read.
    """
    monkeypatch.setattr(wl, "_now_iso", lambda: stamp)
    if first:
        status, body = await _record(conductor, {"action": "goal", "goal": "g", "round": 1})
        assert status == 200, body
    status, body = await _record(
        conductor, {"action": "create", "title": title, "acceptance": {"kind": "human_approval"}}
    )
    assert status == 200, body
    return body["item"]["item_id"]


async def _three_dated_items(monkeypatch) -> list[str]:
    """Three items a day apart, returned OLDEST first — the store's own order."""
    ids = []
    for day, first in ((1, True), (2, False), (3, False)):
        ids.append(
            await _create_stamped(
                monkeypatch,
                CONDUCTOR_A,
                f"item {day}",
                f"2026-03-0{day}T10:00:00+00:00",
                first=first,
            )
        )
    return ids


async def _decide_n_times(conductor: str, item_id: str, count: int) -> None:
    """*count* distinct decision events on one item (event ids are content-addressed,
    so the texts must differ for the lines not to collapse)."""
    for n in range(count):
        status, body = await _record(
            conductor, {"action": "decide", "item_id": item_id, "decision": f"decision {n}"}
        )
        assert status == 200, body


@pytest.mark.asyncio
async def test_read_keeps_the_stores_order_for_items_and_the_batch(monkeypatch):
    """An argument-less read is unchanged: rows and ``accept_batch`` come in the
    store's own order, oldest first. Only the budget trim ranks by age."""
    ids = await _three_dated_items(monkeypatch)
    _, body = await _read(CONDUCTOR_A)
    assert [r["item_id"] for r in body["items"]] == ids
    assert [e["id"] for e in body["accept_batch"]["items"]] == ids
    assert [it.item_id for it in wl.list_work_items(CONDUCTOR_A)] == ids


@pytest.mark.asyncio
async def test_the_trim_ranks_rows_by_the_stamp_as_time_across_a_dst_fall_back(monkeypatch):
    """``created_at`` is local time with an offset, and across a DST fall-back a
    later moment can read lexically earlier: 01:15-05:00 is forty-five minutes
    AFTER 01:30-04:00. The store lists the two as text, so the later moment comes
    FIRST there — and the trim must still sacrifice the earlier one, because "the
    newest item is always kept" is a promise about time."""
    older = await _create_stamped(
        monkeypatch, CONDUCTOR_A, "before the fall-back", "2026-11-01T01:30:00-04:00", first=True
    )
    newer = await _create_stamped(
        monkeypatch, CONDUCTOR_A, "after the fall-back", "2026-11-01T01:15:00-05:00"
    )
    assert "2026-11-01T01:30:00-04:00" > "2026-11-01T01:15:00-05:00", "text order is wrong here"
    # Wide tails on both, so emptying ONE is worth far more than the trim's own
    # markers cost and the budget below can sit well above them.
    for item_id in (older, newer):
        for n in range(12):
            status, body = await _record(
                CONDUCTOR_A,
                {"action": "decide", "item_id": item_id, "decision": f"{n} " + "x" * 1500},
            )
            assert status == 200, body
    _, full = await _read(CONDUCTOR_A)
    assert [r["item_id"] for r in full["items"]] == [newer, older], "the store's text order"
    # A budget one tail short of the whole: stage one must empty exactly one tail,
    # and by age it has to be the EARLIER moment's — not the row listed last.
    monkeypatch.setattr(routes, "_RESPONSE_BUDGET_CHARS", len(routes._dumps(full)) - 1000)
    resp = await routes.api_work_ledger_get(_req("GET", "/api/work-ledger", sk=CONDUCTOR_A))
    body = json.loads(resp.text)
    assert len(resp.text) <= routes._RESPONSE_BUDGET_CHARS
    assert body["truncated"] is True
    assert body["dropped_events_for"] == [older], "the earlier moment loses its tail first"
    rows = {r["item_id"]: r for r in body["items"]}
    assert rows[older]["events"] == [] and len(rows[newer]["events"]) == 13
    assert "omitted_items" not in body
    assert [e["id"] for e in body["accept_batch"]["items"]] == [newer, older], "untouched"


def test_age_key_never_raises_on_an_unreadable_stamp():
    """A stamp that will not parse takes the floor — the OLDEST rank, so the trim
    sacrifices that item first — rather than raising inside the read."""
    unreadable = wl.WorkItem(item_id="it_0000000b", created_at="not a stamp")
    blank = wl.WorkItem(item_id="it_0000000a", created_at="")
    real = wl.WorkItem(item_id="it_00000001", created_at="2000-01-01T00:00:00+00:00")
    keys = {it.item_id: routes._age_key(it) for it in (unreadable, blank, real)}
    assert keys["it_0000000b"][:2] == keys["it_0000000a"][:2] == (False, routes._STAMP_FLOOR)
    assert keys["it_00000001"][0] is True
    # The parse result leads: a REAL stamp at the calendar's start with a positive
    # offset is an earlier moment than the floor, yet still ranks after garbage.
    earliest = wl.WorkItem(item_id="it_00000003", created_at="0001-01-01T00:00:00+14:00")
    assert [it.item_id for it in sorted((earliest, unreadable), key=routes._age_key)] == [
        "it_0000000b",
        "it_00000003",
    ]
    ordered = sorted((real, unreadable, blank), key=routes._age_key)
    assert [it.item_id for it in ordered] == ["it_0000000a", "it_0000000b", "it_00000001"]


@pytest.mark.asyncio
async def test_read_default_tail_is_the_ceiling_and_events_narrows_or_drops_it(monkeypatch):
    """The default tail is unchanged — the 20 ceiling the Crew board reads too —
    and ``events=<n>`` can only shorten it."""
    ids = await _three_dated_items(monkeypatch)
    await _decide_n_times(CONDUCTOR_A, ids[2], 12)
    assert len(wl.read_events(CONDUCTOR_A, ids[2])) == 13, "create + 12 decisions on disk"

    _, body = await _read(CONDUCTOR_A)
    row = next(r for r in body["items"] if r["item_id"] == ids[2])
    assert len(row["events"]) == 13 <= routes._MAX_EVENT_TAIL == 20
    assert [e["text"] for e in row["events"]][1:] == [f"decision {n}" for n in range(12)]

    _, body = await _read_with(CONDUCTOR_A, "events=10")
    row = next(r for r in body["items"] if r["item_id"] == ids[2])
    assert len(row["events"]) == 10
    # The NEWEST ten, in log order.
    assert [e["text"] for e in row["events"]] == [f"decision {n}" for n in range(2, 12)]
    _, body = await _read_with(CONDUCTOR_A, "events=0")
    assert all(r["events"] == [] for r in body["items"])
    assert "events" in body["items"][0], "events=0 keeps the key, so the shape is stable"
    _, body = await _read_with(CONDUCTOR_A, f"events={routes._MAX_EVENT_TAIL}")
    row = next(r for r in body["items"] if r["item_id"] == ids[2])
    assert len(row["events"]) == 13


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,field",
    [
        ("events=21", "events"),
        ("events=-1", "events"),
        ("events=five", "events"),
        ("item_id=it_nothex", "item_id"),
        ("item_id=..%2Fetc", "item_id"),
        ("state=closed", "state"),
        ("since=yesterday", "since"),
        ("compact=maybe", "compact"),
        ("status=done", "status"),
        # An EMPTY value is refused for every parameter: the schema skips a pattern
        # on an empty string, so ``item_id=`` would otherwise select nothing and
        # answer 200 with no rows beside a whole-board batch.
        ("item_id=", "item_id"),
        ("state=", "state"),
        ("since=", "since"),
        ("events=", "events"),
        ("compact=", "compact"),
    ],
)
async def test_read_refuses_an_out_of_shape_query_and_names_the_field(monkeypatch, query, field):
    """The route validates the query with the tool's own schema, so a loopback
    caller that bypasses the tool layer meets the same bounds — and an unknown
    parameter is refused, never silently ignored into a whole-board answer."""
    await _three_dated_items(monkeypatch)
    status, body = await _read_with(CONDUCTOR_A, query)
    assert status == 400, body
    assert body["code"] == wl.CODE_INVALID_VALUE
    assert body["field"] == field


@pytest.mark.asyncio
async def test_read_filters_by_item_id(monkeypatch):
    ids = await _three_dated_items(monkeypatch)
    status, body = await _read_with(CONDUCTOR_A, f"item_id={ids[1]}")
    assert status == 200
    assert [r["item_id"] for r in body["items"]] == [ids[1]]
    # The batch is the WHOLE bar, whatever the rows were narrowed to.
    assert [e["id"] for e in body["accept_batch"]["items"]] == ids
    # Well-formed but not on this board: an empty answer, not an error.
    status, body = await _read_with(CONDUCTOR_A, "item_id=it_00000000")
    assert status == 200
    assert body["items"] == []


@pytest.mark.asyncio
async def test_read_filters_by_state(monkeypatch):
    ids = await _three_dated_items(monkeypatch)
    status, body = await _record(
        CONDUCTOR_A, {"action": "close", "item_id": ids[0], "state": "accepted", "decision": "ok"}
    )
    assert status == 200, body
    _, body = await _read_with(CONDUCTOR_A, "state=open")
    assert [r["item_id"] for r in body["items"]] == [ids[1], ids[2]]
    _, body = await _read_with(CONDUCTOR_A, "state=accepted")
    assert [r["item_id"] for r in body["items"]] == [ids[0]]
    _, body = await _read_with(CONDUCTOR_A, "state=rejected")
    assert body["items"] == []


@pytest.mark.asyncio
async def test_read_filters_by_since_on_created_reported_and_closed(monkeypatch):
    """``since`` reads the stamps another party writes: a create, a worker's
    report, a close. Inclusive at the stamp, so a caller can pass back the newest
    stamp it saw and still receive the item that carried it."""
    ids = await _three_dated_items(monkeypatch)
    _, body = await _read_with(CONDUCTOR_A, "since=2026-03-02T10:00:00%2B00:00")
    assert [r["item_id"] for r in body["items"]] == [ids[1], ids[2]]
    _, body = await _read_with(CONDUCTOR_A, "since=2026-03-04T00:00:00%2B00:00")
    assert body["items"] == []

    # A worker report on the OLDEST item moves it into a later window.
    _dispatched(WORKER_A, CONDUCTOR_A)
    status, body = await _record(
        CONDUCTOR_A, {"action": "bind", "item_id": ids[0], "worker_session_key": WORKER_A}
    )
    assert status == 200, body
    monkeypatch.setattr(wl, "_now_iso", lambda: "2026-03-05T10:00:00+00:00")
    status, body = await _report(WORKER_A, {"status": "progress", "summary": "moving"})
    assert status == 200, body
    _, body = await _read_with(CONDUCTOR_A, "since=2026-03-05T00:00:00%2B00:00")
    assert [r["item_id"] for r in body["items"]] == [ids[0]]

    # And so does a close, on the middle one.
    monkeypatch.setattr(wl, "_now_iso", lambda: "2026-03-06T10:00:00+00:00")
    status, body = await _record(
        CONDUCTOR_A, {"action": "close", "item_id": ids[1], "state": "abandoned", "decision": "x"}
    )
    assert status == 200, body
    _, body = await _read_with(CONDUCTOR_A, "since=2026-03-06T00:00:00%2B00:00")
    assert [r["item_id"] for r in body["items"]] == [ids[1]]
    # A stamp with no offset is read as local time, like the store's own reader.
    _, body = await _read_with(CONDUCTOR_A, "since=2000-01-01T00:00:00")
    assert len(body["items"]) == 3


@pytest.mark.asyncio
async def test_read_filters_compose_and_leave_the_batch_whole(monkeypatch):
    ids = await _three_dated_items(monkeypatch)
    status, body = await _record(
        CONDUCTOR_A, {"action": "close", "item_id": ids[2], "state": "rejected", "decision": "no"}
    )
    assert status == 200, body
    _, whole = await _read(CONDUCTOR_A)
    _, body = await _read_with(CONDUCTOR_A, "state=open&since=2026-03-02T00:00:00%2B00:00&events=1")
    assert [r["item_id"] for r in body["items"]] == [ids[1]]
    assert len(body["items"][0]["events"]) == 1
    # The batch is the bar's own document (every OPEN item with a concrete
    # acceptance) and the row filters do not reach it.
    assert body["accept_batch"] == whole["accept_batch"]
    assert {e["id"] for e in body["accept_batch"]["items"]} == {ids[0], ids[1]}


@pytest.mark.asyncio
async def test_read_compact_mode_has_no_events_acceptance_or_batch(monkeypatch):
    """The patrol read: the columns that decide who moves next, and no document
    that is large in its own right. Still the store's order, still filterable."""
    ids = await _three_dated_items(monkeypatch)
    await _decide_n_times(CONDUCTOR_A, ids[2], 3)
    status, body = await _read_with(CONDUCTOR_A, "compact=true")
    assert status == 200, body
    assert body["compact"] is True
    assert "accept_batch" not in body
    assert [r["item_id"] for r in body["items"]] == ids
    for row in body["items"]:
        assert set(row) == set(routes._COMPACT_ROW_FIELDS)
        assert "events" not in row and "acceptance" not in row and "artifacts" not in row
        # The derived flags ride along: a patrol read must still see a dead worker.
        assert {"orphaned", "stale", "acceptance_concrete"} <= set(row)
        assert row["acceptance_concrete"] is True
    newest = body["items"][-1]
    assert newest["decision"] == "decision 2"
    assert newest["state"] == "open"
    # Composes with the filters, and the full read carries no ``compact`` key.
    _, body = await _read_with(CONDUCTOR_A, f"compact=1&item_id={ids[0]}")
    assert [r["item_id"] for r in body["items"]] == [ids[0]]
    _, body = await _read_with(CONDUCTOR_A, "compact=false")
    assert "compact" not in body
    assert "accept_batch" in body
    assert "events" in body["items"][0]


@pytest.mark.asyncio
async def test_read_under_budget_carries_no_truncated_marker(monkeypatch):
    ids = await _three_dated_items(monkeypatch)
    await _decide_n_times(CONDUCTOR_A, ids[0], 4)
    _, body = await _read(CONDUCTOR_A)
    for key in ("truncated", "truncation_hint", "dropped_events_for", "omitted_items"):
        assert key not in body, key
    assert all(r["events"] for r in body["items"])


@pytest.mark.asyncio
async def test_read_over_budget_drops_oldest_tails_first_and_marks_truncated(monkeypatch):
    """The wired-in budget: with the ceiling lowered to what the newest row and
    the batch need, the OLDEST items lose their tails and the newest keeps its
    events — and the reply is still one valid JSON document that says what went."""
    ids = await _three_dated_items(monkeypatch)
    for item_id in ids:
        await _decide_n_times(CONDUCTOR_A, item_id, 5)
    _, full = await _read(CONDUCTOR_A)
    full_text = routes._dumps(full)
    newest_events = routes._dumps(full["items"][-1]["events"])
    # A budget the newest row's tail fits in and the second one's does not.
    monkeypatch.setattr(routes, "_RESPONSE_BUDGET_CHARS", len(full_text) - len(newest_events) - 200)
    resp = await routes.api_work_ledger_get(_req("GET", "/api/work-ledger", sk=CONDUCTOR_A))
    assert resp.status == 200
    body = json.loads(resp.text)  # valid JSON, or this raises
    assert body["truncated"] is True
    assert "budget" in body["truncation_hint"]
    assert "omitted_items" not in body, "stage two was not needed"
    rows = {r["item_id"]: r for r in body["items"]}
    assert len(rows) == 3
    assert rows[ids[2]]["events"], "the newest item keeps its tail"
    assert rows[ids[0]]["events"] == [], "the oldest loses its tail first"
    assert body["dropped_events_for"][0] == ids[0]
    assert set(body["dropped_events_for"]) <= {ids[0], ids[1]}
    assert len(resp.text) <= routes._RESPONSE_BUDGET_CHARS
    # The batch is untouched by the trim.
    assert body["accept_batch"] == full["accept_batch"]


def test_fit_to_budget_drops_tails_then_rows_by_age_and_never_the_newest():
    """The pure trim, on a synthetic board wide enough to need both stages. The
    rows are handed over SCRAMBLED, because what goes is decided by the age order
    the caller supplies, never by where a row sits in the response."""

    def _row(n: int, *, events: int = 20) -> dict[str, Any]:
        return {
            "item_id": f"it_{n:08x}",
            "title": f"item {n}",
            # Bulk outside the tail, so emptying every tail still leaves the
            # board over a 20,000-char budget and stage two has to run.
            "summary": "s" * 3000,
            "events": [{"id": f"e{n}-{k}", "text": "x" * 500} for k in range(events)],
        }

    positions = [3, 8, 1, 5, 2, 7, 4, 6]
    rows = [_row(n) for n in positions]
    oldest_first = [f"it_{n:08x}" for n in range(1, 9)]
    payload: dict[str, Any] = {"conductor": {"goal": "g"}, "items": rows}
    text = routes._fit_to_budget(payload, 20_000, oldest_first)
    body = json.loads(text)
    assert len(text) <= 20_000
    assert body["truncated"] is True
    kept = [r["item_id"] for r in body["items"]]
    assert "it_00000008" in kept, "the newest row survives"
    # Stage one emptied tails from the oldest up; stage two dropped the oldest rows.
    assert body["dropped_events_for"][0] == "it_00000001"
    assert body["omitted_items"][0] == "it_00000001"
    assert body["omitted_items"] == oldest_first[: len(body["omitted_items"])]
    assert set(body["omitted_items"]).isdisjoint(kept)
    # The survivors keep their RESPONSE positions relative to each other.
    scrambled_ids = [f"it_{n:08x}" for n in positions]
    assert kept == [item_id for item_id in scrambled_ids if item_id in kept]
    # Every survivor has had its tail emptied except, at most, the newest few.
    with_tails = {r["item_id"] for r in body["items"] if r["events"]}
    assert with_tails == set(sorted(kept)[len(kept) - len(with_tails) :])


def test_fit_to_budget_keeps_one_row_even_when_it_alone_is_over():
    """The last row is never dropped; what is too large IN it is elided instead,
    and the text still lands under the budget."""
    payload: dict[str, Any] = {
        "conductor": {},
        "items": [{"item_id": "it_0000000a", "summary": "y" * 5000, "events": []}],
    }
    text = routes._fit_to_budget(payload, 2000, ["it_0000000a"])
    body = json.loads(text)
    assert len(text) <= 2000
    assert [r["item_id"] for r in body["items"]] == ["it_0000000a"]
    assert body["truncated"] is True
    assert "omitted_items" not in body
    assert body["elided_fields"] == ["items[0].summary"]
    assert body["items"][0]["summary"] == {
        "elided": True,
        "chars": 5002,
        "reason": routes._FIELD_ELISION_REASON,
    }


def test_fit_to_budget_leaves_a_fitting_payload_unmarked():
    payload: dict[str, Any] = {"conductor": {}, "items": [{"item_id": "it_0000000a", "events": []}]}
    text = routes._fit_to_budget(dict(payload), 10_000, ["it_0000000a"])
    assert json.loads(text) == payload
    assert text == json.dumps(payload, indent=2, ensure_ascii=False)


# ── the last two stages: the batch, and a bar too large to show ────────────
#
# Every field the store writes is capped except ``acceptance`` (up to
# ``MAX_RECORD_BYTES // 2``), and that one document stands twice in a full read:
# on its row and in ``accept_batch``. Dropping rows down to one therefore does not
# bound the reply — one row plus its own batch entry can still be two copies of a
# document larger than the budget, and larger than the runtime's own cut. These
# pin the two stages that close that: the batch is trimmed oldest-created first,
# and a lone bar that still does not fit is replaced by a marker.

#: The store's own ceiling on one acceptance, in bytes of compact JSON.
_ACCEPTANCE_CAP = wl.MAX_RECORD_BYTES // 2


def _capped_row(n: int, acceptance: dict[str, Any]) -> dict[str, Any]:
    """A row at EVERY store cap but the acceptance: title 200, summary 500,
    decision 2000, 16 artifacts of 64 + 512, so the fit it proves is the worst case."""
    return {
        "item_id": f"it_{n:08x}",
        "title": "t" * wl.MAX_TITLE_CHARS,
        "summary": "s" * wl.MAX_SUMMARY_CHARS,
        "decision": "d" * wl.MAX_DECISION_CHARS,
        "artifacts": {
            f"{i:02d}" + "k" * (wl.MAX_ARTIFACT_KEY_CHARS - 2): "v" * wl.MAX_ARTIFACT_VALUE_CHARS
            for i in range(wl.MAX_ARTIFACT_KEYS)
        },
        "acceptance": acceptance,
        "events": [{"id": f"e{n}-{k}", "text": "x" * wl.MAX_EVENT_TEXT_CHARS} for k in range(5)],
    }


def _oldest_first(rows: list[dict[str, Any]]) -> list[str]:
    """The age order the route computes: here the ids ascend with age."""
    return sorted(r["item_id"] for r in rows)


def _capped_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "conductor": {"goal": "g" * wl.MAX_GOAL_CHARS},
        "items": rows,
        "accept_batch": {
            "items": [{"id": r["item_id"], "accept": r["acceptance"], "status": None} for r in rows]
        },
    }


def _widest_acceptance() -> dict[str, Any]:
    """An acceptance the store still accepts: just under its half-record cap, and
    concrete, so ``accept_batch`` carries it."""
    filler = _ACCEPTANCE_CAP - len(json.dumps({"kind": "file", "path": "/x", "note": ""}))
    acceptance = {"kind": "file", "path": "/x", "note": "n" * filler}
    assert len(json.dumps(acceptance).encode()) <= _ACCEPTANCE_CAP
    assert wl.is_acceptance_concrete(acceptance)
    return acceptance


def _elided(chars: int) -> dict[str, Any]:
    return {"elided": True, "chars": chars, "reason": routes._ELISION_REASON}


def test_fit_to_budget_trims_the_batch_oldest_first_after_the_rows():
    """Thirty-two open items with ~3 KB bars — the widest board the store allows,
    with bars an order of magnitude past the documented shape. Rows go down to one,
    then the batch loses its OLDEST entries until the text fits; no bar is elided."""
    bar = {"kind": "file", "path": "/p" + "q" * 2950, "exists": True}
    rows = [_capped_row(n, dict(bar)) for n in range(1, wl.MAX_ITEMS_PER_CONDUCTOR + 1)]
    payload = _capped_payload(rows)
    text = routes._fit_to_budget(payload, routes._RESPONSE_BUDGET_CHARS, _oldest_first(rows))
    body = json.loads(text)
    assert len(text) <= routes._RESPONSE_BUDGET_CHARS
    assert body["truncated"] is True
    assert [r["item_id"] for r in body["items"]] == ["it_00000020"], "rows down to the newest"
    kept = [e["id"] for e in body["accept_batch"]["items"]]
    assert kept[-1] == "it_00000020" and len(kept) > 1, "the newest bar is never dropped"
    assert kept == sorted(kept), "the newest suffix, still in the store's order"
    assert body["omitted_accept_batch_for"][0] == "it_00000001", "the oldest bar went first"
    assert set(kept).isdisjoint(body["omitted_accept_batch_for"])
    assert len(kept) + len(body["omitted_accept_batch_for"]) == wl.MAX_ITEMS_PER_CONDUCTOR
    assert "elided_acceptance_for" not in body
    assert body["accept_batch"]["items"][0]["accept"] == bar, "a kept bar is intact"
    assert "accept_batch" in body["truncation_hint"] and "elided" in body["truncation_hint"]


def test_fit_to_budget_elides_a_lone_acceptance_too_large_to_show():
    """The newest item carries the largest acceptance the store accepts. Both of its
    copies are replaced by the marker, the marker says how large the bar was, the
    text is under the budget — and so under the runtime's cut — and every other
    field is at its cap, so this is the worst case the store can produce."""
    widest = _widest_acceptance()
    rows = [_capped_row(3, widest), _capped_row(2, {"kind": "human_approval"})]
    payload = _capped_payload(rows)
    text = routes._fit_to_budget(payload, routes._RESPONSE_BUDGET_CHARS, _oldest_first(rows))
    body = json.loads(text)
    assert len(text) <= routes._RESPONSE_BUDGET_CHARS < validation.MAX_RESPONSE_LEN
    assert body["omitted_items"] == ["it_00000002"]
    assert body["omitted_accept_batch_for"] == ["it_00000002"]
    assert body["elided_acceptance_for"] == ["it_00000003"]
    chars = len(json.dumps(widest, ensure_ascii=False))
    assert body["items"][0]["acceptance"] == _elided(chars)
    assert body["accept_batch"]["items"] == [
        {"id": "it_00000003", "accept": _elided(chars), "status": None}
    ]
    assert widest["note"].startswith("n"), "the caller's document itself is untouched"
    # A board of 32 such bars is no worse: the earlier stages leave one of each.
    rows = [_capped_row(n, dict(widest)) for n in range(1, wl.MAX_ITEMS_PER_CONDUCTOR + 1)]
    text = routes._fit_to_budget(
        _capped_payload(rows), routes._RESPONSE_BUDGET_CHARS, _oldest_first(rows)
    )
    body = json.loads(text)
    assert len(text) <= routes._RESPONSE_BUDGET_CHARS
    assert body["elided_acceptance_for"] == ["it_00000020"]
    assert len(body["omitted_accept_batch_for"]) == wl.MAX_ITEMS_PER_CONDUCTOR - 1


def test_fit_to_budget_elides_the_largest_copy_first_and_only_as_many_as_it_takes():
    """When the row and the batch's last entry are different items — a filter kept
    a small-barred row, the batch's newest open bar is the wide one — only the copy
    that does not fit goes; a bar that fits is shown. On a tie the row's copy goes
    first, so the evaluator keeps a bar it can run."""
    wide = _widest_acceptance()
    row = _capped_row(9, {"kind": "human_approval"})
    payload = _capped_payload([row])
    payload["accept_batch"] = {"items": [{"id": "it_00000008", "accept": wide, "status": None}]}
    body = json.loads(
        routes._fit_to_budget(
            payload, routes._RESPONSE_BUDGET_CHARS, ["it_00000008", "it_00000009"]
        )
    )
    assert body["items"][0]["acceptance"] == {"kind": "human_approval"}
    assert body["accept_batch"]["items"][0]["accept"]["elided"] is True
    assert body["elided_acceptance_for"] == ["it_00000008"]
    # Two copies of one bar that together overflow but alone fit: one marker.
    half = {"kind": "file", "path": "/h", "note": "h" * 30_000}
    body = json.loads(
        routes._fit_to_budget(_capped_payload([_capped_row(7, half)]), 50_000, ["it_00000007"])
    )
    assert body["items"][0]["acceptance"]["elided"] is True
    assert body["accept_batch"]["items"][0]["accept"] == half
    assert body["elided_acceptance_for"] == ["it_00000007"]


@pytest.mark.asyncio
async def test_read_with_the_widest_acceptance_is_valid_json_under_budget_and_evaluates_to_error(
    monkeypatch,
):
    """Through the route and the real store, at the wired-in budget: an item whose
    acceptance is the largest the store accepts comes back as one valid document
    under the budget, its bar elided on the row and in the batch — and the real
    ``accept_eval.py`` answers ``error`` for that entry, which is the right verdict
    for a bar that could not be read."""
    ids = await _three_dated_items(monkeypatch)
    # Past the route's own per-write cap (one crew-log line), so written the way an
    # item that predates the record would have been: straight into the store.
    monkeypatch.setattr(wl, "_now_iso", lambda: "2026-03-04T10:00:00+00:00")
    widest = wl.apply_conductor_action(
        CONDUCTOR_A, "create", title="wide bar", acceptance=_widest_acceptance()
    )["item"].item_id
    resp = await routes.api_work_ledger_get(_req("GET", "/api/work-ledger", sk=CONDUCTOR_A))
    assert resp.status == 200
    assert len(resp.text) <= routes._RESPONSE_BUDGET_CHARS < validation.MAX_RESPONSE_LEN
    body = json.loads(resp.text)  # valid JSON, or this raises
    assert body["truncated"] is True
    assert [r["item_id"] for r in body["items"]] == [widest]
    assert body["omitted_items"] == ids, "the three older rows, oldest popped first"
    assert body["items"][0]["acceptance"]["elided"] is True
    assert body["items"][0]["acceptance"]["chars"] > routes._RESPONSE_BUDGET_CHARS
    assert body["elided_acceptance_for"] == [widest]
    assert [e["id"] for e in body["accept_batch"]["items"]] == [widest]
    assert body["accept_batch"]["items"][0]["accept"]["elided"] is True
    stored = wl.list_work_items(CONDUCTOR_A)[-1]
    assert stored.item_id == widest and len(stored.acceptance["note"]) > 400_000, "store untouched"

    script = (
        Path(__file__).resolve().parents[1]
        / "src/kiro_crew/builtin_skills/goal-conductor/scripts/accept_eval.py"
    )
    proc = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(body["accept_batch"]).encode(),
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    results = {row["id"]: row["verdict"] for row in json.loads(proc.stdout.decode())["results"]}
    assert results == {widest: "error"}


@pytest.mark.asyncio
async def test_read_of_a_full_board_of_wide_bars_trims_the_batch_oldest_first(monkeypatch):
    """Thirty-two items through the route, each with a ~3 KB bar, at the wired-in
    budget: one valid document under the budget, the newest row kept, and the batch
    the newest suffix of the board with the OLDEST bars named as omitted."""
    ids: list[str] = []
    for n in range(wl.MAX_ITEMS_PER_CONDUCTOR):
        monkeypatch.setattr(wl, "_now_iso", lambda n=n: f"2026-03-01T10:{n:02d}:00+00:00")
        if n == 0:
            status, body = await _record(CONDUCTOR_A, {"action": "goal", "goal": "g", "round": 1})
            assert status == 200, body
        status, body = await _record(
            CONDUCTOR_A,
            {
                "action": "create",
                "title": f"item {n}",
                "acceptance": {"kind": "file", "path": "/p" + "q" * 2950, "exists": True},
            },
        )
        assert status == 200, body
        ids.append(body["item"]["item_id"])
    resp = await routes.api_work_ledger_get(_req("GET", "/api/work-ledger", sk=CONDUCTOR_A))
    assert resp.status == 200
    assert len(resp.text) <= routes._RESPONSE_BUDGET_CHARS
    body = json.loads(resp.text)
    assert body["truncated"] is True
    assert body["items"][0]["item_id"] == ids[-1], "the newest row survives"
    assert body["items"][0]["acceptance"]["path"].startswith("/pq"), "and its bar is shown"
    kept = [e["id"] for e in body["accept_batch"]["items"]]
    assert 1 < len(kept) < wl.MAX_ITEMS_PER_CONDUCTOR
    assert kept == ids[-len(kept) :], "the newest suffix of the board, in the store's order"
    assert body["omitted_accept_batch_for"][0] == ids[0], "the oldest bar went first"
    assert set(kept) | set(body["omitted_accept_batch_for"]) == set(ids)
    assert "elided_acceptance_for" not in body


# ── the fifth stage: whatever the writers did not cap ──────────────────────
#
# Every field a writer produces is capped, so the four stages above bound every
# record the store's own code can write. A record grown by hand under the reader's
# 1 MB ceiling still reads, and the header is touched by none of them — so the
# guarantee is made unconditional by one generic last stage, and then asserted.


def _fully_capped_row(n: int) -> dict[str, Any]:
    """A row as ``to_dict`` + the derived flags would render it with EVERY field at
    its store cap, including twenty events at the event-text cap."""
    row = _capped_row(n, {"kind": "pr_checks", "pr": 999_999_999, "repo": "o" * 100})
    row.update(
        {
            "schema": 1,
            "state": "open",
            "verdict": "pending",
            "worker_session_key": "w" * 512,
            "round": 999_999,
            "fails": 999,
            "status": "progress",
            "pr": 999_999_999,
            "last_report_at": "2026-09-30T10:00:00+00:00",
            "created_at": "2026-09-30T10:00:00+00:00",
            "closed_at": None,
            "recorded_at": "2026-09-30T10:00:00+00:00",
            "orphaned": False,
            "stale": False,
            "acceptance_concrete": True,
            "events": [
                {
                    "id": "e" * 16,
                    "kind": "decision",
                    "text": "x" * wl.MAX_EVENT_TEXT_CHARS,
                    "ts": "2026-09-30T10:00:00+00:00",
                    "by": "c" * 64,
                }
                for _ in range(routes._MAX_EVENT_TAIL)
            ],
        }
    )
    return row


def _capped_conductor() -> dict[str, Any]:
    return {
        "goal": "g" * wl.MAX_GOAL_CHARS,
        "round": 999_999,
        "slot_key": "k" * 128,
        "depth": wl.MAX_DEPTH,
        "created_total": wl.MAX_STORED_ITEMS_PER_CONDUCTOR,
        "created_at": "2026-09-30T10:00:00+00:00",
        "recorded_at": "2026-09-30T10:00:00+00:00",
    }


def test_fit_to_budget_elides_an_oversized_header_field_last_and_names_its_path():
    """A hand-grown ``goal`` of 200,000 chars is nothing the first four stages
    touch. The fifth replaces it — the largest string leaf left — with the marker,
    names it by path, and the newest item is still there."""
    rows = [_capped_row(n, {"kind": "human_approval"}) for n in (1, 2, 3)]
    payload = _capped_payload(rows)
    payload["conductor"] = {**_capped_conductor(), "goal": "g" * 200_000}
    text = routes._fit_to_budget(payload, routes._RESPONSE_BUDGET_CHARS, _oldest_first(rows))
    body = json.loads(text)
    assert len(text) <= routes._RESPONSE_BUDGET_CHARS
    assert body["elided_fields"] == ["conductor.goal"]
    assert body["conductor"]["goal"] == {
        "elided": True,
        "chars": 200_002,
        "reason": routes._FIELD_ELISION_REASON,
    }
    assert body["conductor"]["round"] == 999_999, "the rest of the header is intact"
    assert [r["item_id"] for r in body["items"]] == ["it_00000003"], "the newest item"
    assert body["items"][0]["acceptance"] == {"kind": "human_approval"}
    assert "elided_acceptance_for" not in body, "a bar smaller than its marker is never elided"
    assert "elided_fields" in body["truncation_hint"]
    # The walk skips its own markers and keys: a payload of nothing but markers
    # nominates nothing, so it stops rather than loops.
    assert routes._string_leaves({"truncation_hint": "h" * 999, "conductor": {}}) == []
    nested = {"a": [{"elided": True, "reason": "r" * 999}, {"bb": "leaf"}]}
    leaves = routes._string_leaves(nested)
    assert [(kind, routes._render_path(chain, {})) for kind, _c, _k, chain in leaves] == [
        ("value", "a[1].bb"),
        ("key", "a[1]"),
    ]
    # A nested KEY competes; a top-level one (the reply's own schema) never does.
    # Largest first: the 40-char key outranks its one-char value.
    keyed = {"conductor": {"k" * 40: "v"}}
    leaves = routes._string_leaves(keyed)
    assert [(kind, key) for kind, _c, key, _chain in leaves] == [
        ("key", "k" * 40),
        ("value", "k" * 40),
    ]
    assert routes._render_path(leaves[0][3], {}) == "conductor"
    assert routes._render_path(leaves[1][3], {}) == "conductor." + "k" * 40
    # A path under a renamed key names the marker key, not the bulk that is gone.
    holder = keyed["conductor"]
    assert (
        routes._render_path(leaves[1][3], {(id(holder), "k" * 40): "[elided key: 40 chars]"})
        == "conductor.[elided key: 40 chars]"
    )


@pytest.mark.asyncio
async def test_read_with_a_hand_grown_goal_is_valid_json_under_budget(monkeypatch):
    """Through the route: a ``conductor.json`` rewritten on disk with a 200,000-char
    goal still reads (the reader's ceiling is 1 MB), and the reply is one valid
    document under the budget with the goal elided by path and every item present."""
    ids = await _three_dated_items(monkeypatch)
    record_path = wl.conductor_dir(CONDUCTOR_A) / "conductor.json"
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    stored["goal"] = "g" * 200_000
    record_path.write_text(json.dumps(stored), encoding="utf-8")
    assert wl.MAX_GOAL_CHARS < 200_000 < wl.MAX_RECORD_BYTES
    resp = await routes.api_work_ledger_get(_req("GET", "/api/work-ledger", sk=CONDUCTOR_A))
    assert resp.status == 200
    assert len(resp.text) <= routes._RESPONSE_BUDGET_CHARS
    body = json.loads(resp.text)  # valid JSON, or this raises
    assert body["truncated"] is True
    assert body["elided_fields"] == ["conductor.goal"]
    assert body["conductor"]["goal"]["elided"] is True
    assert ids[-1] in [r["item_id"] for r in body["items"]], "the newest item is kept"
    assert ids[-1] in [e["id"] for e in body["accept_batch"]["items"]]


def test_fit_to_budget_leaves_a_board_at_every_store_cap_to_the_first_four_stages():
    """What the writers can produce never reaches the fifth stage. One item with
    every field at its cap and twenty capped events fits untouched — no marker at
    all — and the widest board the store allows is trimmed by the first stages
    alone, with no field elided by path."""
    one = _capped_payload([_fully_capped_row(1)])
    one["conductor"] = _capped_conductor()
    text = routes._fit_to_budget(one, routes._RESPONSE_BUDGET_CHARS, ["it_00000001"])
    assert len(text) <= routes._RESPONSE_BUDGET_CHARS
    assert not (set(json.loads(text)) & routes._TRIM_MARKER_KEYS), "nothing trimmed"

    rows = [_fully_capped_row(n) for n in range(1, wl.MAX_ITEMS_PER_CONDUCTOR + 1)]
    wide = _capped_payload(rows)
    wide["conductor"] = _capped_conductor()
    assert len(routes._dumps(wide)) > 10 * routes._RESPONSE_BUDGET_CHARS, "far over"
    text = routes._fit_to_budget(wide, routes._RESPONSE_BUDGET_CHARS, _oldest_first(rows))
    body = json.loads(text)
    assert len(text) <= routes._RESPONSE_BUDGET_CHARS
    assert body["truncated"] is True
    assert "elided_fields" not in body and "elided_acceptance_for" not in body
    newest = body["items"][-1]
    assert newest["item_id"] == "it_00000020", "the newest row survives, intact"
    assert newest["decision"] == "d" * wl.MAX_DECISION_CHARS
    assert newest["acceptance"]["pr"] == 999_999_999


# ── the budget is the DELIVERED size, and the trim never gives up ──────────
#
# The MCP layer redacts the serialized reply before the model reads it, and a
# redaction can lengthen text: a 14-char credential-bearing link becomes a
# 35-char placeholder. A reply fitted on its raw length could leave the route
# under the budget and reach the runtime's cut anyway, so every stage measures
# the redacted text through the same shim. And a record can carry bulk no string
# elision reaches — a huge KEY, or numbers — so the last stage nominates keys and,
# failing everything, answers with an envelope rather than an assertion.

#: A link the redaction pass rewrites to a placeholder more than twice its size.
_SHORT_LINK = "//a.co?token=x "


def _links(count: int) -> str:
    return _SHORT_LINK * count


def test_fit_to_budget_measures_the_text_as_delivered_after_redaction():
    """A payload whose raw text fits but whose REDACTED text does not is trimmed
    until the redacted text fits; the delivered size is the measured size."""
    note = _links(500)  # ~7.5 KB raw, ~18 KB redacted; the bar stands twice
    acceptance = {"kind": "file", "path": "/x", "note": note}
    rows = [_capped_row(2, dict(acceptance)), _capped_row(1, {"kind": "human_approval"})]
    payload = _capped_payload(rows)
    raw = routes._dumps(payload)
    assert len(raw) <= routes._RESPONSE_BUDGET_CHARS < len(redact_via_context(raw)), "the case"
    assert redact_via_context(redact_via_context(raw)) == redact_via_context(raw), "idempotent"
    text = routes._fit_to_budget(payload, routes._RESPONSE_BUDGET_CHARS, _oldest_first(rows))
    body = json.loads(text)  # valid JSON, or this raises
    delivered = redact_via_context(json.dumps(body, indent=2, ensure_ascii=False))
    assert len(delivered) <= routes._RESPONSE_BUDGET_CHARS
    assert delivered == redact_via_context(text), "the tool layer's re-dump is byte-identical"
    assert body["truncated"] is True
    assert body["elided_acceptance_for"] == ["it_00000002"], "the link-heavy bar went"
    assert len(json.loads(delivered)["items"]) >= 1


def test_fit_to_budget_measures_the_text_as_delivered_after_nfc_normalization():
    """The tool layer's response sanitizer NFC-normalizes the reply before the model
    reads it and cuts it at ``MAX_RESPONSE_LEN``. A character from the Unicode
    composition exclusions decomposes under NFC — U+FB2C becomes three code points
    — so a goal of 30,000 of them is 30 KB serialized and 90 KB delivered, and
    beside one capped row the reply is under this budget raw and past the runtime's
    cut delivered: torn mid-document if only the raw length were measured. The
    delivered size is the measured size."""
    payload = _capped_payload([_capped_row(1, {"kind": "human_approval"})])
    payload["conductor"] = {**_capped_conductor(), "goal": "\ufb2c" * 30_000}
    raw = routes._dumps(payload)
    assert len(raw) <= routes._RESPONSE_BUDGET_CHARS < len(sanitize_string(raw)), "the case"
    assert validation.MAX_RESPONSE_LEN < len(sanitize_string(raw)), "past the runtime's cut"
    assert sanitize_string(sanitize_string(raw)) == sanitize_string(raw), "idempotent"
    text = routes._fit_to_budget(payload, routes._RESPONSE_BUDGET_CHARS, ["it_00000001"])
    body = json.loads(text)  # valid JSON, or this raises
    delivered = routes._delivered(json.dumps(body, indent=2, ensure_ascii=False))
    assert len(delivered) <= routes._RESPONSE_BUDGET_CHARS
    assert delivered == routes._delivered(text), "the tool layer's re-dump is byte-identical"
    assert json.loads(delivered)["items"][0]["item_id"] == "it_00000001", "the item arrives"
    assert body["truncated"] is True
    assert body["elided_fields"] == ["conductor.goal"]


def test_the_budget_stays_clear_of_the_tool_layers_cut():
    """The runtime truncates a tool response at ``MAX_RESPONSE_LEN`` — with a suffix,
    tearing a JSON document. A delivered reply under this budget never reaches it."""
    assert routes._RESPONSE_BUDGET_CHARS < validation.MAX_RESPONSE_LEN


@pytest.mark.asyncio
async def test_read_of_a_record_with_decomposing_characters_fits_as_delivered(monkeypatch):
    """Through the route: a ``conductor.json`` rewritten on disk with a goal of
    35,000 U+FB2C (readable: well under the 1 MB ceiling) answers one document
    whose DELIVERED form — redacted, then NFC-normalized as the tool layer frames
    it — is under the budget and still valid JSON with the newest item present."""
    ids = await _three_dated_items(monkeypatch)
    record_path = wl.conductor_dir(CONDUCTOR_A) / "conductor.json"
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    stored["goal"] = "\ufb2c" * 35_000
    record_path.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")
    # Control: on the RAW length alone this board fits untrimmed, and the text the
    # tool layer would then deliver is past the runtime's own cut.
    with pytest.MonkeyPatch.context() as raw_only:
        raw_only.setattr(routes, "_fits", lambda text, budget: len(text) <= budget)
        control = await routes.api_work_ledger_get(_req("GET", "/api/work-ledger", sk=CONDUCTOR_A))
    assert "truncated" not in json.loads(control.text)
    assert len(control.text) <= routes._RESPONSE_BUDGET_CHARS
    assert len(routes._delivered(control.text)) > validation.MAX_RESPONSE_LEN
    # The real read measures what is delivered.
    resp = await routes.api_work_ledger_get(_req("GET", "/api/work-ledger", sk=CONDUCTOR_A))
    assert resp.status == 200
    delivered = routes._delivered(json.dumps(json.loads(resp.text), indent=2, ensure_ascii=False))
    assert len(delivered) <= routes._RESPONSE_BUDGET_CHARS
    body = json.loads(delivered)  # what the model receives is valid JSON
    assert body["truncated"] is True
    assert body["elided_fields"] == ["conductor.goal"]
    assert ids[-1] in [r["item_id"] for r in body["items"]], "the newest item is kept"


@pytest.mark.asyncio
async def test_read_with_link_heavy_bars_fits_after_redaction(monkeypatch):
    """Through the route and the real store: bars full of short credential-bearing
    links, whose redacted form is more than twice their size. The reply as the MCP
    layer delivers it — re-dumped and redacted — stays under the budget.

    The bars are written straight into the store: the record route redacts caller
    text on the way IN, so what can still grow on the way OUT is a record written
    outside it — by hand, or before that pass existed — and the read must hold for
    those too."""
    await _create_stamped(
        monkeypatch, CONDUCTOR_A, "first", "2026-03-01T10:00:00+00:00", first=True
    )
    for n in range(2):
        monkeypatch.setattr(wl, "_now_iso", lambda n=n: f"2026-03-02T10:0{n}:00+00:00")
        wl.apply_conductor_action(
            CONDUCTOR_A,
            "create",
            title=f"links {n}",
            acceptance={"kind": "file", "path": "/x", "note": _links(500)},
        )
    # Control: measured on the RAW length alone, this board fits untrimmed -- and
    # the text the MCP layer would deliver is then far over the budget.
    with pytest.MonkeyPatch.context() as raw_only:
        raw_only.setattr(routes, "_fits", lambda text, budget: len(text) <= budget)
        control = await routes.api_work_ledger_get(_req("GET", "/api/work-ledger", sk=CONDUCTOR_A))
    assert "truncated" not in json.loads(control.text)
    assert len(control.text) <= routes._RESPONSE_BUDGET_CHARS
    assert len(redact_via_context(control.text)) > routes._RESPONSE_BUDGET_CHARS
    # The real read measures what is delivered.
    resp = await routes.api_work_ledger_get(_req("GET", "/api/work-ledger", sk=CONDUCTOR_A))
    assert resp.status == 200
    delivered = redact_via_context(json.dumps(json.loads(resp.text), indent=2, ensure_ascii=False))
    assert len(delivered) <= routes._RESPONSE_BUDGET_CHARS
    body = json.loads(delivered)  # what the model receives is valid JSON
    assert body["truncated"] is True
    assert len(body["items"]) >= 1


def test_fit_to_budget_elides_an_oversized_key_and_keeps_its_value():
    """A hand-edited record can carry its bulk in a KEY. The fifth stage renames it
    to a marker key, keeps the value, and records the dict's path plus the marker."""
    rows = [_capped_row(1, {"kind": "human_approval"})]
    rows[0]["artifacts"] = {"K" * 60_000: "kept", "small": "also kept"}
    payload = _capped_payload(rows)
    text = routes._fit_to_budget(payload, routes._RESPONSE_BUDGET_CHARS, ["it_00000001"])
    body = json.loads(text)
    assert len(text) <= routes._RESPONSE_BUDGET_CHARS
    assert body["elided_fields"] == ["items[0].artifacts.[elided key: 60000 chars]"]
    assert body["items"][0]["artifacts"] == {
        "small": "also kept",
        "[elided key: 60000 chars]": "kept",
    }
    # Two oversized keys in one dict get distinct marker keys.
    rows = [_capped_row(1, {"kind": "human_approval"})]
    rows[0]["artifacts"] = {"A" * 60_000: "a", "B" * 60_000: "b"}
    body = json.loads(
        routes._fit_to_budget(_capped_payload(rows), routes._RESPONSE_BUDGET_CHARS, ["it_00000001"])
    )
    assert body["items"][0]["artifacts"] == {
        "[elided key: 60000 chars]": "a",
        "[elided key: 60000 chars #2]": "b",
    }


@pytest.mark.asyncio
async def test_read_of_an_item_with_an_oversized_key_answers_with_the_key_elided(monkeypatch):
    """The item file rewritten on disk with a 60,000-char artifact key — past the
    64-char cap both writers enforce, under the 1 MB reader ceiling. The read is
    200, valid, under budget, and names the elided key by path; the row is kept."""
    ids = await _three_dated_items(monkeypatch)
    path = wl.item_path(CONDUCTOR_A, ids[-1])
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["artifacts"] = {"K" * 60_000: "kept"}
    path.write_text(json.dumps(stored), encoding="utf-8")
    resp = await routes.api_work_ledger_get(_req("GET", "/api/work-ledger", sk=CONDUCTOR_A))
    assert resp.status == 200
    assert len(resp.text) <= routes._RESPONSE_BUDGET_CHARS
    body = json.loads(resp.text)
    assert body["truncated"] is True
    # The older rows went first (stages one to three), so the keyed row -- the
    # newest -- is the one left, and the key is renamed in place.
    assert [r["item_id"] for r in body["items"]] == [ids[-1]]
    assert body["items"][0]["artifacts"] == {"[elided key: 60000 chars]": "kept"}
    assert body["elided_fields"] == ["items[0].artifacts.[elided key: 60000 chars]"]
    assert "unfittable" not in body


def test_fit_to_budget_collapses_to_an_envelope_when_nothing_can_be_elided():
    """Bulk that is neither a string value nor a key — numbers — defeats every
    stage. The answer is still a valid reply under the budget: the conductor's key,
    no rows, an empty batch, the two markers, a hint, and every item id."""
    rows = [_capped_row(n, {"kind": "human_approval"}) for n in (1, 2)]
    rows[1]["bulk"] = list(range(10**8, 10**8 + 9000))  # ~90 KB of digits
    payload = _capped_payload(rows)
    payload["conductor"] = {**_capped_conductor(), "slot_key": "chat-a-conductor"}
    text = routes._fit_to_budget(payload, routes._RESPONSE_BUDGET_CHARS, _oldest_first(rows))
    body = json.loads(text)
    assert len(text) <= routes._RESPONSE_BUDGET_CHARS
    assert body == {
        "conductor": {"slot_key": "chat-a-conductor"},
        "items": [],
        "accept_batch": {"items": []},
        "truncated": True,
        "unfittable": True,
        "truncation_hint": body["truncation_hint"],
        "omitted_count": 2,
        "omitted_items": ["it_00000001", "it_00000002"],
    }
    assert "work_ledger_rebuild" in body["truncation_hint"]
    assert "neither a string value nor a key" in body["truncation_hint"]
    # The recovery it names must actually recover: a full read of ONE item still
    # carries the whole-board batch (and with it the sibling that cannot fit), so
    # the hint sends the reader to the compact read, which carries no batch.
    assert "compact=true&item_id=<id>" in body["truncation_hint"]
    assert "accept write" in body["truncation_hint"]
    assert "item_id=<id>" in body["truncation_hint"]
    # Compact mode has no batch, and the envelope does not invent one.
    compact = {"conductor": {}, "items": [{"item_id": "it_0000000a", "bulk": list(range(9000))}]}
    body = json.loads(routes._fit_to_budget(compact, 2000, ["it_0000000a"]))
    assert "accept_batch" not in body and body["unfittable"] is True
    assert body["omitted_items"] == ["it_0000000a"]


def test_fit_to_budget_measures_candidates_by_their_serialized_size():
    """A run of control characters is six JSON chars each. Measured by Python
    length, a 100-NUL artifact would look smaller than its marker and the trim would
    give up on a board it can fit; measured serialized, it goes like any other bulk."""
    rows = [_capped_row(1, {"kind": "human_approval"})]
    rows[0]["artifacts"] = {f"a{i:02d}": "\x00" * 100 for i in range(120)}  # ~72 KB serialized
    payload = _capped_payload(rows)
    text = routes._fit_to_budget(payload, routes._RESPONSE_BUDGET_CHARS, ["it_00000001"])
    body = json.loads(text)
    assert len(text) <= routes._RESPONSE_BUDGET_CHARS
    assert "unfittable" not in body, "the bulk was reachable after all"
    assert body["items"][0]["item_id"] == "it_00000001"
    # Largest first: the capped decision and goal (2,000 chars) go before the
    # 602-char NUL values, and those are then reached as values, not skipped as small.
    assert any(p.startswith("items[0].artifacts.a") for p in body["elided_fields"])
    assert any(
        v == {"elided": True, "chars": 602, "reason": routes._FIELD_ELISION_REASON}
        for v in body["items"][0]["artifacts"].values()
    )


def test_lone_surrogates_are_replaced_on_the_object_and_distinct_keys_stay_distinct():
    """``"\\ud800"`` is a JSON escape ``json.loads`` accepts, and a hand-edited record
    can carry it. Kept as a code point, no UTF-8 encoder — aiohttp's, the tool
    layer's stdout — takes it. The trim replaces it in every string and KEY before
    the first serialization; a key whose replacement collides with a neighbour is
    suffixed rather than merged, so the reader's ``json.loads`` drops nothing."""
    lone = json.loads('"\\ud800 tail"')
    assert len(lone) == 6 and lone.startswith("\ud800")
    payload = {
        "conductor": {"goal": lone},
        "items": [
            {
                "item_id": "it_00000001",
                "artifacts": {json.loads('"\\ud800"'): "a", "\ufffd": "b", "x": lone},
                "events": [json.loads('"\\udc00"')],
            }
        ],
    }
    text = routes._fit_to_budget(payload, 50_000, ["it_00000001"])
    text.encode("utf-8")  # encodable, or this raises
    body = json.loads(text)
    assert body["conductor"]["goal"] == "\ufffd tail"
    assert body["items"][0]["artifacts"] == {"\ufffd": "b", "x": "\ufffd tail", "\ufffd #2": "a"}
    assert body["items"][0]["events"] == ["\ufffd"]
    assert "truncated" not in body, "a replacement is not a trim"
    # Ordinary non-ASCII is untouched.
    assert (
        json.loads(routes._fit_to_budget({"g": "ünï ✓", "items": []}, 50_000, []))["g"] == "ünï ✓"
    )


@pytest.mark.asyncio
async def test_read_of_a_record_with_a_lone_surrogate_answers(monkeypatch):
    ids = await _three_dated_items(monkeypatch)
    path = wl.item_path(CONDUCTOR_A, ids[0])
    raw = path.read_text(encoding="utf-8").replace('"item 1"', '"item \\ud800 one"')
    path.write_text(raw, encoding="utf-8")
    assert wl.list_work_items(CONDUCTOR_A)[0].title.startswith("item \ud800"), "read back as-is"
    resp = await routes.api_work_ledger_get(_req("GET", "/api/work-ledger", sk=CONDUCTOR_A))
    assert resp.status == 200
    resp.text.encode("utf-8")
    body = json.loads(resp.text)
    assert body["items"][0]["title"] == "item \ufffd one"
    assert "truncated" not in body


def test_fit_to_budget_answers_with_the_envelope_when_the_record_nests_too_deep():
    """``json.loads`` accepts nesting the indented serializer cannot follow, so a
    ~2 KB record can raise RecursionError on the first dump. The ids are collected
    first and the reply is the envelope, never a failed read."""
    deep: list[Any] = []
    cursor = deep
    for _ in range(3000):
        nested: list[Any] = []
        cursor.append(nested)
        cursor = nested
    rows = [
        _capped_row(1, {"kind": "human_approval", "meta": deep}),
        _capped_row(2, {"kind": "human_approval"}),
    ]
    payload = _capped_payload(rows)
    payload["conductor"] = {"slot_key": "chat-a-conductor"}
    text = routes._fit_to_budget(payload, routes._RESPONSE_BUDGET_CHARS, _oldest_first(rows))
    body = json.loads(text)
    assert body["unfittable"] is True and body["items"] == []
    assert body["omitted_items"] == ["it_00000001", "it_00000002"]
    assert "nests deeper" in body["truncation_hint"]
    # A FILTERED read narrows the rows while the batch stays the whole board: the
    # envelope still names and counts the sibling whose bar tripped the fallback.
    filtered = {
        "conductor": {"slot_key": "chat-a-conductor"},
        "items": [_capped_row(2, {"kind": "human_approval"})],
        "accept_batch": {
            "items": [
                {
                    "id": "it_00000001",
                    "accept": {"kind": "human_approval", "meta": deep},
                    "status": None,
                },
                {"id": "it_00000002", "accept": {"kind": "human_approval"}, "status": None},
            ]
        },
    }
    body = json.loads(
        routes._fit_to_budget(
            filtered, routes._RESPONSE_BUDGET_CHARS, ["it_00000001", "it_00000002"]
        )
    )
    assert body["unfittable"] is True
    assert body["omitted_items"] == ["it_00000002", "it_00000001"] and body["omitted_count"] == 2


def test_the_envelope_bounds_its_own_id_list():
    """A hand-grown board can hold thousands of readable item files. The envelope
    keeps the count and as many ids as fit — halving the list until the text fits
    as delivered — so the fallback itself never exceeds the budget."""
    ids = [f"it_{n:08x}" for n in range(5000)]
    envelope = routes._unfittable_envelope(
        {"conductor": {"slot_key": "chat-a-conductor"}}, 50_000, ids, True, why="numbers"
    )
    text = routes._dumps(envelope)
    assert len(text) <= 50_000 and len(redact_via_context(text)) <= 50_000
    assert envelope["omitted_count"] == 5000
    assert 0 < len(envelope["omitted_items"]) < 5000
    assert envelope["omitted_items"] == ids[: len(envelope["omitted_items"])]
    assert "numbers" in envelope["truncation_hint"]
    # Ids that are not shaped like one never enter the list; the count still counts them.
    envelope = routes._unfittable_envelope({}, 50_000, ["it_00000001", "x" * 65], False, why="w")
    assert envelope["omitted_items"] == ["it_00000001"] and envelope["omitted_count"] == 2
    assert "accept_batch" not in envelope and "slot_key" not in envelope["conductor"]
    # Through the trim, on a board small enough to walk: the same envelope.
    # Ten rows of ~60 KB of digits each: not even the newest alone can fit.
    rows = [{"item_id": f"it_{n:08x}", "bulk": list(range(10**8, 10**8 + 6000))} for n in range(10)]
    row_ids = [r["item_id"] for r in rows]  # the trim edits ``rows`` in place
    body = json.loads(
        routes._fit_to_budget(
            {"conductor": {"slot_key": "chat-a-conductor"}, "items": rows},
            routes._RESPONSE_BUDGET_CHARS,
            row_ids,
        )
    )
    assert body["unfittable"] is True and body["omitted_count"] == 10
    assert body["omitted_items"] == row_ids


@pytest.mark.asyncio
async def test_a_compact_read_shows_a_stale_worker(monkeypatch):
    """The patrol read the skill recommends is the compact one, and the dead-worker
    signal is exactly what a patrol is there to notice — so ``stale`` (and its two
    siblings) are on the compact row, not only on the full one."""
    ids = await two_by_two()
    _SLOTS[CONDUCTOR_A] = _Slot()  # the conductor's own slot is open: not orphaned
    real_is_stale = wl.is_stale
    monkeypatch.setattr(wl, "is_stale", lambda item, **kw: real_is_stale(item, window_secs=0, **kw))
    _, body = await _read_with(CONDUCTOR_A, "compact=true")
    row = next(r for r in body["items"] if r["item_id"] == ids["item_a"])
    assert row["stale"] is True, "a bound worker whose slot is not running, past the window"
    assert row["orphaned"] is False
    assert row["acceptance_concrete"] is True
    assert "events" not in row and "acceptance" not in row
    _, full = await _read(CONDUCTOR_A)
    full_row = next(r for r in full["items"] if r["item_id"] == ids["item_a"])
    assert (row["stale"], row["orphaned"]) == (full_row["stale"], full_row["orphaned"])


def test_fit_to_budget_serializes_a_wide_board_a_bounded_number_of_times():
    """A 256-item board with twenty events each (the stored ceiling) is what needs
    trimming most. A re-serialization of the whole payload per dropped unit would
    be hundreds of multi-hundred-KB runs of the indented encoder — past the tool
    layer's read timeout on exactly these boards — so units are measured once and
    the payload is re-serialized only to confirm at a stage boundary: the
    full-serialization count is the bound asserted here (a wall-clock bound would
    measure the runner, not the code)."""
    rows = [_fully_capped_row(n) for n in range(1, wl.MAX_STORED_ITEMS_PER_CONDUCTOR + 1)]
    for row in rows:
        row["events"] = row["events"][:20]
    payload = _capped_payload(rows)
    payload["conductor"] = _capped_conductor()
    assert len(routes._dumps(payload)) > 40 * routes._RESPONSE_BUDGET_CHARS, "a wide board"
    dumps = {"n": 0}
    real_dumps = routes._dumps

    def counting(document: Any) -> str:
        dumps["n"] += 1
        return real_dumps(document)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(routes, "_dumps", counting)
        text = routes._fit_to_budget(payload, routes._RESPONSE_BUDGET_CHARS, _oldest_first(rows))
    body = json.loads(text)
    assert len(text) <= routes._RESPONSE_BUDGET_CHARS
    assert len(routes._delivered(text)) <= routes._RESPONSE_BUDGET_CHARS
    assert body["truncated"] is True
    assert body["items"][-1]["item_id"] == f"it_{wl.MAX_STORED_ITEMS_PER_CONDUCTOR:08x}"
    assert dumps["n"] <= 12, f"{dumps['n']} full serializations"


def test_fit_to_budget_drains_thousands_of_leaves_in_bounded_time():
    """A hand-grown item under the reader's ceiling can hold thousands of strings
    (here 3,000 artifact values of 240 chars, ~790 KB serialized). The fifth stage
    ranks them once and drains them on measured sizes, so the read answers in well
    under the tool layer's timeout — a re-walk and a re-dump per elided leaf would
    be minutes. The bound asserted is the full-serialization count (a wall-clock
    bound would measure the runner, not the code). No marker can bring this record
    under the budget, so the answer is the envelope, and it is still the envelope
    that fits."""
    row = _capped_row(1, {"kind": "human_approval"})
    row["artifacts"] = {f"key_{n}": "v" * 240 for n in range(3_000)}
    payload = _capped_payload([row])
    assert len(routes._dumps(payload)) > 700_000
    dumps = {"n": 0}
    real_dumps = routes._dumps

    def counting(document: Any) -> str:
        dumps["n"] += 1
        return real_dumps(document)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(routes, "_dumps", counting)
        text = routes._fit_to_budget(payload, routes._RESPONSE_BUDGET_CHARS, ["it_00000001"])
    body = json.loads(text)
    assert len(text) <= routes._RESPONSE_BUDGET_CHARS
    assert body["unfittable"] is True and body["omitted_items"] == ["it_00000001"]
    assert dumps["n"] <= 12, f"{dumps['n']} full serializations"


def test_fit_to_budget_elides_many_leaves_largest_first_and_fits():
    """Where the leaves CAN bring the record under the budget, the ranked drain
    elides the largest ones and stops: 40 artifact values of 3,000 chars beside a
    capped row; the first stages cannot touch a single row, so the fifth trims it,
    and the answer is one document under the budget naming each elided path."""
    row = _capped_row(1, {"kind": "human_approval"})
    row["artifacts"] = {f"big_{n}": "v" * 3_000 for n in range(40)}
    row["artifacts"]["tiny"] = "x"
    payload = _capped_payload([row])
    text = routes._fit_to_budget(payload, routes._RESPONSE_BUDGET_CHARS, ["it_00000001"])
    body = json.loads(text)
    assert len(text) <= routes._RESPONSE_BUDGET_CHARS
    assert body["truncated"] is True and "unfittable" not in body
    assert body["items"][0]["artifacts"]["tiny"] == "x", "a small leaf is never worth eliding"
    assert all(path.startswith("items[0].artifacts.big_") for path in body["elided_fields"])
    assert 0 < len(body["elided_fields"]) <= 40


def test_the_schema_ceiling_restates_the_route_caps():
    """``validation`` cannot import the handler, so the ``events`` ceiling is
    spelled twice; this is what keeps the two spellings one number."""
    events = next(f for f in validation.WORK_LEDGER_READ_SCHEMA.fields if f.name == "events")
    assert events.max_val == routes._MAX_EVENT_TAIL
    assert events.min_val == 0
    assert routes._MAX_EVENT_TAIL <= wl.MAX_EVENTS_PER_ITEM
    state = next(f for f in validation.WORK_LEDGER_READ_SCHEMA.fields if f.name == "state")
    assert state.allowed == wl.ITEM_STATES
    item_id = next(f for f in validation.WORK_LEDGER_READ_SCHEMA.fields if f.name == "item_id")
    assert item_id.pattern is not None and item_id.pattern.pattern == wl._ITEM_ID_RE.pattern
