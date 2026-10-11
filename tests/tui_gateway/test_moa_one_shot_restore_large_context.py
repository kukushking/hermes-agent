"""A ``/moa`` one-shot lands on the preset and restores the pre-MoA model after its turn, however
large the session is (#136710).

Both swaps go through ``_apply_model_switch`` so the live client is switched in place. With the
selection guards live on that path, a session at or above ``model.switch_context_confirm_tokens``
got a confirm-required answer instead of a switch, and nothing consumes that answer on either side
of a one-shot: the ``/moa`` turn silently ran on the old model under the "queued" notice, and a
restore after a MoA turn silently left the session pinned to the preset for every later turn. The
classic CLI's ``/moa`` never offers a one-shot for confirmation; the TUI gateway now matches.
"""

from types import SimpleNamespace

import pytest

from tui_gateway import server

PREVIOUS_MODEL, PREVIOUS_PROVIDER = "model-a", "nous"
MOA_PRESET = "skill-moa"
GUARD_THRESHOLD = 100_000


class _Agent:
    """A live agent with a measured context, on either side of the one-shot."""

    def __init__(self, prompt_tokens: int, model: str = PREVIOUS_MODEL, provider: str = PREVIOUS_PROVIDER):
        self.model, self.provider = model, provider
        self.base_url, self.api_key, self.api_mode = "", "", "chat_completions"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=prompt_tokens)

    def switch_model(self, **kw):
        self.model, self.provider = kw["new_model"], kw["new_provider"]


@pytest.fixture
def quiet_switch(monkeypatch):
    """Resolve any model as-is, keep the real selection guards, silence the server side effects."""

    def fake_switch_model(**kw):
        return SimpleNamespace(
            success=True, new_model=kw["raw_input"], target_provider=kw.get("explicit_provider") or PREVIOUS_PROVIDER,
            base_url="", api_key="key", api_mode="chat_completions", warning_message="", model_info=None,
            error_message="", runtime_capabilities=None)

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch_model)
    monkeypatch.setattr("hermes_cli.model_cost_guard.expensive_model_warning", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_selection_guards._context_cache_threshold", lambda: GUARD_THRESHOLD)
    for name in ("_restart_slash_worker", "_persist_live_session_runtime", "_persist_live_session_system_prompt",
                 "_append_model_switch_marker", "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)


def _finish_moa_turn(agent: _Agent) -> dict:
    session = {
        "agent": agent, "history": [],
        "model_override": {"provider": "moa", "model": MOA_PRESET},
        "moa_one_shot_restore": {"override": None, "model": PREVIOUS_MODEL, "provider": PREVIOUS_PROVIDER},
    }
    turn = SimpleNamespace(result=None, agent=agent, history=[], history_version=0)
    server._absorb_turn_result("sid", session, turn, None, None, None)
    return session


_SESSION_SIZES = pytest.mark.parametrize(
    "prompt_tokens", [GUARD_THRESHOLD * 6, GUARD_THRESHOLD // 100], ids=["past-guard", "small"])


@_SESSION_SIZES
def test_moa_one_shot_switches_to_the_preset(quiet_switch, prompt_tokens):
    agent = _Agent(prompt_tokens)
    session = {"agent": agent, "history": []}

    answer = server._cmd_moa("1", {"session_id": "sid"}, session, "moa", "compare these designs")

    assert answer["result"]["type"] == "send"
    assert agent.provider == "moa"
    assert session["moa_one_shot_restore"] == {
        "override": None, "model": PREVIOUS_MODEL, "provider": PREVIOUS_PROVIDER}


@_SESSION_SIZES
def test_moa_one_shot_restores_the_previous_model(quiet_switch, prompt_tokens):
    agent = _Agent(prompt_tokens, MOA_PRESET, "moa")

    session = _finish_moa_turn(agent)

    assert (agent.model, agent.provider) == (PREVIOUS_MODEL, PREVIOUS_PROVIDER)
    assert "moa_one_shot_restore" not in session
    assert "model_override" not in session
