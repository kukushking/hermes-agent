"""Every direct Firecrawl request is bounded (#136720).

An upstream that accepts the connection and then never answers used to hang the keyed SDK client
forever (``timeout=None`` reached ``requests.post``), so ``web_search`` / ``web_extract`` only failed
at the tool-dispatch deadline, 420 s later, against an API that was healthy. The client must be
built with a request timeout shorter than that deadline. The keyless REST client already was; the
keyed, self-hosted and tool-gateway paths all go through the SDK constructor under test here.
"""

from types import SimpleNamespace

import pytest

from agent.tool_executor import _DEFAULT_CONCURRENT_TOOL_TIMEOUT_S
from plugins.web.firecrawl import provider as prov


@pytest.fixture()
def build_direct_client(monkeypatch):
    """Build the direct client for an env with a recording stand-in for the SDK class."""
    import tools.web_tools as wt

    monkeypatch.setattr(wt, "_firecrawl_client", None, raising=False)
    monkeypatch.setattr(wt, "_firecrawl_client_config", None, raising=False)
    monkeypatch.setattr("tools.tool_backend_helpers.read_web_capability_selection", lambda capability=None: None)
    monkeypatch.setattr(prov, "_is_explicit_firecrawl_selection", lambda: False)
    built: dict = {}
    monkeypatch.setattr(prov, "Firecrawl", lambda **kwargs: built.update(kwargs) or SimpleNamespace(**kwargs))

    def build(env: dict[str, str]) -> dict:
        monkeypatch.setattr(prov, "_env", lambda name: env.get(name, ""))
        prov._get_firecrawl_client("search")
        return built

    return build


@pytest.mark.parametrize("env", [
    {"FIRECRAWL_API_KEY": "fc-test"},
    {"FIRECRAWL_API_URL": "http://firecrawl.internal:3002"},
    {"FIRECRAWL_API_KEY": "fc-test", "FIRECRAWL_API_URL": "http://firecrawl.internal:3002"},
], ids=["keyed-cloud", "self-hosted", "keyed-self-hosted"])
def test_direct_sdk_client_is_built_with_a_bounded_request_timeout(build_direct_client, env):
    kwargs = build_direct_client(env)

    timeout = kwargs.get("timeout")
    assert isinstance(timeout, (int, float)) and timeout > 0
    assert timeout < _DEFAULT_CONCURRENT_TOOL_TIMEOUT_S
    # The credentials still reach the SDK unchanged.
    if "FIRECRAWL_API_KEY" in env:
        assert kwargs["api_key"] == env["FIRECRAWL_API_KEY"]
    if "FIRECRAWL_API_URL" in env:
        assert kwargs["api_url"] == env["FIRECRAWL_API_URL"]


def test_keyless_client_requests_are_bounded_the_same_way(monkeypatch):
    seen: dict = {}

    def fake_post(url, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": {"web": []}})

    monkeypatch.setattr(prov.httpx, "post", fake_post)

    prov._KeylessFirecrawlClient().search(query="hermes", limit=1)

    assert 0 < seen["timeout"] < _DEFAULT_CONCURRENT_TOOL_TIMEOUT_S
