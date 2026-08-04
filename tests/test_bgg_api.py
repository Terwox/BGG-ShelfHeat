import pytest

from shelfheat.match import (
    BGGApiClient,
    BGGConfigurationError,
    _fetch_collection,
    _fetch_plays,
    _fetch_related_game_ids,
)


COLLECTION_XML = """<?xml version="1.0"?>
<items>
  <item objectid="123">
    <name>Arcs</name>
    <yearpublished>2024</yearpublished>
    <numplays>2</numplays>
    <stats><rating value="8.5" /></stats>
    <thumbnail>https://example.invalid/arcs.jpg</thumbnail>
  </item>
</items>
"""

PLAYS_XML = """<?xml version="1.0"?>
<plays>
  <play date="2024-06-01">
    <item objectid="123" />
  </play>
</plays>
"""


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def test_fetch_collection_uses_bearer_token_and_retries_queued_response():
    sleeps = []
    session = FakeSession([
        FakeResponse(202),
        FakeResponse(200, COLLECTION_XML),
    ])
    client = BGGApiClient("server-token", session=session, sleep=sleeps.append)

    games = _fetch_collection("alice", client=client)

    assert games[0]["name"] == "Arcs"
    assert games[0]["bgg_id"] == 123
    assert session.calls[0]["headers"]["Authorization"] == "Bearer server-token"
    assert sleeps == [3]


def test_fetch_plays_uses_tokenized_client():
    session = FakeSession([FakeResponse(200, PLAYS_XML)])
    client = BGGApiClient("server-token", session=session, sleep=lambda _: None)

    plays = _fetch_plays("alice", client=client)

    assert plays == {123: "2024-06-01"}
    assert session.calls[0]["headers"]["Authorization"] == "Bearer server-token"
    assert "page" in session.calls[0]["params"]


def test_token_failure_does_not_leak_token_value():
    session = FakeSession([FakeResponse(401)])
    client = BGGApiClient("secret-token-value", session=session, sleep=lambda _: None)

    with pytest.raises(BGGConfigurationError) as exc:
        _fetch_collection("alice", client=client)

    assert "secret-token-value" not in str(exc.value)


def test_related_game_lookup_does_not_call_bgg_without_client():
    assert _fetch_related_game_ids(123, client=None) == []


def test_successful_collection_response_is_cached_in_memory_only():
    session = FakeSession([FakeResponse(200, COLLECTION_XML)])
    client = BGGApiClient("server-token", session=session, sleep=lambda _: None)

    assert _fetch_collection("alice", client=client)[0]["name"] == "Arcs"
    assert _fetch_collection("alice", client=client)[0]["name"] == "Arcs"

    assert len(session.calls) == 1
