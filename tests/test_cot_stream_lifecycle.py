from types import SimpleNamespace

import pytest

from dynasor.core.cot import openai_chat_completion_stream


def event(text="", finish=None):
    return SimpleNamespace(choices=[SimpleNamespace(text=text, finish_reason=finish)])


class Response:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __iter__(self):
        for item in self.events:
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self):
        self.closed = True


class Client:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses[len(self.calls) - 1]
        if isinstance(response, Exception):
            raise response
        return response


def generate(client, **kwargs):
    return openai_chat_completion_stream(
        client, "test", "prompt", max_tokens=kwargs.pop("max_tokens", 65),
        dynasor_saving_effort=kwargs.pop("effort", (3, 32)), **kwargs)


def test_natural_stop_closes_unread_probe_and_preserves_text():
    main = Response([event("answer", "stop"), SimpleNamespace(choices=[])])
    probe = Response([event("unused}")])
    client = Client([main, probe])
    assert "".join(generate(client)) == "answer"
    assert main.closed and probe.closed
    assert len(client.calls) == 2


@pytest.mark.parametrize("effort", [(3, 32), None])
def test_consumer_close_releases_all_started_responses(effort):
    main, probe = Response([event("first "), event("second ")]), Response([])
    client = Client([main, probe])
    iterator = generate(client, effort=effort)
    assert next(iterator) == "first "
    iterator.close()
    assert main.closed
    if effort:
        assert probe.closed


def test_budget_rounds_down_final_chunk():
    responses = []
    for _ in range(3):
        responses.extend([Response([event("x ", "length")]), Response([event("no answer")])])
    client = Client(responses)
    assert "".join(generate(client)) == "x x x "
    assert [call["max_tokens"] for call in client.calls[::2]] == [32, 32, 1]
    assert all(response.closed for response in responses)


@pytest.mark.parametrize("failure", ["main", "probe", "probe_create"])
def test_exception_releases_all_acquired_streams(failure):
    error = RuntimeError("upstream failed")
    main = Response([error] if failure == "main" else [event("x ", "length")])
    probe = error if failure == "probe_create" else Response([error])
    client = Client([main, probe])
    with pytest.raises(RuntimeError, match="upstream failed"):
        list(generate(client))
    assert main.closed
    if isinstance(probe, Response):
        assert probe.closed


def test_empty_generation_closes_probe_without_unbound_chunk():
    main, probe = Response([]), Response([])
    assert list(generate(Client([main, probe]))) == []
    assert main.closed and probe.closed


def test_adaptive_answer_preserved_and_all_probes_closed():
    responses = []
    for _ in range(2):
        responses.extend([Response([event("step ", "length")]), Response([event("42}")])])
    client = Client(responses)
    result = "".join(generate(client, effort=(2, 32)))
    assert result.startswith("step step ")
    assert "\\boxed{42}" in result
    assert all(response.closed for response in responses)


@pytest.mark.parametrize("effort", [(0, 32), (-1, 32), (3, 0), (3, -1)])
def test_invalid_effort_fails_before_creating_responses(effort):
    client = Client([])
    with pytest.raises(ValueError, match="threshold and chunk_size must be positive"):
        list(generate(client, effort=effort))
    assert client.calls == []


@pytest.mark.parametrize("budget", [1, 7, 31, 32])
def test_budget_at_most_one_chunk_does_not_round_up(budget):
    main = Response([event("answer", "length")])
    probe = Response([event("no answer")])
    client = Client([main, probe])
    assert "".join(generate(client, max_tokens=budget)) == "answer"
    assert len(client.calls) == 2
    assert client.calls[0]["max_tokens"] == budget
    assert client.calls[1]["max_tokens"] == 20
    assert main.closed and probe.closed
