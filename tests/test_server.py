from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from voice_rag.agent import KnowledgeAgent
from voice_rag.server import create_app


def _make_test_agent():
    agent = KnowledgeAgent(_lazy_init=True)
    agent._voice_adapter = MagicMock()
    agent._voice_adapter.format_response_headers.return_value = {"content-type": "text/event-stream"}

    async def fake_stream(*args, **kwargs):
        yield 'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"test","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        yield 'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    agent._llm_client = MagicMock()
    agent._llm_client.stream_chat_completion = fake_stream
    agent._dense_embedder = MagicMock()
    agent._dense_embedder.embed.return_value = [[0.1, 0.2]]
    agent._sparse_embedder = MagicMock()
    agent._sparse_embedder.embed.return_value = [MagicMock()]
    agent._vector_store = MagicMock()
    agent._vector_store.query.return_value = []
    agent._vector_store.collection_stats.return_value = {"collection_exists": True, "points_count": 10}
    return agent


def test_health_endpoint():
    agent = _make_test_agent()
    app = create_app(agent)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_completions_stream():
    agent = _make_test_agent()
    app = create_app(agent)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "custom", "stream": True, "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "[DONE]" in response.text


def test_chat_completions_compat_route():
    agent = _make_test_agent()
    app = create_app(agent)
    client = TestClient(app)
    response = client.post(
        "/chat/completions",
        json={"model": "custom", "stream": True, "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    assert "[DONE]" in response.text


def test_chat_completions_rejects_non_stream():
    agent = _make_test_agent()
    app = create_app(agent)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "custom", "stream": False, "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 400


def test_chat_completions_maps_custom_llm_placeholder_to_configured_model():
    agent = _make_test_agent()

    async def fake_stream(*args, **kwargs):
        yield 'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"test","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        yield 'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    stream_mock = MagicMock(side_effect=fake_stream)
    agent._llm_client.stream_chat_completion = stream_mock
    app = create_app(agent)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "custom-llm", "stream": True, "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 200
    assert "[DONE]" in response.text
    assert agent._llm_client.stream_chat_completion.call_args.kwargs["model"] == agent.config.llm.model
