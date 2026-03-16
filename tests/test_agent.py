from voice_rag.agent import KnowledgeAgent
from voice_rag.core.config import VoiceRagConfig


def test_agent_default_creation():
    config = VoiceRagConfig()
    agent = KnowledgeAgent(config=config, _lazy_init=True)
    assert agent.config.llm.provider == "openai"
    assert agent.config.voice.provider == "elevenlabs"


def test_agent_with_string_providers():
    agent = KnowledgeAgent(llm="anthropic", voice="deepgram", _lazy_init=True)
    assert agent.config.llm.provider == "anthropic"
    assert agent.config.voice.provider == "deepgram"


def test_agent_create_app():
    agent = KnowledgeAgent(_lazy_init=True)
    app = agent.create_app()
    route_paths = [route.path for route in app.routes]
    assert "/v1/chat/completions" in route_paths
    assert "/chat/completions" in route_paths
    assert "/health" in route_paths
