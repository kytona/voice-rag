from voice_rag.core.config import EmbeddingConfig, LLMConfig, VoiceRagConfig

def test_default_config():
    config = VoiceRagConfig()
    assert config.llm.provider == "openai"
    assert config.voice.provider == "elevenlabs"
    assert config.vector_store.provider == "qdrant"
    assert config.llm.model == "gpt-4o-mini"
    assert config.vector_store.url == ""
    assert config.vector_store.use_local is True

def test_config_from_dict():
    config = VoiceRagConfig(
        llm={"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
        voice={"provider": "deepgram"},
    )
    assert config.llm.provider == "anthropic"
    assert config.voice.provider == "deepgram"

def test_config_from_yaml(tmp_path):
    yaml_content = """
llm:
  provider: openai
  model: gpt-4o
voice:
  provider: elevenlabs
vector_store:
  provider: qdrant
  collection_name: my_collection
"""
    config_file = tmp_path / "voice-rag.yaml"
    config_file.write_text(yaml_content)
    config = VoiceRagConfig.from_yaml(config_file)
    assert config.llm.model == "gpt-4o"
    assert config.vector_store.collection_name == "my_collection"


def test_legacy_in_memory_config_still_uses_local_store():
    config = VoiceRagConfig(vector_store={"in_memory": True, "url": "http://localhost:6333"})
    assert config.vector_store.url == ""
    assert config.vector_store.use_local is True


def test_settings_do_not_auto_load_dotenv():
    assert LLMConfig.model_config.get("env_file") is None
    assert EmbeddingConfig.model_config.get("env_file") is None
