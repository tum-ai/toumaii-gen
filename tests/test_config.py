import pytest

from tool_calling_data_generation.config import ConfigError, validate_config


def test_minimal_config_gets_standalone_defaults():
    config = validate_config({"generation": {"model_id": "local/model"}})
    assert config["source"]["type"] == "toucan"
    assert config["hub"]["private"] is True
    assert config["generation"]["backend"] == "auto"


def test_public_upload_configuration_is_rejected():
    with pytest.raises(ConfigError, match="private"):
        validate_config(
            {"generation": {"model_id": "local/model"}, "hub": {"private": False}}
        )


def test_tokens_are_rejected_from_yaml_configuration():
    with pytest.raises(ConfigError, match="Do not store tokens"):
        validate_config(
            {"generation": {"model_id": "local/model"}, "hub": {"token": "secret"}}
        )

