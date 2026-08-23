import json

from tool_calling_data_generation.cli import main


def test_validate_command_does_not_load_generation_backend(capsys):
    assert main(["validate", "--config", "config.yaml"]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["source"]["toucan"]["access_mode"] == "full"
    assert rendered["hub"]["private"] is True
