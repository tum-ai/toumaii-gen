# Copyright (C) 2026 Moritz Pötzsch, Andrea Gentilini, Lisa Rossgoderer, Tom Osterfeld,
# Kesava Prasad Sanjivi Arul, and Swanand Ravindra Kadhe
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json

from tool_calling_data_generation.cli import main


def test_validate_command_does_not_load_generation_backend(capsys):
    assert main(["validate", "--config", "config.yaml"]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["source"]["toucan"]["access_mode"] == "full"
    assert rendered["hub"]["private"] is True
