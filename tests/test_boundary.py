# Copyright (C) 2026 Moritz Pötzsch, Andrea Gentilini, Lisa Rossgoderer, Tom Osterfeld,
# Kesava Prasad Sanjivi Arul, and Swanand Ravindra Kadhe
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path


def test_extracted_package_has_no_parent_repo_imports():
    package = Path(__file__).parents[1] / "tool_calling_data_generation"
    forbidden = ("from src.", "import src.", "from rewards import")
    offenders = []
    for path in package.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                offenders.append(f"{path.relative_to(package)}: {marker}")
    assert offenders == []
