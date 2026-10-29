"""ToolACE-specific regression tests for the production cleaner."""

import pytest

from tool_calling_data_generation.post_processing import post_process_rejected as cleaner


def test_toolace_turn_labels_decode_to_numeric_order():
    assert cleaner._strict_turn_id("single_0") == (0, None)
    assert cleaner._strict_turn_id("multi_12") == (12, None)
    assert cleaner._strict_turn_id("7") == (7, None)
    assert cleaner._strict_turn_id(3) == (3, None)


def test_toolace_turn_label_mode_is_preserved_for_validation():
    assert cleaner._turn_id_mode("single_2") == "single"
    assert cleaner._turn_id_mode("multi_9") == "multi"
    assert cleaner._turn_id_mode("2") is None
    assert cleaner._turn_id_mode(2) is None


def test_invalid_toolace_turn_labels_are_not_silently_coerced():
    turn_id, error = cleaner._strict_turn_id("parallel_1")
    assert turn_id is None
    assert error is not None


def test_toolace_metadata_subset_names_match_file_kinds():
    for file_kind, (subset_name, toolcall_mode) in (
        cleaner.FILE_KIND_TO_METADATA.items()
    ):
        assert subset_name == file_kind
        assert toolcall_mode == (
            "multi" if file_kind.endswith("_multicall") else "single"
        )


def test_cleaner_requires_external_repository_configuration(monkeypatch):
    monkeypatch.setattr(cleaner, "SOURCE_DATASET_ID", None)
    monkeypatch.setattr(cleaner, "SOURCE_REVISION", None)
    monkeypatch.setattr(cleaner, "DESTINATION_DATASET_ID", None)

    with pytest.raises(ValueError, match="Required cleaning configuration") as error:
        cleaner._validate_configuration()

    message = str(error.value)
    assert "TOUCAN_CLEANING_SOURCE_DATASET_ID" in message
    assert "TOUCAN_CLEANING_SOURCE_REVISION" in message
    assert "TOUCAN_CLEANING_DESTINATION_DATASET_ID" in message
