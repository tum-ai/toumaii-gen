import pytest

import tool_calling_data_generation.hub as hub_module
from tool_calling_data_generation.hub import _assert_repo_owner_visible


def test_private_upload_owner_must_be_visible_to_token():
    identity = {"name": "alice", "orgs": [{"name": "research-org"}]}
    _assert_repo_owner_visible(identity, "research-org/smoke")
    with pytest.raises(PermissionError):
        _assert_repo_owner_visible(identity, "another-org/smoke")


def test_upload_creates_private_repo_and_pushes_three_configs(monkeypatch, tmp_path):
    class Missing(Exception):
        pass

    class Info:
        private = True

    class FakeApi:
        def __init__(self):
            self.info_calls = 0
            self.created = []

        def whoami(self):
            return {"name": "alice", "orgs": [{"name": "research-org"}]}

        def dataset_info(self, **kwargs):
            self.info_calls += 1
            if self.info_calls == 1:
                raise Missing()
            return Info()

        def create_repo(self, **kwargs):
            self.created.append(kwargs)

    pushes = []

    class FakeDataset:
        def __len__(self):
            return 3

        def push_to_hub(self, **kwargs):
            pushes.append(kwargs)

    class DatasetFactory:
        @staticmethod
        def from_parquet(path):
            return FakeDataset()

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"status":"generated","artifacts":{'
        '"canonical_parquet":"canonical.parquet",'
        '"dpo_parquet":"dpo.parquet",'
        '"sft_parquet":"sft.parquet"}}',
        encoding="utf-8",
    )
    api = FakeApi()
    monkeypatch.setattr(hub_module, "RepositoryNotFoundError", Missing)
    monkeypatch.setattr(hub_module, "get_token", lambda: "token")
    monkeypatch.setattr(hub_module, "HfApi", lambda token: api)
    monkeypatch.setattr(hub_module, "Dataset", DatasetFactory)

    result = hub_module.upload_private_run(
        {
            "hub": {
                "repo_id": "research-org/smoke",
                "private": True,
                "split": "train",
                "canonical_config": "canonical",
                "dpo_config": "dpo",
                "sft_config": "sft",
            }
        },
        manifest,
    )

    assert api.created[0]["private"] is True
    assert [push["config_name"] for push in pushes] == ["canonical", "dpo", "sft"]
    assert all(push["private"] is True for push in pushes)
    assert result["rows_by_config"] == {"canonical": 3, "dpo": 3, "sft": 3}


def test_upload_refuses_existing_public_repo(monkeypatch, tmp_path):
    class PublicInfo:
        private = False

    class FakeApi:
        def whoami(self):
            return {"name": "alice", "orgs": []}

        def dataset_info(self, **kwargs):
            return PublicInfo()

    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"status":"generated","artifacts":{}}', encoding="utf-8")
    monkeypatch.setattr(hub_module, "get_token", lambda: "token")
    monkeypatch.setattr(hub_module, "HfApi", lambda token: FakeApi())

    with pytest.raises(PermissionError, match="not private"):
        hub_module.upload_private_run(
            {"hub": {"repo_id": "alice/public-target", "private": True}}, manifest
        )
