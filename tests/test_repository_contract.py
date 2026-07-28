from __future__ import annotations

from jsonschema import Draft202012Validator

from tools.validate_repository import (
    EXPECTED_CONDITIONS,
    ROOT,
    load_json,
    load_yaml,
    validate_metadata_schema,
    validate_project_config,
    validate_readme_links,
    validate_required_files,
)


def test_required_repository_files_exist() -> None:
    assert validate_required_files() == []


def test_shared_project_config_is_valid() -> None:
    config = load_yaml(ROOT / "shared/project_config.yaml")
    assert config["conditions"] == EXPECTED_CONDITIONS
    assert validate_project_config(config) == []


def test_metadata_schema_accepts_minimal_episode() -> None:
    schema = load_json(ROOT / "shared/metadata_schema.json")
    assert validate_metadata_schema(schema) == []

    record = {
        "episode_id": "0__0__seed7__baseline",
        "paired_group_id": "0__0__seed7",
        "group": "groupA",
        "model": "openvla",
        "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-object",
        "model_code_commit": "example-model-commit",
        "suite": "libero_object",
        "task_id": 0,
        "task_name": "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "initial_state_index": 0,
        "random_seed": 7,
        "condition": "baseline",
        "instruction": "Pick the alphabet soup and place it in the basket",
        "success": True,
        "episode_length": 120,
        "policy_query_count": 15,
        "config_path": "shared/project_config.yaml",
        "config_hash": "example-config-hash",
        "git_commit": "example-repo-commit",
    }
    Draft202012Validator(schema).validate(record)


def test_readme_navigation_targets_exist() -> None:
    assert validate_readme_links() == []
