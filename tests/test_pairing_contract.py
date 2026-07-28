from __future__ import annotations

from shared.contracts import EpisodePairKey, validate_paired_episodes


def make_record(condition: str) -> dict[str, object]:
    pair = EpisodePairKey(task_id=0, initial_state_index=1, random_seed=7)
    return {
        "episode_id": f"{pair.paired_group_id}__{condition}",
        "paired_group_id": pair.paired_group_id,
        "condition": condition,
        "model": "openvla",
        "checkpoint": "checkpoint",
        "model_code_commit": "model-commit",
        "suite": "libero_object",
        "task_id": 0,
        "task_name": "task",
        "initial_state_index": 1,
        "random_seed": 7,
        "instruction": "instruction",
        "num_open_loop_steps": 8,
    }


def test_pair_key_is_stable() -> None:
    key = EpisodePairKey(task_id=3, initial_state_index=2, random_seed=11)
    assert key.paired_group_id == "3__2__seed11"


def test_complete_paired_group_is_valid() -> None:
    records = [
        make_record("baseline"),
        make_record("target_mask"),
        make_record("background_control"),
    ]
    assert validate_paired_episodes(records) == []


def test_pairing_detects_invariant_mismatch() -> None:
    records = [
        make_record("baseline"),
        make_record("target_mask"),
        make_record("background_control"),
    ]
    records[1]["random_seed"] = 99
    errors = validate_paired_episodes(records)
    assert any("random_seed" in error for error in errors)
