"""Small shared contracts for pairing VLA episodes and policy queries.

This module is intentionally independent of OpenVLA and LIBERO so it can be used
by repository validation, Group A tracing, and Group B intervention code.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

EXPECTED_CONDITIONS = frozenset({"baseline", "target_mask", "background_control"})
PAIR_INVARIANT_FIELDS = (
    "model",
    "checkpoint",
    "model_code_commit",
    "suite",
    "task_id",
    "task_name",
    "initial_state_index",
    "random_seed",
    "instruction",
    "num_open_loop_steps",
)


@dataclass(frozen=True)
class EpisodePairKey:
    task_id: int
    initial_state_index: int
    random_seed: int

    @property
    def paired_group_id(self) -> str:
        return f"{self.task_id}__{self.initial_state_index}__seed{self.random_seed}"

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "EpisodePairKey":
        return cls(
            task_id=int(metadata["task_id"]),
            initial_state_index=int(metadata["initial_state_index"]),
            random_seed=int(metadata["random_seed"]),
        )


@dataclass(frozen=True)
class PolicyQueryKey:
    paired_group_id: str
    condition: str
    policy_query_index: int

    def __post_init__(self) -> None:
        if self.condition not in EXPECTED_CONDITIONS:
            raise ValueError(f"Unknown condition: {self.condition}")
        if self.policy_query_index < 0:
            raise ValueError("policy_query_index must be non-negative")


def config_sha256(path: str | Path) -> str:
    file_path = Path(path)
    digest = sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_paired_episodes(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return protocol errors for one baseline/target/background paired group."""

    rows = list(records)
    errors: list[str] = []
    if not rows:
        return ["paired episode group is empty"]

    expected_pair_id = EpisodePairKey.from_metadata(rows[0]).paired_group_id
    conditions = {str(row.get("condition")) for row in rows}
    if conditions != EXPECTED_CONDITIONS:
        errors.append(
            f"conditions mismatch: expected {sorted(EXPECTED_CONDITIONS)}, got {sorted(conditions)}"
        )

    if len(rows) != len(EXPECTED_CONDITIONS):
        errors.append(f"expected 3 condition rows, got {len(rows)}")

    reference = rows[0]
    for index, row in enumerate(rows):
        pair_id = str(row.get("paired_group_id", ""))
        if pair_id != expected_pair_id:
            errors.append(
                f"row {index} paired_group_id mismatch: expected {expected_pair_id}, got {pair_id}"
            )
        for field in PAIR_INVARIANT_FIELDS:
            if row.get(field) != reference.get(field):
                errors.append(
                    f"row {index} invariant field {field!r} differs: "
                    f"{row.get(field)!r} != {reference.get(field)!r}"
                )

    return errors
