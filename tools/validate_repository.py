"""Validate the repository-level experiment contract without loading VLA dependencies."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    "shared/project_config.yaml",
    "shared/metadata_schema.json",
    "groupA/README.md",
    "groupB/README.md",
    "docs/PROJECT_CHARTER.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/PAPER_LIST.md",
    "docs/WEEK1_PLAN.md",
    "docs/WEEK2_PLAN.md",
    "docs/STUDENT_WORKFLOW.md",
    "docs/RESULT_REPORT_TEMPLATE.md",
)

EXPECTED_CONDITIONS = ["baseline", "target_mask", "background_control"]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data


def validate_required_files(root: Path = ROOT) -> list[str]:
    return [path for path in REQUIRED_FILES if not (root / path).is_file()]


def validate_project_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_sections = {
        "project",
        "model",
        "environment",
        "conditions",
        "pairing",
        "trace",
        "mask",
        "metrics",
        "validation",
        "output",
    }
    missing = sorted(required_sections - set(config))
    if missing:
        errors.append(f"missing config sections: {missing}")
        return errors

    if config["conditions"] != EXPECTED_CONDITIONS:
        errors.append(
            "conditions must remain ordered as baseline/target_mask/background_control"
        )

    environment = config["environment"]
    if environment.get("task_suite_name") != "libero_object":
        errors.append("environment.task_suite_name must be libero_object")

    states = environment.get("initial_state_indices")
    if not isinstance(states, list) or not states:
        errors.append("environment.initial_state_indices must be a non-empty list")
    elif len(states) != len(set(states)):
        errors.append("environment.initial_state_indices contains duplicates")

    open_loop = config["model"].get("num_open_loop_steps")
    if not isinstance(open_loop, int) or open_loop <= 0:
        errors.append("model.num_open_loop_steps must be a positive integer")

    tolerance = config["mask"].get("background_area_tolerance")
    if not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 1:
        errors.append("mask.background_area_tolerance must be within [0,1]")

    pairing = config["pairing"]
    for key in (
        "require_same_checkpoint",
        "require_same_instruction",
        "require_same_initial_state",
        "require_same_seed",
        "require_same_open_loop_steps",
    ):
        if pairing.get(key) is not True:
            errors.append(f"pairing.{key} must remain true")

    if config["output"].get("save_large_tensors_to_git") is not False:
        errors.append("output.save_large_tensors_to_git must remain false")

    return errors


def validate_metadata_schema(schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # jsonschema provides detailed exception types.
        errors.append(f"invalid metadata schema: {exc}")
        return errors

    required = set(schema.get("required", []))
    contract_fields = {
        "episode_id",
        "paired_group_id",
        "checkpoint",
        "model_code_commit",
        "initial_state_index",
        "random_seed",
        "condition",
        "policy_query_count",
        "config_hash",
        "git_commit",
    }
    missing = sorted(contract_fields - required)
    if missing:
        errors.append(f"metadata schema missing required contract fields: {missing}")
    return errors


def validate_readme_links(root: Path = ROOT) -> list[str]:
    readme = (root / "README.md").read_text(encoding="utf-8")
    errors: list[str] = []
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme):
        if target.startswith(("http://", "https://", "#")):
            continue
        resolved = (root / target).resolve()
        if not resolved.exists():
            errors.append(f"README link target does not exist: {target}")
    return errors


def main() -> None:
    errors: list[str] = []
    missing_files = validate_required_files()
    if missing_files:
        errors.append(f"missing required files: {missing_files}")

    config_path = ROOT / "shared/project_config.yaml"
    schema_path = ROOT / "shared/metadata_schema.json"
    if config_path.is_file():
        errors.extend(validate_project_config(load_yaml(config_path)))
    if schema_path.is_file():
        errors.extend(validate_metadata_schema(load_json(schema_path)))
    if (ROOT / "README.md").is_file():
        errors.extend(validate_readme_links())

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(f"Repository validation failed with {len(errors)} error(s).")

    print("PASS: repository files, shared config, metadata schema, and README links are valid.")


if __name__ == "__main__":
    main()
