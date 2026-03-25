import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _read_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_version = _read_optional(repo_root / "data" / "processed" / ".data_version")
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.environ.get("GITHUB_SHA"),
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "image": os.environ.get("IMAGE_NAME"),
        "image_digest": os.environ.get("IMAGE_DIGEST"),
        "dataset_version_hash": data_version,
        "mlflow_run_id": os.environ.get("MLFLOW_RUN_ID"),
        "mlflow_model_version": os.environ.get("MLFLOW_MODEL_VERSION"),
        "mlflow_model_alias": os.environ.get("MLFLOW_MODEL_ALIAS", "production"),
        "test_summary": {
            "unit_passed": os.environ.get("UNIT_TESTS_PASSED"),
            "integration_passed": os.environ.get("INTEGRATION_TESTS_PASSED"),
            "e2e_passed": os.environ.get("E2E_TESTS_PASSED"),
            "coverage_xml": "coverage.xml",
        },
    }
    output = repo_root / "release-metadata.json"
    output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"release metadata written: {output}")


if __name__ == "__main__":
    main()
