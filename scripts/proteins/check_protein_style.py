#!/usr/bin/env python3
"""Check the protein integration against the repository Ruff policy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (
    "configs/proteins",
    "data/protein_multimodal.py",
    "data/protein_multimodal_sampler.py",
    "data/protein_structure_codec.py",
    "data/protein_tasks.py",
    "data/proteins.py",
    "data/uniref50.py",
    "data/uniref50_sampler.py",
    "evaluation/evaluation_drivers/proteins",
    "models/protein_multimodal.py",
    "scripts/upload_to_hf.py",
    "scripts/proteins",
    "tests/protein_fixture.py",
    "tests/test_protein_multimodal_e2e.py",
    "tests/test_protein_multimodal_model.py",
    "tests/test_protein_multimodal_tasks.py",
    "tests/test_protein_multimodal_train.py",
    "tests/test_protein_structure_codec.py",
    "tests/test_protein_upload.py",
    "tests/test_protein_workflow.py",
    "trainers/multimodal_step.py",
    "utils/proteins",
)


def main() -> int:
    """Run Ruff over files owned by the protein integration."""
    command = [sys.executable, "-m", "ruff"]
    passed = True
    for arguments in (["check", *TARGETS],):
        result = subprocess.run(
            command + arguments, cwd=ROOT, check=False
        )
        passed = passed and result.returncode == 0
    if passed:
        print("Protein integration style checks passed.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
