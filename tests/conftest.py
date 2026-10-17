"""Shared fixtures: a throw-away project folder with a tiny synthetic dataset."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from heuriforge.config import ROOT, load_settings, load_splits
from heuriforge.instances import prepare_dataset

DATA = Path(__file__).parent / "data"

TINY = {
    "dataset": "tiny",
    "smoke": ["S-n21-s101", "S-n21-s102", "S-n26-s103"],
    "train": ["S-n31-s1", "S-n36-s2", "S-n41-s3"],
    "val": ["S-n33-s4", "S-n38-s5"],
    "test": ["S-n35-s6", "S-n40-s7"],
}


@pytest.fixture(scope="session")
def project(tmp_path_factory) -> Path:
    """A project root with the real settings and a 'tiny' dataset of 10 small instances."""
    root = tmp_path_factory.mktemp("project")
    (root / "config" / "splits").mkdir(parents=True)
    shutil.copy(ROOT / "config" / "settings.yaml", root / "config" / "settings.yaml")
    (root / "config" / "splits" / "tiny.yaml").write_text(yaml.safe_dump(TINY))
    settings = load_settings(root=root)
    prepare_dataset(settings.paths.data, "tiny", load_splits("tiny", root), ref_iterations=500)
    return root


@pytest.fixture
def settings(project):
    s = load_settings(root=project)
    s.eval.workers = 2
    s.run.parallel = 2
    s.tasks["pyvrp_accept"].update(iterations=100, smoke_iterations=50)
    return s
