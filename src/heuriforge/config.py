"""Settings (config/settings.yaml) and data splits (config/splits/<dataset>.yaml)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]  # the repository root
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


class Paths(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in settings.yaml is an error

    data: Path = Path("data/instances")
    runs: Path = Path("runs")
    site: Path = Path("site")
    bench: Path = Path("bench")

    def resolved(self, root: Path) -> Paths:
        return Paths(**{k: (v if v.is_absolute() else root / v) for k, v in self})


class ModelEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "local"
    cost: float = 1.0


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # server: local llama-server; random: no language model (baseline); mock: offline fake model
    provider: Literal["server", "random", "mock"] = "server"
    models: list[ModelEndpoint] = Field(default_factory=lambda: [ModelEndpoint(name="local")])
    mutations: list[Literal["diff", "rewrite"]] = ["diff", "rewrite"]
    thinking: bool = False
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 1500
    timeout_s: float = 600.0
    retries: int = 2


class EvolutionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    islands: int = 2
    migration_interval: int = 40
    complexity_bins: list[int] = [40, 80, 160]
    params_bins: list[int] = [2, 3, 4]
    exploit_prob: float = 0.3
    selection_lambda: float = 10.0
    inspirations: int = 2
    max_param_variants: int = 6
    reflection_interval: int = 25
    max_lessons_in_prompt: int = 3

    @field_validator("islands")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("islands must be at least 1")
        return v


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_candidates: int = 500
    max_hours: float = 10.0
    parallel: int = 2
    seed: int = 0


class EvalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workers: int = 4
    memory_mb: int = 2048


class TrackingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mlflow: bool = False


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: Paths = Paths()
    llm: LLMSettings = LLMSettings()
    evolution: EvolutionSettings = EvolutionSettings()
    run: RunSettings = RunSettings()
    eval: EvalSettings = EvalSettings()
    tracking: TrackingSettings = TrackingSettings()
    tasks: dict[str, dict] = Field(default_factory=dict)
    root: Path = ROOT

    def task_options(self, task: str) -> dict:
        return dict(self.tasks.get(task, {}))


def load_settings(path: str | Path | None = None, root: Path | None = None) -> Settings:
    root = Path(root or os.environ.get("HEURIFORGE_ROOT", ROOT))
    path = Path(path or os.environ.get("HEURIFORGE_SETTINGS", root / "config" / "settings.yaml"))
    raw = yaml.safe_load(path.read_text()) or {}
    settings = Settings(**raw, root=root)
    settings.paths = settings.paths.resolved(root)
    return settings


def load_splits(dataset: str, root: Path | None = None) -> dict[str, list[str]]:
    """Instance names per split: smoke, train, val and test."""
    root = Path(root or os.environ.get("HEURIFORGE_ROOT", ROOT))
    path = root / "config" / "splits" / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no split file for dataset {dataset!r}: {path}")
    raw = yaml.safe_load(path.read_text())
    splits = {k: list(raw.get(k) or []) for k in ("smoke", "train", "val", "test")}
    seen: dict[str, str] = {}
    for split, names in splits.items():
        for name in names:
            if name in seen and {seen[name], split} != {"smoke"}:
                raise ValueError(f"{name} appears in both {seen[name]} and {split}")
            seen[name] = split
    return splits
