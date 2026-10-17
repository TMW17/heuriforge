"""Optional experiment tracking with MLflow (``tracking.mlflow: true`` in the settings).

Install with ``uv sync --extra tracking``. Runs are stored in a local SQLite file
(``mlflow.db`` in the repository) unless ``MLFLOW_TRACKING_URI`` says otherwise; browse
them with ``uv run mlflow ui --backend-store-uri sqlite:///mlflow.db``.
"""

from __future__ import annotations

import os
from pathlib import Path


def _flatten(data: dict, prefix: str = "") -> dict[str, str]:
    out = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}."))
        else:
            out[name] = str(value)[:500]
    return out


class MlflowTracker:
    def __init__(self, run_name: str, params: dict, root: Path):
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow is not installed. Install it with: uv sync --extra tracking"
            ) from exc
        self.mlflow = mlflow
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').resolve()}")
        mlflow.set_experiment("heuriforge")
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(_flatten(params))

    def attempt(self, child, best) -> None:
        metrics = {}
        if best is not None and best.train_score is not None:
            metrics["best_train_gap"] = best.train_score
        if child.status == "evaluated":
            metrics["train_gap"] = child.train_score
        if child.val_score is not None and child.val_score != float("inf"):
            metrics["val_gap"] = child.val_score
        if child.completion_tokens:
            metrics["completion_tokens"] = child.completion_tokens
        if metrics:
            self.mlflow.log_metrics(metrics, step=int(child.attempt or 0))

    def finish(self, summary: dict, paths) -> None:
        final = {
            k: v
            for k, v in summary.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        self.mlflow.log_metrics({f"final_{k}": float(v) for k, v in final.items()})
        for path in (paths.best, paths.summary, paths.root / "run.yaml"):
            if Path(path).exists():
                self.mlflow.log_artifact(str(path))
        self.mlflow.end_run()
