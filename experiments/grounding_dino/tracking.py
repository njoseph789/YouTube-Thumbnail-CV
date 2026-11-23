"""MLflow setup shared by run_baseline.py and evaluate.py.

MLflow 3.x blocks the plain-folder (file) backend, so tracking uses a local SQLite
database. Both the database and the artifacts live under the results folder
(gitignored via experiments/*/results/).
"""
from pathlib import Path

import mlflow


def setup_mlflow(out_dir: Path, experiment: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (out_dir / "mlflow.db").resolve().as_posix())
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=(out_dir / "artifacts").resolve().as_uri()
        )
    mlflow.set_experiment(experiment)
