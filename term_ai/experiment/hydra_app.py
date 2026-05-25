from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from term_ai.experiment.runner import create_run_dir, init_matrix, write_resume_state, write_run_manifest


@hydra.main(version_base=None, config_path="../../configs/experiment", config_name="default")
def main(cfg: DictConfig) -> None:
    matrix_path = Path(cfg.model.matrix_path)
    init_matrix(matrix_path)

    run_dir = create_run_dir(Path(cfg.logging.runs_dir), str(cfg.model.experiment_id))
    config_snapshot = OmegaConf.to_container(cfg, resolve=True)
    write_run_manifest(run_dir, str(cfg.model.experiment_id), config_snapshot)
    write_resume_state(run_dir, {"stage": "initialized", "hydra_config_snapshot": True})
    print(OmegaConf.to_yaml(OmegaConf.create({"run_dir": str(run_dir), "matrix_path": str(matrix_path)}), resolve=True))


if __name__ == "__main__":
    main()
