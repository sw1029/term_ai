from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from term_ai.augmentation.pipeline import prepare_artifacts
from term_ai.contracts import normalize_openai_model_id


@hydra.main(version_base=None, config_path="../../configs/augmentation", config_name="default")
def main(cfg: DictConfig) -> None:
    judge_model = cfg.judge.model
    if bool(cfg.judge.enforce_model_separation) and judge_model:
        if normalize_openai_model_id(str(judge_model)) == normalize_openai_model_id(str(cfg.teacher.model)):
            raise ValueError("judge.model must differ from teacher.model when model separation is enabled")
    manifest = prepare_artifacts(
        input_path=Path(cfg.data.input_path),
        output_dir=Path(cfg.data.output_dir),
        seed=int(cfg.data.seed),
    )
    print(OmegaConf.to_yaml(OmegaConf.create(manifest), resolve=True))


if __name__ == "__main__":
    main()
