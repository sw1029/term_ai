from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from term_ai.augmentation.pipeline import prepare_artifacts


@hydra.main(version_base=None, config_path="../../configs/augmentation", config_name="default")
def main(cfg: DictConfig) -> None:
    manifest = prepare_artifacts(
        input_path=Path(cfg.data.input_path),
        output_dir=Path(cfg.data.output_dir),
        seed=int(cfg.data.seed),
    )
    print(OmegaConf.to_yaml(OmegaConf.create(manifest), resolve=True))


if __name__ == "__main__":
    main()
