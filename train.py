import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import os

@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig):
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    logger = hydra.utils.instantiate(cfg.logger)
    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.callbacks]

    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    trainer.logger.watch(model)
    trainer.fit(model, datamodule)

if __name__ == '__main__':
    main()