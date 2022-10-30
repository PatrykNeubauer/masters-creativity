import pytorch_lightning as pl
import omegaconf

def get_wandb_logger(trainer: pl.Trainer) -> pl.loggers.WandbLogger:
    """
    Gets WandbLogger from the trainer object.
    """
    if isinstance(trainer.logger, pl.loggers.WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, pl.loggers.LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, pl.loggers.WandbLogger):
                return logger

    raise Exception(
        "Wandb callback used without a WandbLogger."
    )


class LogCode(pl.Callback):
    """
    Logs the code to w&b.
    """
    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment
        experiment.log_code(self.code_dir)


class LogHydraParameters(pl.Callback):
    """
    Logs Hydra configs as parameters to w&b.
    """
    def __init__(self):
        self.config_path = './.hydra/config.yaml'

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        conf = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(self.config_path), resolve=True, throw_on_missing=True
            )
        logger.log_hyperparams(conf)
