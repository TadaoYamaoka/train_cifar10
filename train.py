from lightning.pytorch.cli import LightningCLI

from data import CIFAR10DataModule
from model import SwinCIFAR10Classifier


def main() -> None:
    LightningCLI(
        model_class=SwinCIFAR10Classifier,
        datamodule_class=CIFAR10DataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
