# from pytorch_lightning import LightningModule
import lightning.pytorch as pl
import torch
from torch.optim import AdamW, Adam
import torch.optim as optim
import os

from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


HOME = os.path.expanduser("~")


class LitModule(pl.LightningModule):
    # def __init__(self, args):
    #     super().__init__()
    #     self.model = self.get_model(args)
    #     self.args = args
    #     self.save_hyperparameters()

    def get_model(self, args):
        NotImplementedError

    def _step(self):
        NotImplementedError

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "predict")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        # return optimizer
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def fit(
        self, dataloader, device="cuda", max_epochs=None, max_steps=None, verbose=True
    ):
        import tqdm
        import numpy as np

        self.to(device)
        self.configure_optimizers()
        max_epochs = int(np.ceil(max_steps / len(dataloader)))
        with tqdm(range(max_epochs), total=max_epochs, desc="Epochs") as tq:
            for epoch in tq:
                tk0 = tqdm(
                    enumerate(dataloader),
                    total=len(dataloader),
                    leave=False,
                    desc="Iterations",
                    disable=(not verbose),
                )
                for i, batch in tk0:
                    loss = self._step(batch, i, "train")
                    self.optimizer.zero_grad()
                    loss.backword()
                    self.optimizer.step()

                    tk0.set_postfix_str(f"loss: {loss.item()}")

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def get_metrics(self):
        # don't show the version number
        items = super().get_metrics()
        items.pop("v_num", None)
        return items

    def add_grad_histgram(self):
        for name, params in self.named_parameters():
            if self.logger:
                self.logger.experiment.add_histogram(name, params, self.current_epoch)

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("LitModel")
    #     # parser.add_argument("--encoder_layers", type=int, default=12)
    #     # parser.add_argument("--data_path", type=str, default="/some/path")
    #     return parent_parser

    # @property
    # def example_input_array(self):
    #     # import pdb; pdb.set_trace()
    #     batch = next(iter(self.datamodule.train_dataloader()))
    #     return [batch]

    # def training_epoch_end(self, outputs):
    # self.add_grad_histgram()
    # self.log_dict(outputs[0], on_epoch=True, prog_bar=True)


from pytorch_lightning.callbacks import Callback


class LogGraphCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        if pl_module.example_input_array is not None:
            pl_module.logger.experiment.add_graph(
                pl_module, pl_module.example_input_array
            )


def set_schedule(
    pl_module,
):  # https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/vilt_utils.py
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "nlvr2_classifier"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    names = [n for n, p in pl_module.named_parameters()]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )
