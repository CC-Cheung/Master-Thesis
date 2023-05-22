import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import logging


def make_class_data(num_domains, num_points, coef=None, permute=None):
    # domain, point, x_dim
    x = np.arange(num_points).reshape((1, num_points, 1)) / 20
    x = x.repeat(num_domains, axis=0)

    mask_sep=coef[:,:1, np.newaxis].repeat(num_points, axis=1)
    mask_flip=coef[:,1:, np.newaxis].repeat(num_points, axis=1)

    y = (x>mask_sep)==mask_flip
    #all_points, dim
    return [torch.Tensor(i.reshape((num_points * num_domains, -1))) for i in [x, y, x[:, permute, :], y[:, permute, :]]]


def make_class_data_val(num_points, num_exist, coef=None):

    # point
    x = np.arange(num_points) / 20
    mask_sep = coef[0].repeat(num_points)
    mask_flip = coef[1].repeat(num_points)
    y = (x>mask_sep)==mask_flip

    num_non_exist = num_points - num_exist

    sep=(num_points-1)//(num_exist-1)
    rem=num_points%num_exist

    idx_exist=[i for i in range(num_points) if i % sep == 0 and i<num_points-rem-sep or i==num_points-1]
    idx_not=[i for i in range(num_points) if (i % sep != 0 or i>=num_points-rem-sep) and i!=num_points-1]

    #non_exist, exist, x_dim
    x_exist = x[idx_exist]
    x_exist = x_exist[np.newaxis, :,np.newaxis].repeat(num_non_exist, axis=0)

    y_exist = y[idx_exist]
    y_exist = y_exist[np.newaxis, :,np.newaxis].repeat(num_non_exist, axis=0)

    #non_exist, exist, x_dim
    x_not = x[idx_not]
    x_not = x_not[:, np.newaxis, np.newaxis].repeat(num_exist, axis=1)

    y_not = y[idx_not]

    #non_exist, y_dim
    y_not = y_not[:, np.newaxis]
    return [torch.Tensor(i) for i in [x_not, y_not, x_exist, y_exist]]


class linear_relu(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
        for i in range(depth):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ReLU())

        self.net.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class BaselineProf(pl.LightningModule):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.val_loss=nn.MSELoss()

    def forward(self, x1, x2):
        a = self.lin(x1)
        b = self.lin(x2)
        result = torch.linalg.norm(a - b, dim=-1, keepdim=True) + 1e-12
        return result

    def on_train_start(self):
        self.log("hp/epochs", self.trainer.max_epochs)
        self.log( "hp/num_domains", num_domains)
        self.log( "hp/num_points", num_points)
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        #
        x1, y1, x2, y2 = batch
        distance = self(x1, x2)
        loss = (torch.abs(y1 - y2) / distance).mean()+torch.norm(distance)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        for i, weight in enumerate(self.lin.weight):
            self.log(f"weights {i}", weight)

        return loss

    def validation_step(self, batch, batch_idx):
        # not exist, exist, x_dim
        x_not, y_not, x_exist, y_exist = batch

        # not exist, exist, 1
        distance = self(x_exist, x_not)
        weight = (1 / (distance.abs() + 1e-12))
        normalized_weights = torch.nn.functional.normalize(weight, 1, 1)
        # not_exist,1
        out = (y_exist * normalized_weights).sum(dim=1)

        loss = self.val_loss(out, y_not)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        self.log("val_inaccuracy", (out.round()-y_not).abs().mean())
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    # define any number of nn.Modules (or use your current ones)
    in_dim = 1
    hidden_dim = 4
    out_dim = 1
    depth = 4
    num_domains = 2
    train_coef = np.array([[0.1, False], [0.1, True]])

    # num_domains = 3
    #
    # train_coef = np.array([[0.1, False], [0.1, True],[0.2, True]])

    # train_coef = np.array([[0, 0, 0], [0, 0, 10], [0, 0, 100], [0, 0, 1000]])

    val_coef = np.array([0.1, False])
    permute = [5, 6, 4, 7, 3, 8, 2, 9, 1, 0]
    num_points = 10
    # permute = [13,11,5, 6, 4, 7, 3, 8, 2, 9, 1, 0,10, 14,12]
    # num_points = 15
    num_val_io_pairs = 3


    train_dataset = TensorDataset(*make_class_data(num_domains, num_points, coef=train_coef, permute=permute))
    val_dataset = TensorDataset(*make_class_data_val(num_points, num_val_io_pairs, coef=val_coef))

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10)

    # configure logging at the root level of Lightning
    # logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/trash")
    # model_checkpoint=ModelCheckpoint(save_top_k=2, monitor="val_loss")
    base=BaselineProf(in_dim, out_dim)
    base=BaselineProf.load_from_checkpoint(
        "lightning_logs/trash/lightning_logs/version_4/checkpoints/epoch=999-step=2000.ckpt",
        in_dim=in_dim, out_dim=out_dim)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=1000)
    trainer1 = pl.Trainer(limit_train_batches=100,
                          logger=tb_logger,
                          max_epochs=1000,
                          log_every_n_steps = 5,
                          accelerator="gpu", devices=1,
                          callbacks=[early_stop_callback,
                                     # model_checkpoint
                                     ],
                          fast_dev_run=True
                          )
    # trainer2 = pl.Trainer(limit_train_batches=100,
    #                  logger=tb_logger,
    #                   max_epochs=2400,
    #                   log_every_n_steps = 1,
    # accelerator = "gpu", devices = 1,
    #
    # # fast_dev_run=True
    #
    #                   )

    # idea_module=Idea.load_from_checkpoint("/content/drive/MyDrive/Masters/Thesis/lightning_logs/version_5/checkpoints/epoch=1199-step=2400.ckpt")
    #
    # idea_module

    # trainer1.fit(idea_module2, train_loader,val_loader)

    trainer1.fit(base, train_loader, val_loader)

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir /content/drive/MyDrive/Masters/Thesis/lightning_logs --port=8008
#
# val_dataset[6]
#
# train_dataset[0]
#
# idea_module2(val_dataset[0:1][0],val_dataset[0:1][2] )

# val_dataset[0:1][1]


# Commented out IPython magic to ensure Python compatibility.
# %debug
