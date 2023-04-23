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
from pytorch_lightning.loggers import WandbLogger
import glob
import wandb
from torch.autograd import Variable


def make_quad_data(num_points, in_dim, coef=None, permute=None):
    # coef should be num_domain, 2*in_dim+mixed+1, 1

    num_domains, _ = coef.shape

    x = torch.rand(num_domains, num_points, in_dim)

    powers_of_x_temp = [x ** i for i in range(1, 3)]
    # 2| num_domain, num_points, in_dim

    mixed = torch.stack([powers_of_x_temp[0][:, :, i] * powers_of_x_temp[0][:, :, j]
                         for i in range(1, in_dim) for j in range(0, i)]) \
        .reshape((num_domains, num_points, -1))
    # num_domains, num_points, mixed

    powers_of_x = torch.stack(powers_of_x_temp).reshape((num_domains, num_points, -1))
    # num_domains, num_points, 2*in_dim

    terms = torch.concat((powers_of_x, mixed, torch.ones(num_domains, num_points, 1)), dim=-1)
    # num_domains, num_points, 2*in_dim+mixed+1

    y = (terms @ coef).round()
    # num_domains, num_points, 1

    x_perm, y_perm = torch.cat((x, y), dim=-1)[:, torch.randperm(num_points), :] \
        .split(in_dim, dim=-1)

    return x, y, x_perm, y_perm


def make_quad_data_val(num_points, num_exist, in_dim, coef=None):
    # coef should be 2*in_dim+mixed+1, 1

    x = torch.rand(num_points, in_dim)

    powers_of_x_temp = [x ** i for i in range(1, 3)]
    # 2|  num_points, in_dim

    mixed = torch.stack([powers_of_x_temp[0][:, i] * powers_of_x_temp[0][:, j]
                         for i in range(1, in_dim) for j in range(0, i)]) \
        .reshape((num_domains, num_points, -1))
    # num_points, mixed

    powers_of_x = torch.stack(powers_of_x_temp).reshape((num_points, -1))
    # num_points, 2*in_dim

    terms = torch.concat((powers_of_x, mixed, torch.ones(num_points, 1)), dim=-1)
    # num_points, 2*in_dim+mixed+1

    y = (terms @ coef).round()
    #  num_points, 1
    io_pairs=torch.cat((x,y), dim=1)

    num_non_exist=num_points-num_exist


    # non_exist, exist, x_dim
    #check
    x_exist,y_exist = io_pairs[np.newaxis, :num_exist]\
        .repeat(num_non_exist, axis=0).split(in_dim, dim=-1)

    x_not, y_not = io_pairs[num_exist:, np.newaxis] \
        .repeat(num_exist, axis=1).split(in_dim, dim=-1)

    return x_not, y_not, x_exist, y_exist


class LinearKernel(pl.LightningModule):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.kernel = nn.Linear(in_dim, embed_dim)
        self.val_loss = nn.MSELoss()

    def forward(self, x1, x2):
        a = self.kernel(x1)
        b = self.kernel(x2)
        result = torch.linalg.norm(a - b, dim=-1, keepdim=True) + 1e-12

        return result

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        #
        x1, y1, x2, y2 = batch
        distance = self(x1, x2)
        loss = (torch.abs(y1 - y2) / distance).mean()
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        # for i, layer in enumerate(self.kernel.net):
        #     if i%2==0:
        #         self.log(f"weights {i}", layer.weight.mean())
        #         self.log(f"bias {i}", layer.bias.mean())

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
        self.log("val_accuracy", (out.round() - y_not).abs().mean())
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class QuadraticKernel(pl.LightningModule):
    def __init__(self, in_dim, embed_dim):
        super().__init__()

        self.kernel = Variable(torch.rand(2 * in_dim + 1 + (in_dim - 1) * in_dim), embed_dim)
        # in_dim square and singles, 1 for 1, for mixed: in_dim first, in_dim-1 second choice
        self.val_loss = nn.MSELoss()

    def forward(self, x1, x2):
        # batch, in_dim
        batch_size, in_dim = x1.shape

        xs = torch.cat((x1, x2))
        # 2batch, in_dim

        powers_of_x_temp = [xs ** i for i in range(1, 3)]
        # 2|2batch, in_dim

        mixed = torch.stack([powers_of_x_temp[0][:, i] * powers_of_x_temp[0][:, j]
                             for i in range(1, in_dim) for j in range(0, i)]) \
            .reshape((2 * batch_size, -1))
        # 2batch, mixed

        powers_of_x = torch.stack(powers_of_x_temp).reshape((2 * batch_size, 2 * in_dim))

        terms = torch.concat((powers_of_x, mixed, torch.ones(2 * batch_size, 1)), dim=1)
        # 2batch, 2*in_dim+mixed+1

        x1_terms, x2_terms = (terms @ self.kernel).split(batch_size)

        result = torch.linalg.norm(x1_terms - x2_terms, dim=-1, keepdim=True) + 1e-12

        return result

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        #
        x1, y1, x2, y2 = batch
        distance = self(x1, x2)
        loss = (torch.abs(y1 - y2) / distance).mean()
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        # for i, layer in enumerate(self.kernel.net):
        #     if i%2==0:
        #         self.log(f"weights {i}", layer.weight.mean())
        #         self.log(f"bias {i}", layer.bias.mean())

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
        self.log("val_accuracy", (out.round() - y_not).abs().mean())
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    in_dim = 3
    embed_dim = 4
    out_dim = 4
    depth = 4
    num_domains = 3

    train_coef = np.array(num_domains, 2 * in_dim + 1 + (in_dim - 1) * in_dim)

    # train_coef = np.array([[0, 0, 0], [0, 0, 10], [0, 0, 100], [0, 0, 1000]])

    val_coef = np.array([3, 2, 1])
    num_points = 50
    permute = np.random.permutation(num_points)

    num_val_io_pairs = 21
    max_epoch = 500

    train_dataset = TensorDataset(*make_quad_data(num_domains, num_points, coef=train_coef, permute=permute))
    val_dataset = TensorDataset(*make_quad_data_val(num_points, num_val_io_pairs, coef=val_coef))

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10)

    # configure logging at the root level of Lightning
    # logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/prof_mod")
    wandb_logger = WandbLogger(project="DA Thesis", name="prof mod dim more", log_model="True")
    wandb.init()
    wandb_logger.experiment.config.update({"train_coef": train_coef,
                                           "val_coef": val_coef,
                                           "num_points": num_points,
                                           "num_domains": num_domains,
                                           "num_val_io_pairs": num_val_io_pairs,
                                           "in_dim": in_dim,
                                           "max_epoch": max_epoch,
                                           "file": os.path.basename(__file__)})
    # model_checkpoint=ModelCheckpoint(save_top_k=2, monitor="val_loss")
    base = BaselineProfMod(in_dim, hidden_dim, out_dim, depth)
    wandb_logger.watch(base)
    # list_of_files = glob.glob('DA Thesis/**/*.ckpt', recursive=True)  # * means all if need specific format then *.csv
    # latest_file = max(list_of_files, key=os.path.getctime)
    # base=BaselineProfMod.load_from_checkpoint(
    #     latest_file,
    #     in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, depth=depth)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=1000)
    trainer1 = pl.Trainer(limit_train_batches=100,
                          logger=wandb_logger,
                          max_epochs=max_epoch,
                          # log_every_n_steps = 5,
                          accelerator="gpu", devices=1,
                          callbacks=[early_stop_callback,
                                     # model_checkpoint
                                     ],
                          # fast_dev_run=True
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
    list_of_files = glob.glob('DA Thesis/**/*.ckpt', recursive=True)  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    wandb.save(latest_file)
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
