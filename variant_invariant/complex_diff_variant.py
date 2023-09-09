# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J7bxCUtpGHCEwgT1_1m_lRL3c0rIo2t4
"""


import os
import torch
from pytorch_lightning.callbacks import EarlyStopping

import wandb
from torch import optim, nn, utils, Tensor

import pytorch_lightning as pl
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
import glob
import StormReactor


def make_weird_data_val(num_points, coef):
    full=coef
    for i in range(num_points):
        full=np.concatenate((full,(full[i:i+1]-full[i+1:i+2]+0.5)/2+np.sin(i)))

    x = full[:num_points]
    y = full[ 2:]

    #coef will be 3
    #y will be num_points, out_dim=1

    x = torch.Tensor(x).unsqueeze(dim=0)
    y = torch.Tensor(y).unsqueeze(dim=0)
    return x,y
class linear_relu(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
        for i in range(depth):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ReLU())

        self.net.append(nn.Linear(hidden_dim, out_dim))
        self.net=nn.Sequential(*self.net)
    def forward(self, x):
        return self.net(x)
class linear_elu(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(in_dim, hidden_dim), nn.ELU()])
        for i in range(depth):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ELU())

        self.net.append(nn.Linear(hidden_dim, out_dim))
        self.net=nn.Sequential(*self.net)
    def forward(self, x):
        return self.net(x)
class linear(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
        for i in range(depth):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))

        self.net.append(nn.Linear(hidden_dim, out_dim))
        self.net=nn.Sequential(*self.net)
    def forward(self, x):
        return self.net(x)




class TestInvariant(pl.LightningModule):

    def __init__(self, in_dim, f_embed_dim, g_embed_dim, out_dim, num_domains):
        super().__init__()
        self.invariant = nn.LSTM(in_dim, hidden_size=f_embed_dim, num_layers=2, proj_size=out_dim, batch_first=True)
        self.train_variants = nn.ModuleList(
            [nn.LSTM(in_dim, hidden_size=g_embed_dim, num_layers=2, proj_size=out_dim, batch_first=True) for i in
             range(num_domains)])
        self.test_variant = nn.LSTM(in_dim, hidden_size=g_embed_dim, num_layers=2, proj_size=out_dim, batch_first=True)
        self.num_domains = num_domains
        self.eta = lambda x, y: x + y

        self.loss = nn.MSELoss()

    def forward(self, x):
        result_f = self.invariant(x)
        result_g =self.test_variant(x)

        return self.eta(result_f[0], result_g[0])

    def training_step(self, batch, batch_idx):
        loss=self.loss(self(batch[0]), batch[1])
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) :
        loss = self.loss(self(batch[0]), batch[1])
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        return optimizer
def get_latest_file():
    list_of_files = glob.glob(os.path.dirname(__file__) + '/DA Thesis/**/*.ckpt',
                              recursive=True)  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file
if __name__=="__main__":
    # define any number of nn.Modules (or use your current ones)
    in_dim = 1
    out_dim = 1
    depth = 2
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    key="c20d41ecf28a9b0efa2c5acb361828d1319bc62e"

    val_coef = np.random.rand(2, in_dim)

    num_points = 50
    num_val_io_pairs=10
    f_embed_dim = 20
    g_embed_dim=10
    max_epoch=1500

    # train_dataset = TensorDataset(*make_quad_data(num_domains, num_points,train_coef))
    # val_dataset = TensorDataset(*make_quad_data(num_domains, num_points,train_coef))

    train_dataset = TensorDataset(*make_weird_data_val(num_val_io_pairs, val_coef))
    val_dataset = TensorDataset(*make_weird_data_val(num_points,val_coef))

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=20)


    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/trash")
    wandb_logger=WandbLogger(project="DA Thesis", name="trash",log_model="True")
    wandb.init() ########################################################
    wandb_logger.experiment.config.update({
                                           "val_coef": val_coef,
                                           "num_points": num_points,

                                           "f_embed_dim":f_embed_dim,
                                           "g_embed_dim": g_embed_dim,
                                           "max_epoch": max_epoch,
                                            "num_val_io_pairs":num_val_io_pairs,
                                           "file":os.path.basename(__file__)})

    # base_trans=TestInvariant(in_dim, f_embed_dim=f_embed_dim, g_embed_dim=g_embed_dim,out_dim=out_dim,num_domains=1)

    # list_of_files = glob.glob('DA Thesis/**/*.ckpt', recursive=True)  # * means all if need specific format then *.csv
    # latest_file = max(list_of_files, key=os.path.getctime)
    # base_trans=TestInvariant.load_from_checkpoint(latest_file,
    #                                             in_dim=in_dim, f_embed_dim=f_embed_dim, g_embed_dim=g_embed_dim,out_dim=out_dim,num_domains=1)


    base_trans = TestInvariant.load_from_checkpoint(os.path.dirname(__file__)+"/variant.ckpt",
                                                    in_dim=in_dim, f_embed_dim=f_embed_dim,
                                                    g_embed_dim=g_embed_dim,out_dim=out_dim,num_domains=10)    #
    # for param in base_trans.invariant.parameters():
    #     param.requires_grad = False
    # base_trans.invariant.eval()


    wandb_logger.watch(base_trans, log="all")

    # early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=300)

    trainer1 = pl.Trainer(limit_train_batches=100,
                          logger=wandb_logger,
                          max_epochs=max_epoch,
                          # log_every_n_steps = 5,
                          accelerator="gpu", devices=1,
                          callbacks=[
                              # early_stop_callback,
                                     # model_checkpoint
                                     ],
                    #   fast_dev_run=True
                      )

    trainer1.fit(base_trans, train_loader, val_loader)

    wandb.save(get_latest_file())