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
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
import glob
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

def make_quad_data(num_domains, num_points, coef):
#make double range
    x=np.random.rand(num_domains, num_points, in_dim)*2
    x_transpose=x.transpose(2, 1, 0)
    #coef num_domains, powers
    y=np.array([coef[:, i] * x_transpose ** i
                for i in range(coef.shape[1])]).sum(axis=0)\
        .transpose(2, 1, 0)

    #y will be num_domains, num_points, out_dim=1
    x=torch.Tensor(x)
    y=torch.Tensor(y)
    result=[]
    for i in range(num_domains):
        result.append(x[i, :, :])
        result.append(y[i, :, :])
    #[(num_points, in_dim), num_points(out_dim)]
    return result
def make_quad_data_val(num_points, coef):
    x=np.random.rand(num_points, in_dim)*2
    y=np.array([(coef[i]*x**i) for i in range(coef.shape[1])]).sum(axis=0)
    #y will be num_points, out_dim=1
    x=torch.Tensor(x)
    y=torch.Tensor(y)
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
        self.net = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for i in range(depth):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))

        self.net.append(nn.Linear(hidden_dim, out_dim))
        self.net=nn.Sequential(*self.net)
    def forward(self, x):
        return self.net(x)


class TrainInvariant(pl.LightningModule):
    def __init__(self, in_dim, f_embed_dim, g_embed_dim, out_dim, num_domains):
        super().__init__()
        self.invariant=nn.Linear(f_embed_dim, out_dim)
        self.train_variants=nn.ModuleList([nn.Linear(g_embed_dim, out_dim) for i in range (num_domains)])
        #not used
        self.test_variant=nn.Linear(g_embed_dim, out_dim)
        self.num_domains=num_domains
        self.eta=lambda x,y: x+y

        self.loss=nn.MSELoss()
    def forward(self, x, domain_num):
        #num_points, in_dim
        powers_f = torch.cat([x ** i for i in range(1,f_embed_dim+1)], dim=1)
        powers_g = torch.cat([x ** i for i in range(1,g_embed_dim+1)], dim=1)

        result_f = self.invariant(powers_f)
        result_g =self.train_variants[domain_num](powers_g)

        return self.eta(result_f, result_g)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        losses=[self.loss(self(batch[2*i], i), batch[2*i+1]) for i in range (self.num_domains)]
        loss=0
        for i in range(self.num_domains):
            loss+=losses[i]
        loss=loss/len(losses)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) :
        losses = [self.loss(self(batch[2*i], i), batch[2*i+1]) for i in range(self.num_domains)]

        loss = 0
        for i in range(self.num_domains):
            loss += losses[i]
        loss = loss / len(losses)        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
    # def on_fit_end(self):

    def configure_optimizers(self):
        #check if all there
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        return optimizer

class TestInvariant(pl.LightningModule):

    def __init__(self, in_dim, f_embed_dim, g_embed_dim, out_dim, num_domains):
        super().__init__()
        self.invariant = nn.Linear(f_embed_dim, out_dim)
        self.train_variants = nn.ModuleList([nn.Linear(g_embed_dim, out_dim) for i in range(num_domains)])
        # not used
        self.test_variant = nn.Linear(g_embed_dim, out_dim)
        self.num_domains = num_domains
        self.eta = lambda x, y: x + y

        self.loss = nn.MSELoss()

    def forward(self, x, domain_num):
        # num_points, in_dim
        powers_f = torch.cat([x ** i for i in range(1,f_embed_dim+1)], dim=1)
        powers_g = torch.cat([x ** i for i in range(1,g_embed_dim+1)], dim=1)

        result_f = self.invariant(powers_f)
        result_g = self.test_variant(powers_g)

        return self.eta(result_f, result_g)

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
    hidden_dim_nature = 3
    hidden_dim_pred = 4
    num_heads = 1
    out_dim = 1
    depth = 2
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    key="c20d41ecf28a9b0efa2c5acb361828d1319bc62e"
    f_embed_dim = 6
    g_embed_dim = 2
    gen_dim=4
    num_domains = 50
    # train_coef = np.array([[1, 0, 1, 1],[0, 1, 1,1.5], [1, 1, 0, 1], [0.5, 0.5, 0.5,0.5],[0.25, 0.25, 0.25,0.25]])
    train_coef = np.concatenate((np.random.rand(num_domains,g_embed_dim+1),
                                 np.random.normal(loc=1, scale=0, size=(num_domains, gen_dim-g_embed_dim))), axis=1)

    val_coef = np.array([0.8, 0.1, .2])
    num_domains = train_coef.shape[0]

    num_points = 100
    num_val_io_pairs=100

    max_epoch=3000

    train_dataset = TensorDataset(*make_quad_data(num_domains, num_points,train_coef))
    val_dataset = TensorDataset(*make_quad_data(num_domains, num_points,train_coef))

    # train_dataset = TensorDataset(*make_quad_data_val(num_val_io_pairs, val_coef))
    # val_dataset = TensorDataset(*make_quad_data_val(num_points, val_coef))

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=50)


    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/trash")
    wandb_logger=WandbLogger(project="DA Thesis", name="trash",log_model="True")
    wandb.init() ########################################################
    wandb_logger.experiment.config.update({"train_coef": train_coef,
                                           "val_coef": val_coef,
                                           "num_points": num_points,
                                           "num_domains": num_domains,
                                           "f_embed_dim":f_embed_dim,
                                           "g_embed_dim": g_embed_dim,
                                           "max_epoch": max_epoch,
                                           "file":os.path.basename(__file__)})

    base_trans=TrainInvariant(in_dim, f_embed_dim=f_embed_dim, g_embed_dim=g_embed_dim,out_dim=out_dim,num_domains=num_domains)
    # base_trans=TestInvariant(in_dim, g_embed_dim=g_embed_dim,out_dim=out_dim)

    # list_of_files = glob.glob('DA Thesis/**/*.ckpt', recursive=True)  # * means all if need specific format then *.csv
    # latest_file = max(list_of_files, key=os.path.getctime)
    # base_trans=TestInvariant.load_from_checkpoint(latest_file,
    #                                             in_dim=in_dim, g_embed_dim=g_embed_dim, out_dim=out_dim)
    base_trans = TrainInvariant.load_from_checkpoint(os.path.dirname(__file__) + '/epoch=1999-step=4000.ckpt',
        in_dim=in_dim, f_embed_dim=f_embed_dim, g_embed_dim=g_embed_dim,out_dim=out_dim,num_domains=num_domains)

    #Zeroing f
    # base_trans.invariant.weight.data.fill_(0.00)
    # base_trans.invariant.bias.data.fill_(0.00)
    # for param in base_trans.invariant.parameters():
    #     param.requires_grad = False
    # base_trans.invariant.eval()

    #
    # base_trans.invariant.weight.data=torch.tensor([[0,0.0,1,1,0,0]])
    # base_trans.invariant.bias.data=torch.tensor([0.0])
    #
    # base_trans.train_variants[0].weight.data=torch.tensor(train_coef[:1, 1:3],dtype=torch.float32)
    # base_trans.train_variants[0].bias.data=torch.tensor(train_coef[:1, 0],dtype=torch.float32)


    wandb_logger.watch(base_trans, log="all")

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=1000)

    trainer1 = pl.Trainer(limit_train_batches=100,
                          logger=wandb_logger,
                          max_epochs=max_epoch,
                          # log_every_n_steps = 5,
                          accelerator="gpu", devices=1,
                          callbacks=[early_stop_callback,
                                     # model_checkpoint
                                     ],
                    #   fast_dev_run=True
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


    #
    # idea_module

    # trainer1.fit(idea_module2, train_loader,val_loader)

    trainer1.fit(base_trans, train_loader, val_loader)

    wandb.save(get_latest_file())
