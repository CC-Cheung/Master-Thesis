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
    x=np.random.rand(num_domains, num_points, in_dim)
    x_reshaped=x.reshape(in_dim, num_points, num_domains)
    y=(coef[:,0]*np.e**x_reshaped + \
      coef[:,1] *x_reshaped**2 + \
      coef[:,2]*np.sin(10*x_reshaped))\
        .reshape(num_domains, num_points, in_dim)
    #coef will be num_domain, 3
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
    x=np.random.rand(num_points, in_dim)

    y=(coef[0]*np.e**x+ \
      coef[1] *x**2 + \
      coef[2]*np.sin(10*x)).sum(axis=0)\

    #coef will be 3
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
# class reshape_linear_relu(linear_relu):
#     def __init__(self, *args, **kwargs):
#         super().__init__( *args, **kwargs)
#     def forward(self, x):
#         shape=x.shape
#         return self.net(x.reshape(shape[0]*shape[1], shape [2])).reshape(shape[0], shape[1], -1)






class TrainInvariant(pl.LightningModule):
    def __init__(self, in_dim, f_embed_dim, g_embed_dim, out_dim, num_domains):
        super().__init__()
        self.invariant=linear_elu(in_dim, f_embed_dim, out_dim, 2)
        self.train_variants=nn.ModuleList([linear_elu(in_dim, g_embed_dim, out_dim, 2) for i in range (num_domains)])
        self.test_variant=linear_elu(in_dim, g_embed_dim, out_dim,2)
        self.num_domains=num_domains
        self.eta=lambda x,y: x+y

        self.loss=nn.MSELoss()
    def forward(self, x, domain_num):
        result_f = self.invariant(x)
        result_g =self.train_variants[domain_num](x)

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

        self.invariant = linear_elu(in_dim, f_embed_dim, out_dim,2)

        self.test_variant=linear_elu(in_dim, g_embed_dim, out_dim,2)
        self.train_variants = nn.ModuleList([linear_elu(in_dim, g_embed_dim, out_dim, 2) for i in range(num_domains)]) #not used
        self.num_domains=num_domains #not used

        self.eta=lambda x,y: x+y
        self.loss=nn.MSELoss()

    def forward(self, x):
        result_f = self.invariant(x)
        result_g =self.test_variant(x)

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
    train_coef = np.array([[1, 0, 1],[0, 1, 1], [1, 1, 0], [0.5, 0.5, 0.5],[0.25, 0.25, 0.25]])
    train_coef = np.random.rand(100,3)

    val_coef = np.array([0.8, 0.1, .2])
    num_points = 100
    num_domains = train_coef.shape[0]
    num_val_io_pairs=5
    f_embed_dim = 8
    g_embed_dim=2
    max_epoch=500

    train_dataset = TensorDataset(*make_quad_data(num_domains, num_points,train_coef))
    val_dataset = TensorDataset(*make_quad_data(num_domains, num_points,train_coef))

    # train_dataset = TensorDataset(*make_quad_data_val(num_points,val_coef))
    # val_dataset = TensorDataset(*make_quad_data_val(num_points,val_coef))

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=20)


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

    base_trans=TrainInvariant(in_dim, f_embed_dim=f_embed_dim, g_embed_dim=g_embed_dim,out_dim=out_dim,num_domains=1)

    list_of_files = glob.glob(os.path.dirname(__file__) + '/DA Thesis/**/*.ckpt',
                              recursive=True)  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    # base_trans = TrainInvariant.load_from_checkpoint(in_dim, f_embed_dim=f_embed_dim,
    #                                                  g_embed_dim=g_embed_dim,
    #                                                  out_dim=out_dim,
    #                                                  num_domains=1)
    # base_trans = DoubleTrans.load_from_checkpoint("epoch=2999-step=15000.ckpt",
    #                                                 in_dim=in_dim + out_dim, embed_dim=embed_dim, out_dim=3,
    #                                                 num_heads=num_heads)

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
    list_of_files = glob.glob(os.path.dirname(__file__) + '/DA Thesis/**/*.ckpt',
                              recursive=True)  # * means all if need specific format then *.csv
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

# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J7bxCUtpGHCEwgT1_1m_lRL3c0rIo2t4
"""