# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J7bxCUtpGHCEwgT1_1m_lRL3c0rIo2t4
"""


import os
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

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
from scipy.integrate import odeint
import pandas as pd





def make_water_data(raw, data_length, drop_columns):
    df_raw = [pd.read_csv(name).drop(drop_columns,1).iloc[::20, :] for name in raw]
    # df_raw = [pd.read_csv(name).loc[:,[
    #                                       # "rain_t_2",
    #                                    "s_t_2.1.nsteps.", "Ctot", "Ntot"]] for name in raw]

    all_domains=pd.concat(df_raw,axis=0)
    mean=all_domains.mean(axis=0)
    std=all_domains.std(axis=0)
    result=[]
    for domain in df_raw:
        normalized_domain=((domain - mean) / std).values


        # temp=[(normalized_domain[i:i+data_length, 0:1], normalized_domain[i:i+data_length, 1:])
        #  for i in range(0, len(domain), data_length)]
        # return [torch.tensor(i).float() for i in list(zip(*temp))]
        temp=np.stack([normalized_domain[i:i+data_length]
         for i in range(0, len(domain), data_length//20)][:-20])
        result.append(torch.tensor(temp).float())
    return result


    # result=[]
    # for i in range(num_domains):
    #     result.append(x[i:i+1, :, :])
    #     result.append(y[i:i+1, :, :])
    # #[(num_points, in_dim), num_points(out_dim)]
    # return result


class LinearRelu(nn.Module):
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
class LinearElu(nn.Module):
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
class Linear(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for i in range(depth):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))

        self.net.append(nn.Linear(hidden_dim, out_dim))
        self.net=nn.Sequential(*self.net)
    def forward(self, x):
        return self.net(x)

class SequenceModule(nn.Module):
    def __init__(self, in_dim, seq_hidden_dim, init_hidden_dim, out_dim, warmup_length):
        super().__init__()
        self.num_layers=2
        self.seq_hidden_dim=seq_hidden_dim
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.warmup_length=warmup_length

        self.lstm=nn.LSTM(in_dim, hidden_size=seq_hidden_dim, num_layers=self.num_layers, proj_size=out_dim, batch_first=True)
        self.lstm_warmup=nn.LSTM(in_dim+out_dim,
                                 hidden_size=init_hidden_dim,
                                 proj_size=out_dim,
                                 num_layers=self.num_layers,
                                 batch_first=True
                                 )
    def forward(self, x):
        #batch, length, dim
        _, (h0, c0) =self.lstm_warmup(x[:, :self.warmup_length])
        return self.lstm(x[:,self.warmup_length:,:self.in_dim], (h0, c0))


class TrainInvariant(pl.LightningModule):
    def __init__(self, in_dim, f_embed_dim,  g_embed_dim, out_dim, num_domains, warmup_length):
        super().__init__()
        self.in_dim=in_dim
        self.warmup_length=warmup_length
        self.invariant=SequenceModule(in_dim, f_embed_dim, f_embed_dim, out_dim, warmup_length)
        self.train_variants=nn.ModuleList([SequenceModule(in_dim, g_embed_dim, g_embed_dim, out_dim, warmup_length) for i in range (num_domains)])
        self.test_variant=SequenceModule(in_dim, g_embed_dim, g_embed_dim, out_dim, warmup_length)
        self.num_domains=num_domains
        self.eta=lambda x,y: x+y

        self.loss=nn.MSELoss()
    def forward(self, x, domain_num):
        result_f = self.invariant(x)
        result_g =self.train_variants[domain_num](x)

        return self.eta(result_f[0], result_g[0])

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        losses=[self.loss(self(batch[i], i), batch[i][:, self.warmup_length:,self.in_dim:]) for i in range (self.num_domains)]
        loss=0
        for i in range(self.num_domains):
            loss+=losses[i]
        loss=loss/len(losses)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) :
        losses = [self.loss(self(batch[i], i), batch[i][:, self.warmup_length:,self.in_dim:]) for i in range(self.num_domains)]
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



def get_latest_file():
    list_of_files = glob.glob(os.path.dirname(__file__) + '/DA Thesis/**/*.ckpt',
                              recursive=True)  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file
if __name__=="__main__":
    # define any number of nn.Modules (or use your current ones)
    drop_columns=["Unnamed: 0", "time2.1.nsteps.",
                  "rain_t_2"]
    all_data=[]
    # for file in os.listdir("data"):
    #     one_domain = pd.read_csv("data/"+file)
    #     one_domain=one_domain.drop("Unnamed: 0",1)
    #     all_data[file]=one_domain

    raw = [
        # "data/data_exportrate=0.125.csv",
        #    "data/data_exportconstrain.csv",
        # "data/data_exportsindiv30rain.csv",
        "data/data_exportCN.add=25.csv",
        "data/data_exportCN.add=40.csv",
        "data/data_exportCN.add=55.csv",
        # "data/data_exportCN.add=70.csv",
        "data/data_exportCN.add=85.csv",
        "data/data_exportCN.add=100.csv",

    ]


    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    key="c20d41ecf28a9b0efa2c5acb361828d1319bc62e"


    predict_length = 200
    warmup_length=100
    data_length= predict_length + warmup_length
    f_embed_dim = 1000
    g_embed_dim=100
    # f_layers=2
    # g_layers=2
    max_epoch=3

    all_dataset = TensorDataset(*make_water_data(raw=raw, data_length=data_length, drop_columns=drop_columns))
    in_dim = 1
    out_dim = 10
    num_domains=len(raw)
    split_fraction=0.7
    split_point=int(len(all_dataset)*split_fraction)

    #no random
    train_dataset = all_dataset[:split_point]
    val_dataset = all_dataset[:split_point]

    #random
    train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, (split_fraction,1-split_fraction))

    # train_dataset = TensorDataset(*make_quad_data_val(num_val_io_pairs, val_coef))
    # val_dataset = TensorDataset(*make_quad_data_val(num_points, val_coef))

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=20)
    # val_loader=train_loader
    wandb_logger=WandbLogger(project="DA Thesis", name="trash",log_model="True")
    wandb.init() ########################################################
    wandb_logger.experiment.config.update({
                                           "num_domains": num_domains,
                                           "f_embed_dim":f_embed_dim,
                                           "g_embed_dim": g_embed_dim,
                                           "max_epoch": max_epoch,
                                           "warmup_length": warmup_length,
                                           "file":os.path.basename(__file__)})

    base_trans=TrainInvariant(in_dim, f_embed_dim=f_embed_dim, g_embed_dim=g_embed_dim,out_dim=out_dim,num_domains=num_domains, warmup_length=warmup_length)
    #

    # base_trans = TrainInvariant.load_from_checkpoint(get_latest_file(),in_dim=in_dim, f_embed_dim=f_embed_dim,
    #                                                  g_embed_dim=g_embed_dim,
    #                                                  out_dim=out_dim,
    #                                                  num_domains=num_domains)
    base_trans = TrainInvariant.load_from_checkpoint("epoch=462-step=7871.ckpt",
                                                    in_dim=in_dim,
                                                     f_embed_dim=f_embed_dim,
                                                     g_embed_dim=g_embed_dim,
                                                     out_dim=out_dim,
                                                     num_domains=num_domains,
                                                     warmup_length=warmup_length)


    wandb_logger.watch(base_trans, log="all")

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=50)
    # small_error_callback = EarlyStopping(monitor="val_loss", stopping_threshold=0.02)
    model_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",

    )
    trainer1 = pl.Trainer(limit_train_batches=100,
                          logger=wandb_logger,
                          max_epochs=max_epoch,
                          # log_every_n_steps = 5,
                          accelerator="gpu", devices=1,
                          callbacks=[
                              model_callback,
                              early_stop_callback,
                                     # small_error_callback
                                     # model_checkpoint
                                     ],
                      # fast_dev_run=True
                      )



    #
    # idea_module

    # trainer1.fit(idea_module2, train_loader,val_loader)

    trainer1.fit(base_trans, train_loader, val_loader)

    wandb.save(get_latest_file())

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