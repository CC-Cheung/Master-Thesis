import matplotlib.pyplot as plt


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
    df_raw = [pd.read_csv(name).drop(drop_columns,1) for name in raw]
    # df_raw = [pd.read_csv(name).loc[:, [
    #                                        # "rain_t_2",
    #                                        "s_t_2.1.nsteps.", "Ctot", "Ntot"]] for name in raw]

    all_domains=pd.concat(df_raw,axis=0)
    mean=all_domains.mean(axis=0)
    std=all_domains.std(axis=0)
    for domain in df_raw:
        normalized_domain=((domain - mean) / std).values


        # temp=[(normalized_domain[i:i+data_length, 0:1], normalized_domain[i:i+data_length, 1:])
        #  for i in range(0, len(domain), data_length)]
        # return [torch.tensor(i).float() for i in list(zip(*temp))]

        return torch.tensor(normalized_domain).float()



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
    def __init__(self, in_dim, seq_hidden_dim, init_hidden_dim, out_dim, depth=3):
        super().__init__()
        self.num_layers=2
        self.seq_hidden_dim=seq_hidden_dim
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.lstm=nn.LSTM(in_dim, hidden_size=seq_hidden_dim, num_layers=self.num_layers, proj_size=out_dim, batch_first=True)
        self.mlp=LinearElu(in_dim+out_dim, init_hidden_dim, self.num_layers*seq_hidden_dim, depth)

    def forward(self, x):
        init=self.mlp(x[:, :1]).reshape(self.num_layers, x.shape[0], self.seq_hidden_dim)
        return self.lstm(x[:,:,:self.in_dim], (torch.zeros(self.num_layers,x.shape[0],self.out_dim).to("cuda"), init))


class TrainInvariant(pl.LightningModule):
    def __init__(self, in_dim, f_embed_dim,  g_embed_dim, out_dim, num_domains):
        super().__init__()
        self.in_dim=in_dim
        self.invariant=SequenceModule(in_dim, f_embed_dim, f_embed_dim, out_dim)
        self.train_variants=nn.ModuleList([SequenceModule(in_dim, g_embed_dim, g_embed_dim, out_dim) for i in range (num_domains)])
        self.test_variant=SequenceModule(in_dim, f_embed_dim, f_embed_dim, out_dim)
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

        losses=[self.loss(self(batch[i], i), batch[i][:, :,self.in_dim:]) for i in range (self.num_domains)]
        loss=0
        for i in range(self.num_domains):
            loss+=losses[i]
        loss=loss/len(losses)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) :
        losses = [self.loss(self(batch[i], i), batch[i][:, :,self.in_dim:]) for i in range(self.num_domains)]
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
    drop_columns=["Unnamed: 0", "time2.1.nsteps."]
    all_data=[]
    # for file in os.listdir("data"):
    #     one_domain = pd.read_csv("data/"+file)
    #     one_domain=one_domain.drop("Unnamed: 0",1)
    #     all_data[file]=one_domain

    raw = [
        # "data/data_exportrate=0.125.csv",
           # "data/data_exportconstrain.csv",
        # "data/data_exportsindiv30rain.csv",
        "data/data_exportlittlerain.csv"

    ]


    np.random.seed(0)
    torch.manual_seed(0)

    key="c20d41ecf28a9b0efa2c5acb361828d1319bc62e"


    predict_length = 800
    warm_up_length=200
    data_length=predict_length+warm_up_length
    f_embed_dim = 100
    g_embed_dim=20
    # f_layers=2
    # g_layers=2
    max_epoch=1000


    in_dim = 1
    out_dim = 1
    num_domains = 1
    start=0
    end=146000
    tensor=make_water_data(raw, data_length, drop_columns).to("cuda")[start:end]
    # base_trans = TrainInvariant.load_from_checkpoint("epoch=757-step=4548.ckpt",
    #                                                 in_dim=in_dim,
    #                                                  f_embed_dim=f_embed_dim,
    #                                                  g_embed_dim=g_embed_dim,
    #                                                  out_dim=out_dim,
    #                                                  num_domains=num_domains).to("cuda")
    # result=base_trans(tensor.reshape((1,end-start,2)),0)
    # plt.plot(result.flatten().cpu().detach())
    # plt.plot(tensor[:,0].cpu().detach())
    df=pd.read_csv(raw[0]).drop(drop_columns, 1)
    for i in range(2,tensor.shape[1]):
        plt.plot(tensor[:, i].cpu().detach(), label = df.columns[i])

    plt.legend()

    plt.show()