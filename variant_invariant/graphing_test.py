# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J7bxCUtpGHCEwgT1_1m_lRL3c0rIo2t4
"""


import os
import torch

from torch import optim, nn, utils, Tensor

import pytorch_lightning as pl
import numpy as np

import glob
import matplotlib.pyplot as plt

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



class TestInvariant(pl.LightningModule):


    def __init__(self, in_dim, f_embed_dim, g_embed_dim, out_dim, num_domains):
        super().__init__()
        self.invariant = linear_elu(in_dim, f_embed_dim, out_dim, 2)
        self.train_variants = nn.ModuleList([linear_elu(in_dim, g_embed_dim, out_dim, 2) for i in range(num_domains)])
        self.test_variant = linear_elu(in_dim, g_embed_dim, out_dim, 2)
        self.num_domains = num_domains
        self.eta = lambda x, y: x + y

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

def get_latest_file():
    list_of_files = glob.glob(os.path.dirname(__file__) + '/DA Thesis/**/*.ckpt',
                              recursive=True)  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file
if __name__=="__main__":
    # define any number of nn.Modules (or use your current ones)


    in_dim = 1
    out_dim = 1
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    key="c20d41ecf28a9b0efa2c5acb361828d1319bc62e"
    f_embed_dim = 8
    g_embed_dim = 2

    model_names=["variant.ckpt",
                 "nofreeze.ckpt",
                 # "9000.ckpt",
                 # "12000.ckpt",
                 ]
    # model_names = ["epoch=2999-step=6000.ckpt"
    #                ]
    x=torch.arange(0,1,0.05).unsqueeze(dim=1)




    # plot the data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    all_fs=[]
    all_gs=[]
    weird=[lambda x: np.e**x, lambda x:x**2, lambda x:np.sin(10*x)]
    gen_coef = np.array([1.3528104691935328, 1.080031441673445,1.195747596821148])
    val_dataset=(np.array([[0.4237],
                    [0.6459],
                    [0.4376],
                    [0.8918],
                    [0.9637]]),
                 np.array([[1.1974],
                    [3.2404],
                    [1.1736],
                    [4.7397],
                    [4.2976]]))
    for name in model_names:
        base_trans = TestInvariant.load_from_checkpoint(os.path.dirname(__file__) + '/' + name,
                                                        in_dim=in_dim, f_embed_dim=f_embed_dim, g_embed_dim=g_embed_dim,
                                                        out_dim=out_dim, num_domains=10)

        f_data = base_trans.invariant(x).detach().cpu().numpy()
        g_data = base_trans.test_variant(x).detach().cpu().numpy()
        actual = np.array([(gen_coef[j] * weird[j](x)) for j in range(gen_coef.shape[0])]).sum(axis=0)

        plt.plot(x, f_data, color= "red", label="f")
        plt.plot(x, g_data, color="green", label="g")
        plt.plot(x, g_data+f_data, color="blue", label="f+g")
        plt.plot(x,actual, color='black', label="actual")
        plt.scatter(val_dataset[0], val_dataset[1], c="g", label="actual")

        plt.title(name)
        plt.legend()
        plt.show()


    color = ["red", "orange", "green", "blue", "purple", "gray"]

