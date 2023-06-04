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
    out_dim = 1
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    key="c20d41ecf28a9b0efa2c5acb361828d1319bc62e"
    f_embed_dim = 6
    g_embed_dim = 2
    gen_dim=4
    num_domains = 50

    model_names=[]
    x=np.arange(0,1,0.05)




    base_trans = TestInvariant.load_from_checkpoint(os.path.dirname(__file__) + '/epoch=4999-step=5000.ckpt',
        in_dim=in_dim, f_embed_dim=f_embed_dim, g_embed_dim=g_embed_dim,out_dim=out_dim,num_domains=num_domains)
    base_trans2 = TestInvariant.load_from_checkpoint(os.path.dirname(__file__) + '/epoch=499-step=500.ckpt',
                                                     in_dim=in_dim, f_embed_dim=f_embed_dim, g_embed_dim=g_embed_dim,
                                                     out_dim=out_dim, num_domains=num_domains)

    f_coef = base_trans.invariant.weight.detach().numpy()
    f_bias = base_trans.invariant.bias.detach().numpy()
    f_coef = np.concatenate((f_bias, f_coef[0,:]))

    f_coef_2 = base_trans2.invariant.weight.detach().numpy()
    f_bias_2 = base_trans2.invariant.bias.detach().numpy()
    f_coef_2 = np.concatenate((f_bias_2, f_coef_2[0, :]))

    # g_coef = base_trans.train_variants[0].weight.detach().numpy()
    # g_bias = base_trans.train_variants[0].bias.detach().numpy()
    # g_coef = np.concatenate((g_bias, g_coef[0, :]))

    g_coef = base_trans.test_variant.weight.detach().numpy()
    g_bias = base_trans.test_variant.bias.detach().numpy()
    g_coef = np.concatenate((g_bias, g_coef[0, :]))

    g_coef_2 = base_trans2.test_variant.weight.detach().numpy()
    g_bias_2 = base_trans2.test_variant.bias.detach().numpy()
    g_coef_2 = np.concatenate((g_bias_2, g_coef_2[0, :]))

    gen_coef =np.random.normal(loc=1, scale=0.2, size=(10, 3))

    f_data = np.array([(f_coef[i] * x ** i) for i in range(f_coef.size)]).sum(axis=0)
    g_data = np.array([(g_coef[i] * x ** i) for i in range(g_coef.size)]).sum(axis=0)

    f_data_2 = np.array([(f_coef_2[i] * x ** i) for i in range(f_coef_2.size)]).sum(axis=0)
    g_data_2 = np.array([(g_coef_2[i] * x ** i) for i in range(g_coef_2.size)]).sum(axis=0)
    # plot the data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for name in model_names:
        base_trans = TrainInvariant.load_from_checkpoint(os.path.dirname(__file__) + '/epoch=4999-step=5000.ckpt',
                                                        in_dim=in_dim, f_embed_dim=f_embed_dim, g_embed_dim=g_embed_dim,
                                                        out_dim=out_dim, num_domains=num_domains)
        f_coef = base_trans.invariant.weight.detach().numpy()
        f_bias = base_trans.invariant.bias.detach().numpy()
        f_coef = np.concatenate((f_bias, f_coef[0, :]))

        g_coefs = [np.concatenate((train_variant.bias.detach().numpy(),
                                  train_variant.weight.detach().numpy()[0, :])) for train_variant in base_trans.train_variants]
        f_data = np.array([(f_coef[i] * x ** i) for i in range(f_coef.size)]).sum(axis=0)
        g_data = [np.array([(g_coef[i] * x ** i) for i in range(g_coef.size)]).sum(axis=0) for g_coef in g_coefs]
        color=["red", "orange", "yellow", "green", "blue", "purple"]
        for i in range(len(g_data)):
            ax.plot(x, g_data[i] + f_data, color='tab:'+color[i])


    # ax.plot(x, f_data, color='tab:blue')
    ax.plot(x, g_data+f_data, color='tab:blue')
    ax.plot(x, g_data_2+f_data_2, color='tab:green')

    # for i in range (10):
    #     ax.plot(x, gen_coef[i,0]*np.e**x+
    #             gen_coef[i,1]*x**2+
    #             gen_coef[i,2]*np.sin(10*x), color='tab:orange')

    gen_coef=np.array([[0.5701967704178796,0.43860151346232035,0.9883738380592262,0.9860526037510328,0.9809344020787358]])
    io_points=np.array([[0.6531],
        [0.2533],
        [0.4663],
        [0.2444],
        [0.1590],
        [0.1104],
        [0.6563],
        [0.1382],
        [0.1966],
        [0.3687]])
    ax.plot(x, gen_coef[0, 0] * np.e ** x +
                        gen_coef[0,1]*x**2+
                        gen_coef[0,2]*np.sin(10*x), color='tab:orange')

    ax.scatter(io_points, gen_coef[0, 0] * np.e ** io_points +
            gen_coef[0, 1] * io_points ** 2 +
            gen_coef[0, 2] * np.sin(10 * io_points), marker='^',color='tab:purple')

    fig.show()

