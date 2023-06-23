# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J7bxCUtpGHCEwgT1_1m_lRL3c0rIo2t4
"""


import os

import matplotlib.pyplot as plt
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
    x=np.random.rand(num_domains, num_points, in_dim)
    x_transpose=x.transpose(2, 1, 0)
    #coef num_domains, powers
    y=np.array([coef[:, i] * x_transpose ** i
                for i in range(coef.shape[1])]).sum(axis=0)\
        .transpose(2, 1, 0)



    return x,y
def make_quad_data_val(num_points, coef):
    x=np.random.rand(num_points, in_dim)
    y=np.array([(coef[i]*x**i) for i in range(coef.shape[0])]).sum(axis=0)
    #y will be num_points, out_dim=1

    return x,y


def make_weird_data(num_domains, num_points, coef):
    x=np.random.rand(num_domains, num_points, in_dim)
    x_transpose=x.transpose(2, 1, 0)
    y=(coef[:,0] * np.e ** x_transpose + \
      coef[:,1] * x_transpose ** 2 + \
      coef[:,2] * np.sin(10 * x_transpose))\
        .transpose(2, 1, 0)
    #coef will be num_domain, 3
    #y will be num_domains, num_points, out_dim=1

    return x,y
def make_weird_data_val(num_points, coef):
    x=np.random.rand(num_points, in_dim)

    y=(coef[0]*np.e**x+ \
      coef[1] *x**2 + \
      coef[2]*np.sin(10*x))


    return x,y




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

    f_embed_dim = 6
    g_embed_dim = 2
    num_domains = 10
    gen_deg=4

    # train_coef = np.array([[1, -18.95, 73.5, 20,50,10,1],
    #                        [1.26, 0, 15, 0,50,10,1],
    #                        [26, -95, .5, 0,50,10,1]])

    # train_coef = np.array([[1.5, -1, 1],
    #                      [1, -1.5, 1],
    #                        [1, -1, 1.5]])
    # train_coef = np.concatenate((np.random.rand(num_domains,g_embed_dim+1),
    #                              np.random.normal(loc=1, scale=0.1, size=(num_domains, gen_deg-g_embed_dim))), axis=1)
    train_coef = np.random.normal(loc=1, scale=0.2, size=(num_domains, 3))
    val_coef=np.random.normal(loc=1, scale=0.2, size=(3))

    num_domains = train_coef.shape[0]

    num_points = 10
    num_val_io=17


    # train_dataset = make_quad_data(num_domains, num_points,train_coef)
    # val_dataset = make_quad_data_val(num_val_io,val_coef)

    train_dataset = make_weird_data(num_domains, num_points, train_coef)
    val_dataset = make_weird_data_val(num_val_io,val_coef)

    # for i in range(2):
    #     train_dataset[i]=np.concatenate((train_dataset[i], val_dataset[i]))
    #num_domain, points, in_dim


    powers_f = np.stack([train_dataset[0] ** i for i in range(0, f_embed_dim + 1)])
    powers_g = np.stack([train_dataset [0]** i for i in range(0, g_embed_dim + 1)])
    #power, num_domain, num_points, in_dim

    powers_f_val = np.stack([val_dataset[0] ** i for i in range(0, f_embed_dim + 1)])
    powers_g_val = np.stack([val_dataset[0] ** i for i in range(0, g_embed_dim + 1)])
    #power, num_points, in_dim

    f_portion_of_a=powers_f \
        .squeeze(axis=-1) \
        .transpose((1,2,0))\
        .reshape((-1, f_embed_dim+1))
    #num_domain*num_points, power (in dim=1)

    f_portion_of_a=np.concatenate((f_portion_of_a,
                                  powers_f_val.transpose((1,2,0))
                                                .reshape((-1, f_embed_dim+1))))
    #num_domain*num_points + num_val_io, power

    g_portion_of_a=np.zeros((num_domains*num_points+num_val_io, (num_domains+1)*(g_embed_dim+1)))

    powers_g_transpose=powers_g.squeeze(axis=-1).transpose((1,2,0))
    #num_domain,num_points, power,  (in dim was 1)

    for i in range(num_domains):
        g_portion_of_a[i*num_points: (i+1)*num_points, i*(g_embed_dim+1): (i+1)*(g_embed_dim+1)]=\
            powers_g_transpose[i, :, :]
    g_portion_of_a[num_domains*num_points:, num_domains*(g_embed_dim+1): ]=powers_g_val.transpose((1,2,0)).reshape((-1, g_embed_dim+1))

    a=np.concatenate((f_portion_of_a,g_portion_of_a), axis=1)
    # a=f_portion_of_a

    b=np.concatenate((train_dataset[1].reshape((-1, in_dim)), val_dataset[1]))

    result=np.linalg.lstsq(a,b)
    x=np.arange(0,1.05,0.05)
    weird=[lambda x: np.e**x, lambda x:x**2, lambda x:np.sin(10*x)]

    # for i in range(num_domains):
    #     plt.tight_layout()
    #     f_coef=result[0][:f_embed_dim+1, 0]
    #     # f=np.array([(f_coef[j] * x ** j) for j in range(f_coef.size)]).sum(axis=0)
    #     # plt.plot(x, f, label="f")
    #
    #     g_coef=result[0][f_embed_dim+1+ (1+g_embed_dim)*i: f_embed_dim+1+(1+g_embed_dim)*(i+1), 0]
    #     # g=np.array([(g_coef[j] * x ** j) for j in range(g_coef.size)]).sum(axis=0)
    #     # plt.plot(x, g, label="g")
    #
    #     coef=f_coef+np.concatenate((g_coef, np.zeros(f_embed_dim-g_embed_dim)))
    #     trained=np.array([(coef[j] * x ** j) for j in range(coef.size)]).sum(axis=0)
    #     plt.plot(x,trained, label="trained")
    #
    #     actual=np.array([(train_coef[i,j] * weird[j](x)) for j in range(train_coef.shape[1])]).sum(axis=0)
    #     plt.plot(x,actual, label="actual")
    #     plt.scatter(train_dataset[0][i], train_dataset[1][i], c="g", label="actual")
    #     plt.title("train_coef="+str(train_coef[i])+
    #               " f="+str (f_coef.round(decimals=2))+
    #               " g="+str (g_coef.round(decimals=3))+
    #               " R^2="+str (((actual-trained)**2).mean()),
    #               wrap=True, fontsize=7)
    #     plt.legend()
    #     plt.show()

    plt.tight_layout()
    f_coef=result[0][:f_embed_dim+1, 0]
    # f=np.array([(f_coef[j] * x ** j) for j in range(f_coef.size)]).sum(axis=0)
    # plt.plot(x, f, label="f")

    g_coef=result[0][f_embed_dim+1+ (1+g_embed_dim)*num_domains:, 0]
    # g=np.array([(g_coef[j] * x ** j) for j in range(g_coef.size)]).sum(axis=0)
    # plt.plot(x, g, label="g")

    coef=f_coef+np.concatenate((g_coef, np.zeros(f_embed_dim-g_embed_dim)))
    trained=np.array([(coef[j] * x ** j) for j in range(coef.size)]).sum(axis=0)
    plt.plot(x,trained, label="trained")

    actual=np.array([(val_coef[j] * weird[j](x)) for j in range(val_coef.shape[0])]).sum(axis=0)
    plt.plot(x,actual, label="actual")
    plt.scatter(val_dataset[0], val_dataset[1], c="g", label="actual")
    plt.title("train_coef=" + str(val_coef) +
              " f=" + str(f_coef.round(decimals=2)) +
              " g=" + str(g_coef.round(decimals=3)) +
              " R^2=" + str(((actual - trained) ** 2).mean()),
              wrap=True, fontsize=7)
    plt.legend()
    plt.show()




    # power, num_points, in_dim

    f_portion_of_a = powers_f_val \
        .squeeze(axis=-1) \
        .transpose((1, 0)) \
    # num_points, power (in dim=1)

    g_portion_of_a = powers_g_val.squeeze(axis=-1).transpose((1, 0))
    # num_points, power,  (in dim was 1)

    # a = np.concatenate((f_portion_of_a, g_portion_of_a), axis=1)
    a=f_portion_of_a

    b = np.concatenate(val_dataset[1].reshape((1,-1)))

    result = np.linalg.lstsq(a, b)
    plt.tight_layout()
    f_coef = result[0][:f_embed_dim + 1]
    # f=np.array([(f_coef[j] * x ** j) for j in range(f_coef.size)]).sum(axis=0)
    # plt.plot(x, f, label="f")

    # g_coef = result[0][f_embed_dim + 1 :]
    # g=np.array([(g_coef[j] * x ** j) for j in range(g_coef.size)]).sum(axis=0)
    # plt.plot(x, g, label="g")

    coef = f_coef
    trained = np.array([(coef[j] * x ** j) for j in range(coef.size)]).sum(axis=0)
    plt.plot(x, trained, label="trained")

    actual = np.array([(val_coef[j] * weird[j](x)) for j in range(val_coef.shape[0])]).sum(axis=0)
    plt.plot(x, actual, label="actual")
    plt.scatter(val_dataset[0], val_dataset[1], c="g", label="actual")
    plt.title("train_coef=" + str(val_coef) +
              " f=" + str(f_coef.round(decimals=2)) +
              " g=" + str(g_coef.round(decimals=3)) +
              " R^2=" + str(((actual - trained) ** 2).mean()),
              wrap=True, fontsize=7)
    plt.legend()
    plt.show()

    for i in range (100):
        coef=np.random.normal(loc=1, scale=0.2, size=(3))
        y=np.array([(coef[j] * weird[j](x)) for j in range(coef.shape[0])]).sum(axis=0)
        plt.plot(x,y, 'b', linewidth=0.2)
    plt.show()
    print(result[0], result [1], result [2])

    #deg 6 error [7.56572675e-27]
    #deg 7, error [1.19885257e-05]
    #deg 8 [0.00114999]
    # result_f = self.invariant(powers_f)
    # result_g = self.train_variants[domain_num](powers_g)
    #
    # return self.eta(result_f, result_g)

