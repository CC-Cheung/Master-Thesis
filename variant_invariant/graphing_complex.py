import os
import torch
from pytorch_lightning.callbacks import EarlyStopping

import wandb
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint



def z_derivatives(x, t, g,h,b, input):
    return [x[1]-input, -(1 / x[0]) * (x[1] ** 2 + b * x[1] + g * x[0] - g * h)]
def input_func(input,t, duration):
    return 0.1 * np.sin(2 * np.pi / duration * input* t)
def make_water_data_val(num_points, start=0, end=3):
    duration=end-start
    time = np.arange(start, end, duration/(num_points+1))
    full=np.zeros((num_points+1, 2))

    # full=np.random.rand(num_domains, num_points+1, in_dim)
    full[0, 0]=2e-3
    def gen_func(x, t):
        return z_derivatives(x, t, g, h, b, input_func(input,t,3))

    temp= odeint(gen_func, full[0, :], time)
    full=temp[np.newaxis,:]
    #full 1, num_points+1,2 (1 because one domain, batch first)

    x=full[:, :num_points]
    y=full[:,1:]
    #x,y will be 1, num_points, out_dim=1

    y=torch.Tensor(y)

    # x=torch.Tensor(
    #     np.concatenate(
    #         (x,
    #          input_func(input.reshape((-1,1)),
    #                     time[:-1].reshape((1,-1)),
    #                     duration)[:, :,np.newaxis]), axis=-1))
    x = torch.Tensor(input_func(
                        input,
                        time[:-1].reshape((1, -1)),
                        3)[:, :,np.newaxis])
    #y num_domain, num_points, in_dim, input num_domain, time num_points+1

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

if __name__=="__main__":
    # define any number of nn.Modules (or use your current ones)

    in_dim = 1
    out_dim = 2

    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    key = "c20d41ecf28a9b0efa2c5acb361828d1319bc62e"

    input, g,h,b=2.882026172983832, 10.210157208367225, 0.1097873798410574, 0.3060223299800364

    num_points = 200
    num_val_io_pairs = 25
    f_embed_dim = 50
    g_embed_dim = 10
    max_epoch = 1500

    model_names=["variant.ckpt",
                 "nofreeze.ckpt",
                 # "9000.ckpt",
                 # "12000.ckpt",
                 ]
    # model_names = ["epoch=2999-step=6000.ckpt"
    #                ]
    x,y=make_water_data_val(num_points)
    y=y.squeeze().cpu().detach().numpy().T
    end=3
    start=0
    duration = end - start
    t = np.arange(start, end, duration / (num_points + 1))[:-1]


    # plot the data

    for name in model_names:
        base_trans = TestInvariant.load_from_checkpoint(os.path.dirname(__file__) + '/' + name,
                                                        in_dim=in_dim, f_embed_dim=f_embed_dim,
                                                        g_embed_dim=g_embed_dim, out_dim=out_dim, num_domains=10)
        f = base_trans.invariant(x)[0].cpu().detach().numpy().squeeze().T
        g=base_trans.test_variant(x)[0].cpu().detach().numpy().squeeze().T
        for i in range (2):
            plt.plot (t, x.cpu().detach().numpy().squeeze(), label="input")
            plt.plot(t, f[i],  label="f")
            plt.plot(t, g[i], label="g")
            plt.plot(t, f[i]+g[i], label="pred")
            plt.plot(t, y[i], label="actual")
            plt.title(name)
            plt.legend()
            plt.show()
            print(y[i], f[i]+g[i])

    color = ["red", "orange", "green", "blue", "purple", "gray"]
