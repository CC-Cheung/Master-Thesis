import copy
import datetime
import os
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import wandb
from torch import optim, nn, utils, Tensor

import pytorch_lightning as pl
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
import glob
import pandas as pd
import matplotlib.pyplot as plt


class LinearRelu(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
        for i in range(depth):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ReLU())

        self.net.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*self.net)

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
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class Linear(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for i in range(depth):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))

        self.net.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class SequenceModule(nn.Module):
    def __init__(self, in_dim, out_dim, total_dim, hidden_dim, num_layers, warmup_length):
        super().__init__()
        self.num_layers = 2
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.warmup_length = warmup_length

        self.lstm_warmup = nn.LSTM(total_dim,
                                   hidden_size=hidden_dim,
                                   num_layers=num_layers,
                                   batch_first=True,
                                   # dropout=0.2
                                   )
        self.lstm = nn.LSTM(in_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            # dropout=0.2
                            )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # batch, length, dim
        _, (h0, c0) = self.lstm_warmup(x[:, :self.warmup_length])
        output = self.lstm(x[:, self.warmup_length:, :self.in_dim], (h0, c0))[0]
        return self.proj(output)


class TrainInvariant(pl.LightningModule):
    def __init__(self,
                 in_dim,
                 out_dim,
                 total_dim,
                 num_domains,
                 f_hidden_dim,
                 g_hidden_dim,
                 f_num_layers,
                 g_num_layers,
                 warmup_length
                 ):
        super().__init__()
        self.warmup_length = warmup_length
        self.invariant = SequenceModule(in_dim=in_dim,
                                        out_dim=out_dim,
                                        total_dim=total_dim,
                                        hidden_dim=f_hidden_dim,
                                        num_layers=f_num_layers,
                                        warmup_length=warmup_length)
        self.test_variant = SequenceModule(in_dim=in_dim,
                                           out_dim=out_dim,
                                           total_dim=total_dim,
                                           hidden_dim=g_hidden_dim,
                                           num_layers=g_num_layers,
                                           warmup_length=warmup_length)
        self.train_variants = nn.ModuleList([SequenceModule(in_dim=in_dim,
                                                            out_dim=out_dim,
                                                            total_dim=total_dim,
                                                            hidden_dim=g_hidden_dim,
                                                            num_layers=g_num_layers,
                                                            warmup_length=warmup_length) for i in range(num_domains)])
        self.test_variant = SequenceModule(in_dim=in_dim,
                                           out_dim=out_dim,
                                           total_dim=total_dim,
                                           hidden_dim=g_hidden_dim,
                                           num_layers=g_num_layers,
                                           warmup_length=warmup_length)
        self.num_domains = num_domains
        self.eta = lambda x, y: x + y

        self.loss = nn.MSELoss(reduction="none")

    def forward(self, x):
        result_f = self.invariant(x)
        result_g = self.test_variant(x)

        return self.eta(result_f, result_g)
        # return result_f

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        batch = batch[0]
        loss = 0
        value, mask = get_predict(batch, warmup_length=ex_warmup_length, in_dim=in_dim)
        if mask.sum() != 0:
            loss = (mask * self.loss(self(batch), value)).sum() / mask.sum()
            self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch[0]
        value, mask = get_predict(batch, warmup_length=ex_warmup_length, in_dim=in_dim)
        if mask.sum() != 0:
            loss = (mask * self.loss(self(batch), value)).sum() / mask.sum()
            self.log("val_loss", loss)

    # def on_fit_end(self):

    def configure_optimizers(self):
        # check if all there
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        return optimizer


class TrainInvariantCoolEta(pl.LightningModule):
    def __init__(self,
                 in_dim,
                 out_dim,
                 total_dim,
                 num_domains,
                 f_hidden_dim,
                 g_hidden_dim,
                 f_num_layers,
                 g_num_layers,
                 warmup_length
                 ):
        super().__init__()
        self.warmup_length = warmup_length
        self.invariant = SequenceModule(in_dim=in_dim,
                                        out_dim=f_hidden_dim,
                                        total_dim=total_dim,
                                        hidden_dim=f_hidden_dim,
                                        num_layers=f_num_layers,
                                        warmup_length=warmup_length)
        self.test_variant = SequenceModule(in_dim=f_hidden_dim,
                                           out_dim=out_dim,
                                           total_dim=f_hidden_dim,
                                           hidden_dim=g_hidden_dim,
                                           num_layers=g_num_layers,
                                           warmup_length=warmup_length)
        self.train_variants = nn.ModuleList([SequenceModule(in_dim=f_hidden_dim,
                                                            out_dim=out_dim,
                                                            total_dim=f_hidden_dim,
                                                            hidden_dim=g_hidden_dim,
                                                            num_layers=g_num_layers,
                                                            warmup_length=warmup_length) for i in range(num_domains)])

        self.num_domains = num_domains
        self.in_dim = in_dim
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, x):
        out_warmup_temp, (hi0, ci0) = self.invariant.lstm_warmup(x[:, :self.warmup_length])
        _, (hv0, cv0) = self.test_variant.lstm_warmup(out_warmup_temp)
        out_pred_temp = self.invariant.lstm(x[:, self.warmup_length:, :self.in_dim], (hi0, ci0))[0]
        return self.test_variant.proj(
            self.test_variant.lstm(out_pred_temp, (hv0, cv0))[0])

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        loss = 0
        batch = batch[0]
        value, mask = get_predict(batch, warmup_length=self.warmup_length, in_dim=in_dim)
        if mask.sum() != 0:
            loss += (mask * self.loss(self(batch), value)).sum() / mask.sum()

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = 0
        batch = batch[0]
        value, mask = get_predict(batch, warmup_length=self.warmup_length, in_dim=in_dim)
        if mask.sum() != 0:
            loss += (mask * self.loss(self(batch), value)).sum() / mask.sum()

        loss = loss
        self.log("val_loss", loss)

    # def on_fit_end(self):

    def configure_optimizers(self):
        # check if all there
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        return optimizer


def get_latest_file():
    list_of_files = glob.glob(os.path.dirname(__file__) + '/DA Thesis/**/*.ckpt',
                              recursive=True)  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def get_predict(data, warmup_length=0, in_dim=5):
    predict = data[:, warmup_length:, in_dim:]

    return predict[..., 1::2], predict[..., ::2]


def plot(graphs, mask=None, batch_num=0, feature=0):
    for graph in graphs:
        plt.plot(numpy_it(graph[batch_num, :, feature]))
    if mask is not None:
        plt.vlines(numpy_it(mask[batch_num, :, feature]).nonzero(),
                   ymin=plt.ylim()[0],
                   ymax=plt.ylim()[1],
                   colors='r', linewidth=0.5)

    plt.show()


def numpy_it(t):
    return t.detach().cpu().numpy()


def errors(pred, real, mask):
    mean_real = (real * mask).sum() / mask.sum()
    numerator = ((real - pred) ** 2 * mask).sum()
    denominator = ((real - mean_real) ** 2 * mask).sum()
    nse = 1 - (numerator / denominator)

    mse = numerator / numerator.numel()

    mean_pred = (pred * mask).sum() / mask.sum()

    numerator = (((pred - mean_pred) * (real - mean_real)) * mask).sum()
    denominator = (((pred - mean_pred) ** 2).sum() * ((real - mean_real) ** 2).sum()).sqrt()
    pcc = numerator / denominator
    return {"nse": nse, "mse": mse, "pcc": pcc}


if __name__ == "__main__":

    all_data = []
    to_use_path = "Data/to use/"
    to_use_path = "../Data/to use/log"



    # df_tried = pd.read_csv("Download/run data/cool_eta_variant_for_test.csv").drop("Unnamed: 0", axis=1)
    # df_tried=df_tried[["f_hidden_dim","f_num_layers","g_hidden_dim","g_num_layers"]].drop_duplicates()
    # tried=[[row["f_hidden_dim"], row["f_num_layers"],row["g_hidden_dim"],row["g_num_layers"]]
    #        for _,row in df_tried.iterrows()]


    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    # key="c20d41ecf28a9b0efa2c5acb361828d1319bc62e"

    f_hidden_dim = 32
    g_hidden_dim = 8
    f_num_layers = 2
    g_num_layers = 2

    max_epoch = 500

    # window length / drop columnsnot used
    in_dim = 5
    out_dim = 9
    total_dim = 23  # in_dim (time and flow missing/not) + out_dim*2 (missing/not)
    size_tried = []
    seed=2

    # purpose=f"cool eta log no pretrain seed {seed}"
    # purpose="cool eta log no pretrain no add mixed"
    purpose=f"various seed {seed}"

    # purpose="various"

    models_path = os.path.join("Download", "models", purpose)
    #refer to data processing (log)
    num_domains = 7


    temp=True
    result = torch.load(os.path.join(to_use_path, f"res_decrease 0.05", "Lost"))

    train_loader = result["train_dataloader"]
    val_loader = result["val_dataloader"]
    ex_predict_length = result["info"]["ex_predict_length"]
    ex_warmup_length = result["info"]["ex_warmup_length"]
    window_distance = result["info"]["window_distance"]
    ex_window_length = ex_warmup_length + ex_predict_length
    no_pretrain=TrainInvariantCoolEta.load_from_checkpoint("../f32,2 g8,2 005 no pretrain.ckpt",
        in_dim=in_dim,
          out_dim=out_dim,
          total_dim=total_dim,
          num_domains=1,
          f_hidden_dim=f_hidden_dim,
          g_hidden_dim=g_hidden_dim,
          f_num_layers=f_num_layers,
          g_num_layers=g_num_layers,
          warmup_length=ex_warmup_length)

    default = TrainInvariantCoolEta.load_from_checkpoint(
        os.path.join("../f32,2 g8,2 005.ckpt"),
        in_dim=in_dim,
        out_dim=out_dim,
        total_dim=total_dim,
        num_domains=num_domains,
        f_hidden_dim=f_hidden_dim,
        g_hidden_dim=g_hidden_dim,
        f_num_layers=f_num_layers,
        g_num_layers=g_num_layers,
        warmup_length=ex_warmup_length)
    unreduced=torch.load("../Data/processed/log/logLost_samples")
    p=0

