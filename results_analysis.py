import datetime
import os
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import wandb
from torch import optim, nn, utils, Tensor

import pytorch_lightning as pl
from collections import Counter,deque
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
import glob
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import re

#Model classes
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
        value, mask = get_actual(batch, warmup_length=self.warmup_length, in_dim=in_dim)
        if mask.sum() != 0:
            loss += (mask * self.loss(self(batch), value)).sum() / mask.sum()

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = 0
        batch = batch[0]
        value, mask = get_actual(batch, warmup_length=self.warmup_length, in_dim=in_dim)
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


def get_actual(data, warmup_length=0, in_dim=5):
    predict = data[:, warmup_length:, in_dim:]

    return predict[..., 1::2], predict[..., ::2]


def numpy_it(t):
    return t.detach().cpu().numpy()
#Undo data processing transformations
def rescale(result):
    return (10**(numpy_it(result)*all_logged_std+all_logged_mean)-0.1)\
           **2+val_cols_neg_min
#Mask missing and normalize
def mask_norm(tensor, mask):
    non_inf=tensor!=-np.inf
    inf=tensor==-np.inf
    return (np.nansum(tensor* mask.sum(axis=1, keepdims=True)*non_inf,axis=0)
            /(mask.sum(axis=(0,1)))-inf.sum(axis=0).squeeze()).squeeze()

#Calculating Metrics
def error_metrics(model, batch):
    #batch, length, feat
    real, mask= get_actual(batch, warmup_length=ex_warmup_length)
    pred=model(batch)
    mask=numpy_it(mask)
    real_values, pred_values=rescale(real), rescale(pred)

    mean_real = (real_values * mask).sum(axis=1, keepdims=True) / mask.sum(axis=1, keepdims=True)
    mean_pred = (pred_values * mask).sum(axis=1, keepdims=True) / mask.sum(axis=1, keepdims=True)
    std_real=np.sqrt(((real_values - mean_real) ** 2 * mask).sum(axis=1,keepdims=True)
                     / mask.sum(axis=1, keepdims=True))
    std_pred = np.sqrt(((pred_values - mean_real) ** 2 * mask).sum(axis=1, keepdims=True)
                       / mask.sum(axis=1, keepdims=True))

    # numerator = ((real_values - pred_values) ** 2 * mask).sum()
    # denominator = ((real_values - mean_real) ** 2 * mask).sum()

    numerator_sep = ((real_values - pred_values) ** 2*mask).sum(axis=1, keepdims=True)
    denominator_sep = ((real_values - mean_real) ** 2*mask).sum(axis=1, keepdims=True)

    # nse = 1 - (numerator / denominator)
    nse_sep_vals=mask_norm(1 - (numerator_sep / denominator_sep), mask)
    nse_sep={feature+" nse": nse_sep_vals[i] for i,feature in enumerate(features)}


    # mse = numerator/ mask.sum()
    mse_sep_vals=mask_norm(numerator_sep,mask)
    mse_sep={feature+" mse": mse_sep_vals[i] for i,feature in enumerate(features)}


    # numerator = (((pred_values - mean_pred) * (real_values - mean_real)) * mask).sum()
    # denominator = np.sqrt((((pred_values - mean_pred) ** 2).sum() * ((real_values - mean_real) ** 2).sum()))
    numerator_sep = (((pred_values - mean_pred) * (real_values - mean_real)) * mask).sum(axis=1, keepdims=True)
    denominator_sep = np.sqrt(((pred_values - mean_pred) ** 2*mask).sum(axis=1, keepdims=True) *
                              ((real_values - mean_real) ** 2*mask).sum(axis=1, keepdims=True))
    #pcc dependent on the whole predict
    # pcc = numerator / denominator
    pcc_sep_sep = (numerator_sep / denominator_sep)
    pcc_sep_vals=mask_norm(pcc_sep_sep, mask)
    pcc_sep={feature+" pcc": pcc_sep_vals[i] for i,feature in enumerate(features)}

    kge_sep_sep=(1-np.sqrt((pcc_sep_sep-1)**2+(mean_pred/mean_real-1)**2+(std_pred/std_real-1)**2))
    kge_sep_vals=mask_norm(kge_sep_sep, mask)
    kge_sep={feature+" kge": kge_sep_vals[i] for i,feature in enumerate(features)}
    # kge=(kge_vals * mask.sum(axis=(0,1))/mask.sum()).sum()
    return nse_sep|mse_sep|pcc_sep|kge_sep
def numpy_it(t):
    return t.detach().cpu().numpy()

def plot(graphs, mask=None, batch_num=0, feature=0, label=[]):
    for i,graph in enumerate(graphs):
        if type(graph).__module__ != np.__name__:
            graph=numpy_it(graph)
        plt.plot(graph[batch_num, :, feature], label=label[i])
    if mask is not None:
        plt.vlines(numpy_it(mask[batch_num, :, feature]).nonzero(),
                   ymin=plt.ylim()[0],
                   ymax=plt.ylim()[1],
                   colors='r', linewidth=0.5)
    # plt.title("f32,2 g8,2 No Pretrain TSS")
    plt.title(f"f32,2 g8,2 {label[1]} TSS")
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("TSS")
    # plt.savefig("f32,2 g8,2 No Pretrain TSS.png")
    # plt.savefig("f32,2 g8,2 Default TSS.png")

    plt.show()
def mse(a,b):
    return ((a-b)**2).mean()
if __name__=="__main__":
    model_folder=os.path.join("Download","models")
    folder=os.path.join(model_folder, "cool eta logvariant")
    to_use_path = "Data/to use/log"
    d = torch.load("/home/cbcheung/Work/Thesis/Code/Real/Data/cleaned/log/global_stats")
    all_logged_mean, all_logged_std, val_cols_neg_min= d['all_logged_mean'].values[1:],\
                                         d['all_logged_std'].values[1:],\
                                        d['val_cols_neg_min'].values[1:]

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
    results=None
    features=d['all_logged_mean'].index[1:]
    all_errors_dict={"name": [], "id": [], "f_hidden_dim": [],
     "f_num_layers": [], "g_hidden_dim": [],
     "g_num_layers": [], "res_decrease": [], "freeze": [],
     }|\
     {feature+" nse":[] for feature in features} | \
     {feature + " mse": [] for feature in features} |\
    {feature+" pcc":[] for feature in features}|\
    {feature + " kge": [] for feature in features}
    all_errors= pd.DataFrame(all_errors_dict)

    result = torch.load(os.path.join(to_use_path, f"res_decrease {0.05}", "Lost"))

    train_loader = result["train_dataloader"]
    val_loader = result["val_dataloader"]
    ex_predict_length = result["info"]["ex_predict_length"]
    ex_warmup_length = result["info"]["ex_warmup_length"]
    window_distance = result["info"]["window_distance"]
    ex_window_length = ex_warmup_length + ex_predict_length
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    model1 = TrainInvariantCoolEta.load_from_checkpoint(
        "f32,2 g8,2 005.ckpt",
        in_dim=in_dim,
        out_dim=out_dim,
        total_dim=total_dim,
        num_domains=7,
        f_hidden_dim=f_hidden_dim,
        g_hidden_dim=g_hidden_dim,
        f_num_layers=f_num_layers,
        g_num_layers=g_num_layers,
        warmup_length=ex_warmup_length)
    model2 = TrainInvariantCoolEta.load_from_checkpoint(
        "f32,2 g8,2 005 no pretrain.ckpt",
        in_dim=in_dim,
        out_dim=out_dim,
        total_dim=total_dim,
        num_domains=1,
        f_hidden_dim=f_hidden_dim,
        g_hidden_dim=g_hidden_dim,
        f_num_layers=f_num_layers,
        g_num_layers=g_num_layers,
        warmup_length=ex_warmup_length)
    model3 = TrainInvariantCoolEta.load_from_checkpoint(
        "f32,2 g8,2 005 no pretrain both.ckpt",
        in_dim=in_dim,
        out_dim=out_dim,
        total_dim=total_dim,
        num_domains=1,
        f_hidden_dim=f_hidden_dim,
        g_hidden_dim=g_hidden_dim,
        f_num_layers=f_num_layers,
        g_num_layers=g_num_layers,
        warmup_length=ex_warmup_length)
    # all_tensors = val_loader.dataset.tensors[0]
    for batch in val_loader:
        batch=batch[0]
        real, mask = get_actual(batch, warmup_length=ex_warmup_length)
        pred1 = model1(batch)
        pred2 = model2(batch)
        pred3 = model3(batch)
        real_values, pred_values1, pred_values2, pred_values3= rescale(real), rescale(pred1), rescale(pred2),rescale(pred3),
        print(mse(real_values[0, :, 0], pred_values1[0, :, 0])/np.mean(real_values[0, :, 0]))
        print(mse(real_values[0, :, 0], pred_values2[0, :, 0])/np.mean(real_values[0, :, 0]))
        print(mse(real_values[0, :, 0], pred_values3[0, :, 0])/np.mean(real_values[0, :, 0]))
        print("__________")
        # plot([real,pred1,pred2,pred3], label=["True", "Default", "No Pretrain Target Only", "No Pretrain Source+Target"])
        plot([real_values,pred_values1], label=["True", "Default",])
        plot([real_values,pred_values2], label=["True",  "No Pretrain Target Only"])
        plot([real_values,pred_values3], label=["True",  "No Pretrain Source+Target"])

    #Create csv
    # seed=2
    # df=pd.read_csv(f"Download/run data/metrics{seed}.csv").drop("Unnamed: 0",  axis=1)
    # parts = df['name'].map(lambda x:x.split(' '))
    # df["freeze"]=parts.map(lambda x:x[5].replace('freeze','').replace('.ckpt','')=="True")
    # # df['id'] = parts.map(lambda x:x[0])
    # # f_parts = parts.map(lambda x:x[1].replace("f", ''))
    # # df['f_hidden_dim'], df['f_num_layers'] = [f_parts.map(lambda x: int(x.split(",")[i])) for i in range(2)]
    # g_parts = parts.map(lambda x: x[2].replace("g", ''))
    # df['g_hidden_dim'], df['g_num_layers'] = [g_parts.map(lambda x: int(x.split(",")[i])) for i in range(2)]
    # # df['res_decrease'] = parts.map(lambda x:x[4])
    # df.to_csv(f"Download/run data/metrics{seed}.csv")