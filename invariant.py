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

#A submodel
class SequenceModule(nn.Module):
    def __init__(self, in_dim, out_dim, total_dim, hidden_dim,num_layers, warmup_length):
        super().__init__()
        self.num_layers=2
        self.hidden_dim=hidden_dim
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.warmup_length=warmup_length
        self.lstm_warmup = nn.LSTM(total_dim,
                                   hidden_size=hidden_dim,
                                   num_layers=num_layers,
                                   batch_first=True,
                                   # dropout=0.2
                                   )
        self.lstm=nn.LSTM(in_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          # dropout=0.2
        )
        self.proj=nn.Linear(hidden_dim,out_dim)

    def forward(self, x):
        #batch, length, dim
        _, (h0, c0) =self.lstm_warmup(x[:, :self.warmup_length])
        output= self.lstm(x[:,self.warmup_length:,:self.in_dim], (h0, c0))[0]
        return self.proj(output)

#Default
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
        self.in_dim=in_dim
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, x, domain_num):
        out_warmup_temp, (hi0, ci0) = self.invariant.lstm_warmup(x[:, :self.warmup_length])
        _, (hv0, cv0) = self.train_variants[domain_num].lstm_warmup(out_warmup_temp)
        out_pred_temp = self.invariant.lstm(x[:, self.warmup_length:, :self.in_dim], (hi0, ci0))[0]
        return self.train_variants[domain_num].proj(
            self.train_variants[domain_num].lstm(out_pred_temp, (hv0, cv0))[0])

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        loss = 0
        valid = 0

        for i in range(self.num_domains):
            value, mask = get_actual(batch[i], warmup_length=self.warmup_length, in_dim=in_dim)
            if mask.sum() != 0:
                loss += (mask * self.loss(self(batch[i], i), value)).sum() / mask.sum()
                valid += 1

        loss = loss / valid
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = 0
        valid = 0
        for i in range(self.num_domains):
            value, mask = get_actual(batch[i], warmup_length=self.warmup_length, in_dim=in_dim)
            if mask.sum() != 0:
                loss += (mask * self.loss(self(batch[i], i), value)).sum() / mask.sum()
                valid += 1

        loss = loss / valid
        self.log("val_loss", loss)

    # def on_fit_end(self):

    def configure_optimizers(self):
        # check if all there
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        return optimizer

#Baselines
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
        self.warmup_length=warmup_length
        self.invariant=SequenceModule(in_dim=in_dim,
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
        self.train_variants=nn.ModuleList([SequenceModule(in_dim=in_dim,
                                        out_dim=out_dim,
                                        total_dim=total_dim,
                                        hidden_dim=g_hidden_dim,
                                        num_layers=g_num_layers,
                                        warmup_length=warmup_length) for i in range (num_domains)])
        self.test_variant=SequenceModule(in_dim=in_dim,
                                        out_dim=out_dim,
                                        total_dim=total_dim,
                                        hidden_dim=g_hidden_dim,
                                        num_layers=g_num_layers,
                                        warmup_length=warmup_length)
        self.num_domains=num_domains
        self.eta=lambda x,y: x+y

        self.loss=nn.MSELoss(reduction="none")
    def forward(self, x, domain_num):
        result_f = self.invariant(x)
        result_g =self.train_variants[domain_num](x)

        return self.eta(result_f, result_g)
        # return self.train_variants[domain_num](self.invariant(x))

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        loss = 0
        valid=0

        for i in range(self.num_domains):
            value, mask = get_actual(batch[i], warmup_length=warmup_length, in_dim=in_dim)
            if mask.sum() != 0:
                loss += (mask * self.loss(self(batch[i], i), value)).sum() / mask.sum()
                valid += 1

        loss = loss / valid
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) :
        loss = 0
        valid=0
        for i in range(self.num_domains):
            value, mask = get_actual(batch[i], warmup_length=warmup_length, in_dim=in_dim)
            if mask.sum() != 0:
                loss += (mask*self.loss(self(batch[i], i), value)).sum()/mask.sum()
                valid+=1

        loss = loss / valid
        self.log("val_loss", loss)

    def configure_optimizers(self):
        #check if all there
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        return optimizer



def get_latest_file():
    list_of_files = glob.glob(os.path.dirname(__file__) + '/DA Thesis/**/*.ckpt',
                              recursive=True)  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

#getting the supposed output from a window
def get_actual(data, warmup_length=0, in_dim=5):
    predict=data[:, warmup_length:, in_dim:]
    return predict[..., 1::2], predict[..., ::2]

#for plotting
def plot(real, pred, batch_num=0, feature=0):
    plt.plot(numpy_it(real[batch_num,:,feature]))
    plt.plot(numpy_it(pred[batch_num,:,feature]))
    plt.show()

def numpy_it(t):
    return t.detach().cpu().numpy()
if __name__=="__main__":

    all_data = []
    # path = "Data/to use/pretrain 1"
    # path = "Data/to use/Maumee Basin"
    # data="Maumee Basin"
    data = "various"
    # data = "pretrain 14"
    # data = "Just Cuyahoga"
    # data = "Just Maumee"
    # data= "Just Lost"


    path = "Data/to use/log/"+data



    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    # key="c20d41ecf28a9b0efa2c5acb361828d1319bc62e"

    result = torch.load(path)
    ex_predict_length = result["info"]["ex_predict_length"]
    ex_warmup_length = result["info"]["ex_warmup_length"]
    window_distance=result["info"]["window_distance"]
    num_domains=len(result["info"]["domains"])

    ex_window_length = ex_warmup_length + ex_predict_length
    res_decrease = 1

    predict_length = ex_predict_length // res_decrease
    warmup_length = ex_warmup_length // res_decrease
    window_length = ex_window_length // res_decrease



    max_epoch=1000
    # result = torch.load(path)
    train_loader = result["train_dataloader"]
    val_loader = result["val_dataloader"]
    in_dim = 5
    out_dim = 9
    total_dim=23 #in_dim (time and flow missing/not) + out_dim*2 (missing/not)
    for seed in [1,2]:
        for f_hidden_dim in [32,8,128]:
            for f_num_layers in [2,1]:
                for g_hidden_dim in [8,32,128]:
                    for g_num_layers in [2,1]:

                        #Help resume training
                        # if [f_hidden_dim, f_num_layers, g_hidden_dim, g_num_layers] in [[32,2,8,2],[32,2,8,1],[32,2,32,2]]:
                        #     continue

                        if (np.array([f_hidden_dim, f_num_layers, g_hidden_dim, g_num_layers])
                            ==np.array([32,2,8,2])).sum() <3 :
                            continue

                        np.random.seed(seed)
                        torch.use_deterministic_algorithms(True)
                        torch.manual_seed(seed)
                        wandb_logger=WandbLogger(project="DA Thesis",
                                                 name=f"f{f_hidden_dim},{f_num_layers} g{g_hidden_dim},{g_num_layers}",
                                                 log_model="True")

                        wandb.init()

                        wandb_logger.experiment.config.update({
                            "num_domains": num_domains,
                            "f_hidden_dim": f_hidden_dim,
                            "g_hidden_dim": g_hidden_dim,
                            "f_num_layers": f_num_layers,
                            "g_num_layers": g_num_layers,
                            "max_epoch": max_epoch,
                            "predict_len": predict_length,
                            "file": os.path.basename(__file__),
                            "purpose": f"various {seed}",
                            "seed": seed,
                            "data": data

                        })
                        wandb.run.define_metric("val_loss", summary="min")

                        #Uncomment the right one
                        model=TrainInvariant(in_dim=in_dim,
                                                  out_dim=out_dim,
                                                  total_dim=total_dim,
                                                  num_domains=num_domains,
                                                  f_hidden_dim=f_hidden_dim,
                                                  g_hidden_dim=g_hidden_dim,
                                                  f_num_layers=f_num_layers,
                                                  g_num_layers=g_num_layers,
                                                  warmup_length=warmup_length)
                        # model = TrainInvariantCoolEta(in_dim=in_dim,
                        #                             out_dim=out_dim,
                        #                             total_dim=total_dim,
                        #                             num_domains=num_domains,
                        #                             f_hidden_dim=f_hidden_dim,
                        #                             g_hidden_dim=g_hidden_dim,
                        #                             f_num_layers=f_num_layers,
                        #                             g_num_layers=g_num_layers,
                        #                             warmup_length=warmup_length)
                        # model = TrainInvariantCoolEta.load_from_checkpoint("/home/cbcheung/Work/Thesis/Code/Real/Download/models/cool eta log/z0cmjwmq f128,2 g8,2.ckpt",
                        #                                                    in_dim=in_dim,
                        #                                                    out_dim=out_dim,
                        #                                                    total_dim=total_dim,
                        #                                                    num_domains=num_domains,
                        #                                                    f_hidden_dim=128,
                        #                                                    f_num_layers=2,
                        #                                                    g_hidden_dim=8,
                        #                                                    g_num_layers=2,
                        #                                                    warmup_length=ex_warmup_length)

                        # wandb_logger.watch(base_trans, log="all")
                        #
                        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=30)
                        # small_error_callback = EarlyStopping(monitor="val_loss", stopping_threshold=0.02)
                        model_callback_train = ModelCheckpoint(
                            save_top_k=1,
                            monitor="train_loss",
                            mode="min",
                            auto_insert_metric_name=True
                        )
                        model_callback_val = ModelCheckpoint(
                            save_top_k=1,
                            monitor="val_loss",
                            mode="min",
                            auto_insert_metric_name=True
                        )
                        trainer1 = pl.Trainer(
                                              logger=wandb_logger,
                                              max_epochs=max_epoch,
                                              # log_every_n_steps = 5,
                                              accelerator="gpu", devices=1,
                                              callbacks=[
                                                  model_callback_val,
                                                  early_stop_callback,
                                                         # small_error_callback
                                                         # model_checkpoint
                                                         ],
                                          # fast_dev_run=True
                                          )

                        trainer1.fit(model, train_loader, val_loader)

                        wandb.save(get_latest_file())
                        wandb.finish()



