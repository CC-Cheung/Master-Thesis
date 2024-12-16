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

#A Submodel
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

#Just vector adding f and g
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

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        loss = 0
        value, mask = get_actual(batch, warmup_length=ex_warmup_length, in_dim=in_dim)
        if mask.sum() != 0:
            loss = (mask * self.loss(self(batch), value)).sum() / mask.sum()
            self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch[0]
        value, mask = get_actual(batch, warmup_length=ex_warmup_length, in_dim=in_dim)
        if mask.sum() != 0:
            loss = (mask * self.loss(self(batch), value)).sum() / mask.sum()
            self.log("val_loss", loss)


    def configure_optimizers(self):
        # check if all there
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        return optimizer

#Feeding input through f then g
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

#For wanddb
def get_latest_file():
    list_of_files = glob.glob(os.path.dirname(__file__) + '/DA Thesis/**/*.ckpt',
                              recursive=True)  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

#Get the supposed output from 128-day window [64:, just the WQ values]
def get_actual(data, warmup_length=64, in_dim=5):
    predict = data[:, warmup_length:, in_dim:]

    return predict[..., 1::2], predict[..., ::2]

#Give matrices, will select the batch and feature to plot.
def plot(graphs, mask=None, batch_num=0, feature=0):
    label=["Predicted", "True"]
    for i,graph in enumerate(graphs):
        plt.plot(numpy_it(graph[batch_num, :, feature]), label=label[i])
    if mask is not None:
        plt.vlines(numpy_it(mask[batch_num, :, feature]).nonzero(),
                   ymin=plt.ylim()[0],
                   ymax=plt.ylim()[1],
                   colors='r', linewidth=0.5)
    # plt.title("f32,2 g8,2 No Pretrain TSS")
    plt.title("f32,2 g8,2 Default TSS")
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("TSS")
    # plt.savefig("f32,2 g8,2 No Pretrain TSS.png")
    # plt.savefig("f32,2 g8,2 Default TSS.png")

    plt.show()

#if in pycharm, makes visualizes tensors by converting to numpy
def numpy_it(t):
    return t.detach().cpu().numpy()


if __name__ == "__main__":

    all_data = []
    to_use_path = "Data/to use/"
    to_use_path = "Data/to use/log"

    max_epoch = 500

    in_dim = 5
    out_dim = 9
    total_dim = 23  # in_dim (3time and flow missing/not) + out_dim*2 (missing/not)
    num_domains = 7

    seed=2

    purpose=f"cool eta log no pretrain both data seed {seed}"
    # purpose="trash"
    models_path = os.path.join("Download", "models", purpose)

    #for the combining dataset baseline
    result_various_data=torch.cat(torch.load(os.path.join(to_use_path, "various"))['train_dataloader'].dataset.tensors, dim=0)
    for res_decrease in [
                        0.02,
                         0.05,
                         0.1,
                        0.2,
                        0.4,
    ]:
        result = torch.load(os.path.join(to_use_path, f"res_decrease {res_decrease}", "Lost"))


        train_loader = result["train_dataloader"]
        # for the combining dataset baseline
        train_loader = DataLoader(TensorDataset(
                                                torch.cat((result_various_data,
                                                           result['train_dataloader'].dataset.tensors[0]), dim=0))
                        , batch_size=32, shuffle=True)



        val_loader = result["val_dataloader"]
        ex_predict_length = result["info"]["ex_predict_length"]
        ex_warmup_length = result["info"]["ex_warmup_length"]
        window_distance = result["info"]["window_distance"]
        ex_window_length = ex_warmup_length + ex_predict_length

        #go through everything in folder
        # for file in list_of_models:

        #Baseline experiments
        for f_hidden_dim, f_num_layers, g_hidden_dim, g_num_layers in [
            [32, 2, 8, 2],
            [32, 2, 8, 1],
            [32, 1, 8, 2],
            [128, 2, 8, 2],
            [8, 2, 8, 2],
            [32, 2, 128, 2],
            [32, 2, 32, 2],
        ]:
            file="N/A"


            # parts = file.split(' ')
            # id = parts[0]
            # f_parts = parts[1].replace("f", '')
            # f_hidden_dim, f_num_layers = [int(i) for i in f_parts.split(',')]
            # g_parts = parts[2].replace(".ckpt", '').replace("g", '')
            # g_hidden_dim, g_num_layers = [int(i) for i in g_parts.split(',')]

            # if [f_hidden_dim, f_num_layers, g_hidden_dim, g_num_layers] in tried:
            #     continue
            # if [f_hidden_dim, f_num_layers, g_hidden_dim, g_num_layers] not in [[32,2,8,2], [32,2,32,2], [32,2,8,1]]:
            #     continue

            # size_tried.append([f_hidden_dim, f_num_layers, g_hidden_dim, g_num_layers])
            # if f_hidden_dim==64 or f_num_layers!=4 or g_num_layers!=4 or g_hidden_dim!=2:
            #     continue
            # for split in [0.2, 0.01]:
                # if res_decrease<0.2 and g_hidden_dim==8:
                #     continue

            #Comment True if doing baselines
            for freeze in [
                            # True,
                           False
            ]:

                np.random.seed(seed)
                torch.use_deterministic_algorithms(True)
                torch.manual_seed(seed)


                wandb_logger = WandbLogger(project="DA Thesis",
                                           name=f"f{f_hidden_dim},{f_num_layers} g{g_hidden_dim},{g_num_layers} res_dec {res_decrease} freeze{freeze}",
                                           # name=f"f{f_hidden_dim},{f_num_layers} g{g_hidden_dim},{g_num_layers} split {split} freeze{freeze}",
                                           # name=f"f{f_hidden_dim},{f_num_layers} g{g_hidden_dim},{g_num_layers} res_dec {res_decrease} no pretrain",
                                           # name="trash",

                                           log_model="True")
                wandb.init()

                wandb_logger.experiment.config.update({
                    "num_domains": num_domains,
                    "f_hidden_dim": f_hidden_dim,
                    "g_hidden_dim": g_hidden_dim,
                    "f_num_layers": f_num_layers,
                    "g_num_layers": g_num_layers,
                    "max_epoch": max_epoch,
                    "ex_predict_length": ex_predict_length,
                    "ex_warmup_length": ex_warmup_length,
                    "file": os.path.basename(__file__),
                    "res_decrease": res_decrease,
                    "from": file,
                    "purpose": purpose,
                    # "purpose":"cool eta log no pretrain no add",
                    # "purpose": "trash",

                    "freeze": freeze,
                    "seed": seed

                })
                wandb.run.define_metric("val_loss", summary="min")

                # Uncomment the right one
                # model=TrainInvariant.load_from_checkpoint(os.path.join(models_path, file),
                #                         in_dim=in_dim,
                #                           out_dim=out_dim,
                #                           total_dim=total_dim,
                #                           num_domains=num_domains,
                #                           f_hidden_dim=f_hidden_dim,
                #                           g_hidden_dim=g_hidden_dim,
                #                           f_num_layers=f_num_layers,
                #                           g_num_layers=g_num_layers,
                #                           warmup_length=ex_warmup_length)
                # model = TrainInvariantCoolEta.load_from_checkpoint("f32,2 g8,2 005 no pretrain.ckpt",
                #                                             in_dim=in_dim,
                #                                             out_dim=out_dim,
                #                                             total_dim=total_dim,
                #                                             num_domains=1,
                #                                             f_hidden_dim=f_hidden_dim,
                #                                             g_hidden_dim=g_hidden_dim,
                #                                             f_num_layers=f_num_layers,
                #                                             g_num_layers=g_num_layers,
                #                                             warmup_length=ex_warmup_length)
                # model = TrainInvariantCoolEta.load_from_checkpoint("f32,2 g8,2 005 no pretrain both.ckpt",
                #                                                    in_dim=in_dim,
                #                                                    out_dim=out_dim,
                #                                                    total_dim=total_dim,
                #                                                    num_domains=1,
                #                                                    f_hidden_dim=f_hidden_dim,
                #                                                    g_hidden_dim=g_hidden_dim,
                #                                                    f_num_layers=f_num_layers,
                #                                                    g_num_layers=g_num_layers,
                #                                                    warmup_length=ex_warmup_length)
                # model = TrainInvariantCoolEta.load_from_checkpoint("f32,2 g8,2 005.ckpt",
                #                                                    in_dim=in_dim,
                #                                                    out_dim=out_dim,
                #                                                    total_dim=total_dim,
                #                                                    num_domains=num_domains,
                #                                                    f_hidden_dim=f_hidden_dim,
                #                                                    g_hidden_dim=g_hidden_dim,
                #                                                    f_num_layers=f_num_layers,
                #                                                    g_num_layers=g_num_layers,
                #                                                    warmup_length=ex_warmup_length)
                # model = TrainInvariant(
                #     in_dim=in_dim,
                #     out_dim=out_dim,
                #     total_dim=total_dim,
                #     num_domains=num_domains,
                #     f_hidden_dim=f_hidden_dim,
                #     g_hidden_dim=g_hidden_dim,
                #     f_num_layers=f_num_layers,
                #     g_num_layers=g_num_layers,
                #     warmup_length=ex_warmup_length)
                # model = TrainInvariantCoolEta.load_from_checkpoint(
                #     os.path.join(models_path, file),
                #     in_dim=in_dim,
                #     out_dim=out_dim,
                #     total_dim=total_dim,
                #     num_domains=num_domains,
                #     f_hidden_dim=f_hidden_dim,
                #     g_hidden_dim=g_hidden_dim,
                #     f_num_layers=f_num_layers,
                #     g_num_layers=g_num_layers,
                #     warmup_length=ex_warmup_length)
                # model = TrainInvariantCoolEta.load_from_checkpoint(
                #     "/home/cbcheung/Work/Thesis/Code/Real/Download/models/cool eta log no pretrain no addvariant/jmboy04j f8,2 g8,2 res_dec 0.7 freezeFalse.ckpt",
                #     in_dim=in_dim,
                #     out_dim=out_dim,
                #     total_dim=total_dim,
                #     num_domains=num_domains,
                #     f_hidden_dim=f_hidden_dim,
                #     g_hidden_dim=g_hidden_dim,
                #     f_num_layers=f_num_layers,
                #     g_num_layers=g_num_layers,
                #     warmup_length=ex_warmup_length)
                model = TrainInvariantCoolEta(
                                               in_dim=in_dim,
                                               out_dim=out_dim,
                                               total_dim=total_dim,
                                               num_domains=num_domains,
                                               f_hidden_dim=f_hidden_dim,
                                               g_hidden_dim=g_hidden_dim,
                                               f_num_layers=f_num_layers,
                                               g_num_layers=g_num_layers,
                                               warmup_length=ex_warmup_length)
                if freeze:
                    for param in model.invariant.parameters():
                        param.requires_grad = False
                    model.invariant.eval()

                # wandb_logger.watch(model, log="all")
                #
                early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=10)
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
                    accelerator="gpu", devices=1,
                    callbacks=[
                        model_callback_val,
                        early_stop_callback,
                    ],
                    # fast_dev_run=True
                )

                trainer1.fit(model, train_loader, val_loader)

                wandb.save(get_latest_file())
                wandb.finish()

        # break
