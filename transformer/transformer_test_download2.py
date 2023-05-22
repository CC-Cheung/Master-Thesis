import os
import torch
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
def make_quad_data(num_domains, num_points, coef):
  x=np.random.rand(num_domains, num_points, in_dim)

  y=coef[:,:1, np.newaxis]*x**2 + coef[:,1:2, np.newaxis]*x + coef[:,-1:, np.newaxis]
  return torch.Tensor(np.concatenate((x,y) ,axis=-1)), torch.Tensor(coef)

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
        self.net = nn.ModuleList([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
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
class reshape_linear_relu(linear_relu):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
    def forward(self, x):
        shape=x.shape
        return self.net(x.reshape(shape[0]*shape[1], shape [2])).reshape(shape[0], shape[1], -1)

class TransformerBlock(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, num_heads=1):
        super().__init__()
        self.lin1 = linear(in_dim, embed_dim, embed_dim, 2)

        self.att1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)

        self.lin2 = linear(embed_dim, embed_dim, out_dim, 2)
        self.a = nn.Linear(3,3)

    def forward(self, x):
        embed = self.lin1(x)
        after_layer1 = self.layer_norm1(self.att1(embed, embed, embed)[0] + embed)  # arbitrary
        result = self.layer_norm2(self.lin2(after_layer1))
        return result

class BaselineTrans(pl.LightningModule):
    def __init__(self, in_dim, embed_dim, out_dim, num_heads=1):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = TransformerBlock(in_dim, embed_dim, out_dim, num_heads=num_heads)


        self.loss=nn.MSELoss()
    def forward(self, x):
        result = self.transformer(x).mean(dim=1)
        # result = self.transformer(x)[:, 0]

        return result

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        x, y= batch
        result=self(x)
        loss = self.loss(y, result)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) :
        x, y = batch
        result = self(x)
        loss = self.loss(y, result)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__=="__main__":
    # define any number of nn.Modules (or use your current ones)
    #hello
    in_dim = 1
    hidden_dim_nature = 3
    hidden_dim_pred = 4
    num_heads = 1
    out_dim = 1
    depth = 2
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    key="c20d41ecf28a9b0efa2c5acb361828d1319bc62e"
    train_coef = np.array([[1, 0, 1],[0, 1, 1], [1, 1, 0]])
    train_coef = np.random.rand(100,3)

    # train_coef=np.random.rand(100,3)
    val_coef = np.array([[0.8, 0.1, .2]])
    num_points = 10
    num_domains = train_coef.shape[0]
    num_val_io_pairs=5
    embed_dim = 8
    max_epoch=1000

    train_dataset = TensorDataset(*make_quad_data(num_domains, num_points,train_coef))
    val_dataset = TensorDataset(*make_quad_data(1, num_points, val_coef))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=10)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/trash")
    wandb_logger=WandbLogger(project="DA Thesis", name="trash",log_model="True")
    wandb.init()
    wandb.save("transformer_test.py")
    # add multiple parameters
    wandb_logger.experiment.config.update({"train_coef": train_coef,
                                           "val_coef": val_coef,
                                           "num_points": num_points,
                                           "num_domains": num_domains,
                                           "embed_dim":embed_dim,
                                           "max_epoch": max_epoch})

    # base_trans=BaselineTrans(in_dim+out_dim,embed_dim, 3, num_heads)
    # wandb_logger.watch(base_trans)
    # list_of_files = glob.glob('DA Thesis/**/*.ckpt', recursive=True)  # * means all if need specific format then *.csv
    # latest_file = max(list_of_files, key=os.path.getctime)
    # base_trans=BaselineTrans.load_from_checkpoint(latest_file,
    #                                             in_dim=in_dim + out_dim, embed_dim=embed_dim, out_dim=3, num_heads=num_heads)

    base_trans = BaselineTrans.load_from_checkpoint("/home/cbcheung/Work/Thesis/Code/epoch=999-step=50000.ckpt",
                                                    in_dim=in_dim + out_dim, embed_dim=embed_dim, out_dim=3,
                                                    num_heads=num_heads)
    trainer1 = pl.Trainer(limit_train_batches=100,
                     logger=wandb_logger,
                      max_epochs=max_epoch,
                      # log_every_n_steps = 5,
                    accelerator="gpu", devices=1,
                    #   fast_dev_run=True
                      )
    # trainer2 = pl.Trainer(limit_train_batches=100,
    #                  logger=tb_logger,
    #                   max_epochs=2400,
    #                   log_every_n_steps = 1,
    # accelerator = "gpu", devices = 1,
    #
    # # fast_dev_run=True
    #
    #                   )


    #
    # idea_module

    # trainer1.fit(idea_module2, train_loader,val_loader)

    trainer1.fit(base_trans, train_loader, val_loader)
    list_of_files = glob.glob('DA Thesis/**/*.ckpt',recursive=True)  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    wandb.save(latest_file)