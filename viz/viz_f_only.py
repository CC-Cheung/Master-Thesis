import pandas as pd
import wandb
api = wandb.Api()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import numpy as np
import os
folder=os.path.join("..", "Download", "run data")

df=pd.read_csv(os.path.join(folder,"default_invariant.csv")).drop("Unnamed: 0", axis=1)
# fig, axs = plt.subplots(5, 4)
df = df.drop(df[df["name"] == "trash"].index)

gs=[2, 4, 8, 16, 32]
cm_subsection = [i / len(gs) for i in range(len(gs))]
colors = {gs[i]: cm.viridis(cm_subsection[i]) for i in range(len(gs))}

for row, d in enumerate(np.sort(pd.unique(df["num_domains"]))):
    for col, f in enumerate(np.sort(pd.unique(df["f_embed_dim"]))):
        # for d in pd.unique(df["num_domains"].sort()):

        df_df=df[(df["num_domains"]==d) & (df["f_embed_dim"]==f)]

        if len(df_df)<2:
            continue
        # fig = plt.figure()
        # ax = fig.add_subplot()

        fs=np.sort(pd.unique(df_df["f_embed_dim"]))

        for g in gs:
            # Generate the random data for the y=k 'layer'.

            df_dfg=df_df[df_df["g_embed_dim"] == g].sort_values("split")

            splits_prev = 0
            to_delete = []
            for i in range(len(df_dfg["split"])):
                if df_dfg.iloc[i, :]["split"] == splits_prev or \
                        df_dfg.iloc[i, :]["split"] not in (0.01, 0.02, 0.05, 0.1, 0.2, 0.4):
                    to_delete.append(i)
                splits_prev = df_dfg.iloc[i, :]["split"]
            if len(df_dfg)==0:
                continue



            # ax.plot(df_fgd_no_freeze["split"], df_fgd_no_freeze["val_loss"],
            #         color=c,
            #         alpha=0.8,
            #         marker=".",
            #         label=f"{d} domain, no freeze")
            # ax.plot(df_fgd_freeze["split"], df_fgd_freeze["val_loss"],
            #         color=c,
            #         alpha=0.8,
            #         label=f"{d} domain",
            #         marker="P")

            plt.plot(df_dfg["split"], df_dfg["val_loss"] ,
                     color=colors[g],
                     alpha=0.8,
                     label=f"{g} g",
                     marker="P")

        # axs[row,col].set_xlabel('Split')
        # axs[row,col].set_ylabel('Freeze Loss - No freeze')
        # axs[row,col].set_title(f"F{f}G{g}")
        # axs[row,col].set_xscale("log")
        plt.xlabel('Split')
        # plt.ylabel('Freeze Loss - No freeze')
        plt.title(f"D{d}F{f} f only")
        plt.xscale("log")
        plt.legend()
        plt.savefig(os.path.join("../Output", "fd_const", f"D{d}F{f} f only.png"))

        plt.show()

print("hi")