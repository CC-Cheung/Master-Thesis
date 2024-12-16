import pandas as pd
import wandb
api = wandb.Api()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import os
import torch


size_idx=["f_hidden_dim", "f_num_layers", "g_hidden_dim", "g_num_layers",]
model_idx=size_idx + ["freeze", "res_decrease"]
all_idx=model_idx+["name", "val_loss"]
folder=os.path.join("..", "Download", "run data")
infos=[("pretrain 14","dashed" ),("Maumee Basin", "dotted"), ("various", "solid")]
# infos=[("Just Maumee","dashed" ),("Just Cuyahoga", "dotted"), ("various", "solid")]
# infos=[("add", "dotted"), ("default eta", "solid")]
infos=[("no pretrain", "dotted"), ("default DA", "solid")]
mode_dict= {'diff_val_loss':"Validation Loss Difference",  'freeze_val_loss':"Validation Loss"}


purpose="default"
purpose="res"
# purpose="split_double"
purpose="cool_eta"
purpose="cool eta log"
# purpose="various"
# purpose="cool eta log Maumee"
# purpose="cool eta log pretrain 14"
purpose="cool eta log no pretrain no add mixed"
output_folder="../Output/log"

col="res_decrease" if purpose=="res" or "cool_eta" else "split"

def get_seed_df(csvs):
    no_freeze="no pretrain" in csvs[0]
    dfs=[pd.read_csv(os.path.join(folder,csv)).drop("Unnamed: 0", axis=1)[all_idx].sort_values(model_idx)
         for csv in csvs]

    dfs_processed=[]
    if no_freeze:
        for df in dfs:
            temp = df.rename(columns={"val_loss": 'freeze_val_loss'})
            dfs_processed.append(temp)
    else:
        for df in dfs:
            temp=df[df["freeze"]==True].reset_index(drop=True)
            temp=temp.rename(columns={"val_loss":'freeze_val_loss'})
            temp['no_freeze_val_loss']=df[df["freeze"]==False]['val_loss'].reset_index(drop=True)
            temp['diff_val_loss']=temp['freeze_val_loss']-temp['no_freeze_val_loss']
            dfs_processed.append(temp)
    df_all=pd.concat(dfs_processed, axis=0)
    together=df_all.groupby(model_idx)
    together_mean = together.mean().add_suffix("_mean")
    together_max = together.max().add_suffix("_max")
    together_min = together.min().add_suffix("_min")
    df_stats = pd.concat([together_mean, together_min, together_max], axis=1)

    return df_stats



# get_seed_df(["cool eta log Maumee"+ s+"_variant.csv" for s in ('', ' seed 1', ' seed 2')])

p=0
# df_no_pretrains=pd.read_csv(os.path.join(folder,"cool eta log no pretrain seed 1_variant.csv")).drop("Unnamed: 0", axis=1)

# df=pd.read_csv(os.path.join(folder,purpose+"_variant.csv")).drop("Unnamed: 0", axis=1)

# df_orig = df.drop(df[df["name"] == "trash"].index)

# df=pd.read_csv(os.path.join(folder,"metrics0.csv")).drop("Unnamed: 0", axis=1)
# df_orig = df.drop(df[df["name"] == "trash"].index)
# df2=pd.read_csv(os.path.join(folder,"metrics1.csv")).drop("Unnamed: 0", axis=1)
# df_orig2 = df2.drop(df2[df2["name"] == "trash"].index)
# df3=pd.read_csv(os.path.join(folder,"metrics2.csv")).drop("Unnamed: 0", axis=1)
# df_orig3 = df3.drop(df3[df3["name"] == "trash"].index)
# val_col=df.columns[df.columns.str.contains("Value") & df.columns.str.contains("nse")].str.replace(" nse", '')

# df2=pd.read_csv(os.path.join(folder,"cool eta log seed 1"+"_variant.csv")).drop("Unnamed: 0", axis=1)
# df_orig2 = df2.drop(df2[df2["name"] == "trash"].index)
# df3=pd.read_csv(os.path.join(folder,"cool eta log seed 2"+"_variant.csv")).drop("Unnamed: 0", axis=1)
# df_orig3 = df3.drop(df3[df3["name"] == "trash"].index)
# #
# merge=pd.concat([df_orig2,df_orig3,df_orig], axis=0).drop(["name","id"], axis=1)
# # merge=merge.drop(merge.columns[merge.columns.str.contains("Conductivity")], axis=1)
# together=merge.groupby(size_idx+ ["freeze", "res_decrease"])
# together_mean=together.mean().add_suffix("_mean")
# together_min=together.min().add_suffix("_min")
# together_max=together.max().add_suffix("_max")
# df_stats=pd.concat([together_mean, together_min,together_max], axis=1)
# df_stats.to_csv(os.path.join(folder, "metric.csv"))
#
# df_stats_mean=df_stats[df_stats.columns[df_stats.columns.str.contains("_mean")]].mean()
# df_stats_min=df_stats[df_stats.columns[df_stats.columns.str.contains("_min")]].min()
# df_stats_max=df_stats[df_stats.columns[df_stats.columns.str.contains("_max")]].max()
# df_stats_stats=pd.concat([df_stats_min,df_stats_max,df_stats_mean])
# df_stats_stats=pd.DataFrame({"Feature": val_col}|
# {metric+group: df_stats_stats[df_stats_stats.index.str.contains(metric+group)].reset_index(drop=True)
#  for metric in ("nse", "mse", "pcc", "kge")
#  for group in ("_mean", "_min", "_max")})
# df_stats_stats.to_csv(os.path.join(folder, "metric stats.csv"))
#
# df_stats=df_stats.reset_index()
# for merge_method in ("_mean", "_max", "_min"):
    # for metric in ("nse", "pcc","kge"):
        # df_stats[metric+merge_method]=df_stats.loc[:,df_stats.columns.str.contains(metric+merge_method)].mean(axis=1)
# # together_diff=merge[merge['freeze']==True].sort_values(size_idx+["res_decrease"]).reset_index(drop=True)
# # together_diff["val_loss"]= together_diff["val_loss"]-merge[merge['freeze']==False].sort_values(size_idx+["res_decrease"])["val_loss"].reset_index(drop=True)
# # together_diff=together_diff.groupby(["f_hidden_dim", "f_num_layers", "g_hidden_dim", "g_num_layers",  "res_decrease"])
# # together_diff_mean=together_diff.mean().add_suffix("_mean")
# # together_diff_min=together_diff.min().add_suffix("_min")
# # together_diff_max=together_diff.max().add_suffix("_max")
# # df_stats_diff=pd.concat([together_diff_mean, together_diff_min,together_diff_max], axis=1).reset_index()
# df_stats_diff=None


x_axis=[0.4,0.2,0.1,0.05,0.02,0.01] if purpose=="default" else [0.02, 0.05, 0.1, 0.2,0.4]

def clean_df(df,freeze=None):
    if freeze is None:
        df_f=df
    else:
        df_f = df[df["freeze"] == freeze]
    df_f=df_f.reset_index()
    cols_prev = 0
    to_delete = []

    for i in range(len(df_f[col])):
        if df_f.iloc[i, :][col] == cols_prev or \
                df_f.iloc[i, :][col] not in x_axis:
            to_delete.append(i)
        cols_prev = df_f.iloc[i, :][col]

    return df_f

# def plot_fix2_for1_vary1(fix2, for1, vary1, diff=True):
#     df = df_orig[(df_orig[fix2[0][0]] == fix2[0][1]) & (df_orig[fix2[1][0]] == fix2[1][1])]
#
#
#     for for1_var in np.sort(pd.unique(df[for1])):
#         # for d in pd.unique(df["num_domains"].sort()):
#
#         df_gl=df[df[for1]==for1_var]
#
#         if len(df_gl)<10:
#             continue
#         # fig = plt.figure()
#         # ax = fig.add_subplot()
#
#         for vary1_var in np.sort(pd.unique(df_gl[vary1])):
#             # Generate the random data for the y=k 'layer'.
#
#             df_glgd=df_gl[df_gl[vary1] == vary1_var].sort_values(col)
#             df_glgd_freeze=clean_df(df_glgd, True)
#             df_glgd_no_freeze=clean_df(df_glgd, False)
#             if len(df_glgd_freeze)!=len(df_glgd_no_freeze):
#                 continue
#             if diff:
#                 plt.plot(df_glgd_freeze[col], df_glgd_freeze["val_loss"]-df_glgd_no_freeze ["val_loss"],
#                          # color=colors[g],
#                          alpha=0.8,
#                          label=f"{vary1_var} {vary1}",
#                          marker="P")
#             else:
#                 plt.plot(df_glgd_freeze[col], df_glgd_freeze["val_loss"],
#                          # color=colors[g],
#                          alpha=0.8,
#                          label=f"{vary1_var} {vary1}",
#                          marker="P")
#
#         plt.xlabel(col)
#         # plt.ylabel('Freeze Loss - No freeze')
#         title=f"cool eta{fix2[0][0]} {fix2[0][1]} {fix2[1][0]} {fix2[1][1]} {for1}{for1_var} diff {diff}"
#         plt.title(title)
#         plt.xscale("log")
#         plt.legend()
#         plt.savefig(os.path.join(output_folder, "variant", title+".png"))
#
#         plt.show()
# def plot_fix2(fix2):
#     df = df_orig[(df_orig[fix2[0][0]] == fix2[0][1]) & (df_orig[fix2[1][0]] == fix2[1][1])].copy()
#     plot_all(df, diff=True, title=fix2)
#     plot_all(df, diff=False, title=fix2)
def plot_fix3(fix3):
    df = df_orig[(df_orig[fix3[0][0]] == fix3[0][1]) &
                 (df_orig[fix3[1][0]] == fix3[1][1]) &
                 (df_orig[fix3[2][0]] == fix3[2][1])].copy()

    size_vars=['f_hidden_dim',
    'f_num_layers',
     'g_hidden_dim',
    'g_num_layers']
    provided=[i[0] for i in fix3]
    size_strs = []
    count=0
    for i, var in enumerate(size_vars):
        if var in provided:
            size_strs.append(fix3[count][1])
            count+=1
        else:
            size_strs.append("_")
    size_str=f"f{size_strs[0]}, {size_strs[1]} g{size_strs[2]},{size_strs[3]}"

    # plot_all(df, diff=True, title=purpose.replace("cool eta log", "") +size_str)
    # plot_all(df, diff=False, title=purpose.replace("cool eta log", "") +size_str)
    # plot_all(df, diff=True, title="no pretrain " +size_str)
    plot_all(df, diff=False, title="Temporally Mixed " +size_str)

def plot_all(df, diff=True, title=""):
    f_hidden_dims=np.sort(df['f_hidden_dim'].unique())
    g_hidden_dims=np.sort(df['g_hidden_dim'].unique())
    f_num_layers=np.sort(df['f_num_layers'].unique())
    g_num_layers=np.sort(df['g_num_layers'].unique())
    counter=0
    for f_hidden_dim in f_hidden_dims:
        for g_hidden_dim in g_hidden_dims:
            for f_num_layer in f_num_layers:
                for g_num_layer in g_num_layers:
                    # if [f_hidden_dim,f_num_layer,g_hidden_dim,g_num_layer] ==[32,2,8,2]:
                    #     continue
                    df_temp = df[(df['f_hidden_dim'] == f_hidden_dim) &
                                 (df['g_hidden_dim'] == g_hidden_dim) &
                                 (df['f_num_layers'] == f_num_layer) &
                                 (df['g_num_layers'] == g_num_layer) ]
                    # df_no_pretrain=df_no_pretrains[(df_no_pretrains['f_hidden_dim'] == f_hidden_dim) &
                    #              (df_no_pretrains['g_hidden_dim'] == g_hidden_dim) &
                    #              (df_no_pretrains['f_num_layers'] == f_num_layer) &
                    #              (df_no_pretrains['g_num_layers'] == g_num_layer) ]

                    df_glgd=df_temp.sort_values(col)
                    df_glgd_freeze=clean_df(df_glgd, True)
                    df_glgd_no_freeze=clean_df(df_glgd, False)
                    # if len(df_glgd_freeze)!=len(df_glgd_no_freeze):
                    #     continue
                    # if len(df_no_pretrain) == 0:
                    #     continue
                    if diff:
                        # plt.plot(df_glgd_freeze[col], df_glgd_freeze["val_loss"]-df_glgd_no_freeze ["val_loss"],
                        #          color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                        #          label=f"f{f_hidden_dim},{f_num_layer} g{g_hidden_dim},{g_num_layer}",
                        #          marker="P")
                        plt.plot(df_glgd_no_freeze[col], df_glgd_no_freeze["val_loss"] - df_glgd_no_freeze["train_loss"],
                                 color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                                 label=f"f{f_hidden_dim},{f_num_layer} g{g_hidden_dim},{g_num_layer}",
                                 marker="P")
                    else:
                        # plt.plot(df_glgd_freeze[col], df_glgd_freeze["val_loss"],
                        #          color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                        #          label=f"f{f_hidden_dim},{f_num_layer} g{g_hidden_dim},{g_num_layer}",
                        #          marker="P")
                        plt.plot(df_glgd_no_freeze[col], df_glgd_no_freeze["val_loss"],
                                 color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                                 label=f"f{f_hidden_dim},{f_num_layer} g{g_hidden_dim},{g_num_layer}, no pretrain",
                                 marker="P")
                        # plt.plot(df_glgd_no_freeze[col], df_glgd_no_freeze["train_loss"],
                        #          color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                        #          label=f"f{f_hidden_dim},{f_num_layer} g{g_hidden_dim},{g_num_layer}, no pretrain",
                        #          marker="P")
                        # plt.plot(df_no_pretrain[col], df_no_pretrain["val_loss"],
                        #          color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                        #          alpha=0.8,
                        #          linestyle='dashed',
                        #          label=f"f{f_hidden_dim},{f_num_layer} g{g_hidden_dim},{g_num_layer}, no pretrain",
                        #          marker="X")
                    counter+=1

    plt.xlabel("Resolution Decrease")
    # plt.ylabel('Freeze Loss - No freeze')
    # diff_text= "Validation Loss Difference" if diff else "Freeze Validation Loss"
    # diff_text= "Val train diff" if diff else "Train Loss"
    diff_text= "Val train diff" if diff else "Validation Loss"


    title=f"{title} {diff_text}"
    plt.title(title)
    plt.xscale("log")
    plt.xticks(df_glgd_no_freeze[col].unique(), labels=df_glgd_no_freeze[col].unique())

    plt.legend()
    # plt.savefig(os.path.join(output_folder, "variant", title+".png"))

    plt.show()

def plot_fix3_seed(fix3):
    df = df_stats[(df_stats[fix3[0][0]] == fix3[0][1]) &
                 (df_stats[fix3[1][0]] == fix3[1][1]) &
                 (df_stats[fix3[2][0]] == fix3[2][1])].copy()
    df_diff = df_stats_diff[(df_stats_diff[fix3[0][0]] == fix3[0][1]) &
                  (df_stats_diff[fix3[1][0]] == fix3[1][1]) &
                  (df_stats_diff[fix3[2][0]] == fix3[2][1])].copy()
    df_diff = None
    size_vars=['f_hidden_dim',
    'f_num_layers',
     'g_hidden_dim',
    'g_num_layers']
    provided=[i[0] for i in fix3]
    size_strs = []
    count=0
    for i, var in enumerate(size_vars):
        if var in provided:
            size_strs.append(fix3[count][1])
            count+=1
        else:
            size_strs.append("_")
    size_str=f"f{size_strs[0]},{size_strs[1]} g{size_strs[2]},{size_strs[3]}"

    plot_all_seed(df, df_diff, diff=True, title="3 seeds " +size_str)
    plot_all_seed(df, df_diff, diff=False, title=" " +size_str)

    # plot_all(df, diff=False, title="baseline " +size_str)

def plot_all_seed(df,df_diff, diff=True, title=""):
    f_hidden_dims=np.sort(df['f_hidden_dim'].unique())
    g_hidden_dims=np.sort(df['g_hidden_dim'].unique())
    f_num_layers=np.sort(df['f_num_layers'].unique())
    g_num_layers=np.sort(df['g_num_layers'].unique())
    counter=0
    # metric="nse"
    # metric="pcc"
    # metric="kge"

    for f_hidden_dim in f_hidden_dims:
        for g_hidden_dim in g_hidden_dims:
            for f_num_layer in f_num_layers:
                for g_num_layer in g_num_layers:
                    df_temp = df[(df['f_hidden_dim'] == f_hidden_dim) &
                                 (df['g_hidden_dim'] == g_hidden_dim) &
                                 (df['f_num_layers'] == f_num_layer) &
                                 (df['g_num_layers'] == g_num_layer) ]
                    # df_temp_diff=df_diff[(df_diff['f_hidden_dim'] == f_hidden_dim) &
                    #              (df_diff['g_hidden_dim'] == g_hidden_dim) &
                    #              (df_diff['f_num_layers'] == f_num_layer) &
                    #              (df_diff['g_num_layers'] == g_num_layer) ]

                    df_glgd=df_temp.sort_values(col)
                    df_glgd_freeze=clean_df(df_glgd, True)
                    # df_glgd_diff = df_temp_diff.sort_values(col)
                    # df_glgd_diff = clean_df(df_glgd_diff)

                    # if len(df_no_pretrain) == 0:
                    #     continue
                    if diff:
                        plt.plot(df_glgd_freeze[col], df_glgd_diff["val_loss_mean"],
                                 color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                                 label=f"f{f_hidden_dim},{f_num_layer} g{g_hidden_dim},{g_num_layer}",
                                 marker="P")
                        plt.fill_between(df_glgd_freeze[col],
                                         df_glgd_diff["val_loss_min"] ,
                                         df_glgd_diff["val_loss_max"] ,
                                         color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                                         alpha=0.2,
                                         )
                        pass
                    else:
                        # plt.plot(df_glgd_freeze[col], df_glgd_freeze["val_loss_mean"],
                        #          color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                        #          label=f"f{f_hidden_dim},{f_num_layer} g{g_hidden_dim},{g_num_layer}",
                        #          marker="P")
                        # plt.fill_between(df_glgd_freeze[col],
                        #                  df_glgd_freeze["val_loss_min"] ,
                        #                  df_glgd_freeze["val_loss_max"] ,
                        #                  color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                        #                  alpha=0.2,
                        #                  )
                        plt.plot(df_glgd_freeze[col], df_glgd_freeze[metric+"_mean"],
                                 color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                                 label=f"f{f_hidden_dim},{f_num_layer} g{g_hidden_dim},{g_num_layer}",
                                 marker="P")
                        plt.fill_between(df_glgd_freeze[col],
                                         df_glgd_freeze[metric+"_min"],
                                         df_glgd_freeze[metric+"_max"],
                                         color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                                         alpha=0.2,
                                         )
                    counter+=1


    plt.xlabel('Resolution Decrease')

    # plt.ylabel('Freeze Loss - No freeze')
    # diff_text= "Validation Loss Difference" if diff else "Freeze Validation Loss"
    diff_text= metric

    # title=f"{title} {diff_text}"
    title=f"{diff_text}{title}"

    plt.title(title)
    plt.xscale("log")
    plt.xticks(df_glgd_freeze[col].unique(), labels=df_glgd_freeze[col].unique())
    plt.legend()
    plt.savefig(os.path.join(output_folder, "variant", title+".png"))

    plt.show()
# #
#
def plot_fix3s(fix3):
    # name="Many Domains"
    # df1 = pd.read_csv(os.path.join(folder, "cool eta log pretrain 14" + "_variant.csv")).drop("Unnamed: 0", axis=1)
    # df2 = pd.read_csv(os.path.join(folder, "cool eta log Maumee"  + "_variant.csv")).drop("Unnamed: 0", axis=1)
    # df3= pd.read_csv(os.path.join(folder, "cool eta log"  + "_variant.csv")).drop("Unnamed: 0", axis=1)
    # dfs=[df1,df2,df3]

    # name="Little Domains"
    # df1 = pd.read_csv(os.path.join(folder, "cool eta log Just Maumee" + "_variant.csv")).drop("Unnamed: 0", axis=1)
    # df2 = pd.read_csv(os.path.join(folder, "cool eta log Just Cuyahoga" + "_variant.csv")).drop("Unnamed: 0", axis=1)
    # df3 = pd.read_csv(os.path.join(folder, "cool eta log" + "_variant.csv")).drop("Unnamed: 0", axis=1)
    # dfs=[df1,df2,df3]

    # name="Eta"
    # df1 = pd.read_csv(os.path.join(folder, "various" + "_variant.csv")).drop("Unnamed: 0", axis=1)
    # df2 = pd.read_csv(os.path.join(folder, "cool eta log" + "_variant.csv")).drop("Unnamed: 0", axis=1)
    # dfs=[df1,df2]

    name = "Baseline"
    df1 = pd.read_csv(os.path.join(folder, "cool eta log no pretrain" + "_variant.csv")).drop("Unnamed: 0", axis=1)
    df2 = pd.read_csv(os.path.join(folder, "cool eta log" + "_variant.csv")).drop("Unnamed: 0", axis=1)
    df1['freeze']=True
    dfs = [df1, df2]


    dfs = [df[(df[fix3[0][0]] == fix3[0][1]) &
                 (df[fix3[1][0]] == fix3[1][1]) &
                 (df[fix3[2][0]] == fix3[2][1])] for df in dfs]

    size_vars=['f_hidden_dim',
    'f_num_layers',
     'g_hidden_dim',
    'g_num_layers']
    provided=[i[0] for i in fix3]
    size_strs = []
    count=0
    for i, var in enumerate(size_vars):
        if var in provided:
            size_strs.append(fix3[count][1])
            count+=1
        else:
            size_strs.append("_")
    size_str=f"f{size_strs[0]}, {size_strs[1]} g{size_strs[2]},{size_strs[3]}"

    # plot_alls(dfs, diff=True, title=name+" "+size_str)
    plot_alls(dfs, diff=False, title=name+" "+size_str)

def plot_alls(dfs, diff=True, title=""):
    f_hidden_dims = np.sort(dfs[0]['f_hidden_dim'].unique())
    g_hidden_dims = np.sort(dfs[0]['g_hidden_dim'].unique())
    f_num_layers = np.sort(dfs[0]['f_num_layers'].unique())
    g_num_layers = np.sort(dfs[0]['g_num_layers'].unique())
    counter=0
    for f_hidden_dim in f_hidden_dims:
        for g_hidden_dim in g_hidden_dims:
            for f_num_layer in f_num_layers:
                for g_num_layer in g_num_layers:
                    df_temps = [df[(df['f_hidden_dim'] == f_hidden_dim) &
                                 (df['g_hidden_dim'] == g_hidden_dim) &
                                 (df['f_num_layers'] == f_num_layer) &
                                 (df['g_num_layers'] == g_num_layer) ] for df in dfs]
                    # df_no_pretrain=df_no_pretrains[(df_no_pretrains['f_hidden_dim'] == f_hidden_dim) &
                    #              (df_no_pretrains['g_hidden_dim'] == g_hidden_dim) &
                    #              (df_no_pretrains['f_num_layers'] == f_num_layer) &
                    #              (df_no_pretrains['g_num_layers'] == g_num_layer) ]

                    df_glgds=[df_temp.sort_values(col) for df_temp in df_temps]
                    df_glgd_freezes=[clean_df(df_glgd, True) for df_glgd in df_glgds]
                    df_glgd_no_freezes=[clean_df(df_glgd, False) for df_glgd in df_glgds]
                    # if len(df_glgd_freeze)!=len(df_glgd_no_freeze):
                    #     continue
                    for info, df_glgd_freeze, df_glgd_no_freeze in zip (infos, df_glgd_freezes,df_glgd_no_freezes):
                        if diff:
                            plt.plot(df_glgd_freeze[col], df_glgd_freeze["val_loss"]-df_glgd_no_freeze ["val_loss"],
                                     color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                                     alpha=0.8,
                                     label=f"f{f_hidden_dim},{f_num_layer} g{g_hidden_dim},{g_num_layer}, {info[0]}",
                                     marker="P",
                                     linestyle=info[1]
                                     )
                        else:
                            plt.plot(df_glgd_freeze[col], df_glgd_freeze["val_loss"],
                                     color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                                     alpha=0.8,
                                     label=f"f{f_hidden_dim},{f_num_layer} g{g_hidden_dim},{g_num_layer},{info[0]}",
                                     marker="P",
                                     linestyle=info[1]
                                     )
                        # plt.plot(df_no_pretrain[col], df_no_pretrain["val_loss"],
                        #          color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                        #          alpha=0.8,
                        #          linestyle='dashed',
                        #          label=f"f{f_hidden_dim},{f_num_layer} g{g_hidden_dim},{g_num_layer}, no pretrain",
                        #          marker="X")
                    counter+=1

    plt.xlabel('Resolution Decrease')
    plt.ylabel('Loss')

    # plt.ylabel('Freeze Loss - No freeze')
    diff_text= "Validation Loss Difference" if diff else "Freeze Validation Loss"
    title=f"{title} {diff_text}"
    plt.title(title)
    plt.xscale("log")
    plt.xticks(df_temps[0][col].unique(), labels=df_temps[0][col].unique())

    # leg = plt.legend(framealpha=0.2)
    leg = plt.legend()

    # for _txt in leg.texts:
    #     _txt.set_alpha(0.4)
    # for line in leg.legendHandles:
    #     line.set_alpha(0.3)
    plt.savefig(os.path.join(output_folder, "variant", title+".png"))

    plt.show()

def plot_source_seed_fix3s(fix3s):
    # dfs = [get_seed_df(["cool eta log" + purpose + s + "_variant.csv"
    #                     for s in (
    #                         # '',
    #                         #       ' seed 1',
    #                               ' seed 2',
    #                               )]).reset_index()
    #        for purpose in (" Maumee", " pretrain 14", "")]
    # name="Large Pretrain Set"
    # infos = [("Maumee Basin", "dashed"), ("Pretrain 14", "dotted"), ('Various', "solid")]

    # dfs = [get_seed_df(["cool eta log" + purpose + s + "_variant.csv" for s in ('', ' seed 1', ' seed 2')]).reset_index()
    #        for purpose in (" Just Maumee", " Just Cuyahoga", "")]
    # name="Small Pretrain Set"
    # infos = [("Just Maumee", "dashed"), ("Just Cuyahoga", "dotted"), ('Various', "solid")]
    # dfs = [get_seed_df(["cool eta log" + purpose + s + "_variant.csv"
    #                     for s in (
    #                         '',
    #                               ' seed 1',
    #                         ' seed 2',
    #                     )]).reset_index()
    #        for purpose in (" no pretrain", "")]
    # name = "Baseline"
    # infos = [("Baseline", "dashed"), ("Default", "solid")]
    dfs = [get_seed_df([purpose + s + "_variant.csv"
                        for s in (
                            '',
                            ' seed 1',
                            ' seed 2',
                        )]).reset_index()
           for purpose in ("various", "cool eta log")]
    name = "Different Utilization of f and g"
    infos = [("Adding", "dashed"), ("Default", "solid")]
    # dfs[-1] = dfs[-1][(dfs[-1][size_idx] == (32, 2, 8, 2)).sum(axis=1) >= 3]
    # dfs[-1] = dfs[-1][(dfs[-1]['f_hidden_dim'].isin((8, 32, 128))) &
    #                   (dfs[-1]['g_hidden_dim'].isin((8, 32, 128))) &
    #                   (dfs[-1]['f_num_layers'].isin((1, 2))) &
    #                   (dfs[-1]['g_num_layers'].isin((1, 2)))]


    # dfs = [get_seed_df(["cool eta log" + purpose + s + "_variant.csv"
    #                     for s in (
    #                         '',
    #                         ' seed 1',
    #                         ' seed 2',
    #                     )]).reset_index()
    #        for purpose in ("",)]
    # name = "Various sizes"
    # infos = [("", "solid")]

    dfs = [df[(df[fix3s[0][0]] == fix3s[0][1]) &
              (df[fix3s[1][0]] == fix3s[1][1]) &
              (df[fix3s[2][0]] == fix3s[2][1])] for df in dfs]
    size_vars=['f_hidden_dim',
    'f_num_layers',
     'g_hidden_dim',
    'g_num_layers']
    provided=[i[0] for i in fix3s]
    size_strs = []
    count=0
    vary=''
    for i, var in enumerate(size_vars):
        if var in provided:
            size_strs.append(fix3s[count][1])
            count+=1
        else:
            vary=var
            size_strs.append("_")
    size_str=f"f{size_strs[0]},{size_strs[1]} g{size_strs[2]},{size_strs[3]}"
    plot_source_seed_fix3_alls(dfs, modes=[
        'freeze_val_loss',
        "diff_val_loss"
    ],
                               vary=vary, size_str=size_str,infos=infos, title=name+" "+size_str)

def plot_source_seed_fix3_alls(dfs, modes, vary, size_str, infos, title=""):
    vary_values = np.sort(dfs[0][vary].unique())
    for mode in modes:
        counter = 0
        for vary_value in vary_values:
            df_temps = [df[df[vary] == vary_value] for df in dfs]
            # if len(df_glgd_freeze)!=len(df_glgd_no_freeze):
            #     continue
            size_str_label=''
            for size_str_split in size_str.split(' '):
                if '_' in size_str_split:
                    size_str_label=size_str_split.replace('_', str(vary_value))
            for df, info in zip(df_temps, infos):

                plt.plot([1, 3, 6, 12, 25], df[mode+"_mean"],
                 color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                 alpha=0.8,
                 # label=f"{size_str.replace('_', str(vary_value))}, {info[0]}",
                 label=f"{size_str_label}, {info[0]}",
                 marker="P",
                 linestyle=info[1]
                 )
                # plt.fill_between(df[col],
                #                  df[mode+"_min"],
                #                  df[mode+"_max"],
                #                  color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                #                  alpha=0.2,
                #                  )

            counter+=1

        plt.xlabel('Number of Measured Timesteps per 64 Days')
        plt.ylabel(mode_dict[mode])
        # plt.ylabel('Freeze Loss - No freeze')
        diff_text= mode_dict[mode]
        actual_title=f"{title} {diff_text}"
        plt.title(actual_title)
        plt.xscale("log")
        plt.xticks([1, 3, 6, 12, 25], [1, 3, 6, 12, 25])
        # leg = plt.legend(framealpha=0.2)
        # for _txt in leg.texts:
        #     _txt.set_alpha(0.4)
        # for line in leg.legendHandles:
        #     line.set_alpha(0.3)

        leg = plt.legend()

        plt.savefig(os.path.join(output_folder, "variant", actual_title+".png"))

        plt.show()

# filts=[[['f_hidden_dim', 16], ['f_num_layers', 2]],[['g_num_layers', 4], ['g_hidden_dim', 2]]]
# for fix_which in [0,1]:
#     for vary_which in [0,1]:
#         for diff in [True, False]:
#             plot_fix2_for1_vary1(filts[fix_which], filts[(fix_which+1)%2][vary_which][0],
#                                  filts[(fix_which+1)%2][(vary_which+1)%2][0], diff=diff)

# plot_fix2_for1_vary1([['f_hidden_dim', 16], ['f_num_layers', 2]], "g_hidden_dim", "g_num_layers",)
# plot_fix2_for1_vary1([['f_hidden_dim', 16], ['f_num_layers', 2]],  "g_num_layers", "g_hidden_dim")
# plot_fix2_for1_vary1([['g_num_layers', 4], ['g_hidden_dim', 2]],  "f_hidden_dim", "f_num_layers")
# plot_fix2_for1_vary1([['g_num_layers', 4], ['g_hidden_dim', 2]],  "f_num_layers", "f_hidden_dim")

# plot_fix2([['f_hidden_dim', 32], ['f_num_layers', 2]])
# plot_fix2([['g_hidden_dim', 8],['g_num_layers', 2] ])
# plot_all(df_orig, True)
# plot_all(df_orig, False)

# plot_fix3([['f_hidden_dim', 32],['g_hidden_dim', 8],['g_num_layers', 2] ])
# plot_fix3([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_hidden_dim', 8]])
# plot_fix3([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_num_layers', 2] ])
# plot_fix3([['f_num_layers', 2] ,['g_hidden_dim', 8],['g_num_layers', 2] ])

plot_fix3_seed([['f_hidden_dim', 32],['g_hidden_dim', 8],['g_num_layers', 2] ])
plot_fix3_seed([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_hidden_dim', 8]])
plot_fix3_seed([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_num_layers', 2] ])
plot_fix3_seed([['f_num_layers', 2] ,['g_hidden_dim', 8],['g_num_layers', 2] ])

# plot_fix3s([['f_hidden_dim', 32],['g_hidden_dim', 8],['g_num_layers', 2] ])
# plot_fix3s([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_hidden_dim', 8]])
# plot_fix3s([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_num_layers', 2] ])
# plot_fix3s([['f_num_layers', 2] ,['g_hidden_dim', 8],['g_num_layers', 2] ])
plot_source_seed_fix3s([['f_hidden_dim', 32],['g_hidden_dim', 8],['g_num_layers', 2] ])
plot_source_seed_fix3s([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_hidden_dim', 8]])
plot_source_seed_fix3s([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_num_layers', 2] ])
plot_source_seed_fix3s([['f_num_layers', 2] ,['g_hidden_dim', 8],['g_num_layers', 2] ])



print("hi")