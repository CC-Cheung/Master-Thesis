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
#Uncomment the right one
infos=[("pretrain 14","dashed" ),("Maumee Basin", "dotted"), ("various", "solid")]
# infos=[("Just Maumee","dashed" ),("Just Cuyahoga", "dotted"), ("various", "solid")]
# infos=[("add", "dotted"), ("default eta", "solid")]

infos=[("no pretrain", "dotted"), ("default DA", "solid")]
mode_dict= {'diff_val_loss':"Validation Loss Difference",
            'freeze_val_loss':"Validation Loss",
            'no_freeze_val_loss': "Not Freezing Validation Loss",
            'nse':'NSE',
            'kge':"KGE"}


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

#Get what you want to plot from csvs
def get_seed_df(csvs):
    no_freeze="no pretrain" in csvs[0]
    use_metrics="metrics" in csvs[0]
    # dfs=[pd.read_csv(os.path.join(folder,csv)).drop("Unnamed: 0", axis=1)[all_idx].sort_values(model_idx)
         # for csv in csvs]
    dfs = [pd.read_csv(os.path.join(folder, csv)).drop("Unnamed: 0", axis=1).sort_values(model_idx)
           for csv in csvs]

    dfs_processed=[]
    if use_metrics:
        for i in range (3):
            for metric in ("nse", "pcc","kge"):
                dfs[i][metric]=dfs[i].loc[:,dfs[i].columns.str.contains(metric)].mean(axis=1)
            dfs[i]=dfs[i].loc[:, dfs[i].columns.str.len() < 20]
        dfs_processed=dfs
    else:
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

# p=0
# # df_no_pretrains=pd.read_csv(os.path.join(folder,"cool eta log no pretrain seed 1_variant.csv")).drop("Unnamed: 0", axis=1)
#
# # df=pd.read_csv(os.path.join(folder,purpose+"_variant.csv")).drop("Unnamed: 0", axis=1)
# #
# # df_orig = df.drop(df[df["name"] == "trash"].index)
#
# df=pd.read_csv(os.path.join(folder,"metrics0.csv")).drop("Unnamed: 0", axis=1)
# df_orig = df.drop(df[df["name"] == "trash"].index)
# df2=pd.read_csv(os.path.join(folder,"metrics1.csv")).drop("Unnamed: 0", axis=1)
# df_orig2 = df2.drop(df2[df2["name"] == "trash"].index)
# df3=pd.read_csv(os.path.join(folder,"metrics2.csv")).drop("Unnamed: 0", axis=1)
# df_orig3 = df3.drop(df3[df3["name"] == "trash"].index)
# val_col=df.columns[df.columns.str.contains("Value") & df.columns.str.contains("nse")].str.replace(" nse", '')
#
# merge=pd.concat([df_orig2,df_orig3,df_orig], axis=0).drop(["name","id"], axis=1)
# # merge=merge.drop(merge.columns[merge.columns.str.contains("Conductivity")], axis=1)
# together=merge.groupby(size_idx+ ["freeze", "res_decrease"])
# together_mean=together.mean().add_suffix("_mean")
# together_min=together.min().add_suffix("_min")
# together_max=together.max().add_suffix("_max")
# df_stats=pd.concat([together_mean, together_min,together_max], axis=1)
# # df_stats.to_csv(os.path.join(folder, "metric.csv"))
# #
# # df_stats_mean=df_stats[df_stats.columns[df_stats.columns.str.contains("_mean")]].mean()
# # df_stats_min=df_stats[df_stats.columns[df_stats.columns.str.contains("_min")]].min()
# # df_stats_max=df_stats[df_stats.columns[df_stats.columns.str.contains("_max")]].max()
# # df_stats_stats=pd.concat([df_stats_min,df_stats_max,df_stats_mean])
# # df_stats_stats=pd.DataFrame({"Feature": val_col}|
# # {metric+group: df_stats_stats[df_stats_stats.index.str.contains(metric+group)].reset_index(drop=True)
# #  for metric in ("nse", "mse", "pcc", "kge")
# #  for group in ("_mean", "_min", "_max")})
# # df_stats_stats.to_csv(os.path.join(folder, "metric stats.csv"))
# #
# df_stats=df_stats.reset_index()
# for merge_method in ("_mean", "_max", "_min"):
#     for metric in ("nse", "pcc","kge"):
#         df_stats[metric+merge_method]=df_stats.loc[:,df_stats.columns.str.contains(metric+merge_method)].mean(axis=1)
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

#One graph for a size paramater (fixing the other 3)
def plot_seed_fix3s(fix3s):

    # df = get_seed_df(["cool eta log"+ s + "_variant.csv"
    #                     for s in (
    #                         '',
    #                               ' seed 1',
    #                         ' seed 2',
    #                     )]).reset_index()
    # name="Different Model Sizes"

    df = get_seed_df(["cool eta log no pretrain" + s + "_variant.csv"
                      for s in (
                          '',
                          ' seed 1',
                          ' seed 2',
                      )]).reset_index()
    name = "No Pretrain"
    # df = get_seed_df(["metrics" + s + ".csv"
    #                   for s in (
    #                       '0',
    #                       '1',
    #                       '2',
    #                   )]).reset_index()
    # name = "Metrics"
    df = df[(df[fix3s[0][0]] == fix3s[0][1]) &
              (df[fix3s[1][0]] == fix3s[1][1]) &
              (df[fix3s[2][0]] == fix3s[2][1])]
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
    plot_seed_fix3_alls(df, modes=[
        # 'no_freeze_val_loss',
        #
        'freeze_val_loss',
        # "diff_val_loss"
    ],
                               vary=vary, size_str=size_str,infos=infos, title=name+" "+size_str)
    # plot_seed_fix3_alls(df, modes=[
    #     'kge',
    #     "nse"
    # ],
    #                            vary=vary, size_str=size_str,infos=infos, title=name+" "+size_str)

def plot_seed_fix3_alls(df, modes, vary, size_str, infos, title=""):
    vary_values = np.sort(df[vary].unique())
    # if "hidden" in vary:
    #     vary_values=np.array([2,8,32,128])
    # else:
    #     pass
    for mode in modes:
        counter = 0
        for vary_value in vary_values:
            df_temp = df[(df[vary] == vary_value)
                         # & (df['freeze']==True) #uncomment when nseing
                         ]
            # if len(df_glgd_freeze)!=len(df_glgd_no_freeze):
            #     continue
            plt.plot([1, 3, 6, 12, 25], df_temp[mode + "_mean"],
                     alpha=0.8,
                     label=f"{vary_value}",
                     marker="P",
                     )
            plt.fill_between([1, 3, 6, 12, 25],
                             df_temp[mode + "_min"],
                             df_temp[mode + "_max"],
                             alpha=0.2,
                             )


        diff_text = mode_dict[mode]
        plt.xlabel('Number of Measured Timesteps per 64 Days')
        plt.ylabel(diff_text)
        # plt.ylabel('Freeze Loss - No freeze')
        actual_title = f"{title} {diff_text}"

        plt.title(actual_title)
        plt.xscale("log")
        plt.xticks([1, 3, 6, 12, 25], [1, 3, 6, 12, 25])
        leg = plt.legend(framealpha=0.2,title=vary)
        # for _txt in leg.texts:
        #     _txt.set_alpha(0.4)
        # for line in leg.legendHandles:
        #     line.set_alpha(0.3)


        plt.savefig(os.path.join(output_folder, "variant", actual_title + "ultra reduced.png"))
        plt.savefig(os.path.join(output_folder, "variant", actual_title + ".png"))

        plt.show()

#One graph for a size parameter with different sources
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
    dfs = [get_seed_df(["cool eta log" + purpose + s + "_variant.csv"
                        for s in (
                            '',
                                  ' seed 1',
                            ' seed 2',
                        )]).reset_index()
           for purpose in (" no pretrain", "")]
    name = "Baseline"
    infos = [("Baseline", "dashed"), ("Default", "solid")]

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
    size_str=f"f{size_strs[0]}, {size_strs[1]} g{size_strs[2]},{size_strs[3]}"
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
            for df, info in zip(df_temps, infos):
                plt.plot(df[col], df[mode+"_mean"],
                 color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                 alpha=0.8,
                 label=f"{size_str.replace('_', str(vary_value))}, {info[0]}",
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
        plt.ylabel('Loss')
        # plt.ylabel('Freeze Loss - No freeze')
        diff_text= mode_dict[mode]
        actual_title=f"{title} {diff_text}"
        plt.title(actual_title)
        plt.xscale("log")
        plt.xticks(df_temps[0][col].unique(), labels=df_temps[0][col].unique())
        # leg = plt.legend(framealpha=0.2)
        # for _txt in leg.texts:
        #     _txt.set_alpha(0.4)
        # for line in leg.legendHandles:
        #     line.set_alpha(0.3)

        leg = plt.legend()

        # plt.savefig(os.path.join(output_folder, "variant", actual_title+".png"))

        plt.show()

#One graph for a single size parameter value
def plot_source_seed_fix4(fix3s):
    # dfs = [get_seed_df(["cool eta log" + purpose + s + "_variant.csv"
    #                     for s in (
    #                         '',
    #                               ' seed 1',
    #                               ' seed 2',
    #                               )]).reset_index()
    #        for purpose in (" Maumee", " pretrain 14", "")]
    # name="Large Pretrain Set"
    # infos = ["Maumee Basin", "Pretrain 14", 'Various']

    # dfs = [get_seed_df(["cool eta log" + purpose + s + "_variant.csv" for s in ('', ' seed 1', ' seed 2')]).reset_index()
    #        for purpose in (" Just Maumee", " Just Cuyahoga", "")]
    # name="Small Pretrain Set"
    # infos = ["Just Maumee", "Just Cuyahoga", 'Various']
    # #
    #
    #

    # dfs = [get_seed_df([ purpose + s + "_variant.csv"
    #                     for s in (
    #                         '',
    #                         ' seed 1',
    #                         ' seed 2',
    #                     )]).reset_index()
    #        for purpose in ("various", "cool eta log")]
    # infos = ['Adding', "Default"]
    # name = "Different Utilizations of f and g"
    # dfs = [get_seed_df(["cool eta log" + purpose + s + "_variant.csv"
    #                     for s in (
    #                         '',
    #                         ' seed 1',
    #                         ' seed 2',
    #                     )]).reset_index()
    #        for purpose in (" no pretrain", "")]
    # name = "No Pretrain vs Default"
    # infos = ["No Pretrain", "Default"]

    dfs = [get_seed_df(["cool eta log" + purpose + s + "_variant.csv"
                        for s in (
                            '',
                            ' seed 1',
                            ' seed 2',
                        )]).reset_index()
           for purpose in (" no pretrain both data", " no pretrain", "", )]
    name = "Different Baselines"
    infos = ["No Pretrain, Use Source and Target","No Pretrain, Use Target Only", "Default"]

    dfs = [df[(df[fix3s[0][0]] == fix3s[0][1]) &
              (df[fix3s[1][0]] == fix3s[1][1]) &
              (df[fix3s[2][0]] == fix3s[2][1])]  for df in dfs]
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
    plot_source_seed_fix4_alls(dfs, modes=[
        'freeze_val_loss',
        # "diff_val_loss"
    ],
                               vary=vary, size_str=size_str,infos=infos, name=name)

def plot_source_seed_fix4_alls(dfs, modes, vary, size_str, infos, name=""):
    vary_values = np.sort(dfs[0][vary].unique())
    for mode in modes:
        for vary_value in vary_values:
            counter = 0

            df_temps = [df[df[vary] == vary_value] for df in dfs]
            # if len(df_glgd_freeze)!=len(df_glgd_no_freeze):
            #     continue
            for df, info in zip(df_temps, infos):
                plt.plot([1,3,6,12,25], df[mode+"_mean"],
                 color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                 alpha=0.8,
                 label=f"{info}",
                 marker="P",
                 )
                plt.fill_between([1,3,6,12,25],
                                 df[mode+"_min"],
                                 df[mode+"_max"],
                                 color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[counter]],
                                 alpha=0.2,
                                 )

                counter+=1

            diff_text= mode_dict[mode]
            plt.xlabel('Number of Measured Timesteps per 64 Days')
            plt.ylabel(diff_text)
            # plt.ylabel('Freeze Loss - No freeze')
            actual_title=f"{name} {size_str.replace('_', str(vary_value))} {diff_text}"
            plt.title(actual_title)
            plt.xscale("log")
            plt.xticks([1,3,6,12,25],[1,3,6,12,25])
            # leg = plt.legend(framealpha=0.2)
            # for _txt in leg.texts:
            #     _txt.set_alpha(0.4)
            # for line in leg.legendHandles:
            #     line.set_alpha(0.3)

            leg = plt.legend()

            plt.savefig(os.path.join(output_folder, "variant", actual_title+".png"))

            plt.show()


plot_seed_fix3s([['f_hidden_dim', 32],['g_hidden_dim', 8],['g_num_layers', 2] ])
plot_seed_fix3s([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_hidden_dim', 8]])
plot_seed_fix3s([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_num_layers', 2] ])
plot_seed_fix3s([['f_num_layers', 2] ,['g_hidden_dim', 8],['g_num_layers', 2] ])


# plot_source_seed_fix3s([['f_hidden_dim', 32],['g_hidden_dim', 8],['g_num_layers', 2] ])
# plot_source_seed_fix3s([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_hidden_dim', 8]])
# plot_source_seed_fix3s([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_num_layers', 2] ])
# plot_source_seed_fix3s([['f_num_layers', 2] ,['g_hidden_dim', 8],['g_num_layers', 2] ])

# plot_source_seed_fix4([['f_hidden_dim', 32], ['g_hidden_dim', 8], ['g_num_layers', 2]])
# plot_source_seed_fix4([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_hidden_dim', 8]])
# plot_source_seed_fix4([['f_hidden_dim', 32],['f_num_layers', 2] ,['g_num_layers', 2] ])
# plot_source_seed_fix4([['f_num_layers', 2] ,['g_hidden_dim', 8],['g_num_layers', 2] ])


print("hi")