import pandas as pd
import wandb
api = wandb.Api()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import numpy as np
import os
def min_max(a):
    return (a-a.min())/(a.max()-a.min())

def default(df):
    # fig, axs = plt.subplots(5, 4)
    df = df.drop(df[df["name"] == "trash"].index)

    gs=[2, 4, 8, 16, 32]
    cm_subsection = [i / len(gs) for i in range(len(gs))]
    colors = {gs[i]: cm.viridis(cm_subsection[i]) for i in range(len(gs))}
    sizes=np.array([[0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [1.6*10e2,2.8*10e2,0,0, 0,0,0],
                    [0,0,0,0,0,0,0],
                    [12.6*10e2,29.5*10e2,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,412*10e2,0,0,0,0,0]])
    sizes_f_cool_eta=np.array([[0,0,0,0,0,0,0],
                    [0,390,0,0,0,0,0],
                    [0,980,0,0,0,0,0],
                    [0,3.9*10e2,0,0,0,0,0],
                    [0,8.7*10e2,0,0,0,0,0],
                    [13.3*10e2,30.2*10e2,0,0,0,0,0],
                    [0,111*10e2,0,0,0,0,0],
                    [0,428*10e2,0,0,0,0,0],
                    ],
                     )
    sizes_g_cool_eta_322=np.array([[0,0,0,0,0,0,0],
                    [0,699,0,0,0,0,0],
                    [0,1*10e2,0,0,0,0,0],
                    [2.8*10e2,3.9*10e2,6.2*10e2,0,0,0,0],
                    [0,10.9*10e2,0,0,0,0,0],
                    [0,34.1*10e2,0,0,0,0,0],
                    [0,117*10e2,0,0,0,0,0],
                    [0,431*10e2,0,0,0,0,0],
                   ],
                     )
    sizes_g_cool_eta_82=np.array([[0,0,0,0,0,0,0],
                    [0,2*10e2,0,0,0,0,0],
                    [0,2.1*10e2,0,0,0,0,0],
                    [0,3.9*10e2,0,0,0,0,0],
                    [0,2.9*10e2,0,0,0,0,0],
                    [3.9*10e2,3.9*10e2,3.9*10e2,0,0,0,0],
                    [0,6.0*10e2,0,0,0,0,0],
                    [0,10.1*10e2,0,0,0,0,0],
                    ],
                     )
    # df['f_size']=sizes[np.log2(df['f_hidden_dim']).astype(int), np.log2(df['f_num_layers']).astype(int)]
    # df['g_size']=sizes[np.log2(df['g_hidden_dim']).astype(int), np.log2(df['g_num_layers']).astype(int)]
    df['f_size']=sizes_f_cool_eta[np.log2(df['f_hidden_dim']).astype(int), np.log2(df['f_num_layers']).astype(int)]
    df_32=df[df["f_hidden_dim"]==32]
    df_n32=df[df["f_hidden_dim"]!=32]
    df["g_size"]=0
    df.loc[:,'g_size'][df["f_hidden_dim"]==32]=sizes_g_cool_eta_322[np.log2(df_32['g_hidden_dim']).astype(int), np.log2(df_32['g_num_layers']).astype(int)]
    df.loc[:,'g_size'][df["f_hidden_dim"]!=32]=sizes_g_cool_eta_82[np.log2(df_n32['f_hidden_dim']).astype(int), np.log2(df_n32['f_num_layers']).astype(int)]

    df_orig=df.copy()

    sm = cm.ScalarMappable(cmap=cm.viridis)
    plt.colorbar(sm)
    # df = df.drop(df_orig[df_orig['f_size'] < 2000].index)
    plt.xlabel('Number of Parameters of f')
    plt.ylabel('Number of Parameters of g')
    plt.title("Source Domain Losses")
    plt.scatter(df['f_size'],df['g_size'],color=cm.viridis(min_max(df['val_loss'])),s=200)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    # plt.savefig(os.path.join(output_folder, purpose.replace("cool eta log",'')))
    plt.savefig(os.path.join(output_folder, purpose))

    plt.show()

    # df=df_orig[df_orig['f_size']==30.2*10e2]
    # sm = cm.ScalarMappable(cmap=cm.viridis)
    # plt.colorbar(sm)
    # plt.xlabel('g_hidden_dim')
    # plt.ylabel('g_num_layers')
    # plt.title("f32,2")
    # plt.scatter(df['g_hidden_dim'],df['g_num_layers'],color=cm.viridis(min_max(df['val_loss'])), s=200)
    # # )
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    # plt.savefig(os.path.join(output_folder, f"{purpose.replace('cool eta log','')} f32,2.png"))
    #
    # plt.show()
    #
    #
    #
    # df=df_orig[(df_orig['g_hidden_dim']==8) &(df_orig['g_num_layers']==2)]
    # sm = cm.ScalarMappable(cmap=cm.viridis)
    # plt.colorbar(sm)
    # plt.xlabel('f_hidden_dim')
    # plt.ylabel('f_num_layers')
    # plt.title("g8,2")
    # plt.scatter(df['f_hidden_dim'],df['f_num_layers'],color=cm.viridis(min_max(df['val_loss'])), s=200)
    # # )
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    # plt.savefig(os.path.join(output_folder, f"{purpose.replace('cool eta log', '')} g8,2.png"))
    #
    # plt.show()
    # print("hi")
def other(dfs, domains, by_size=True):
    all_df_mean=pd.DataFrame({domain : df[0]['val_loss'] for domain, df in zip(domains,dfs)})
    all_df_max=pd.DataFrame({domain : df[1]['val_loss'] for domain, df in zip(domains,dfs)})
    all_df_min=pd.DataFrame({domain : df[2]['val_loss'] for domain, df in zip(domains,dfs)})
    for df in [all_df_mean,all_df_min,all_df_max]:
        df.index=dfs[0][0]['name']
    # all_df['name']=dfs[0]['name']

    if by_size:
        plt.autoscale(tight=True)
        width = 0.15
        x = np.arange(len(dfs[-1][0]))
        plt.xticks(x + width, dfs[-1][0]['name'], fontsize="x-small")
        multiplier = -1
        for i, domain in enumerate(domains):
            offset = width * multiplier
            rects = plt.bar(x + offset, all_df_mean[domain], width, label=domain,
                            yerr=pd.concat((all_df_mean[domain]-all_df_min[domain], all_df_max[domain]-all_df_mean[domain]), axis=1).values.T)

            multiplier += 1
        leg=plt.legend(framealpha=0.5)
        for _txt in leg.texts:
            _txt.set_alpha(0.6)
        for line in leg.legendHandles:
            line.set_alpha(0.4)
        plt.title("Pretrained Model Validation Loss by Model Size")

        plt.ylabel('Validation Loss')
        plt.xlabel('Model Size')
        plt.savefig(os.path.join(output_folder, "Various Pretrain Size.png"))

        plt.show()
    else:
        plt.autoscale(tight=True)
        width = 0.1
        x = np.arange(len(domains))
        plt.xticks(x + width, domains, fontsize="x-small")
        multiplier = -1
        for i, size in enumerate(all_df_mean.index):
            offset = width * multiplier
            rects = plt.bar(x + offset, all_df_mean.loc[size], width, label=size,
                            yerr=pd.concat(
                                (all_df_mean.loc[size] - all_df_min.loc[size], all_df_max.loc[size] - all_df_mean.loc[size]),
                                axis=1).values.T)

            multiplier += 1
        leg = plt.legend(framealpha=0.5)
        for _txt in leg.texts:
            _txt.set_alpha(0.6)
        for line in leg.legendHandles:
            line.set_alpha(0.4)
        plt.title("Pretrained Model Validation Loss by Set of Source Domains")
        plt.ylabel('Validation Loss')
        plt.xlabel('Set of Source Domain')
        plt.savefig(os.path.join(output_folder, "Various Pretrain Source Domains.png"))

        plt.show()

def sort(dfs):
    return [df.sort_values(['f_hidden_dim', 'f_num_layers', 'g_hidden_dim', 'g_num_layers']).reset_index(drop=True) for df in dfs]
def mean_df(csvs):
    df = pd.concat([pd.read_csv(os.path.join(folder, csv)).drop("Unnamed: 0", axis=1) for csv in csvs])
    grouped_df=df.groupby('name')
    return [grouped_df.mean().reset_index(),grouped_df.max().reset_index(), grouped_df.min().reset_index()]

def plot_invariant(df_stats,fix3):
    df = df_stats[(df_stats[fix3[0][0]] == fix3[0][1]) &
                 (df_stats[fix3[1][0]] == fix3[1][1]) &
                 (df_stats[fix3[2][0]] == fix3[2][1])].copy()
    # df_diff = df_stats_diff[(df_stats_diff[fix3[0][0]] == fix3[0][1]) &
    #               (df_stats_diff[fix3[1][0]] == fix3[1][1]) &
    #               (df_stats_diff[fix3[2][0]] == fix3[2][1])].copy()
    size_vars=['f_hidden_dim',
    'f_num_layers',
     'g_hidden_dim',
    'g_num_layers']
    provided=[i[0] for i in fix3]
    size_strs = []
    count=0
    var=None
    for i, size_var in enumerate(size_vars):
        if size_var in provided:
            size_strs.append(fix3[count][1])
            count+=1
        else:
            size_strs.append("_")
            var=size_var
    size_str=f"f{size_strs[0]},{size_strs[1]} g{size_strs[2]},{size_strs[3]}"
    df=df.sort_values(var)
    plot_all_seed(df, var, title="Invariant " +size_str, size_str=size_str)
def plot_all_seed(df,var, title="", size_str=''):

    counter=0
    plt.plot(df[var], df["val_loss_mean"],
             label=size_str,
             marker="P")
    plt.fill_between(df[var], df["val_loss_min"],df["val_loss_max"],
                     alpha=0.2,
                     )
    xlabel=var.replace("_hidden_dim", " Hidden Dimension") if "_hidden_dim" in var else var.replace("_num_layers", " Number of Layers")
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.title(title)
    plt.xscale("log")
    plt.xticks(df[var].unique(), labels=df[var].unique())
    # plt.legend()
    plt.savefig(os.path.join(output_folder, title+".png"))

    plt.show()
if __name__=="__main__":
    folder = os.path.join("..", "Download", "run data")
    purpose = "default"
    purpose = "res"
    purpose = "cool_eta"
    purpose = "cool eta log"
    # purpose="cool eta log Just Cuyahoga"
    # purpose="cool eta log Just Maumee"
    # purpose="cool eta log pretrain 14"
    # purpose="cool eta log Maumee"

    output_folder = "../Output/log/invariant"
    # df = pd.read_csv(os.path.join(folder, purpose + "_invariant.csv")).drop("Unnamed: 0", axis=1)
    # df_cuyahoga = pd.read_csv(os.path.join(folder, "cool eta log Just Cuyahoga_invariant.csv")).drop("Unnamed: 0", axis=1)
    # df_maumee = pd.read_csv(os.path.join(folder, "cool eta log Just Maumee_invariant.csv")).drop("Unnamed: 0", axis=1)
    # df_14 = pd.read_csv(os.path.join(folder, "cool eta log pretrain 14_invariant.csv")).drop("Unnamed: 0", axis=1)
    # df_maumee_basin = pd.read_csv(os.path.join(folder, "cool eta log Maumee_invariant.csv")).drop("Unnamed: 0", axis=1)
    # df=df[df['name'].isin(df_maumee_basin['name'])]
    # dfs=[df,df_cuyahoga,df_maumee,df_maumee_basin,df_14]
    # #
    # dfs=sort(dfs)
    csvs=['',  ' Just Maumee', ' Just Cuyahoga',' Maumee', ' pretrain 14']
    dfs=[mean_df(['cool eta log'+ csv+ s+'_invariant.csv'
           for s in ('', ' seed 1', ' seed 2')]) for csv in csvs]
    domains=['Various',  'Just Maumee', 'Just Cuyahoga','Maumee Basin', 'Pretrain 14']
    # default(df)
    for i in range(3):
        dfs[0][i]=dfs[0][i][dfs[0][i]['name'].isin(dfs[1][i]['name'])].reset_index(drop=True)
    other(dfs, domains=domains,by_size=False)
    print("hi")

    # dfs = mean_df(['cool eta log' +  s + '_invariant.csv'
    #                 for s in ('', ' seed 1', ' seed 2')])
    # df=dfs[0].loc[:,['name', 'val_loss', 'f_hidden_dim', 'f_num_layers', 'g_hidden_dim', 'g_num_layers']]
    # df=df.rename(columns={'val_loss':'val_loss_mean'})
    # df['val_loss_max']=dfs[1]['val_loss']
    # df["val_loss_min"]=dfs[2]['val_loss']
    # plot_invariant(df,[['f_hidden_dim', 32], ['g_hidden_dim', 8], ['g_num_layers', 2]])
    # plot_invariant(df,[['f_hidden_dim', 32], ['f_num_layers', 2], ['g_hidden_dim', 8]])
    # plot_invariant(df,[['f_hidden_dim', 32], ['f_num_layers', 2], ['g_num_layers', 2]])
    # plot_invariant(df,[['f_num_layers', 2], ['g_hidden_dim', 8], ['g_num_layers', 2]])