import pandas as pd
import wandb
import os
import numpy as np

api = wandb.Api(timeout=20)
folder="run data"
#Choose the purpose
# purpose="cool eta log seed"
# purpose="cool eta log Just Cuyahoga seed 1"
# purpose="cool eta log Just Maumee seed 2"

# purpose="cool eta log no pretrain seed 2"
# purpose="various"

# purpose="cool eta log pretrain 14 seed 1"

# purpose="cool eta log no pretrain"
# purpose="cool eta log Maumee seed 2"
purpose="various seed 2"
purpose="cool eta log no pretrain both data seed 0"

#Choose your filters
runs = api.runs("cc-cheung/DA Thesis", filters={
                                                # "config.purpose": "Normal eta + 2 shallower overfit debug",
                                                # "config.purpose": "Normal eta + 2 shallower kfold",
                                                # "config.purpose": "resolution",
                                                # "config.purpose": "cool eta no pretrain",
                                                # "config.purpose": "cool eta",
                                                "config.purpose": purpose ,
                                                # "config.data": "pretrain 14",

                                                # "createdAt": {"$gt": "2024-10-13T04:00:00"},
                                                # "display_name": {"$regex":"^eta +"},
                                                # "$not": {"display_name": "trash"},
                                                "summary_metrics.epoch":{"$gt": 5},
                                                # "$not": {"summary_metrics.epoch": 499},

                                                # "config.file": {"$in":["variant_normal.py", "variant_normal_multi.py"]},
                                                # "config.file": "no_pretrain_multi.py",
                                                # "config.split":0.7,
                                                # "config.res_decrease": 0.05,
                                                # "config.seed":2,

                                                # "config.file": "invariant_real2.py",
                                                "config.file": "variant_real2.py",
                                                #
                                                # "config.f_hidden_dim":32,
                                                # "config.f_num_layers":2,
                                                # "config.g_hidden_dim":8,
                                                # "config.g_num_layers": 2,


})

summary_list, config_list, name_list = [], [], []
keep=["val_loss", "_runtime", "epoch", "trainer/global_step" , "step", "train_loss", "_timestamp", "freeze"]

#Download models
# folder=os.path.join("models", purpose+"variant")
# folder=os.path.join("models", purpose)
#
# if not os.path.exists(folder):
#     os.makedirs(folder)
# for run in runs:
#     # f_hidden_dim = run.config["f_hidden_dim"]
#     # f_num_layers = run.config["f_num_layers"]
#     # g_hidden_dim = run.config["g_hidden_dim"]
#     # g_num_layers = run.config["g_num_layers"]
#
#     # if (np.array([f_hidden_dim, f_num_layers, g_hidden_dim, g_num_layers])
#     #     == np.array([32, 2, 8, 2])).sum() < 3:
#     #     run.config["purpose"] = "cool eta log pretrain 14 trash"
#     #     run.update()
#
#     # run.config["purpose"] = purpose+ "trash"
#     # run.update()
#
#     run.files()
#     for file in run.files():
#         file_name=file._attrs['name']
#         if "epoch" in file_name:
#             file.download(replace=True, root=folder)
#             os.rename(os.path.join(folder, file_name),
#                       os.path.join(folder, f"{run.id} {run.name}.ckpt"))
#             break
# #

#
#
#
#Download Run Data
folder=os.path.join("run data")

for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary={"name": run.name}

    summary.update({key: run.summary._json_dict[key] for key in keep if key in run.summary._json_dict.keys()})
    if "val_loss" in run.summary._json_dict.keys():
        summary["val_loss"]=summary["val_loss"]["min"]

    summary.update({k: v for k,v in run.config.items()
          if not k.startswith('_')})

    summary["train_loss"] = summary["train_loss"]["min"] if type(summary["train_loss"])==dict else\
        run.history(keys=["train_loss"]).min()["train_loss"]
    # run.config["freeze"] = run.name.split()[-1].replace("freeze", "") == "True"
    # run.update()
    summary_list.append(summary)
# # # #
# #
# # #
# #
# # path=os.path.join(folder, "split_double_variant.csv")
# # path=os.path.join(folder, "res_variant.csv")
# # path=os.path.join(folder, "cool_eta_variant_for_test.csv")
# path=os.path.join(folder, purpose+' seed 1'+"_invariant.csv"
# path=os.path.join(folder, purpose+"_invariant.csv")

path=os.path.join(folder, purpose+"_variant.csv")
#
# # path=os.path.join(folder, "cool_eta_no_pretrain.csv")
# # path=os.path.join(folder, "default_invariant.csv")
# # if not os.path.exists(folder):
# #     os.makedirs(folder)
# #
runs_df = pd.DataFrame(summary_list)
# df_test = pd.read_csv(path).drop("Unnamed: 0", axis=1)
# runs_df= pd.concat((df_test, runs_df))
# runs_df.drop_duplicates(("name","val_loss"), keep="last")
# #
# runs_df.to_csv("project.csv")
# runs_df.to_csv("no_pretrain_debug.csv")
# runs_df.to_csv("invariant.csv")
# runs_df.to_csv("g_only.csv")
# runs_df.to_csv("overfit_debug.csv")
runs_df.to_csv(path)


print("hi")