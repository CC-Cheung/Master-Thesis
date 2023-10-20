import wandb
api = wandb.Api()

for i in ["7jemgkhf"]:
    run = api.run(f"cc-cheung/DA Thesis/{i}")
    run.config["num_domains"] = "debug polyinvar"
    run.update()