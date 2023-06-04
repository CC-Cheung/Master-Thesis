import wandb
api = wandb.Api()
for i in ["sb29oyt2","mjhjwcbx","jou3bbjd","7idto2jg"]:
    run = api.run(f"cc-cheung/DA Thesis/{i}")
    run.config["purpose"] = "debug polyinvar"
    run.update()