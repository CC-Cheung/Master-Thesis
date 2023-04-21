import wandb
api = wandb.Api()
for i in ["rjokw4gq", "je6baa2j", "vwphu0iy", "3i9d0m9k", "87vvchfh", "pqqm1eh5", "xtjke1fg", "glbkpe6z", "vdswhcm4", "z8457lj2", "9q80akwt", "qbzc3xcn", "22ue06fq", "96dvpwbc","yj820pso","gfntdbxj", "qhsse8zi", "uczlfysz", "y1j44v6p", "n9bpa41e","m3s4e63x", "cstcoe7u", "jizq01mu", "src76kwx", "usfm0xzq", "aalfj45r", "q8vd7vck","q8vd7vck","skksp7wf"]:
    run = api.run(f"cc-cheung/DA Thesis/{i}")
    run.config["file"] = "transformer_mod.py"
    run.update()