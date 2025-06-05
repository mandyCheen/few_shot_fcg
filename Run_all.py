import warnings
from utils import load_config, save_config
from loadDataset import LoadDataset
from fcgVectorize import FCGVectorize
from trainModule import TrainModule
from trainModule import TestModule
import datetime
import os, torch

warnings.filterwarnings("ignore")

expList = ["5way_5shot", "5way_10shot", "10way_5shot", "10way_10shot"]
seeds = [7, 10, 666, 11, 19, 22, 31, 42, 888] #6, 
# models = ["GCN"]
date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
# # alphaList = [0.6, 0.7, 0.8, 0.9]
# ## always with pretrain
# # for alpha in alphaList:
#     # if alpha == 0.7:
#     #     seeds = [10, 666, 11, 19, 22, 31, 42, 888]
#     # else:
#     #     seeds = [6, 7, 10, 666, 11, 19, 22, 31, 42, 888]
#     # print("alpha: ", alpha)
# parms = [
#     [3, 64, 64],
#     [2, 64, 64],
#     [3, 256, 128],
#     [2, 256, 128]
# ]

for seed in seeds:
    print("seed: ", seed)
    for exp in expList:
        if seed == 7 and (exp == "5way_5shot" or exp == "10way_5shot" or exp == "5way_10shot"):
            continue
        # options = load_config("./config/config_label_prop_pretrain.json")
        options = load_config("./config/config_match_gcn.json")
        print("exp: ", exp)
        options["settings"]["name"] = exp+f"_match_GCN"
        shots = int(exp.split("_")[1].split("shot")[0])
        way = int(exp.split("_")[0].split("way")[0])
        # options["dataset"]["addition_note"] = "modelExp"
        # options["settings"]["model"]["pretrained_model_folder"] = "x86_pretrained_GraphSAGE_3_layers_20250428_1936"
        options["settings"]["few_shot"]["train"]["support_shots"] = shots
        options["settings"]["few_shot"]["train"]["query_shots"] = 20 - shots
        options["settings"]["few_shot"]["test"]["support_shots"] = shots
        options["settings"]["few_shot"]["test"]["query_shots"] = 20 - shots
        options["settings"]["few_shot"]["train"]["class_per_iter"] = way
        options["settings"]["few_shot"]["test"]["class_per_iter"] = way
        # options["settings"]["model"]["model_name"] = model
        # options["settings"]["model"]["num_layers"] = parm[0]
        # options["settings"]["model"]["hidden_size"] = parm[1]
        # options["settings"]["model"]["output_size"] = parm[2]
        # options["settings"]["few_shot"]["parameters"]["relation_model"] = model
        # options["settings"]["few_shot"]["parameters"]["relation_layer"] = parm[0] - 1
        # options["settings"]["few_shot"]["parameters"]["dim_in"] = parm[2]
        # options["settings"]["few_shot"]["parameters"]["dim_hidden"] = parm[2] // 2
        # options["settings"]["few_shot"]["parameters"]["dim_out"] = parm[2] // 4

        # # options["settings"]["few_shot"]["parameters"]["alpha"] = alpha
        # options["settings"]["few_shot"]["parameters"]["k"] = 20
        options["settings"]["seed"] = seed
        save_config(options, "./config/config_match_gcn.json")

        dataset = LoadDataset(options)
        # vectorizer = FCGVectorize(options, dataset)
        # vectorizer.node_embedding(dataset.rawDataset)
        try:
            trainModule = TrainModule(options, dataset)
            trainModule.train()

            test = TestModule(os.path.join(trainModule.model_folder, "config.json"), dataset)
            test.eval()
        except Exception as e:
            now = datetime.datetime.now()
            open(f"./logs/Error_log_{date}.txt", "a").write(f"{now}: {exp} seed:{seed} {e}\n")

        torch.cuda.empty_cache()