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
seeds = [6, 7, 10, 666, 11, 19, 22, 31, 42, 888] 
date = datetime.datetime.now().strftime("%Y%m%d_%H%M")

for seed in seeds:
    print("seed: ", seed)
    for exp in expList:
        options = load_config("./config/config_relation_gcn.json")
        print("exp: ", exp)
        options["settings"]["name"] = exp+f"_relation_GCN"
        shots = int(exp.split("_")[1].split("shot")[0])
        way = int(exp.split("_")[0].split("way")[0])
        options["settings"]["few_shot"]["train"]["support_shots"] = shots
        options["settings"]["few_shot"]["train"]["query_shots"] = 20 - shots
        options["settings"]["few_shot"]["test"]["support_shots"] = shots
        options["settings"]["few_shot"]["test"]["query_shots"] = 20 - shots
        options["settings"]["few_shot"]["train"]["class_per_iter"] = way
        options["settings"]["few_shot"]["test"]["class_per_iter"] = way
        options["settings"]["seed"] = seed
        save_config(options, "./config/config_relation_gcn.json")

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