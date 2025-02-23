import warnings
from utils import load_config, save_config
from loadDataset import LoadDataset
from fcgVectorize import FCGVectorize
from trainModule import TrainModule
from trainModule import TestModule
import os


options = load_config("./config/config.json")
warnings.filterwarnings("ignore")

expList = ["5way_1shot", "10way_1shot"]
seeds = [19, 22, 31, 42, 888]
## always with pretrain
for seed in seeds:
    for exp in expList:
        options["settings"]["name"] = exp+"_NnNet"
        shots = int(exp.split("_")[1].split("shot")[0])
        way = int(exp.split("_")[0].split("way")[0])
        options["settings"]["few_shot"]["train"]["support_shots"] = shots
        options["settings"]["few_shot"]["train"]["query_shots"] = 20 - shots
        options["settings"]["few_shot"]["test"]["support_shots"] = shots
        options["settings"]["few_shot"]["test"]["query_shots"] = 20 - shots
        options["settings"]["few_shot"]["train"]["class_per_iter"] = way
        options["settings"]["few_shot"]["test"]["class_per_iter"] = way
        options["settings"]["seed"] = seed
        save_config(options, "./config/config.json")

        dataset = LoadDataset(options, pretrain=False)
        vectorizer = FCGVectorize(options, dataset)
        vectorizer.node_embedding(dataset.rawDataset)
        trainModule = TrainModule(options, dataset)
        trainModule.train()

        test = TestModule(os.path.join(trainModule.model_folder, "config.json"), dataset, options)
        test.eval()
        test.eval_ablation()