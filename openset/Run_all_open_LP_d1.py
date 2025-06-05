import warnings, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, save_config
from loadDataset import LoadDataset
from fcgVectorize import FCGVectorize
from trainModule import TrainModule
from trainModule import TestModule
import torch

warnings.filterwarnings("ignore")

expList = ["5way_5shot"] #, "5way_10shot", "10way_5shot", "10way_10shot"
seeds = [22, 31, 42, 888] # 6, 7, 10, 666, 11, 19
lambdaList = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]

for seed in seeds:
    print("seed: ", seed)
    for exp in expList:
        for lambda_ in lambdaList:
            options = load_config("../config/config_label_prop_openset_meta_nict_d1.json")
            print("exp: ", exp)
            print("lambda: ", lambda_)
            options["settings"]["name"] = exp+f"_LabelPropagation_alpha0.7_k20_lambda{lambda_}"
            shots = int(exp.split("_")[1].split("shot")[0])
            way = int(exp.split("_")[0].split("way")[0])
            options["settings"]["few_shot"]["train"]["support_shots"] = shots
            options["settings"]["few_shot"]["train"]["query_shots"] = 20 - shots
            options["settings"]["few_shot"]["test"]["support_shots"] = shots
            options["settings"]["few_shot"]["test"]["query_shots"] = 20 - shots
            options["settings"]["few_shot"]["train"]["class_per_iter"] = way
            options["settings"]["few_shot"]["test"]["class_per_iter"] = way
            options["settings"]["few_shot"]["parameters"]["alpha"] = 0.7
            options["settings"]["few_shot"]["parameters"]["k"] = 20
            options["settings"]["openset"]["loss_weight"] = lambda_
            options["settings"]["seed"] = seed
            # ## tune the model from base model
            # base_path = f"/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_{seed}_baseline"
            # dir_pattern = f"{exp}_LabelPropagation_alpha{alpha}_k20_2"

            # matching_dirs = os.path.join(base_path, [d for d in os.listdir(base_path) if dir_pattern in d][0])
            # print("matching_dirs: ", matching_dirs)
            # model_path = os.path.join(matching_dirs, [f for f in os.listdir(matching_dirs) if "best" in f][0])
            # print("model_path: ", model_path)

            # options["settings"]["model"]["load_weights"] = model_path
            save_config(options, "../config/config_label_prop_openset_meta_nict_d1.json")

            dataset = LoadDataset(options)
            # vectorizer = FCGVectorize(options, dataset)
            # vectorizer.node_embedding(dataset.opensetData, openset=True)
            trainModule = TrainModule(options, dataset)
            trainModule.train()

            test = TestModule(os.path.join(trainModule.model_folder, "config.json"), dataset)
            test.eval(mode="openset")

            torch.cuda.empty_cache()