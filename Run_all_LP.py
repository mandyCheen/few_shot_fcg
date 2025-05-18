import warnings
from utils import load_config, save_config
from loadDataset import LoadDataset
from fcgVectorize import FCGVectorize
from trainModule import TrainModule
from trainModule import TestModule
import datetime
import os, torch

warnings.filterwarnings("ignore")

expList = ["5way_1shot", "5way_2shot", "5way_3shot", "5way_4shot", "5way_6shot", "5way_7shot", "5way_8shot", "5way_9shot"]
seeds = [6, 7, 10, 666, 11, 19, 22, 31, 42, 888]
models = ["GCN"]
# alphaList = [0.6, 0.7, 0.8, 0.9]
## always with pretrain
# for alpha in alphaList:
    # if alpha == 0.7:
    #     seeds = [10, 666, 11, 19, 22, 31, 42, 888]
    # else:
    #     seeds = [6, 7, 10, 666, 11, 19, 22, 31, 42, 888]
    # print("alpha: ", alpha)

for seed in seeds:
    print("seed: ", seed)
    for model in models:
        for exp in expList:
            # options = load_config("./config/config_label_prop_pretrain.json")
            options = load_config("./config/config_label_prop.json")
            print("exp: ", exp)
            options["settings"]["name"] = exp+f"_LabelPropagation_alpha0.7_k20_{model}"
            shots = int(exp.split("_")[1].split("shot")[0])
            way = int(exp.split("_")[0].split("way")[0])
            options["dataset"]["addition_note"] = "modelExp"
            # options["settings"]["model"]["pretrained_model_folder"] = "x86_pretrained_GraphSAGE_3_layers_20250428_1936"
            options["settings"]["few_shot"]["train"]["support_shots"] = shots
            options["settings"]["few_shot"]["train"]["query_shots"] = 20 - shots
            options["settings"]["few_shot"]["test"]["support_shots"] = shots
            options["settings"]["few_shot"]["test"]["query_shots"] = 20 - shots
            options["settings"]["few_shot"]["train"]["class_per_iter"] = way
            options["settings"]["few_shot"]["test"]["class_per_iter"] = way
            options["settings"]["model"]["model_name"] = model
            options["settings"]["few_shot"]["parameters"]["relation_model"] = model
            # options["settings"]["few_shot"]["parameters"]["alpha"] = alpha
            options["settings"]["few_shot"]["parameters"]["k"] = 20
            options["settings"]["seed"] = seed
            save_config(options, "./config/config_label_prop.json")

            dataset = LoadDataset(options, pretrain=False)
            # vectorizer = FCGVectorize(options, dataset)
            # vectorizer.node_embedding(dataset.rawDataset)
            try:
                trainModule = TrainModule(options, dataset)
                trainModule.train()

                test = TestModule(os.path.join(trainModule.model_folder, "config.json"), dataset)
                test.eval()
            except Exception as e:
                open("Error_lp_models_exp.txt", "a").write(f"{exp} {model} {seed} {e}\n")

            torch.cuda.empty_cache()