import warnings, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, save_config
from loadDataset import LoadDataset
from fcgVectorize import FCGVectorize
from trainModule import TrainModule
from trainModule import TestModule

warnings.filterwarnings("ignore")

expList = ["5way_5shot", "5way_10shot"] #, "10way_5shot", "10way_10shot"
seeds = [6, 7, 10, 666, 11, 19, 22, 31, 42, 888]
alpha = 0.7

for seed in seeds:
    print("seed: ", seed)
    for exp in expList:
        options = load_config("../config/config_label_prop_openset_meta.json")
        print("exp: ", exp)
        options["settings"]["name"] = exp+"_LabelPropagation_alpha0.7_k20"
        shots = int(exp.split("_")[1].split("shot")[0])
        way = int(exp.split("_")[0].split("way")[0])
        options["settings"]["few_shot"]["train"]["support_shots"] = shots
        options["settings"]["few_shot"]["train"]["query_shots"] = 20 - shots
        options["settings"]["few_shot"]["test"]["support_shots"] = shots
        options["settings"]["few_shot"]["test"]["query_shots"] = 20 - shots
        options["settings"]["few_shot"]["train"]["class_per_iter"] = way
        options["settings"]["few_shot"]["test"]["class_per_iter"] = way
        options["settings"]["few_shot"]["parameters"]["alpha"] = alpha
        options["settings"]["few_shot"]["parameters"]["k"] = 20

        # ## tune the model from base model
        # base_path = f"/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_{seed}_baseline"
        # dir_pattern = f"{exp}_LabelPropagation_alpha{alpha}_k20_2"

        # matching_dirs = os.path.join(base_path, [d for d in os.listdir(base_path) if dir_pattern in d][0])
        # print("matching_dirs: ", matching_dirs)
        # model_path = os.path.join(matching_dirs, [f for f in os.listdir(matching_dirs) if "best" in f][0])
        # print("model_path: ", model_path)

        # options["settings"]["model"]["load_weights"] = model_path

        options["settings"]["seed"] = seed
        save_config(options, "../config/config_label_prop_openset_meta.json")

        dataset = LoadDataset(options)
        vectorizer = FCGVectorize(options, dataset)
        vectorizer.node_embedding(dataset.opensetData, openset=True)
        trainModule = TrainModule(options, dataset)
        trainModule.train()

        test = TestModule(os.path.join(trainModule.model_folder, "config.json"), dataset, options)
        test.eval(mode="openset")