from logging import config
from pyexpat import model
import warnings, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset
from utils import load_config, save_config
from loadDataset import LoadDataset
from trainModule import TestModule
warnings.filterwarnings("ignore")

expList = ["5way_5shot", "5way_10shot", "10way_5shot", "10way_10shot"]
seeds = [6, 7, 10, 666, 11, 19, 22, 31, 42, 888]
methodList = ["ProtoNet_without_pretrain", "NnNet_without_pretrain", "SoftNnNet_without_pretrain"]

for seed in seeds:
    
    rootFolder = f"/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_{seed}_baseline"

    for exp in expList:
        print("seed: ", seed)
        print("exp: ", exp)
        modelFolder = [d for d in os.listdir(rootFolder) if exp in d]

        for method in methodList:
            for modelFolder_ in modelFolder:
                if method in modelFolder_:
                    print("modelFolder: ", modelFolder_)
                    modelPath = os.path.join(rootFolder, modelFolder_)

                    configPath = os.path.join(modelPath, "config.json")
                    new_configPath = os.path.join(modelPath, "config_openset.json")
                    opt = load_config(configPath)
                    opt["dataset"]["openset"] = True
                    opt["dataset"]["openset_raw"] = "malware_diec_ghidra_x86_64_fcg_openset_dataset_rm0node.csv"
                    opt["dataset"]["openset_data_ratio"] = 0.2
                    opt["dataset"]["openset_data_mode"] = "random"
                    opt["settings"]["openset"] = {}
                    opt["settings"]["openset"]["test"] = {}
                    opt["settings"]["openset"]["use"] = True
                    opt["settings"]["openset"]["test"]["m_samples"] = 50
                    opt["paths"]["data"]["csv_folder"] = "../dataset/raw_csv"
                    opt["paths"]["data"]["split_folder"] = "../dataset/split"
                    opt["paths"]["data"]["openset_dataset"] = "/mnt/ssd2t/mandy/Projects/few_shot_fcg/dataset/data_ghidra_fcg_openset"
                    save_config(opt, new_configPath)

                    dataset = LoadDataset(opt)
                    test = TestModule(configPath=new_configPath, dataset = dataset)
                    test.eval(mode="openset")