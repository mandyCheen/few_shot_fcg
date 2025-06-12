from logging import config
from pyexpat import model
import warnings, sys, os
import dataset
from utils import load_config, save_config
from loadDataset import LoadDataset
from trainModule import TestModule
warnings.filterwarnings("ignore")

expList = ["5way_5shot", "5way_10shot", "10way_5shot", "10way_10shot"]
seeds = [10] #6, 7, 10, 666, 11, 19, 22, 31, 42, 888

for seed in seeds:
    rootFolder = f"/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_{seed}_modelExp"
    for exp in expList:
        if seed == 19:
            if exp == "5way_5shot" or exp == "5way_10shot":
                continue
        print("seed: ", seed)
        print("exp: ", exp)
        for folder in os.listdir(rootFolder):
            if exp in folder and "layers3_128_256_128_" in folder:
                print("modelFolder: ", folder)
                modelFolder = folder
                modelPath = os.path.join(rootFolder, modelFolder)

                configPath = os.path.join(modelPath, "config.json")
                options = load_config(configPath)
                options["dataset"]["openset"] = True
                options["dataset"]["openset_raw"] = "malware_diec_ghidra_x86_64_fcg_openset_dataset_rm0node.csv"
                options["dataset"]["openset_data_mode"] = "random"
                options["dataset"]["openset_data_ratio"] = 0.2
                opensetSettings = {
                                    "train": {
                                        "use": False,
                                    },
                                    "test": {
                                        "use": True,
                                        "m_samples": 50,
                                        "class_per_iter": 5
                                }
                            }
                options["settings"]["openset"] = opensetSettings
                options["settings"]["train"]["device"] = "cuda:0"
                options["paths"]["data"]["embedding_folder"] = "/mnt/ssd2t/mandy/Projects/few_shot_fcg/embeddings"
            
                save_config(options, configPath)
                dataset = LoadDataset(options)
                test = TestModule(configPath=configPath, dataset=dataset)
                test.eval()