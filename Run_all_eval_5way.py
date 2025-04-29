from logging import config
from pyexpat import model
import warnings, sys, os
import dataset
from utils import load_config, save_config
from loadDataset import LoadDataset
from trainModule import TestModule
warnings.filterwarnings("ignore")

expList = ["10way_5shot", "10way_10shot"]
seeds = [6, 7, 10, 666, 11, 19, 22, 31, 42, 888]

for seed in seeds:
    rootFolder = f"/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_{seed}_baseline"
    for exp in expList:
        print("seed: ", seed)
        print("exp: ", exp)
        for folder in os.listdir(rootFolder):
            # if exp in folder and "LabelPropagation_alpha" in folder and "pretrain" not in folder:
            if exp in folder and "LabelPropagation" in folder and "alpha" not in folder:
                print("modelFolder: ", folder)
                modelFolder = folder
                modelPath = os.path.join(rootFolder, modelFolder)

                configPath = os.path.join(modelPath, "config.json")
                options = load_config(configPath)

                if (options["settings"]["few_shot"]["parameters"]["relation_layer"] == 2 and 
                    options["settings"]["few_shot"]["parameters"]["dim_in"] == 128 and 
                    options["settings"]["few_shot"]["parameters"]["dim_hidden"] == 64 and 
                    options["settings"]["few_shot"]["parameters"]["dim_out"] == 32 and 
                    options["settings"]["few_shot"]["parameters"]["rn"] == 300 and 
                    options["settings"]["few_shot"]["parameters"]["alpha"] == 0.8 and 
                    options["settings"]["few_shot"]["parameters"]["k"] == 20 ):

                    newConfigPath = os.path.join(modelPath, "config_5way.json")
                    
                    options["settings"]["few_shot"]["test"]["class_per_iter"] = 5
                    options["settings"]["train"]["distance"] = "euclidean"
                    
                    save_config(options, newConfigPath)

                    dataset = LoadDataset(options)
                    test = TestModule(configPath=newConfigPath, dataset=dataset)
                    test.eval()