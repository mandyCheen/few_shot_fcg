from logging import config
from pyexpat import model
import warnings, sys, os
import dataset
from utils import load_config, save_config
from loadDataset import LoadDataset
from trainModule import TestModule
warnings.filterwarnings("ignore")

expList = ["5way_5shot", "5way_10shot", "10way_5shot", "10way_10shot"]
seeds = [6, 22, 31, 42, 888, 7, 10, 666, 11, 19]
lambdaList = [1.0, 1.5, 2.0]

for seed in seeds:
    rootFolder = f"/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_{seed}_openset"
    
    for exp in expList:
        print("exp: ", exp)
        for folder in os.listdir(rootFolder):
            if os.path.isdir(os.path.join(rootFolder, folder)):
                if "lambda" not in folder:
                    continue
                if exp not in folder:
                    continue
                lambda_value = float((folder.split("_")[6]).split("lambda")[1])
                if lambda_value not in lambdaList:
                    continue
                
                print("modelFolder: ", folder)
                modelFolder = folder
                modelPath = os.path.join(rootFolder, modelFolder)

                configPath = os.path.join(modelPath, "config.json")
                options = load_config(configPath)
                # options["settings"]["few_shot"]["parameters"]["relation_layer"] = 2
                # options["settings"]["few_shot"]["parameters"]["relation_model"] = "GCN"
                # options["settings"]["train"]["device"] = "cuda:2"

                # save_config(options, configPath)
                dataset = LoadDataset(options)
                test = TestModule(configPath=configPath, dataset=dataset)
                test.eval()