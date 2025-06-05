from logging import config
from pyexpat import model
import warnings, sys, os
import dataset
from utils import load_config, save_config
from loadDataset import LoadDataset
from trainModule import TestModule
warnings.filterwarnings("ignore")

expList = ["5way_5shot", "5way_10shot", "10way_5shot", "10way_10shot"]
seeds = [6, 7, 10, 666, 11, 19, 22, 31, 42, 888]

for seed in seeds:
    rootFolder = f"/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_{seed}_others"
    for exp in expList:
        print("seed: ", seed)
        print("exp: ", exp)
        if not os.path.exists(rootFolder):
            print(f"Root folder {rootFolder} does not exist.")
            continue
        for folder in os.listdir(rootFolder):
            # if exp in folder and "LabelPropagation_alpha" in folder and "pretrain" not in folder:
            if exp in folder:
                print("modelFolder: ", folder)
                modelFolder = folder
                modelPath = os.path.join(rootFolder, modelFolder)

                configPath = os.path.join(modelPath, "config.json")
                options = load_config(configPath)

                dataset = LoadDataset(options)
                test = TestModule(configPath=configPath, dataset=dataset)
                test.eval()