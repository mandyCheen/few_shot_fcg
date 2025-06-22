from logging import config
from pyexpat import model
import warnings, sys, os
import dataset
from utils import load_config, save_config
from loadDataset import LoadDataset
from trainModule import TestModule
warnings.filterwarnings("ignore")

expList = ["5way_5shot", "5way_10shot", "10way_5shot", "10way_10shot"]

rootFolder = "/home/manying/Projects/fcgFewShot/checkpoints/i386_withVal_ghidra_42_openset"
modelRootFolder = "/home/manying/Projects/fcgFewShot/checkpoints/x86_64_withVal_withPretrain_ghidra_6_modelExp"
for exp in expList:
    print("exp: ", exp)
    for folder in os.listdir(modelRootFolder):
        prefix = exp + "_LabelPropagation_alpha0.7_k20_GCN_layers3_128_256_128_"
        if folder.startswith(prefix):
            modelFolder = os.path.join(modelRootFolder, folder)
            model_path = os.path.join(modelFolder, [f for f in os.listdir(modelFolder) if "best" in f][0]) 
            break
    for folder in os.listdir(rootFolder):
        if os.path.isdir(os.path.join(rootFolder, folder)):
            if exp not in folder:
                continue
            print("modelFolder: ", folder)
            modelFolder = folder
            modelPath = os.path.join(rootFolder, modelFolder)

            configPath = os.path.join(modelPath, "config.json")
            options = load_config(configPath)
            options["settings"]["few_shot"]["parameters"]["relation_layer"] = 2
            options["settings"]["few_shot"]["parameters"]["relation_model"] = "GCN"
            options["settings"]["train"]["device"] = "cuda:2"

            save_config(options, configPath)
            dataset = LoadDataset(options)
            test = TestModule(configPath=configPath, dataset=dataset)
            test.eval(model_path=model_path)