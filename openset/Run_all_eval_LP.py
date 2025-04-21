from logging import config
from pyexpat import model
import warnings, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset
from utils import load_config, save_config
from loadDataset import LoadDataset
from trainModule import TestModule
warnings.filterwarnings("ignore")

expList = ["5way_5shot", "5way_10shot"]
seeds = [6, 7, 10, 666, 11, 19, 22, 31, 42, 888]

for seed in seeds:
    
    rootFolder = f"/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_{seed}_openset"
    evalLogPath = os.path.join(rootFolder, "evalLog_openset.csv")
    if os.path.exists(evalLogPath):
        # delete the file if it exists
        os.remove(evalLogPath)

    for exp in expList:
        print("seed: ", seed)
        print("exp: ", exp)
        modelFolder = [d for d in os.listdir(rootFolder) if exp in d][0]
        print("modelFolder: ", modelFolder)
        modelPath = os.path.join(rootFolder, modelFolder)

        configPath = os.path.join(modelPath, "config.json")
        opt = load_config(configPath)
        opt["dataset"]["openset"] = True
        save_config(opt, configPath)

        dataset = LoadDataset(opt)
        test = TestModule(configPath=configPath, dataset = dataset)
        test.eval(mode="openset")