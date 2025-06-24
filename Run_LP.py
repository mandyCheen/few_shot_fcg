import warnings, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, save_config
from loadDataset import LoadDataset
from fcgVectorize import FCGVectorize
from trainModule import TrainModule
from trainModule import TestModule
import torch

warnings.filterwarnings("ignore")

options = load_config("/home/manying/Projects/fcgFewShot/checkpoints/arm_withVal_ghidra_7_openset/10way_10shot_LabelPropagation_alpha0.7_k20_gcn_20250623_043600/config.json")

dataset = LoadDataset(options)

trainModule = TrainModule(options, dataset)
trainModule.train()

test = TestModule(os.path.join(trainModule.model_folder, "config.json"), dataset)
test.eval()
