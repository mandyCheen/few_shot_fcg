import warnings
from utils import load_config, save_config
from loadDataset import LoadDataset
from fcgVectorize import FCGVectorize
from trainModule import TrainModule
from trainModule import TestModule
import os
import torch
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
warnings.filterwarnings("ignore")

# # 設定GPU記憶體使用上限
# def limit_gpu_memory(max_memory_mb=6144):
#     if torch.cuda.is_available():
#         # 獲取當前設備
#         device = torch.cuda.current_device()
#         # 設定記憶體上限
#         torch.cuda.set_per_process_memory_fraction(max_memory_mb / torch.cuda.get_device_properties(device).total_memory)
#         # 啟用記憶體快取釋放
#         torch.cuda.empty_cache()
#         print(f"已將GPU記憶體使用上限設為{max_memory_mb}MB")
#     else:
#         print("無可用GPU")

# # 調用函數設定上限（調整為您需要的值）
# limit_gpu_memory(6144)  

expList = ["5way_5shot", "5way_10shot", "10way_5shot", "10way_10shot"]
seeds = [6, 7, 10, 666, 11, 19, 22, 31, 42, 888]
## always with pretrain
for seed in seeds:
    for exp in expList:
        options = load_config("./config/config_SoftNn.json")
        options["settings"]["name"] = exp+"_SoftNnNet_without_pretrain"
        shots = int(exp.split("_")[1].split("shot")[0])
        way = int(exp.split("_")[0].split("way")[0])
        options["settings"]["few_shot"]["train"]["support_shots"] = shots
        options["settings"]["few_shot"]["train"]["query_shots"] = 20 - shots
        options["settings"]["few_shot"]["test"]["support_shots"] = shots
        options["settings"]["few_shot"]["test"]["query_shots"] = 20 - shots
        options["settings"]["few_shot"]["train"]["class_per_iter"] = way
        options["settings"]["few_shot"]["test"]["class_per_iter"] = way
        options["settings"]["seed"] = seed
        save_config(options, "./config/config_SoftNn.json")

        dataset = LoadDataset(options, pretrain=False)
        # vectorizer = FCGVectorize(options, dataset)
        # vectorizer.node_embedding(dataset.rawDataset)
        trainModule = TrainModule(options, dataset)
        trainModule.train()

        test = TestModule(os.path.join(trainModule.model_folder, "config.json"), dataset, options)
        test.eval()
        torch.cuda.empty_cache()
        # test.eval_ablation()