import pandas as pd
import os
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Batch
from loadDataset import LoadDataset
from dataset import FcgSampler
from models import GraphSAGE, GCN
from torch_geometric.loader import DataLoader # !
from loss import *
from datetime import datetime
from train_utils import Training, Testing, load_GE_data
from utils import load_config, record_log

def collate_graphs(batch):
    return Batch.from_data_list(batch)

class TrainModule(Training):
    def __init__(self, opt: dict, dataset: LoadDataset):

        self.embeddingFolder = os.path.join(opt["paths"]["data"]["embedding_folder"], dataset.datasetName, opt["settings"]["vectorize"]["node_embedding_method"])
        self.rawDataset = dataset.rawDataset
        self.trainDataset = dataset.trainData
        self.valDataset = dataset.valData
        self.testDataset = dataset.testData
        self.trainGraph = []
        self.valGraph = []
        self.testGraph = []
        self.loss_fn = None
        self.opt = opt

        self.support_shots_train = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.query_shots_train = opt["settings"]["few_shot"]["train"]["query_shots"]
        self.class_per_iter_train = opt["settings"]["few_shot"]["train"]["class_per_iter"]
        self.support_shots_test = opt["settings"]["few_shot"]["test"]["support_shots"]
        self.query_shots_test = opt["settings"]["few_shot"]["test"]["query_shots"]
        self.class_per_iter_test = opt["settings"]["few_shot"]["test"]["class_per_iter"]

        self.iterations = opt["settings"]["train"]["iterations"]
        self.device = opt["settings"]["train"]["device"]
        self.embeddingSize = opt["settings"]["vectorize"]["node_embedding_size"]
        self.lr_scheduler = opt["settings"]["train"]["lr_scheduler"]["use"]

        self.parallel = opt["settings"]["train"]["parallel"]
        self.parallel_device = opt["settings"]["train"]["parallel_device"]

        now = datetime.now()
        self.model_folder = opt["paths"]["model"]["model_folder"] + "/" + opt["settings"]["name"] + "_" + now.strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.model_folder, exist_ok=True)
        self.log_file = self.model_folder + "/log.txt"
   
        self.setting()

        super().__init__(
            opt=self.opt,
            trainLoader=self.trainLoader,
            valLoader=self.valLoader,
            model=self.model,
            loss_fn=self.loss_fn,
            optim=self.optim,
            scheduler=self.scheduler if self.lr_scheduler else None,
            device=self.device,
            model_path=self.model_folder
        )

    def get_backbone(self):
        info = self.opt["settings"]["model"]
        if info["model_name"] == "GraphSAGE":
            self.model = GraphSAGE(dim_in=info["input_size"], dim_h=info["hidden_size"], dim_out=info["output_size"], num_layers=info["num_layers"], projection = info["projection"])
        elif info["model_name"] == "GCN":
            self.model = GCN(dim_in=info["input_size"], dim_h=info["hidden_size"], dim_out=info["output_size"], num_layers=info["num_layers"], projection = info["projection"])
        else:
            raise ValueError("Model not supported")
        
        if(info["load_weights"]):
            checkpoint = torch.load(info["load_weights"], map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"Model loaded from {info['load_weights']}")
            record_log(self.log_file, f"Model loaded from {info['load_weights']}\n")
            
        
        if torch.cuda.is_available() and self.device == "cpu":
            print("CUDA is available, but you are using CPU")
        print(f"Device: {self.device}")
        record_log(self.log_file, f"Device: {self.device}\n")

        if self.parallel and len(self.parallel_device) > 1:
            self.model = torch.nn.DataParallel(self.model.to(self.device), device_ids=self.parallel_device) 
        elif self.parallel:
            print("You didn't specify the device ids for parallel training")
            record_log(self.log_file, "You didn't specify the device ids for parallel training\n")
            print("Using all available devices", torch.cuda.device_count(), "GPUs")       
            record_log(self.log_file, f"Using all available devices {torch.cuda.device_count()} GPUs\n")
            self.model = torch.nn.DataParallel(self.model.to(self.device))
        else:
            self.model = self.model.to(self.device)
    def get_loss_fn(self):
        if self.opt["settings"]["few_shot"]["method"] == "ProtoNet":
            loss_fn = ProtoLoss(self.opt)
        elif self.opt["settings"]["few_shot"]["method"] == "NnNet":
            loss_fn = NnLoss(self.opt)
        else:
            raise ValueError("Loss method not supported")
        self.loss_fn = loss_fn
    def get_optimizer(self):
        if self.opt["settings"]["train"]["optimizer"] == "Adam":
            if self.opt["settings"]["model"]["projection"]:
                optimizer = torch.optim.Adam([
                    {
                        "params": self.model.output_proj.parameters(),
                        "lr": self.opt["settings"]["train"]["projection_lr"]
                    },
                    {
                        "params": (p for n, p in self.model.named_parameters()
                                   if not n.startswith("output_proj")),
                        "lr": self.opt["settings"]["train"]["lr"]
                    }
                ])
            else:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt["settings"]["train"]["lr"])
        elif self.opt["settings"]["train"]["optimizer"] == "AdamW":
            if self.opt["settings"]["model"]["projection"]:
                optimizer = torch.optim.AdamW([
                    {
                        "params": self.model.output_proj.parameters(),
                        "lr": self.opt["settings"]["train"]["projection_lr"]
                    },
                    {
                        "params": (p for n, p in self.model.named_parameters()
                                   if not n.startswith("output_proj")),
                        "lr": self.opt["settings"]["train"]["lr"]
                    }
                ])
            else:
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.opt["settings"]["train"]["lr"])
        elif self.opt["settings"]["train"]["optimizer"] == "SGD":
            if self.opt["settings"]["model"]["projection"]:
                optimizer = torch.optim.SGD([
                    {
                        "params": self.model.output_proj.parameters(),
                        "lr": self.opt["settings"]["train"]["projection_lr"]
                    },
                    {
                        "params": (p for n, p in self.model.named_parameters()
                                   if not n.startswith("output_proj")),
                        "lr": self.opt["settings"]["train"]["lr"]
                    }
                ])
            else:
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt["settings"]["train"]["lr"])
        else:
            raise ValueError("Optimizer not supported")
        self.optim = optimizer
        
    def get_lr_scheduler(self):
        if self.opt["settings"]["train"]["lr_scheduler"]["method"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=self.opt["settings"]["train"]["lr_scheduler"]["step_size"], gamma=self.opt["settings"]["train"]["lr_scheduler"]["gamma"])
        elif self.opt["settings"]["train"]["lr_scheduler"]["method"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=self.opt["settings"]["train"]["lr_scheduler"]["factor"], patience=self.opt["settings"]["train"]["lr_scheduler"]["patience"])
        else:
            raise ValueError("LR scheduler not supported")
        self.scheduler = scheduler

    def setting(self):
        print("Setting up the training module...")
        print(f"Loading data from {self.embeddingFolder}...")
        print("Loading training data...")
        self.trainGraph, label = load_GE_data(self.trainDataset, self.embeddingFolder, self.embeddingSize, os.path.join(self.embeddingFolder, "trainData.pkl"))
        sampler = FcgSampler(label, self.support_shots_train + self.query_shots_train, self.class_per_iter_train, self.iterations)
        self.trainLoader = DataLoader(self.trainGraph, batch_sampler=sampler, num_workers=4, collate_fn=collate_graphs)    
        if self.valDataset is not None:
            print("Loading validation data...")
            self.valGraph, label =load_GE_data(self.valDataset, self.embeddingFolder, self.embeddingSize, os.path.join(self.embeddingFolder, "valData.pkl"))
            val_sampler = FcgSampler(label, self.support_shots_test + self.query_shots_test, self.class_per_iter_test, self.iterations)
            self.valLoader = DataLoader(self.valGraph, batch_sampler=val_sampler, num_workers=4, collate_fn=collate_graphs)
        else:
            self.valLoader = None

        self.get_backbone()
        print(f"Model: {self.model}")
        record_log(self.log_file, f"Model: {self.model}\n")
        self.get_loss_fn()
        print(f"Loss function: {self.loss_fn}")
        record_log(self.log_file, f"Loss function: {self.loss_fn}\n")
        self.get_optimizer()
        print(f"Optimizer: {self.optim}") 
        record_log(self.log_file, f"Optimizer: {self.optim}\n")
        if self.lr_scheduler:
            self.get_lr_scheduler()
            
        
        print("Finish setting up the training module")

    def train(self):
        print("Start training...")
        record_log(self.log_file, "Start training...\n")
        self.run()
        print("Finish training")
        record_log(self.log_file, "Finish training\n")

            
class TestModule(Testing):
    def __init__(self, configPath: str, dataset: LoadDataset):
        
        opt = load_config(configPath)
        self.model_folder = os.path.dirname(configPath)
        self.embeddingFolder = os.path.join(opt["paths"]["data"]["embedding_folder"], dataset.datasetName, opt["settings"]["vectorize"]["node_embedding_method"])
        self.testDataset = dataset.testData
        self.valDataset = dataset.valData
        self.datasetName = dataset.datasetName
        self.testGraph = []
        self.loss_fn = None
        self.opt = opt

        self.support_shots_test = opt["settings"]["few_shot"]["test"]["support_shots"]
        self.query_shots_test = opt["settings"]["few_shot"]["test"]["query_shots"]
        self.class_per_iter_test = opt["settings"]["few_shot"]["test"]["class_per_iter"]

        self.iterations = opt["settings"]["train"]["iterations"]
        self.device = opt["settings"]["train"]["device"]
        self.embeddingSize = opt["settings"]["vectorize"]["node_embedding_size"]

        self.setting()

        super().__init__(
            device=self.device,
            loss_fn=self.loss_fn,
        )
        
    def get_loss_fn(self):
        if self.opt["settings"]["few_shot"]["method"] == "ProtoNet":
            loss_fn = ProtoLoss(self.opt)
        elif self.opt["settings"]["few_shot"]["method"] == "NnNet":
            loss_fn = NnLoss(self.opt)
        else:
            raise ValueError("Loss method not supported")
        self.loss_fn = loss_fn
    
    def setting(self):
        print("Setting up the testing module...")
        print(f"Loading data from {self.embeddingFolder}...")

        if self.valDataset is not None:
            print("Loading validation data...")
            self.valGraph, label =load_GE_data(self.valDataset, self.embeddingFolder, self.embeddingSize, os.path.join(self.embeddingFolder, "valData.pkl"))
            val_sampler = FcgSampler(label, self.support_shots_test + self.query_shots_test, self.class_per_iter_test, self.iterations)
            self.valLoader = DataLoader(self.valGraph, batch_sampler=val_sampler, num_workers=4, collate_fn=collate_graphs)
        else:
            self.valLoader = None
    
        print("Loading testing data...")
        testGraph, label = load_GE_data(self.testDataset, self.embeddingFolder, self.embeddingSize, os.path.join(self.embeddingFolder, "testData.pkl"))
        sampler = FcgSampler(label, self.support_shots_test + self.query_shots_test, self.class_per_iter_test, self.iterations)
        self.testLoader = DataLoader(testGraph, batch_sampler=sampler, num_workers=4, collate_fn=collate_graphs)    

        self.model = GraphSAGE(dim_in=self.opt["settings"]["model"]["input_size"], dim_h=self.opt["settings"]["model"]["hidden_size"], dim_out=self.opt["settings"]["model"]["output_size"], num_layers=self.opt["settings"]["model"]["num_layers"], projection = self.opt["settings"]["model"]["projection"])
        self.pretrainModel = GraphSAGE(dim_in=self.opt["settings"]["model"]["input_size"], dim_h=self.opt["settings"]["model"]["hidden_size"], dim_out=self.opt["settings"]["model"]["output_size"], num_layers=self.opt["settings"]["model"]["num_layers"], projection = False)
        
        self.get_loss_fn()
        
        print("Finish setting up the testing module")
        
    
    def eval(self, model_path: str = None):
        
        if model_path is None:
            print("Model path is not provided. Using the best model...")
            model_path = os.path.join(self.model_folder, [f for f in os.listdir(self.model_folder) if "best" in f][0])
            
        evalFolder = os.path.dirname(model_path)
        logFolder = self.opt["paths"]["model"]["model_folder"]

        print("Copying split files...")
        splitFolder = self.opt["paths"]["data"]["split_folder"]
        splitFiles = [f for f in os.listdir(splitFolder) if f.endswith(f"{self.datasetName}.txt")]
        for f in splitFiles:
            src = os.path.join(splitFolder, f)
            dst = os.path.join(evalFolder, f)
            os.system(f"cp {src} {dst}")
        
        print("Record evaluation log...")
        evalLogPath = os.path.join(logFolder, "evalLog.csv")
        if not os.path.exists(evalLogPath):
            with open(evalLogPath, "w") as f:
                f.write("timestamp, folderName, model, test_acc, val_acc\n")
    
        print(f"Loading model from {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device)["model_state_dict"])
        print("Best model loaded")
        self.model = self.model.to(self.device)
        
        print(f"Model: {self.model}")
        print("Start evaluation... (testing dataset)")
        testAcc = self.testing(self.model, self.testLoader)

        print("Start evaluation... (validation dataset)")
        valAcc = self.testing(self.model, self.valLoader)

        with open(evalLogPath, "a") as f:
            f.write(f"{datetime.now()}, {os.path.basename(evalFolder)}, {os.path.basename(model_path)}, {testAcc}, {valAcc}\n")

        pretrainModelPath = self.opt["settings"]["model"]["load_weights"]
        if pretrainModelPath != "":
            self.pretrainModel.load_state_dict(torch.load(pretrainModelPath, map_location=self.device)["model_state_dict"], strict=False)
        print(f"Ablation evaluation... (testing dataset)")
        self.pretrainModel = self.pretrainModel.to(self.device)
        
        testAccPretrain = self.testing(self.pretrainModel, self.testLoader)
        print(f"Ablation evaluation... (validation dataset)")
        valAccPretrain = self.testing(self.pretrainModel, self.valLoader)
        
        print("Finish evaluation")
        
        with open(evalLogPath, "a") as f:
            f.write(f"{datetime.now()}, {os.path.basename(evalFolder)}, {os.path.basename(pretrainModelPath)}, {testAccPretrain}, {valAccPretrain}\n")
            