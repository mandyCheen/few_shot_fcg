import pandas as pd
import os
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Batch
from loadDataset import LoadDataset
from dataset import FcgSampler
from graphSAGE import GraphSAGE, GraphClassifier
from torch_geometric.loader import DataLoader # !
from loss import *
from datetime import datetime
from train_utils import Training, load_GE_data


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
        self.model_folder = opt["paths"]["model"]["model_folder"] + "/" + opt["settings"]["name"] + "_" + now.strftime("%Y%m%d_%H%M")
   
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
            self.model = GraphSAGE(dim_in=info["input_size"], dim_h=info["hidden_size"], dim_out=info["output_size"], num_layers=info["num_layers"], projection = True)
        else:
            raise ValueError("Model not supported")
        
        if(info["load_weights"]):
            checkpoint = torch.load(info["load_weights"])
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"Model loaded from {info['load_weights']}")
        
        if torch.cuda.is_available() and self.device == "cpu":
            print("CUDA is available, but you are using CPU")
        print(f"Device: {self.device}")

        if self.parallel and len(self.parallel_device) > 1:
            self.model = torch.nn.DataParallel(self.model.to(self.device), device_ids=self.parallel_device) 
        elif self.parallel:
            print("You didn't specify the device ids for parallel training")
            print("Using all available devices", torch.cuda.device_count(), "GPUs")       
            self.model = torch.nn.DataParallel(self.model.to(self.device))
        else:
            self.model = self.model.to(self.device)
    def get_loss_fn(self):
        if self.opt["settings"]["few_shot"]["method"] == "ProtoNet":
            loss_fn = ProtoLoss(self.opt)
        self.loss_fn = loss_fn
    def get_optimizer(self):
        if self.opt["settings"]["train"]["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt["settings"]["train"]["lr"])
        self.optim = optimizer
    def get_lr_scheduler(self):
        if self.opt["settings"]["train"]["lr_scheduler"]["method"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=self.opt["settings"]["train"]["lr_scheduler"]["step_size"], gamma=self.opt["settings"]["train"]["lr_scheduler"]["gamma"])
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
            self.valGraph, label =load_GE_data(self.trainDataset, self.embeddingFolder, self.embeddingSize, os.path.join(self.embeddingFolder, "valData.pkl"))
            val_sampler = FcgSampler(label, self.support_shots_test + self.query_shots_test, self.class_per_iter_test, self.iterations)
            self.valLoader = DataLoader(self.valGraph, batch_sampler=val_sampler, num_workers=4, collate_fn=collate_graphs)
        else:
            self.valLoader = None

        self.get_backbone()
        print(f"Model: {self.model}")
        self.get_loss_fn()
        print(f"Loss function: {self.loss_fn}")
        self.get_optimizer()
        print(f"Optimizer: {self.optim}") 
        if self.lr_scheduler:
            self.get_lr_scheduler()
            
        
        print("Finish setting up the training module")

    def train(self):
        print("Start training...")
        self.run()
        print("Finish training")
        
    def eval(self, model_path: str):
        
        print(f"Loading data from {self.embeddingFolder}...")
        print("Loading training data...")
        testGraph, label = load_GE_data(self.testDataset, self.embeddingFolder, self.embeddingSize, os.path.join(self.embeddingFolder, "testData.pkl"))
        sampler = FcgSampler(label, self.support_shots_test + self.query_shots_test, self.class_per_iter_test, self.iterations)
        testLoader = DataLoader(testGraph, batch_sampler=sampler, num_workers=4, collate_fn=collate_graphs)    

        print(f"Loading model from {model_path}")
        model = GraphSAGE(dim_in=self.opt["settings"]["model"]["input_size"], dim_h=self.opt["settings"]["model"]["hidden_size"], dim_out=self.opt["settings"]["model"]["output_size"], num_layers=self.opt["settings"]["model"]["num_layers"], projection = True)
        model.load_state_dict(torch.load(model_path)["model_state_dict"], strict=False)
        print("Best model loaded")
        model = model.to(self.device)
        
        print(f"Model: {model}")
        print("Start evaluation... (testing dataset)")
        self.testing(model, testLoader)

        print("Start evaluation... (validation dataset)")
        self.testing(model, self.valLoader)
        print("Finish evaluation")