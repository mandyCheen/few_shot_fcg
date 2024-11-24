
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader 
from loadDataset import LoadDataset
from datetime import datetime
import torch.nn.functional as F
import os
from tqdm import tqdm
from train_utils import Training, load_GE_data
from loadDataset import LoadDataset
from graphSAGE import GraphClassifier

class PretrainModule(Training):
    def __init__(self, opt: dict, dataset: LoadDataset):
        self.opt = opt
        self.device = opt["settings"]["train"]["device"]
        self.device = torch.device(self.device)
        self.batch_size = opt["pretrain"]["batch_size"]

        self.parallel = opt["settings"]["train"]["parallel"]
        self.parallel_device = opt["settings"]["train"]["parallel_device"]

        self.embeddingFolder = os.path.join(opt["paths"]["data"]["embedding_folder"], dataset.datasetName, opt["settings"]["vectorize"]["node_embedding_method"])
        now = datetime.now()
        self.model_folder = opt["paths"]["model"]["pretrained_folder"] + "/" + opt["pretrain"]["name"] + "_" + now.strftime("%Y%m%d_%H%M")
        
        self.rawDataset = dataset.rawDataset
        self.trainDataset = dataset.trainData
        self.valDataset = dataset.valData
        self.trainGraph = []
        self.valGraph = []

        self.get_model(opt)
        self.get_optimizer(opt)
        self.get_loss_fn(opt)
        self.get_lr_scheduler(opt)
        self.get_pretrain_loader(opt)

        self.load_weights = opt["settings"]["model"]["load_weights"]
        if self.load_weights:
            self.resume_training()

        super().__init__(
            opt=self.opt,
            trainLoader=self.train_loader,
            valLoader=self.val_loader,
            model=self.model,
            loss_fn=self.criterion,
            optim=self.optimizer,
            scheduler=self.scheduler if self.lr_scheduler else None,
            device=self.device,
            model_path=self.model_folder
        )


    def resume_training(self):
        checkpoint = torch.load(self.load_weights)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.init_epoch = checkpoint["epoch"]
        print(f"Model loaded from epoch {self.init_epoch}")

    def get_optimizer(self, opt: dict):
        if opt["settings"]["train"]["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt["settings"]["train"]["lr"])
        elif opt["settings"]["train"]["optimizer"] == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt["settings"]["train"]["lr"])
        else:
            raise ValueError("Optimizer not supported")
    def get_loss_fn(self, opt: dict):
        if opt["settings"]["train"]["loss"] == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        elif opt["settings"]["train"]["loss"] == "MSELoss":
            self.criterion = nn.MSELoss().to(self.device)
        elif opt["settings"]["train"]["loss"] == "NLLLoss":
            self.criterion = nn.NLLLoss().to(self.device)
        else:
            raise ValueError("Loss function not supported")
    def get_lr_scheduler(self, opt: dict):
        if opt["settings"]["train"]["lr_scheduler"]["use"]:
            self.lr_scheduler = True
            if opt["settings"]["train"]["lr_scheduler"]["method"] == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt["settings"]["train"]["lr_scheduler"]["step_size"], gamma=opt["settings"]["train"]["lr_scheduler"]["gamma"])
            self.scheduler = scheduler

    def get_pretrain_loader(self, opt: dict):

        trainGraph, label = load_GE_data(self.trainDataset, self.embeddingFolder, opt["settings"]["vectorize"]["node_embedding_size"], os.path.join(self.embeddingFolder, "trainData.pkl"))
        self.train_loader = DataLoader(trainGraph, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        valGraph, label = load_GE_data(self.valDataset, self.embeddingFolder, opt["settings"]["vectorize"]["node_embedding_size"], os.path.join(self.embeddingFolder, "valData.pkl"))
        self.val_loader = DataLoader(valGraph, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    def get_model(self, opt: dict):
        info = self.opt["settings"]["model"]
        if info["model_name"] == "GraphSAGE":
            model = GraphClassifier(backbone_dim_in=info["input_size"], backbone_dim_h=info["hidden_size"], backbone_dim_out=info["output_size"], backbone_num_layers=info["num_layers"], num_classes=len(opt["dataset"]["pretrain_family"]))
        else:
            raise ValueError("Model not supported")
        if torch.cuda.is_available() and self.device == "cpu":
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        print(f"Device: {self.device}")

        if self.parallel and len(self.parallel_device) > 1:
            model = nn.DataParallel(model.to(self.device), device_ids=self.parallel_device, output_device=self.parallel_device[0]) 
        elif self.parallel:
            print("You didn't specify the device ids for parallel training")
            print("Using all available devices", torch.cuda.device_count(), "GPUs")       
            model = nn.DataParallel(model.to(self.device), output_device=0)
        else:
            model = model.to(self.device)

        self.model = model

    def train(self):
        training = Training(self.opt, self.train_loader, self.val_loader, self.model, self.criterion, self.optimizer, self.scheduler, self.device, self.model_folder)
        print("Start training...")
        self.run(backbone=True, mode="classification")
        print("Finish training")
