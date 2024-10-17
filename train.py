import pandas as pd
import os
from tqdm import tqdm
import pickle
import torch
import numpy as np
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, Batch
from loadDataset import LoadDataset
import sklearn.preprocessing as labelEncoder
from dataset import FcgSampler
from graphSAGE import SAGE
from torch.utils.data import DataLoader
from loss import ProtoLoss


class TrainModule():

    def __init__(self, opt: dict, dataset: LoadDataset):
        self.dataset = dataset
        self.nodeEmbedding = opt["settings"]["vectorize"]["node_embedding_method"]
        self.embeddingFolder = os.path.join(opt["paths"]["data"]["embedding_folder"], self.nodeEmbedding)
        self.rawDataset = dataset.rawDataset
        self.trainDataset = dataset.trainData
        self.valDataset = dataset.valData
        self.trainGraph = []
        self.valGraph = []
        self.loss_fn = None
        self.opt = opt

        self.model_name = opt["settings"]["model"]["model_name"]
        self.input_size = opt["settings"]["model"]["input_size"]
        self.hidden_size = opt["settings"]["model"]["hidden_size"]
        self.output_size = opt["settings"]["model"]["output_size"]

        self.method = opt["settings"]["few_shot"]["method"]
        self.support_shots = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.class_per_iter = opt["settings"]["few_shot"]["train"]["class_per_iter"]
        self.iterations = opt["settings"]["train"]["iterations"]
        self.cuda = opt["settings"]["train"]["device"]
        self.training = opt["settings"]["train"]["training"]
        self.lr = opt["settings"]["train"]["lr"]
        self.lr_scheduler = opt["settings"]["train"]["lr_scheduler"]
        self.optimizer = opt["settings"]["train"]["optimizer"]
        self.epochs = opt["settings"]["train"]["num_epochs"]
        self.iterations = opt["settings"]["train"]["iterations"]
        self.save_model = opt["settings"]["train"]["save_model"]
        self.model_path = opt["paths"]["data"]["model_folder"]

        self.seed = opt["settings"]["seed"]
        self.parallel = opt["settings"]["train"]["parallel"]
        self.parallel_device = opt["settings"]["train"]["parallel_device"]

    def load_GE_data(self, dataset: pd.DataFrame):
        labels = []
        graphList = []
        for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            cpu = row["CPU"]
            family = row["family"]
            fileName = row["file_name"]
            filePath = f"{self.embeddingFolder}/{cpu}/{family}/{fileName}.gpickle"
            with open(filePath, "rb") as f:
                fcg = pickle.load(f)
            torch_data = from_networkx(fcg)
            labels.append(family)
            graphList.append(torch_data)
        le = labelEncoder.LabelEncoder()
        le.fit(labels)
        labels_ = le.transform(labels)
        for i, data in enumerate(graphList):
            data.y = torch.tensor(labels_[i])
        
        return graphList, labels_

    def get_model(self):
        if self.model_name == "GraphSAGE":
            model = SAGE(in_channels=self.input_size, hidden_channels=self.hidden_size, out_channels=self.output_size)
        return model
    def get_loss_fn(self):
        if self.method == "ProtoNet":
            loss_fn = ProtoLoss(self.opt)
        return loss_fn
    def get_optimizer(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    def get_lr_scheduler(self):
        if self.lr_scheduler["method"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=self.lr_scheduler["step_size"], gamma=self.lr_scheduler["gamma"])
        return scheduler

    def setting(self):
        print("Setting up the training module...")
        print(f"Loading data from {self.embeddingFolder}...")
        print("Loading training data...")
        self.trainGraph, label = self.load_GE_data(self.trainDataset)
        sampler = FcgSampler(label, self.support_shots, self.class_per_iter, self.iterations)
        self.trainLoader = DataLoader(self.trainGraph, batch_sampler=sampler, num_workers=4)    
        if self.valDataset is not None:
            print("Loading validation data...")
            self.valGraph, label = self.load_GE_data(self.valDataset)
            val_sampler = FcgSampler(label, self.support_shots, self.class_per_iter, self.iterations)
            self.valLoader = DataLoader(self.valGraph, batch_sampler=val_sampler, num_workers=4)
        else:
            self.valLoader = None
        self.model = self.get_model()

        print(f"Model: {self.model}")
        self.loss_fn = self.get_loss_fn()
        print(f"Loss function: {self.loss_fn}")
        self.optim = self.get_optimizer()
        print(f"Optimizer: {self.optim}") 
        if self.lr_scheduler["use"]:
            self.scheduler = self.get_lr_scheduler()
        self.device = torch.device(self.cuda)
        self.model.to(self.device)
        if self.parallel:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.parallel_device)
        
        print("Finish setting up the training module")



    def train(self):
        print("Start to train...")

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        best_train_acc = 0
        best_val_acc = 0

        for epoch in range(self.epochs):
            print('====== Epoch: {} ======'.format(epoch))
            self.model.train()
            for i, (data, labels) in enumerate(self.trainLoader):
                self.optim.zero_grad()
                data = data.squeeze(1)
                data = data.float()
                data = data.to(self.device)
                labels = labels.to(self.device)
                embeddings = self.model(data)
                loss, acc = self.loss_fn(embeddings, labels)
                loss.backward()
                self.optim.step()   
                train_loss.append(loss.item())
                train_acc.append(acc.item()) 
            if self.lr_scheduler["use"]:
                self.scheduler.step()
            avg_loss = np.mean(train_loss[-(self.iterations):])
            avg_acc = np.mean(train_acc[-(self.iterations):])
            postfix = ' (Best)' if avg_acc >= best_train_acc else ' (Best: {})'.format(best_train_acc)
            print('Avg Train Loss: {}, Avg Train Acc: {}{}'.format(avg_loss, avg_acc, postfix))
            if self.valLoader is not None: 
                self.model.eval()
                for i, (data, labels) in enumerate(self.valLoader):
                    data = data.squeeze(1)
                    data = data.float()
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    model_output = self.model(data)
                    loss, acc= self.loss_fn(embeddings, labels)
                    val_loss.append(loss.item())
                    val_acc.append(acc.item())
                avg_loss = np.mean(val_loss[-(self.iterations):])
                avg_acc = np.mean(val_acc[-(self.iterations):])
                postfix = ' (Best)' if avg_acc >= best_val_acc else ' (Best: {})'.format(best_val_acc)
                print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))
                if avg_acc >= best_val_acc:
                    best_val_acc = avg_acc
                    best_train_acc = avg_acc
                    if avg_acc >= best_train_acc:
                        best_train_acc = avg_acc
                    if self.save_model:
                        torch.save(self.model.state_dict(), self.model_path)
            else:
                if avg_acc >= best_train_acc:
                    best_train_acc = avg_acc
                    if self.save_model:
                        torch.save(self.model.state_dict(), self.model_path)
            
            
        print("Finish training")
        return True
