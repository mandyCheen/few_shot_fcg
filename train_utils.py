import os
from utils import save_checkpoint, save_config, record_log, save_model_architecture
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import pickle
from torch_geometric.utils.convert import from_networkx
import sklearn.preprocessing as labelEncoder
from datetime import datetime

class Training:
    def __init__(self, opt, trainLoader, valLoader, model, loss_fn, optim, scheduler, device):

        self.opt = opt
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.scheduler = scheduler
        self.device = device

        self.epochs = opt["settings"]["train"]["num_epochs"]

        self.save_model = opt["settings"]["train"]["save_model"]
        self.early_stopping = opt["settings"]["train"]["early_stopping"]["use"]
        self.early_stopping_patience = opt["settings"]["train"]["early_stopping"]["patience"]

        now = datetime.now()
        self.model_folder = opt["paths"]["model"]["pretrained_folder"] + "/" + opt["pretrain"]["name"] + "_" + now.strftime("%Y%m%d_%H%M")
        self.log_file = self.model_folder + "/log.txt"



    def end_of_epoch(self, avg_acc, best_acc, epoch, patience, save_backbone):

        if avg_acc >= best_acc:
            best_acc = avg_acc
            if self.save_model:
                save_checkpoint(model_state=self.model.state_dict(), optim_state=self.optim.state_dict(), sche_state=self.scheduler.state_dict(), is_best=True, epoch=epoch+1, checkpoint=self.model_folder)
            if save_backbone:
                save_checkpoint(model_state=self.model.backbone.state_dict(), optim_state=self.optim.state_dict(), sche_state=self.scheduler.state_dict(), is_best=True, epoch=epoch+1, checkpoint=self.model_folder, backbone=True)
            if self.early_stopping:
                patience = 0
        else:
            if self.save_model and (epoch+1) % 10 == 0:
                save_checkpoint(model_state=self.model.state_dict(), optim_state=self.optim.state_dict(), sche_state=self.scheduler.state_dict(), is_best=False, epoch=epoch+1, checkpoint=self.model_folder)
            if save_backbone and (epoch+1) % 10 == 0:
                save_checkpoint(model_state=self.model.backbone.state_dict(), optim_state=self.optim.state_dict(), sche_state=self.scheduler.state_dict(), is_best=False, epoch=epoch+1, checkpoint=self.model_folder, backbone=True)
            if self.early_stopping:
                patience += 1
                if patience == self.early_stopping_patience:
                    print("Early stopping")
                    record_log(self.log_file, "Early stopping in epoch {}\n".format(epoch+1))
                    
                    return best_acc, patience, True
        if self.early_stopping:
            print(f"Patience: {patience}/{self.early_stopping_patience}")
        return best_acc, patience, False


    def run(self, backbone=False, mode = "few-shot"):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        best_train_acc = 0
        best_val_acc = 0
        patience = 0
        stop = False

        os.makedirs(self.model_folder, exist_ok=True)
        # save config
        save_config(self.opt, self.model_folder + "/config.json")
        # save model architecture
        save_model_architecture(self.model, self.model_folder + "/model_architecture.txt")

        for epoch in range(self.epochs):
            self.model.train()

            with tqdm(self.trainLoader, desc=f"Epoch {epoch+1}/{self.epochs} (Training)") as pbar:
                for data in pbar:
                    self.optim.zero_grad()
                    data = data.to(self.device)
                    predicts = self.model(data)
                    if mode == "few-shot":
                        loss, acc = self.loss_fn(predicts, data.y)
                    elif mode == "pretrain":
                        loss = self.loss_fn(predicts, data.y)
                        acc = torch.sum(torch.argmax(predicts, dim=1) == data.y) / len(data.y)
                    loss.backward()
                    self.optim.step()   
                    
                    # Update progress bar with current batch metrics
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{acc.item():.4f}',
                    })

                    train_loss.append(loss.item())
                    train_acc.append(acc.item()) 

                if self.scheduler:
                    self.scheduler.step()
                avg_loss = np.mean(train_loss)
                avg_acc = np.mean(train_acc)
                postfix = ' (Best)' if avg_acc >= best_train_acc else f' (Best: {best_train_acc:.4f})'
                content = f'Avg Train Loss: {avg_loss:.4f}, Avg Train Acc: {avg_acc:.4f}{postfix}'
                if self.valLoader is not None and avg_acc >= best_train_acc:
                    best_train_acc = avg_acc
                print(content)
                record_log(self.log_file, f"Epoch {epoch+1}/{self.epochs}: {content}\n")

            if self.valLoader is not None: 
                self.model.eval()                
                with tqdm(self.valLoader, desc=f"Epoch {epoch+1}/{self.epochs} (Validation)") as pbar:
                    for data in pbar:
                        data = data.to(self.device)
                        with torch.no_grad():
                            model_output = self.model(data)
                            if mode == "few-shot":
                                loss, acc = self.loss_fn(model_output, data.y)
                            elif mode == "pretrain":
                                loss= self.loss_fn(model_output, data.y)
                                acc = torch.sum(torch.argmax(model_output, dim=1) == data.y) / len(data.y)
                        val_loss.append(loss.item())
                        val_acc.append(acc.item())
                        
                        # Update progress bar with current batch metrics
                        pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'acc': f'{acc.item():.4f}',
                        })
                    avg_loss = np.mean(val_loss)
                    avg_acc = np.mean(val_acc)
                    postfix = ' (Best)' if avg_acc >= best_val_acc else f' (Best: {best_val_acc:.4f})'
                    content = f'Avg Val Loss: {avg_loss:.4f}, Avg Val Acc: {avg_acc:.4f}{postfix}'
                    print(content)
                    record_log(self.log_file, f"Epoch {epoch+1}/{self.epochs}: {content}\n")
                    
                    best_val_acc, patience, stop = self.end_of_epoch(avg_acc, best_val_acc, epoch, patience, backbone)
            else:
                best_train_acc, patience, stop = self.end_of_epoch(avg_acc, best_train_acc, epoch, patience, backbone)
            if stop:
                break
        return True


def load_GE_data(dataset: pd.DataFrame, embeddingFolder: str, embeddingSize: int, dataPath: str):
    if not os.path.exists(dataPath):
        labels = []
        graphList = []
        for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            cpu = row["CPU"]
            family = row["family"]
            fileName = row["file_name"]
            filePath = f"{embeddingFolder}/{cpu}/{family}/{fileName}.gpickle"
            with open(filePath, "rb") as f:
                fcg = pickle.load(f)
            for node in fcg.nodes:
                if not isinstance(fcg.nodes[node]["x"], torch.Tensor):
                    fcg.nodes[node]["x"] = torch.tensor(fcg.nodes[node]["x"], dtype=torch.float)
                if len(fcg.nodes[node]["x"]) == 0:
                    fcg.nodes[node]["x"] = torch.zeros(embeddingSize)
            torch_data = from_networkx(fcg)
            labels.append(family)
            graphList.append(torch_data)
        le = labelEncoder.LabelEncoder()
        le.fit(labels)
        labels_ = le.transform(labels)
        for i, data in enumerate(graphList):
            data.y = torch.tensor(labels_[i])
        print(f"Saving data to {dataPath}")
        with open(dataPath, "wb") as f:
            pickle.dump([graphList, labels_], f)
    else:
        print(f"Loading data from {dataPath}...")
        with open(dataPath, "rb") as f:
            graphList, labels_ = pickle.load(f)
    
    return graphList, labels_