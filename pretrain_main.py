from loadDataset import LoadDataset
from pretrainModule import PretrainModule
from fcgVectorize import FCGVectorize
from utils import load_config
import os



if __name__ == '__main__':

    options = load_config("./config.json")
    
    dataset = LoadDataset(options, pretrain=True)

    vectorize = FCGVectorize(options, dataset, pretrain=True)
    vectorize.node_embedding(dataset.rawDataset)
    
    pretrain = PretrainModule(options, dataset)
    
    pretrain.train()
    

    
        


