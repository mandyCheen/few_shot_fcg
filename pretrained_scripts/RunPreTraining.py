from loadDatasetPretrained import LoadDatasetPretrained
from Projects.few_shot_fcg.pretrained_scripts.pretrainModule import PretrainModule
from fcgVectorize import FCGVectorize
from utils import load_config
import os



if __name__ == '__main__':

    options = load_config("./config_pretrain.json")
    dataset = LoadDatasetPretrained(options)
    vectorize = FCGVectorize(options, dataset, pretrain=True)
    vectorize.node_embedding(dataset.rawDataset)
    pretrain = PretrainModule(options, dataset)
    pretrain.train()
