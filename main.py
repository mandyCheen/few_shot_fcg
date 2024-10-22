from utils import load_config
from loadDataset import LoadDataset
from fcgVectorize import FCGVectorize
from train import TrainModule

if __name__ == "__main__":
    options = load_config("./config.json")
    dataset = LoadDataset(options)
    vectorizer = FCGVectorize(options, dataset)
    # node embedding for raw data
    vectorizer.node_embedding(dataset.rawDataset)
    trainModule = TrainModule(options, dataset)
    trainModule.setting()
    trainModule.train()