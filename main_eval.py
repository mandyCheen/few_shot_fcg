import argparse
from utils import load_config
from loadDataset import LoadDataset
from trainModule import TestModule
from fcgVectorize import FCGVectorize


def parse_args():
    parser = argparse.ArgumentParser(description="Few Shot FCG")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    options = load_config(args.config)
    
    dataset = LoadDataset(options)
    
    test = TestModule(args.config, dataset)
    test.eval()
