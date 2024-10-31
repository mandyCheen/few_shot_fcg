
import pandas as pd
import numpy as np
import os

class LoadDataset:
    def __init__(self, opt: dict):
        rawDatasetPath = os.path.join(opt["paths"]["data"]["csv_folder"], opt["dataset"]["raw"])
        self.rawDataset = pd.read_csv(rawDatasetPath)
        self.seed = opt["settings"]["seed"]
        self.cpuArch = opt["dataset"]["cpu_arch"]
        self.datasetSplitFolder = opt["paths"]["data"]["split_folder"]
        self.val = opt["settings"]["train"]["validation"]
        self.splitByCpu = opt["dataset"]["split_by_cpu"]
        splitByCpu = "_splitByCpu" if self.splitByCpu else ""
        self.reverseTool = opt["dataset"]["reverse_tool"]
        val = "_withVal" if self.val else ""
        self.familyCpuList = self.rawDataset.groupby("family")["CPU"].unique().to_dict()
        self.datasetName = f"{self.cpuArch}{splitByCpu}{val}_{self.reverseTool}"

        self.trainData, self.testData, self.valData = self.load_all_datasets()
    
    def write_split_dataset(self, mode, familyList) -> None:
        if not os.path.exists(self.datasetSplitFolder):
            os.makedirs(self.datasetSplitFolder)
        filepath = f"{self.datasetSplitFolder}/{mode}_{self.datasetName}.txt"
        with open(filepath, "w") as f:
            for family in familyList:
                f.write(f"{family}\n")

    def get_split_dataset(self) -> None: 
        if self.val:
            testNum, valNum = 10, 10
        else:
            testNum, valNum = 10, 0
        familyList = self.rawDataset["family"].unique()
        np.random.seed(self.seed)
        np.random.shuffle(familyList)
        
        trainFamily = familyList[:int(len(familyList) - testNum - valNum)]
        testFamily = familyList[int(len(familyList) - testNum - valNum):int(len(familyList) - valNum)]
        valFamily = familyList[int(len(familyList) - valNum):]
        
        self.write_split_dataset("train", trainFamily)
        self.write_split_dataset("test", testFamily)
        if self.val:
            self.write_split_dataset("val", valFamily)
    
    def get_split_dataset_by_cpu(self) -> None:
        if self.val:
            testRate, valRate = 0.75, 0.25
        else:
            testRate, valRate = 1, 0.0
        cpuCounts = self.rawDataset["CPU"].value_counts()
        print(f"CPU counts: {cpuCounts}")

        # train cpu is the cpu with the most samples
        trainCpu = cpuCounts.idxmax()
        testCpu = cpuCounts.index[cpuCounts.index != trainCpu][0]

        trainFamilyList = self.rawDataset[self.rawDataset["CPU"] == trainCpu]["family"].unique()
        testFamilyList = self.rawDataset[self.rawDataset["CPU"] == testCpu]["family"].unique()

        np.random.seed(self.seed)
        np.random.shuffle(testFamilyList)

        trainFamily = trainFamilyList
        testFamily = testFamilyList[:int(len(testFamilyList) * testRate)]
        valFamily = testFamilyList[int(len(testFamilyList) * testRate):]
        self.write_split_dataset("train", trainFamily)
        self.write_split_dataset("test", testFamily)
        if self.val:
            self.write_split_dataset("val", valFamily)

    def load_dataset(self, mode) -> pd.DataFrame:
        filepath = f"{self.datasetSplitFolder}/{mode}_{self.datasetName}.txt"
        if self.splitByCpu:
            if not os.path.exists(filepath):
                print(f"Split dataset for {mode} does not exist, creating split dataset...")
                self.get_split_dataset_by_cpu()        
        else:
            if not os.path.exists(filepath):
                print(f"Split dataset for {mode} does not exist, creating split dataset...")
                self.get_split_dataset()
        
        with open(filepath, "r") as f:
            familyList = f.read().splitlines()
        
        data = self.rawDataset[self.rawDataset["family"].isin(familyList)]
        data = data.reset_index(drop=True)
        print(f"{mode} dataset shape: {data.shape}")
        print(f"{mode} dataset family number: {len(data['family'].unique())}")
        return data

    def load_all_datasets(self) -> pd.DataFrame:
        print("Loading all datasets...")
        trainData = self.load_dataset("train")
        testData = self.load_dataset("test")
        valData = self.load_dataset("val") if self.val else None
        return trainData, testData, valData